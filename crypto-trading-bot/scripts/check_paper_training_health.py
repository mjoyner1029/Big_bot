#!/usr/bin/env python3
"""Check whether the paper-training bot is still alive and healthy.

Verifies:
  - the launchd job is loaded and running
  - exactly one paper-training wrapper process exists
  - the inner trading runner is alive
  - the bot log and trade log are fresh

Exit codes:
  0 = healthy
  1 = unhealthy
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL = "com.bigbot.paper-training"
DEFAULT_TRADE_LOG = PROJECT_ROOT / "logs" / "trade_log_paper_live.csv"
DEFAULT_BOT_LOG = PROJECT_ROOT / "logs" / "bot_paper_training.log"
DEFAULT_LAUNCHD_STDOUT = PROJECT_ROOT / "logs" / "launchd_stdout.log"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _file_age_minutes(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (_utcnow() - mtime).total_seconds() / 60.0


def _run(command: List[str]) -> Tuple[int, str]:
    proc = subprocess.run(command, capture_output=True, text=True)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output.strip()


def _launchctl_print(label: str) -> str:
    code, output = _run(["launchctl", "print", f"gui/{os.getuid()}/{label}"])
    return output if code == 0 else ""


def _process_matches(pattern: str) -> List[str]:
    code, output = _run(["ps", "-axo", "pid,ppid,command"])
    if code != 0:
        return []
    matches = []
    for line in output.splitlines():
        if pattern in line and "grep -v grep" not in line:
            matches.append(line.strip())
    return matches


def _latest_trade_timestamp(trade_log: Path) -> Optional[datetime]:
    if not trade_log.exists():
        return None

    latest: Optional[datetime] = None
    try:
        with trade_log.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                timestamp = (row.get("timestamp") or "").strip()
                if not timestamp:
                    continue
                try:
                    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                if latest is None or parsed > latest:
                    latest = parsed
    except Exception:
        return None
    return latest


def _recent_data_provider_rate_limit(bot_log: Path, lookback_lines: int = 500) -> bool:
    if not bot_log.exists():
        return False
    try:
        lines = bot_log.read_text(errors="ignore").splitlines()[-lookback_lines:]
    except Exception:
        return False

    markers = (
        "YFRateLimitError",
        "yfinance rate-limited",
        "Too Many Requests",
        "No price data for SPY",
    )
    joined = "\n".join(lines)
    return any(m in joined for m in markers)


def _recent_cycle_activity(bot_log: Path, lookback_lines: int = 300) -> bool:
    """Return True when recent bot log lines show active cycle execution."""
    if not bot_log.exists():
        return False
    try:
        lines = bot_log.read_text(errors="ignore").splitlines()[-lookback_lines:]
    except Exception:
        return False
    joined = "\n".join(lines)
    markers = (
        "Cycle",
        "[Main]",
        "[Kill Switch]",
    )
    return any(m in joined for m in markers)


def _recent_stock_market_closed_marker(bot_log: Path, lookback_lines: int = 200) -> bool:
    """Return True when recent logs indicate the stock market is currently closed."""
    if not bot_log.exists():
        return False
    try:
        lines = bot_log.read_text(errors="ignore").splitlines()[-lookback_lines:]
    except Exception:
        return False
    joined = "\n".join(lines)
    return "STOCK MARKETS:" in joined and "CLOSED" in joined


def main() -> int:
    parser = argparse.ArgumentParser(description="Check paper-training bot health")
    parser.add_argument("--trade-log", default=str(DEFAULT_TRADE_LOG))
    parser.add_argument("--bot-log", default=str(DEFAULT_BOT_LOG))
    parser.add_argument("--launchd-stdout", default=str(DEFAULT_LAUNCHD_STDOUT))
    parser.add_argument("--heartbeat-minutes", type=float, default=15.0)
    parser.add_argument("--trade-age-hours", type=float, default=6.0)
    parser.add_argument(
        "--require-recent-trades",
        action="store_true",
        default=False,
        help="Treat stale trade activity as unhealthy/degraded instead of informational",
    )
    parser.add_argument("--expect-single-instance", action="store_true", default=True)
    args = parser.parse_args()

    trade_log = Path(args.trade_log)
    bot_log = Path(args.bot_log)
    launchd_stdout = Path(args.launchd_stdout)

    healthy = True
    degraded = False
    print("PAPER TRAINING HEALTH CHECK")
    print("=" * 60)

    launchd_state = _launchctl_print(LABEL)
    launchd_running = "state = running" in launchd_state or "state = spawning" in launchd_state
    print(f"launchd job: {'OK' if launchd_running else 'FAIL'} ({LABEL})")
    if not launchd_running:
        healthy = False

    wrapper_matches = _process_matches("paper_training_loop.sh")
    runner_matches = _process_matches("run_intraday_evidence.py")
    print(f"wrapper process count: {len(wrapper_matches)}")
    print(f"runner process count: {len(runner_matches)}")
    if args.expect_single_instance and (len(wrapper_matches) != 1 or len(runner_matches) != 1):
        healthy = False

    trade_age = _file_age_minutes(trade_log)
    bot_age = _file_age_minutes(bot_log)
    stdout_age = _file_age_minutes(launchd_stdout)
    cycle_active = _recent_cycle_activity(bot_log)
    stock_market_closed = _recent_stock_market_closed_marker(bot_log)
    process_active = launchd_running and len(wrapper_matches) >= 1 and len(runner_matches) >= 1

    if trade_age is None:
        print(f"trade log: FAIL (missing: {trade_log})")
        healthy = False
    else:
        print(f"trade log age: {trade_age:.1f} min")
        if trade_age > args.trade_age_hours * 60:
            if not args.require_recent_trades and process_active and cycle_active:
                if stock_market_closed:
                    print(
                        f"trade log freshness: OK (older than {args.trade_age_hours:.1f}h, "
                        "stock market closed; engine active)"
                    )
                else:
                    print(
                        f"trade log freshness: OK (older than {args.trade_age_hours:.1f}h, "
                        "no fills yet; engine active)"
                    )
            elif _recent_data_provider_rate_limit(bot_log):
                print(
                    f"trade log freshness: DEGRADED (older than {args.trade_age_hours:.1f}h, "
                    "likely data-provider rate limit)"
                )
                degraded = True
            else:
                print(f"trade log freshness: FAIL (older than {args.trade_age_hours:.1f}h)")
                healthy = False
        else:
            print("trade log freshness: OK")

    if bot_age is None:
        print(f"bot log: FAIL (missing: {bot_log})")
        healthy = False
    else:
        print(f"bot log age: {bot_age:.1f} min")
        if bot_age > args.heartbeat_minutes:
            print(f"bot heartbeat: FAIL (older than {args.heartbeat_minutes:.1f} min)")
            healthy = False
        else:
            print("bot heartbeat: OK")

    if stdout_age is not None:
        print(f"launchd stdout age: {stdout_age:.1f} min")

    latest_trade = _latest_trade_timestamp(trade_log)
    if latest_trade is None:
        print("latest trade: unavailable")
    else:
        trade_delta_hours = (_utcnow() - latest_trade).total_seconds() / 3600.0
        print(f"latest trade age: {trade_delta_hours:.2f} h")
        if trade_delta_hours > args.trade_age_hours:
            if not args.require_recent_trades and process_active and cycle_active:
                if stock_market_closed:
                    print(
                        f"trade activity: OK (no trade in last {args.trade_age_hours:.1f}h, "
                        "stock market closed; engine active)"
                    )
                else:
                    print(
                        f"trade activity: OK (no trade in last {args.trade_age_hours:.1f}h, "
                        "engine active and scanning)"
                    )
            elif _recent_data_provider_rate_limit(bot_log):
                print(
                    f"trade activity: DEGRADED (no trade in last {args.trade_age_hours:.1f}h, "
                    "likely data-provider rate limit)"
                )
                degraded = True
            else:
                print(f"trade activity: FAIL (no trade in last {args.trade_age_hours:.1f}h)")
                healthy = False
        else:
            print("trade activity: OK")

    print("-" * 60)
    if healthy and degraded:
        print("STATUS: DEGRADED")
        return 0
    print(f"STATUS: {'HEALTHY' if healthy else 'UNHEALTHY'}")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())