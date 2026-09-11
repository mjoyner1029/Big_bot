#!/usr/bin/env python3
"""Rolling intraday evidence runner for paper-trading validation.

Runs repeated trading cycles in paper mode, periodically flattens positions
(to realize PnL), and executes readiness checks every N cycles.

Goal: accumulate 100+ closed trades with reproducible artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _set_paper_env(args: argparse.Namespace) -> None:
    """Apply a paper-only runtime profile before importing runtime modules."""
    os.environ.update(
        {
            "TRADING_CAPITAL": str(args.capital),
            "USE_PAPER_TRADING": "true",
            "ENABLE_LIVE_TRADING": "false",
            "ALLOW_LIVE_TRADING": "false",
            "LIVE_TRADING_ENABLED": "false",
            "ASSET_CLASS": args.asset_class,
            "TRADING_MODE": args.trading_mode,
            "CONFIDENCE_THRESHOLD": str(args.confidence_threshold),
            "APPROVAL_THRESHOLD": str(args.approval_threshold),
            "RISK_PER_TRADE_PCT": str(args.risk_per_trade_pct),
            "MAX_OPEN_POSITIONS": str(args.max_open_positions),
            "MAX_POSITION_PCT": str(args.max_position_pct),
            "TRADE_LOG_PATH": args.trade_log_path,
            "BOT_LOG_PATH": args.bot_log_path,
            "STATE_PATH": args.state_path,
            "ENABLE_ALT_DATA_INTEL": "true" if args.enable_alt_data_intel else "false",
            "ENABLE_DYNAMIC_CRYPTO": "true" if args.enable_dynamic_crypto else "false",
            "ENABLE_PENNY_STOCKS": "true" if args.enable_penny_stocks else "false",
            "ENABLE_OPTIONS_TRADING": "true" if args.enable_options_trading else "false",
            "ENABLE_CATALYST_SWING": "true" if args.enable_catalyst_swing else "false",
            # USAspending.gov API is unreliable in paper sessions — disable unless explicitly enabled
            "ENABLE_GOV_CONTRACTS": "true" if getattr(args, "enable_gov_contracts", False) else "false",
        }
    )


def _count_trade_rows(trade_log_path: str) -> Dict[str, int]:
    if not os.path.exists(trade_log_path):
        return {"total_rows": 0, "closed_rows": 0, "open_rows": 0}

    total = 0
    closed = 0
    with open(trade_log_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            pnl = (row.get("pnl") or "").strip()
            if pnl:
                closed += 1

    return {"total_rows": total, "closed_rows": closed, "open_rows": max(0, total - closed)}


def _build_prices(symbols: List[str]) -> Dict[str, float]:
    """Fetch the most recent market price for each symbol.

    Uses 1-minute bars (period="1d") as the primary source so that the exit
    price used for flatten is as close to "right now" as possible — avoiding
    the phantom-PnL problem caused by pairing entry prices from one yfinance
    fetch with exit prices from a different fetch that returns a different last
    hourly bar.  Falls back to hourly (5d/1h) for any symbols that have no
    minute data.
    """
    from config.config import is_crypto
    from data.fetcher import fetch_batch_market_data

    prices: Dict[str, float] = {}

    # 1-minute bars give the freshest available last trade price
    batched_1m = fetch_batch_market_data(symbols, period="1d", interval="1m")
    for sym, df in batched_1m.items():
        if df is not None and not df.empty and "Close" in df.columns:
            last_close = df["Close"].dropna()
            if not last_close.empty:
                prices[sym] = float(last_close.iloc[-1])

    # Fallback: any symbols without minute data → hourly
    missing = [s for s in symbols if s not in prices]
    if missing:
        crypto_miss = [s for s in missing if is_crypto(s)]
        stock_miss  = [s for s in missing if not is_crypto(s)]
        for syms, period in [(crypto_miss, "3mo"), (stock_miss, "5d")]:
            if syms:
                batched = fetch_batch_market_data(syms, period=period, interval="1h")
                for sym, df in batched.items():
                    if df is not None and not df.empty and "Close" in df.columns:
                        last_close = df["Close"].dropna()
                        if not last_close.empty:
                            prices[sym] = float(last_close.iloc[-1])

    return prices


def _flatten_all_positions() -> int:
    from config.config import get_all_symbols
    from trading.trade_executor import flatten_positions, get_portfolio

    portfolio = get_portfolio()
    # Skip swing holds and options — they are intentional multi-day positions
    positions = [
        p for p in portfolio.open_positions
        if str(p.get("hold_type", "") or "") not in ("swing", "options")
    ]
    if not positions:
        return 0

    prices = _build_prices(get_all_symbols())
    return flatten_positions(positions, prices)


def _run_readiness(readiness_script: str) -> Dict[str, Any]:
    result = subprocess.run(
        [sys.executable, readiness_script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    failed_checks: List[str] = []
    in_failed = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "FAILED CHECKS:":
            in_failed = True
            continue
        if in_failed:
            if stripped.startswith("-"):
                failed_checks.append(stripped[1:].strip())
            elif stripped:
                # stop when section ends
                in_failed = False

    status = "GO" if result.returncode == 0 else "NO-GO"
    return {
        "status": status,
        "exit_code": result.returncode,
        "failed_checks": failed_checks,
        "output_tail": "\n".join(output.splitlines()[-30:]),
    }


def _run_snapshot(trade_log_path: str, capital: float, label: str, out_dir: str) -> Dict[str, str]:
    cmd = [
        sys.executable,
        "scripts/paper_metrics_snapshot.py",
        "--log",
        trade_log_path,
        "--capital",
        str(capital),
        "--label",
        label,
        "--out-dir",
        out_dir,
    ]
    run = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    payload = {"snapshot_stdout": (run.stdout or "").strip(), "snapshot_exit": str(run.returncode)}
    if run.stderr:
        payload["snapshot_stderr"] = run.stderr.strip()
    return payload


def _write_rollup(out_dir: str, payload: Dict[str, Any]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"intraday_rollup_{stamp}.json"
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Rolling intraday evidence runner")
    parser.add_argument("--target-closed-trades", type=int, default=100)
    parser.add_argument("--check-every-cycles", type=int, default=10)
    parser.add_argument("--flatten-every-cycles", type=int, default=10)
    parser.add_argument("--max-cycles", type=int, default=600)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)

    parser.add_argument("--capital", type=float, default=2000.0)
    parser.add_argument("--asset-class", choices=["stocks", "crypto", "both"], default="both")
    parser.add_argument("--trading-mode", default="aggressive")
    parser.add_argument("--confidence-threshold", type=float, default=0.50)
    parser.add_argument("--approval-threshold", type=float, default=65.0)
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.01)
    parser.add_argument("--max-open-positions", type=int, default=8)
    parser.add_argument("--max-position-pct", type=float, default=0.12)
    parser.add_argument("--enable-alt-data-intel", action="store_true")
    parser.add_argument("--enable-dynamic-crypto", action="store_true")
    parser.add_argument("--enable-penny-stocks", action="store_true")
    parser.add_argument("--enable-options-trading", action="store_true")
    parser.add_argument("--enable-catalyst-swing", action="store_true")
    parser.add_argument("--enable-gov-contracts", action="store_true",
                        help="Enable USAspending.gov contract signals (off by default in paper mode)")

    parser.add_argument("--trade-log-path", default="logs/trade_log_paper_live.csv")
    parser.add_argument("--bot-log-path", default="logs/bot_intraday_evidence.log")
    parser.add_argument("--state-path", default="state/intraday_evidence_state.json")
    parser.add_argument("--readiness-script", default="scripts/live_readiness_check.py")
    parser.add_argument("--snapshot-out-dir", default="reports/intraday_evidence")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    _set_paper_env(args)

    from logs.trade_logger import initialize_trade_log

    initialize_trade_log(args.trade_log_path)

    # Runtime imports after env/profile is set.
    import main as bot_main

    checkpoints: List[Dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()

    print(
        "[intraday-evidence] start "
        f"target_closed={args.target_closed_trades} check_every={args.check_every_cycles} "
        f"flatten_every={args.flatten_every_cycles} max_cycles={args.max_cycles}"
    )

    bot_main._start_subsystems()
    try:
        for cycle in range(1, args.max_cycles + 1):
            print(f"[intraday-evidence] cycle={cycle}")
            bot_main.run_one_cycle()

            if args.flatten_every_cycles > 0 and cycle % args.flatten_every_cycles == 0:
                closed = _flatten_all_positions()
                print(f"[intraday-evidence] periodic_flatten closed={closed}")

            stats = _count_trade_rows(args.trade_log_path)
            should_check = (
                cycle % max(1, args.check_every_cycles) == 0
                or stats["closed_rows"] >= args.target_closed_trades
                or cycle == args.max_cycles
            )

            if should_check:
                label = f"cycle_{cycle}"
                snap_meta = _run_snapshot(args.trade_log_path, args.capital, label, args.snapshot_out_dir)
                readiness = _run_readiness(args.readiness_script)
                checkpoint = {
                    "cycle": cycle,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "trade_stats": stats,
                    "readiness": readiness,
                    "snapshot": snap_meta,
                }
                checkpoints.append(checkpoint)

                print(
                    "[intraday-evidence] checkpoint "
                    f"cycle={cycle} closed={stats['closed_rows']} status={readiness['status']} "
                    f"failed_checks={len(readiness['failed_checks'])}"
                )

                if stats["closed_rows"] >= args.target_closed_trades:
                    print(
                        "[intraday-evidence] target reached "
                        f"closed_rows={stats['closed_rows']}"
                    )
                    break

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        # Final flatten to realize any remaining PnL.
        final_flattened = _flatten_all_positions()
        final_stats = _count_trade_rows(args.trade_log_path)
        final_readiness = _run_readiness(args.readiness_script)

    finally:
        bot_main._stop_subsystems()

    finished_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "config": {
            "target_closed_trades": args.target_closed_trades,
            "check_every_cycles": args.check_every_cycles,
            "flatten_every_cycles": args.flatten_every_cycles,
            "max_cycles": args.max_cycles,
            "capital": args.capital,
            "asset_class": args.asset_class,
            "trade_log_path": args.trade_log_path,
            "bot_log_path": args.bot_log_path,
            "state_path": args.state_path,
        },
        "final": {
            "final_flattened": final_flattened,
            "trade_stats": final_stats,
            "readiness": final_readiness,
        },
        "checkpoints": checkpoints,
    }

    rollup_path = _write_rollup(args.snapshot_out_dir, summary)
    print(f"[intraday-evidence] rollup={rollup_path}")

    # Return non-zero only if target was not reached.
    if final_stats["closed_rows"] < args.target_closed_trades:
        print(
            "[intraday-evidence] target_not_reached "
            f"closed={final_stats['closed_rows']} target={args.target_closed_trades}"
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
