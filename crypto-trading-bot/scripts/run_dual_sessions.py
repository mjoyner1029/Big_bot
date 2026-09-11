#!/usr/bin/env python3
"""Run isolated paper sessions for stocks and crypto back-to-back."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _run_session(py: str, session: str, capital: float, cycles: int) -> int:
    trade_log = f"logs/trade_log_paper_{session}.csv"
    bot_log = f"logs/bot_{session}.log"
    state_path = f"state/paper_{session}_state.json"

    env = os.environ.copy()
    env.update(
        {
            "TRADING_CAPITAL": str(capital),
            "USE_PAPER_TRADING": "true",
            "ENABLE_LIVE_TRADING": "false",
            "ALLOW_LIVE_TRADING": "false",
            "LIVE_TRADING_ENABLED": "false",
            "ASSET_CLASS": session,
            "TRADE_LOG_PATH": trade_log,
            "BOT_LOG_PATH": bot_log,
            "STATE_PATH": state_path,
        }
    )

    from logs.trade_logger import initialize_trade_log

    initialize_trade_log(trade_log)

    for idx in range(1, cycles + 1):
        print(f"[dual] session={session} cycle={idx}/{cycles}")
        run = subprocess.run([py, "main.py", "--once"], cwd=PROJECT_ROOT, env=env)
        if run.returncode != 0:
            return run.returncode

    snapshot_label = f"{session}_latest"
    run = subprocess.run(
        [
            py,
            "scripts/paper_metrics_snapshot.py",
            "--log",
            trade_log,
            "--capital",
            str(capital),
            "--label",
            snapshot_label,
            "--out-dir",
            "reports/dual_sessions",
        ],
        cwd=PROJECT_ROOT,
        env=env,
    )
    return run.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stocks+crypto paper sessions")
    parser.add_argument("--capital", type=float, default=2000.0, help="Paper capital")
    parser.add_argument("--cycles", type=int, default=1, help="--once cycles per asset class")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    py = sys.executable

    for session in ["stocks", "crypto"]:
        exit_code = _run_session(py, session, args.capital, args.cycles)
        if exit_code != 0:
            print(f"[dual] failed session={session} exit={exit_code}")
            return exit_code

    print("[dual] completed stocks and crypto sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
