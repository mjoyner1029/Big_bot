#!/usr/bin/env python3
"""Run one day slice of a 5-day paper profile with isolated logs and snapshots."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _run(command: list[str], env: dict[str, str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one 5-day profile paper session day")
    parser.add_argument("--day", type=int, required=True, help="Day index (1-5)")
    parser.add_argument("--cycles", type=int, default=1, help="How many --once cycles for this day")
    parser.add_argument("--capital", type=float, default=2000.0, help="Paper capital")
    parser.add_argument(
        "--asset-class",
        default="both",
        choices=["stocks", "crypto", "both"],
        help="Active asset class during this run",
    )
    args = parser.parse_args()

    if args.day < 1 or args.day > 5:
        raise ValueError("--day must be in range 1..5")

    env = os.environ.copy()
    env.update(
        {
            "TRADING_CAPITAL": str(args.capital),
            "USE_PAPER_TRADING": "true",
            "ENABLE_LIVE_TRADING": "false",
            "ALLOW_LIVE_TRADING": "false",
            "LIVE_TRADING_ENABLED": "false",
            "ASSET_CLASS": args.asset_class,
            "TRADE_LOG_PATH": "logs/trade_log_paper_live.csv",
            "BOT_LOG_PATH": "logs/bot_paper_5day.log",
            "STATE_PATH": "state/paper_5day_state.json",
        }
    )

    from logs.trade_logger import initialize_trade_log

    os.chdir(PROJECT_ROOT)
    initialize_trade_log(env["TRADE_LOG_PATH"])

    py = sys.executable
    for idx in range(1, args.cycles + 1):
        print(f"[paper-5day] day={args.day} cycle={idx}/{args.cycles}")
        _run([py, "main.py", "--once"], env=env)

    label = f"day_{args.day}"
    _run(
        [
            py,
            "scripts/paper_metrics_snapshot.py",
            "--log",
            env["TRADE_LOG_PATH"],
            "--capital",
            str(args.capital),
            "--label",
            label,
            "--out-dir",
            "reports/paper_5day/snapshots",
        ],
        env=env,
    )

    print("[paper-5day] completed")
    print("next: run days 2-5 with the same command and incremented --day")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
