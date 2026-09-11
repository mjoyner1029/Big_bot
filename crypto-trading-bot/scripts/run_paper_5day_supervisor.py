#!/usr/bin/env python3
"""Run a multi-day paper profile in one command."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_day(py: str, day: int, cycles: int, capital: float, asset_class: str) -> None:
    cmd = [
        py,
        "scripts/run_paper_5day.py",
        "--day",
        str(day),
        "--cycles",
        str(cycles),
        "--capital",
        str(capital),
        "--asset-class",
        asset_class,
    ]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"day {day} failed with exit code {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper days in one supervisor command")
    parser.add_argument("--start-day", type=int, default=2, help="First day in range (default 2)")
    parser.add_argument("--end-day", type=int, default=5, help="Last day in range (default 5)")
    parser.add_argument("--cycles", type=int, default=1, help="Number of --once cycles per day")
    parser.add_argument("--capital", type=float, default=2000.0, help="Paper capital")
    parser.add_argument(
        "--asset-class",
        default="both",
        choices=["stocks", "crypto", "both"],
        help="Active asset class during each day",
    )
    args = parser.parse_args()

    if args.start_day < 1 or args.end_day > 5 or args.start_day > args.end_day:
        raise ValueError("Valid range is 1..5 and start-day must be <= end-day")

    py = sys.executable
    print(
        "[paper-5day-supervisor] "
        f"running days {args.start_day}-{args.end_day}, cycles/day={args.cycles}, "
        f"capital={args.capital}, asset_class={args.asset_class}"
    )

    for day in range(args.start_day, args.end_day + 1):
        print(f"[paper-5day-supervisor] begin day={day}")
        _run_day(py, day, args.cycles, args.capital, args.asset_class)

    print("[paper-5day-supervisor] completed")
    print("Artifacts: reports/paper_5day/snapshots/day_*.json and day_*.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
