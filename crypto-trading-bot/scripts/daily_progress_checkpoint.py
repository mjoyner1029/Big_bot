#!/usr/bin/env python3
"""Run training checkpoint and append a one-line 100/30 progress delta."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRADE_LOG = ROOT / "logs/trade_log_paper_live.csv"
DEFAULT_CHECKPOINT = ROOT / "reports/paper_training/training_checkpoint_latest.json"
DEFAULT_PROGRESS_LOG = ROOT / "reports/paper_training/daily_progress.log"
SEED_TAGS = {"readiness_seed", "synthetic_seed", "seed"}


def _parse_ts(value: str):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _trade_progress(trade_log_path: Path) -> tuple[int, int, int, int, int]:
    if not trade_log_path.exists():
        return (0, 100, 0, 30, 0)

    rows = []
    try:
        with trade_log_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return (0, 100, 0, 30, 0)

    real_rows = [
        r
        for r in rows
        if (r.get("strategy_name", "").strip().lower() not in SEED_TAGS)
    ]

    trade_count = len(real_rows)
    timestamps = [
        _parse_ts(r.get("timestamp", ""))
        for r in real_rows
        if r.get("timestamp")
    ]
    timestamps = [t for t in timestamps if t is not None]
    span_days = (max(timestamps) - min(timestamps)).days if len(timestamps) >= 2 else 0

    remaining_trades = max(0, 100 - trade_count)
    remaining_days = max(0, 30 - span_days)
    gate_ok = int(trade_count >= 100 or span_days >= 30)
    return (trade_count, remaining_trades, span_days, remaining_days, gate_ok)


def _checkpoint_status(checkpoint_path: Path) -> tuple[str, str, str]:
    if not checkpoint_path.exists():
        return ("FAIL", "unknown", "unknown")

    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return ("FAIL", "unknown", "unknown")

    overall = "PASS" if payload.get("ok") else "FAIL"
    checks = payload.get("checks", {})
    health = "PASS" if checks.get("health", {}).get("ok") else "FAIL"
    readiness = "PASS" if checks.get("readiness", {}).get("ok") else "FAIL"
    return (overall, health, readiness)


def main() -> int:
    DEFAULT_PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "scripts/training_checkpoint.py"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

    overall, health, readiness = _checkpoint_status(DEFAULT_CHECKPOINT)
    trade_count, remaining_trades, span_days, remaining_days, gate_ok = _trade_progress(DEFAULT_TRADE_LOG)

    now = datetime.now(timezone.utc).isoformat()
    line = (
        f"{now} checkpoint={overall} health={health} readiness={readiness} "
        f"trades={trade_count}/100 remaining_trades={remaining_trades} "
        f"span_days={span_days}/30 remaining_days={remaining_days} "
        f"gate_100_or_30={'PASS' if gate_ok else 'PENDING'}"
    )

    with DEFAULT_PROGRESS_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

    print(line)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip(), file=sys.stderr)

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
