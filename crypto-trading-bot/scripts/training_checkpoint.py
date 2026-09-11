#!/usr/bin/env python3
"""Consolidated training checkpoint runner.

Runs all key audits and writes a single JSON artifact for daily tracking.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _run(cmd: List[str]) -> Dict:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return {
        "cmd": cmd,
        "exit_code": proc.returncode,
        "ok": proc.returncode == 0,
        "output_tail": "\n".join(out.splitlines()[-40:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run daily training checkpoint")
    parser.add_argument("--out", default="reports/paper_training/training_checkpoint_latest.json")
    parser.add_argument("--log", default="logs/trade_log_paper_live.csv")
    parser.add_argument("--bot-log", default="logs/bot_paper_training.log")
    args = parser.parse_args()

    checks = {
        "health": _run([sys.executable, "scripts/check_paper_training_health.py", "--trade-age-hours", "24"]),
        "readiness": _run([sys.executable, "scripts/live_readiness_check.py"]),
        "data_quality": _run([
            sys.executable,
            "scripts/data_quality_report.py",
            "--log",
            args.log,
            "--json-out",
            "reports/paper_training/data_quality_latest.json",
        ]),
        "slippage_stress": _run([
            sys.executable,
            "scripts/slippage_stress_audit.py",
            "--log",
            args.log,
            "--json-out",
            "reports/paper_training/slippage_stress_latest.json",
        ]),
        "api_resilience": _run([
            sys.executable,
            "scripts/api_resilience_audit.py",
            "--log",
            args.bot_log,
            "--json-out",
            "reports/paper_training/api_resilience_latest.json",
        ]),
        "strategy_scorecard": _run([
            sys.executable,
            "scripts/strategy_scorecard.py",
            "--lookback",
            "300",
            "--json-out",
            "reports/paper_training/strategy_scorecard_latest.json",
        ]),
    }

    all_ok = all(v.get("ok") for v in checks.values())
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": all_ok,
        "checks": checks,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"TRAINING CHECKPOINT: {'PASS' if all_ok else 'FAIL'}")
    for name, result in checks.items():
        print(f"- {name}: {'PASS' if result['ok'] else 'FAIL'}")
    print(f"Artifact: {out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
