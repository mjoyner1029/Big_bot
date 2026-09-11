#!/usr/bin/env python3
"""API resilience audit from runtime logs.

Parses provider failures (429/403/timeouts/etc.) and reports whether recovery
signals appear, giving a repeatable broker/data API reliability check.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict


ERROR_PATTERNS = {
    "yfinance": re.compile(r"(yfinance|YFRateLimitError|Failed download)", re.IGNORECASE),
    "newsapi": re.compile(r"(NewsAPI failed|newsapi\.org)", re.IGNORECASE),
    "reddit": re.compile(r"(Reddit .* failed|reddit\.com)", re.IGNORECASE),
    "exchange": re.compile(r"(coinbase|alpaca|ccxt|binance|bybit).*?(error|fail|429|403)", re.IGNORECASE),
}

RECOVERY_PATTERNS = {
    "yfinance": re.compile(r"(Fetched .* symbols|All clear|No kill conditions triggered)", re.IGNORECASE),
    "newsapi": re.compile(r"(Total unique headlines|Google News:|News] Total unique)", re.IGNORECASE),
    "reddit": re.compile(r"(Total unique headlines|Google News:)", re.IGNORECASE),
    "exchange": re.compile(r"(order.*(filled|accepted)|Fetched .* balance|health.*OK)", re.IGNORECASE),
}


def audit(log_path: Path, tail_lines: int = 5000) -> Dict:
    if not log_path.exists():
        return {"ok": False, "reason": f"missing log: {log_path}"}

    lines = log_path.read_text(errors="ignore").splitlines()[-tail_lines:]
    errors = {k: 0 for k in ERROR_PATTERNS}
    recoveries = {k: 0 for k in RECOVERY_PATTERNS}

    for ln in lines:
        for source, pattern in ERROR_PATTERNS.items():
            if pattern.search(ln):
                errors[source] += 1
        for source, pattern in RECOVERY_PATTERNS.items():
            if pattern.search(ln):
                recoveries[source] += 1

    total_errors = sum(errors.values())
    total_recoveries = sum(recoveries.values())
    recovery_ratio = (total_recoveries / total_errors) if total_errors else 1.0

    ok = recovery_ratio >= 0.5
    return {
        "ok": ok,
        "tail_lines": tail_lines,
        "errors": errors,
        "recoveries": recoveries,
        "total_errors": total_errors,
        "total_recoveries": total_recoveries,
        "recovery_ratio": round(recovery_ratio, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="API resilience audit from bot logs")
    parser.add_argument("--log", default="logs/bot_paper_training.log")
    parser.add_argument("--tail-lines", type=int, default=5000)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    payload = audit(Path(args.log), tail_lines=args.tail_lines)
    print(f"API RESILIENCE: {'PASS' if payload.get('ok') else 'FAIL'}")
    print(json.dumps(payload, indent=2))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
