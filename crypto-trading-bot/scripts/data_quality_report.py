#!/usr/bin/env python3
"""Data quality audit for paper-trading artifacts.

Checks trade log schema and parse quality so data cleaning is continuously
validated instead of assumed.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict


REQUIRED_COLUMNS = [
    "timestamp",
    "symbol",
    "side",
    "entry_price",
    "qty",
    "result",
    "strategy_name",
]


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _is_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except (TypeError, ValueError):
        return False


def _ratio(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else 0.0


def build_report(log_path: Path) -> Dict:
    if not log_path.exists():
        return {
            "ok": False,
            "reason": f"missing log: {log_path}",
            "rows": 0,
            "checks": {},
        }

    with log_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    row_count = len(rows)
    closed_rows = [r for r in rows if (r.get("pnl") or "").strip()]

    ts_valid = sum(1 for r in rows if _is_timestamp((r.get("timestamp") or "").strip()))
    entry_valid = sum(1 for r in rows if _is_float((r.get("entry_price") or "").strip()))
    qty_valid = sum(1 for r in rows if _is_float((r.get("qty") or "").strip()))
    pnl_valid = sum(1 for r in closed_rows if _is_float((r.get("pnl") or "").strip()))

    key_fields = ["timestamp", "symbol", "side", "entry_price", "qty"]
    complete_rows = sum(
        1
        for r in rows
        if all((r.get(k) or "").strip() for k in key_fields)
    )

    duplicate_keys = set()
    seen = set()
    for r in rows:
        key = (
            (r.get("timestamp") or "").strip(),
            (r.get("symbol") or "").strip(),
            (r.get("side") or "").strip(),
            (r.get("entry_price") or "").strip(),
            (r.get("qty") or "").strip(),
        )
        if key in seen:
            duplicate_keys.add(key)
        seen.add(key)

    checks = {
        "required_columns": all(c in columns for c in REQUIRED_COLUMNS),
        "timestamp_parse_ratio": _ratio(ts_valid, row_count),
        "entry_price_parse_ratio": _ratio(entry_valid, row_count),
        "qty_parse_ratio": _ratio(qty_valid, row_count),
        "closed_pnl_parse_ratio": _ratio(pnl_valid, len(closed_rows)),
        "key_field_completeness_ratio": _ratio(complete_rows, row_count),
        "duplicate_row_ratio": _ratio(len(duplicate_keys), row_count),
    }

    ok = (
        checks["required_columns"]
        and checks["timestamp_parse_ratio"] >= 0.995
        and checks["entry_price_parse_ratio"] >= 0.995
        and checks["qty_parse_ratio"] >= 0.995
        and (checks["closed_pnl_parse_ratio"] >= 0.995 if closed_rows else True)
        and checks["key_field_completeness_ratio"] >= 0.995
        and checks["duplicate_row_ratio"] <= 0.02
    )

    return {
        "ok": ok,
        "rows": row_count,
        "closed_rows": len(closed_rows),
        "columns": columns,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Trade log data quality audit")
    parser.add_argument("--log", default="logs/trade_log_paper_live.csv")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    report = build_report(Path(args.log))
    status = "PASS" if report.get("ok") else "FAIL"
    print(f"DATA QUALITY: {status}")
    print(json.dumps(report, indent=2))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
