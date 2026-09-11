#!/usr/bin/env python3
"""Slippage stress audit on closed paper trades.

Recomputes strategy-level and aggregate expectancy under harsher slippage
assumptions to validate execution robustness.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _f(v: str) -> float:
    return float(v)


def load_closed(log_path: Path) -> List[Dict]:
    out: List[Dict] = []
    with log_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for r in reader:
            if not (r.get("pnl") or "").strip():
                continue
            try:
                out.append(
                    {
                        "symbol": r.get("symbol", ""),
                        "strategy": (r.get("strategy_name") or "").strip() or "(blank)",
                        "entry_price": _f(r.get("entry_price", "0")),
                        "exit_price": _f(r.get("exit_price", "0")),
                        "qty": _f(r.get("qty", "0")),
                        "pnl": _f(r.get("pnl", "0")),
                    }
                )
            except (TypeError, ValueError):
                continue
    return out


def stress(rows: List[Dict], extra_bps: float) -> Dict:
    if not rows:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "expectancy": 0.0,
            "win_rate": 0.0,
        }

    stressed = []
    for r in rows:
        notional_turnover = abs(r["entry_price"] * r["qty"]) + abs(r["exit_price"] * r["qty"])
        extra_cost = notional_turnover * (extra_bps / 10_000.0)
        stressed.append(r["pnl"] - extra_cost)

    wins = sum(1 for p in stressed if p > 0)
    return {
        "trades": len(stressed),
        "total_pnl": round(sum(stressed), 4),
        "expectancy": round(sum(stressed) / len(stressed), 6),
        "win_rate": round((wins / len(stressed)) * 100.0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Slippage stress audit")
    parser.add_argument("--log", default="logs/trade_log_paper_live.csv")
    parser.add_argument("--pass-bps", type=float, default=20.0)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    rows = load_closed(Path(args.log))
    scenarios = [0.0, 5.0, 10.0, 20.0, 35.0]
    results = {str(bps): stress(rows, bps) for bps in scenarios}

    pass_result = results.get(str(args.pass_bps), {"expectancy": -1})
    ok = pass_result["expectancy"] > 0

    payload = {
        "ok": ok,
        "pass_bps": args.pass_bps,
        "rows": len(rows),
        "scenarios": results,
    }

    print(f"SLIPPAGE STRESS: {'PASS' if ok else 'FAIL'}")
    print(json.dumps(payload, indent=2))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
