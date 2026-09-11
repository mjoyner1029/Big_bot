#!/usr/bin/env python3
"""Create daily paper-run metric snapshots from a trade log."""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_closed_trades(log_path: str) -> list[dict]:
    trades: list[dict] = []
    if not os.path.exists(log_path):
        return trades

    with open(log_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pnl_raw = (row.get("pnl") or "").strip()
            if not pnl_raw:
                continue
            try:
                trades.append(
                    {
                        "timestamp": row.get("timestamp", ""),
                        "symbol": row.get("symbol", ""),
                        "pnl": float(pnl_raw),
                    }
                )
            except ValueError:
                continue
    return trades


def _compute_metrics(trades: list[dict], starting_capital: float) -> dict:
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    total_return_pct = (total_pnl / starting_capital * 100.0) if starting_capital > 0 else 0.0
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0

    equity = starting_capital
    peak = equity
    max_drawdown_pct = 0.0
    for trade in trades:
        equity += trade["pnl"]
        peak = max(peak, equity)
        if peak > 0:
            drawdown = (peak - equity) / peak * 100.0
            max_drawdown_pct = max(max_drawdown_pct, drawdown)

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "profit_factor": round(profit_factor, 3),
        "final_equity": round(starting_capital + total_pnl, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a paper-run metrics snapshot")
    parser.add_argument("--log", default="logs/trade_log_paper_live.csv", help="Trade log path")
    parser.add_argument("--capital", type=float, default=2000.0, help="Starting capital")
    parser.add_argument("--label", default="day", help="Snapshot label")
    parser.add_argument(
        "--out-dir",
        default="reports/paper_5day/snapshots",
        help="Directory where snapshot files are written",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = _load_closed_trades(args.log)
    metrics = _compute_metrics(trades, args.capital)
    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "trade_log": args.log,
        "capital": args.capital,
        "metrics": metrics,
    }

    safe_label = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in args.label)
    json_path = out_dir / f"{safe_label}.json"
    md_path = out_dir / f"{safe_label}.md"

    with open(json_path, "w") as handle:
        json.dump(snapshot, handle, indent=2)

    with open(md_path, "w") as handle:
        handle.write(f"# Paper Snapshot: {args.label}\n\n")
        handle.write(f"- Generated (UTC): {snapshot['generated_at_utc']}\n")
        handle.write(f"- Trade log: {args.log}\n")
        handle.write(f"- Starting capital: ${args.capital:.2f}\n")
        handle.write(f"- Total trades: {metrics['total_trades']}\n")
        handle.write(f"- Win rate: {metrics['win_rate_pct']}%\n")
        handle.write(f"- Return: {metrics['total_return_pct']}%\n")
        handle.write(f"- Max drawdown: {metrics['max_drawdown_pct']}%\n")
        handle.write(f"- Final equity: ${metrics['final_equity']:.2f}\n")

    print(f"snapshot_json={json_path}")
    print(f"snapshot_md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
