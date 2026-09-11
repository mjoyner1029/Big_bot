#!/usr/bin/env python3
"""Generate a per-strategy scorecard from the paper trade log.

The scorecard helps decide which strategies are contributing edge and which
should be monitored or paused, without hardcoding a static strategy whitelist.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class StrategyStats:
    strategy: str
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    total_pnl: float
    expectancy: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown_pct: float
    action: str
    size_multiplier: float


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _max_drawdown_pct(pnls: List[float], starting_equity: float = 100.0) -> float:
    if not pnls:
        return 0.0

    equity = starting_equity
    peak = starting_equity
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = ((peak - equity) / peak) * 100.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _compute_strategy_stats(
    strategy: str,
    pnls: List[float],
    min_trades: int,
    dd_limit_pct: float,
) -> StrategyStats:
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    trades = len(pnls)
    total_pnl = sum(pnls)
    win_rate_pct = (len(wins) / trades * 100.0) if trades else 0.0
    expectancy = (total_pnl / trades) if trades else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    max_dd = _max_drawdown_pct(pnls)

    if trades < min_trades:
        action = "watch"
        mult = 1.0
    elif expectancy <= 0 or profit_factor < 1.0 or max_dd > dd_limit_pct:
        action = "pause"
        mult = 0.5
    elif win_rate_pct >= 52.0 and expectancy > 0 and profit_factor >= 1.2 and max_dd <= dd_limit_pct:
        action = "keep"
        mult = 1.2
    else:
        action = "watch"
        mult = 1.0

    return StrategyStats(
        strategy=strategy,
        trades=trades,
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=round(win_rate_pct, 2),
        total_pnl=round(total_pnl, 2),
        expectancy=round(expectancy, 4),
        avg_win=round(avg_win, 4),
        avg_loss=round(avg_loss, 4),
        profit_factor=round(profit_factor, 3),
        max_drawdown_pct=round(max_dd, 2),
        action=action,
        size_multiplier=mult,
    )


def load_closed_trades(
    log_path: Path,
    date_prefix: Optional[str] = None,
    lookback: Optional[int] = None,
) -> Dict[str, List[float]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Trade log not found: {log_path}")

    rows = []
    with log_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pnl_raw = (row.get("pnl") or "").strip()
            ts_raw = (row.get("timestamp") or "").strip()
            if not pnl_raw or not ts_raw:
                continue
            pnl = _safe_float(pnl_raw)
            if pnl is None:
                continue
            if date_prefix and not ts_raw.startswith(date_prefix):
                continue
            strategy = (row.get("strategy_name") or "").strip() or "(blank)"
            rows.append((ts_raw, strategy, pnl))

    rows.sort(key=lambda r: _parse_ts(r[0]))
    if lookback and lookback > 0:
        rows = rows[-lookback:]

    by_strategy: Dict[str, List[float]] = {}
    for _, strategy, pnl in rows:
        by_strategy.setdefault(strategy, []).append(pnl)
    return by_strategy


def print_table(stats: List[StrategyStats]) -> None:
    headers = [
        "strategy",
        "trades",
        "win%",
        "pnl",
        "exp/trade",
        "pf",
        "maxDD%",
        "action",
        "size_mult",
    ]
    print(" | ".join(headers))
    print("-" * 108)
    for s in stats:
        print(
            f"{s.strategy:20} | {s.trades:6d} | {s.win_rate_pct:5.1f} | "
            f"{s.total_pnl:8.2f} | {s.expectancy:9.4f} | {s.profit_factor:5.2f} | "
            f"{s.max_drawdown_pct:6.2f} | {s.action:5} | {s.size_multiplier:8.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate strategy scorecard from paper trade log")
    parser.add_argument("--log", default="logs/trade_log_paper_live.csv", help="Path to trade log")
    parser.add_argument("--date", default=None, help="Filter to date prefix YYYY-MM-DD")
    parser.add_argument("--lookback", type=int, default=0, help="Use only last N closed trades")
    parser.add_argument("--min-trades", type=int, default=8, help="Minimum trades before hard keep/pause")
    parser.add_argument("--dd-limit", type=float, default=12.0, help="Per-strategy max drawdown threshold")
    parser.add_argument("--json-out", default="", help="Optional path to write JSON scorecard")
    args = parser.parse_args()

    by_strategy = load_closed_trades(
        Path(args.log),
        date_prefix=args.date,
        lookback=(args.lookback if args.lookback > 0 else None),
    )

    stats: List[StrategyStats] = []
    for strategy, pnls in by_strategy.items():
        stats.append(
            _compute_strategy_stats(
                strategy=strategy,
                pnls=pnls,
                min_trades=args.min_trades,
                dd_limit_pct=args.dd_limit,
            )
        )

    stats.sort(key=lambda s: (s.action != "keep", -s.expectancy, -s.total_pnl))

    print("STRATEGY SCORECARD")
    print(f"strategies={len(stats)}  min_trades={args.min_trades}  dd_limit={args.dd_limit:.1f}%")
    print_table(stats)

    if stats:
        keeps = [s.strategy for s in stats if s.action == "keep"]
        watches = [s.strategy for s in stats if s.action == "watch"]
        pauses = [s.strategy for s in stats if s.action == "pause"]
        print("\nRECOMMENDED ACTIONS")
        print(f"keep  ({len(keeps)}): {', '.join(keeps) if keeps else '-'}")
        print(f"watch ({len(watches)}): {', '.join(watches) if watches else '-'}")
        print(f"pause ({len(pauses)}): {', '.join(pauses) if pauses else '-'}")

    if args.json_out:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "min_trades": args.min_trades,
            "dd_limit": args.dd_limit,
            "rows": [asdict(s) for s in stats],
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
