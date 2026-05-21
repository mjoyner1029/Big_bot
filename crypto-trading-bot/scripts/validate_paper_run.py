#!/usr/bin/env python3
"""
Paper Trading Validation Script

Reads logs/trade_log.csv and outputs a performance summary:
  - Total return %
  - Max drawdown %
  - Win rate
  - Number of circuit breaker events (from bot.log)
  - Go/No-Go verdict vs PRD thresholds

Usage:
    python scripts/validate_paper_run.py
    python scripts/validate_paper_run.py --log logs/trade_log.csv
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── PRD thresholds ────────────────────────────────────────────────
THRESHOLD_RETURN_PCT = 10.0      # minimum bi-weekly return to pass
THRESHOLD_MAX_DRAWDOWN = 20.0    # maximum acceptable drawdown %
THRESHOLD_WIN_RATE = 50.0        # minimum win rate %
THRESHOLD_CIRCUIT_BREAKERS = 3   # maximum circuit breaker events


def load_trades(log_path: str) -> List[Dict]:
    """Load closed trades from trade_log.csv."""
    if not os.path.exists(log_path):
        print(f"ERROR: Trade log not found: {log_path}")
        sys.exit(1)

    trades = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only include closed trades (have an exit_price and pnl)
            if row.get("exit_price") and row.get("pnl"):
                try:
                    trades.append({
                        "timestamp": row["timestamp"],
                        "symbol": row["symbol"],
                        "side": row["side"],
                        "entry_price": float(row["entry_price"]),
                        "exit_price": float(row["exit_price"]),
                        "qty": float(row["qty"]),
                        "pnl": float(row["pnl"]),
                        "result": row.get("result", ""),
                    })
                except (ValueError, KeyError):
                    continue
    return trades


def compute_metrics(trades: List[Dict], starting_capital: float) -> Dict:
    """Compute performance metrics from closed trades."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
        }

    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    win_rate = (len(wins) / len(trades)) * 100 if trades else 0.0
    total_return_pct = (total_pnl / starting_capital) * 100 if starting_capital > 0 else 0.0

    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown: walk equity curve
    equity = starting_capital
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += t["pnl"]
        if equity > peak:
            peak = equity
        dd = ((peak - equity) / peak) * 100 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "final_equity": round(starting_capital + total_pnl, 2),
    }


def count_circuit_breaker_events(log_path: str) -> int:
    """Count CIRCUIT BREAKER and KILL SWITCH events from bot.log."""
    if not os.path.exists(log_path):
        return 0

    count = 0
    pattern = re.compile(r"(CIRCUIT BREAKER|KILL SWITCH TRIPPED|RAPID LOSS PROTECTION)", re.IGNORECASE)
    try:
        with open(log_path) as f:
            for line in f:
                if pattern.search(line):
                    count += 1
    except Exception:
        pass
    return count


def print_report(metrics: Dict, circuit_breaker_events: int, starting_capital: float) -> bool:
    """Print the validation report and return True if Go/No-Go passes."""
    sep = "=" * 60

    print(f"\n{sep}")
    print("  LIMITLESS — PAPER TRADING VALIDATION REPORT")
    print(sep)
    print(f"  Starting capital:      ${starting_capital:.2f}")
    print(f"  Final equity:          ${metrics['final_equity']:.2f}")
    print(f"  Total P&L:             ${metrics['total_pnl']:.2f}")
    print(f"  Total return:          {metrics['total_return_pct']:.2f}%")
    print(f"  Total trades:          {metrics['total_trades']}")
    print(f"  Winning trades:        {metrics['winning_trades']}")
    print(f"  Losing trades:         {metrics['losing_trades']}")
    print(f"  Win rate:              {metrics['win_rate']:.1f}%")
    print(f"  Max drawdown:          {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Avg win:               ${metrics['avg_win']:.2f}")
    print(f"  Avg loss:              ${metrics['avg_loss']:.2f}")
    print(f"  Profit factor:         {metrics['profit_factor']:.2f}")
    print(f"  Circuit breaker events:{circuit_breaker_events}")
    print(f"{sep}")

    # Go/No-Go checks
    checks = [
        ("Return >= 10%",         metrics["total_return_pct"] >= THRESHOLD_RETURN_PCT,
         f"{metrics['total_return_pct']:.2f}% (need {THRESHOLD_RETURN_PCT}%)"),
        ("Max drawdown < 20%",    metrics["max_drawdown_pct"] < THRESHOLD_MAX_DRAWDOWN,
         f"{metrics['max_drawdown_pct']:.2f}% (limit {THRESHOLD_MAX_DRAWDOWN}%)"),
        ("Win rate >= 50%",       metrics["win_rate"] >= THRESHOLD_WIN_RATE,
         f"{metrics['win_rate']:.1f}% (need {THRESHOLD_WIN_RATE}%)"),
        ("Circuit breakers < 3",  circuit_breaker_events < THRESHOLD_CIRCUIT_BREAKERS,
         f"{circuit_breaker_events} events (limit {THRESHOLD_CIRCUIT_BREAKERS})"),
    ]

    print("\n  GO / NO-GO CHECKLIST")
    print(f"  {'-'*50}")
    all_passed = True
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}: {detail}")
        if not passed:
            all_passed = False

    print(f"\n  VERDICT: {'>>> GO <<<' if all_passed else '>>> NO-GO <<<'}")
    if not all_passed:
        print("  Fix failing metrics before deploying real capital.")
    else:
        print("  Run a second 2-week period to confirm before going live.")
    print(f"{sep}\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate paper trading run")
    parser.add_argument(
        "--log",
        default="logs/trade_log.csv",
        help="Path to trade_log.csv (default: logs/trade_log.csv)",
    )
    parser.add_argument(
        "--bot-log",
        default="logs/bot.log",
        help="Path to bot.log for circuit breaker counting (default: logs/bot.log)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100.0,
        help="Starting paper capital in USD (default: 100)",
    )
    args = parser.parse_args()

    # Change to project root so relative paths work
    os.chdir(PROJECT_ROOT)

    trades = load_trades(args.log)
    if not trades:
        print("WARNING: No closed trades found in log. Has the bot run and closed positions?")

    metrics = compute_metrics(trades, starting_capital=args.capital)
    circuit_breakers = count_circuit_breaker_events(args.bot_log)

    passed = print_report(metrics, circuit_breakers, starting_capital=args.capital)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
