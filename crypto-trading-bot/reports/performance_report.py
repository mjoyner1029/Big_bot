"""Performance Report Generator — produces a human-readable Markdown
report from TradeMemory and StrategyPerformanceTracker.

Output: reports/performance_report.md

The final line of the report MUST be:
  TARGET_STATUS: PASS  or  TARGET_STATUS: FAIL

This report is HONEST — no performance figures are fabricated.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from memory.trade_memory import TradeMemory, get_trade_memory
from ml.strategy_performance import StrategyPerformanceTracker, get_performance_tracker

logger = logging.getLogger(__name__)

REPORTS_DIR = "reports"

# Performance targets
TARGET_CAGR           = 0.30
MAX_DRAWDOWN          = 0.15
MIN_PROFIT_FACTOR     = 2.0
MIN_SHARPE            = 2.0
MIN_WIN_RATE          = 0.55


class PerformanceReporter:
    """Generate a performance report from live paper-trading data."""

    def __init__(
        self,
        trade_memory: Optional[TradeMemory] = None,
        tracker: Optional[StrategyPerformanceTracker] = None,
    ) -> None:
        self._mem     = trade_memory or get_trade_memory()
        self._tracker = tracker or get_performance_tracker()

    # ── Public API ────────────────────────────────────────────────

    def generate(self, initial_capital: float = 10_000.0) -> Dict[str, Any]:
        """Generate and write the performance report.  Returns the stats dict."""
        overall    = self._tracker.get_overall_stats()
        by_strat   = self._tracker.get_all_strategy_stats(min_trades=5)
        best       = self._mem.get_best_setups(limit=5)
        worst      = self._mem.get_worst_setups(limit=5)
        recent_30d = self._mem.get_recent_performance(days=30)

        # Estimate CAGR from trade history
        cagr = self._estimate_cagr(initial_capital)

        n_trades = overall.get("n_trades", 0)
        passed   = (
            n_trades > 0
            and cagr                           >= TARGET_CAGR
            and overall.get("max_drawdown", 1) <= MAX_DRAWDOWN
            and overall.get("profit_factor", 0) >= MIN_PROFIT_FACTOR
            and overall.get("sharpe", 0)        >= MIN_SHARPE
            and overall.get("win_rate", 0)      >= MIN_WIN_RATE
        )

        result = {
            "overall":      overall,
            "by_strategy":  by_strat,
            "recent_30d":   recent_30d,
            "cagr_estimate": round(cagr, 4),
            "best_setups":  best,
            "worst_setups": worst,
            "passed":       passed,
        }

        self._write_report(result)
        return result

    # ── CAGR estimation ───────────────────────────────────────────

    def _estimate_cagr(self, initial_capital: float) -> float:
        trades = self._mem.get_closed_trades()
        if not trades:
            return 0.0
        try:
            first_ts = datetime.fromisoformat(
                sorted(t.get("timestamp_close") or "" for t in trades)[0].replace("Z", "+00:00")
            )
            last_ts  = datetime.fromisoformat(
                sorted(t.get("timestamp_close") or "" for t in trades)[-1].replace("Z", "+00:00")
            )
            days_elapsed = (last_ts - first_ts).days
            if days_elapsed < 1:
                return 0.0
            total_pnl  = sum(float(t.get("pnl") or 0) for t in trades)
            equity_end = initial_capital + total_pnl
            years      = days_elapsed / 365.25
            return (equity_end / initial_capital) ** (1 / years) - 1 if equity_end > 0 else 0.0
        except Exception:
            return 0.0

    # ── Report writer ─────────────────────────────────────────────

    def _write_report(self, result: Dict[str, Any]) -> None:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path    = os.path.join(REPORTS_DIR, "performance_report.md")
        overall = result["overall"]
        n       = overall.get("n_trades", 0)
        passed  = result["passed"]
        cagr    = result["cagr_estimate"]

        lines = [
            "# Bot Performance Report",
            f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"\n## Summary",
            f"- Total closed trades:  {n}",
            f"- Overall win rate:     {overall.get('win_rate', 0)*100:.1f}%",
            f"- Profit factor:        {overall.get('profit_factor', 0):.2f}",
            f"- Expectancy:           ${overall.get('expectancy', 0):.2f}",
            f"- Total P&L:            ${overall.get('total_pnl', 0):.2f}",
            f"- CAGR estimate:        {cagr*100:.1f}%",
            f"- Max drawdown:         {overall.get('max_drawdown', 0)*100:.1f}%",
            f"- Sharpe ratio:         {overall.get('sharpe', 0):.2f}",
            f"- Sortino ratio:        {overall.get('sortino', 0):.2f}",
            f"\n## Target Checklist",
            f"- CAGR ≥ 30%:          {'✓ PASS' if cagr >= TARGET_CAGR else '✗ FAIL'} ({cagr*100:.1f}%)",
            f"- Max DD ≤ 15%:        {'✓ PASS' if overall.get('max_drawdown',1) <= MAX_DRAWDOWN else '✗ FAIL'} ({overall.get('max_drawdown',0)*100:.1f}%)",
            f"- Profit Factor ≥ 2.0: {'✓ PASS' if overall.get('profit_factor',0) >= MIN_PROFIT_FACTOR else '✗ FAIL'} ({overall.get('profit_factor',0):.2f})",
            f"- Sharpe ≥ 2.0:        {'✓ PASS' if overall.get('sharpe',0) >= MIN_SHARPE else '✗ FAIL'} ({overall.get('sharpe',0):.2f})",
            f"- Win Rate ≥ 55%:      {'✓ PASS' if overall.get('win_rate',0) >= MIN_WIN_RATE else '✗ FAIL'} ({overall.get('win_rate',0)*100:.1f}%)",
        ]

        # Last 30 days
        r30 = result.get("recent_30d", {})
        if r30.get("n_trades", 0) > 0:
            lines += [
                f"\n## Last 30 Days",
                f"- Trades: {r30.get('n_trades', 0)}",
                f"- Win rate: {r30.get('win_rate', 0)*100:.1f}%",
                f"- Profit factor: {r30.get('profit_factor', 0):.2f}",
                f"- P&L: ${r30.get('total_pnl', 0):.2f}",
            ]

        # Strategy breakdown
        by_strat = result.get("by_strategy", {})
        if by_strat:
            lines.append(f"\n## Strategy Performance (min 5 trades)")
            for s, stats in sorted(by_strat.items(), key=lambda x: -x[1].get("profit_factor", 0)):
                lines.append(
                    f"- **{s}**: n={stats.get('n_trades',0)}, "
                    f"WR={stats.get('win_rate',0)*100:.0f}%, "
                    f"PF={stats.get('profit_factor',0):.2f}, "
                    f"E={stats.get('expectancy',0):.2f}"
                )

        # Best / worst setups
        if result.get("best_setups"):
            lines.append("\n## Best Setups (regime × strategy)")
            for s in result["best_setups"]:
                lines.append(
                    f"- {s.get('strategy_names','')} @ {s.get('market_regime','')}: "
                    f"avg_return={float(s.get('avg_pnl_pct') or 0)*100:.2f}%, "
                    f"n={s.get('n_trades',0)}, WR={float(s.get('win_rate') or 0)*100:.0f}%"
                )

        if result.get("worst_setups"):
            lines.append("\n## Worst Setups (regime × strategy)")
            for s in result["worst_setups"]:
                lines.append(
                    f"- {s.get('strategy_names','')} @ {s.get('market_regime','')}: "
                    f"avg_return={float(s.get('avg_pnl_pct') or 0)*100:.2f}%, "
                    f"n={s.get('n_trades',0)}, WR={float(s.get('win_rate') or 0)*100:.0f}%"
                )

        lines.append(f"\nTARGET_STATUS: {'PASS' if passed else 'FAIL'}")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        logger.info("[PerformanceReport] Written to %s | TARGET_STATUS: %s", path, "PASS" if passed else "FAIL")


# ── Singleton ─────────────────────────────────────────────────────
_reporter: Optional[PerformanceReporter] = None


def get_performance_reporter() -> PerformanceReporter:
    global _reporter
    if _reporter is None:
        _reporter = PerformanceReporter()
    return _reporter
