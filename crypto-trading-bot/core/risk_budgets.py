"""Drawdown budgets — strategy, family and asset level.

The portfolio kill-switch (SafetyManager) is the last line of defence; these
budgets act EARLIER: when a strategy/family/asset consumes its risk budget its
allocation is reduced, then paused, without waiting for the portfolio breaker.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from core.strategy_correlation import family_of

logger = logging.getLogger(__name__)


@dataclass
class BudgetStatus:
    name: str
    scope: str            # 'strategy' | 'family' | 'asset'
    drawdown: float       # current drawdown ($) of the cumulative pnl curve
    budget: float         # allowed drawdown ($)
    used_frac: float      # drawdown / budget
    action: str           # 'ok' | 'reduce' | 'pause'


class DrawdownBudgetManager:
    """Tracks rolling drawdown per strategy / family / asset from trade memory."""

    def __init__(
        self,
        db_path: str = "data/trade_memory.sqlite",
        capital: float = 10_000.0,
        strategy_budget_pct: float = 0.03,
        family_budget_pct: float = 0.05,
        asset_budget_pct: float = 0.03,
        reduce_threshold: float = 0.7,
        lookback_days: int = 30,
    ) -> None:
        self.db_path = db_path
        self.capital = capital
        self.strategy_budget = capital * strategy_budget_pct
        self.family_budget = capital * family_budget_pct
        self.asset_budget = capital * asset_budget_pct
        self.reduce_threshold = reduce_threshold
        self.lookback_days = lookback_days

    # ── Queries ───────────────────────────────────────────────────────────────

    def _pnls(self, where: str, args: tuple) -> List[float]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    f"SELECT net_pnl FROM trade_memory WHERE {where} "
                    f"AND exit_time >= datetime('now', '-{self.lookback_days} days') "
                    "ORDER BY exit_time ASC",
                    args,
                ).fetchall()
            return [float(r[0]) for r in rows if r[0] is not None]
        except sqlite3.OperationalError:
            return []

    @staticmethod
    def _drawdown(pnls: Sequence[float]) -> float:
        cum = peak = dd = 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            dd = max(dd, peak - cum)
        return dd

    def _status(self, name: str, scope: str, pnls: List[float],
                budget: float) -> BudgetStatus:
        dd = self._drawdown(pnls)
        used = dd / budget if budget > 0 else 0.0
        action = "ok"
        if used >= 1.0:
            action = "pause"
        elif used >= self.reduce_threshold:
            action = "reduce"
        return BudgetStatus(name=name, scope=scope, drawdown=dd, budget=budget,
                            used_frac=used, action=action)

    # ── Public API ────────────────────────────────────────────────────────────

    def strategy_status(self, strategy: str) -> BudgetStatus:
        return self._status(strategy, "strategy",
                            self._pnls("strategy=?", (strategy,)),
                            self.strategy_budget)

    def family_status(self, family: str) -> BudgetStatus:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT strategy, net_pnl FROM trade_memory "
                    f"WHERE exit_time >= datetime('now', '-{self.lookback_days} days') "
                    "ORDER BY exit_time ASC",
                ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        pnls = [float(p) for s, p in rows
                if p is not None and family_of(s or "unknown") == family]
        return self._status(family, "family", pnls, self.family_budget)

    def asset_status(self, symbol: str) -> BudgetStatus:
        return self._status(symbol, "asset",
                            self._pnls("symbol=?", (symbol,)),
                            self.asset_budget)

    def allocation_multiplier(self, strategy: str, symbol: str) -> float:
        """Combined budget multiplier for a candidate trade.

        1.0 = full size, 0.5 = budget pressure (reduce), 0.0 = paused.
        Structured reasons are logged for every reduction.
        """
        statuses = [
            self.strategy_status(strategy),
            self.family_status(family_of(strategy)),
            self.asset_status(symbol),
        ]
        multiplier = 1.0
        for s in statuses:
            if s.action == "pause":
                logger.warning(
                    f"DrawdownBudget: {s.scope} '{s.name}' PAUSED — "
                    f"drawdown ${s.drawdown:.2f} >= budget ${s.budget:.2f}"
                )
                return 0.0
            if s.action == "reduce":
                logger.info(
                    f"DrawdownBudget: {s.scope} '{s.name}' at {s.used_frac:.0%} "
                    f"of budget — reducing allocation"
                )
                multiplier = min(multiplier, 0.5)
        return multiplier
