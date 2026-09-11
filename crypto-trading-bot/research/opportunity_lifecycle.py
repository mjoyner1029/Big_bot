from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PaperTradingExperiment:
    strategy_id: str
    start_date: str
    expected_performance: float = 0.0
    actual_performance: float = 0.0
    expected_slippage: float = 0.0
    actual_slippage: float = 0.0
    expected_win_rate: float = 0.0
    actual_win_rate: float = 0.0
    expected_drawdown: float = 0.0
    actual_drawdown: float = 0.0
    sample_count: int = 0
    status: str = "PAPER_TRADING"
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EdgeMonitor:
    """Compare expected vs realized edge quality and classify health."""

    def evaluate_edge(
        self,
        *,
        expected_performance: float,
        realized_performance: float,
        rolling_sharpe: float,
        rolling_win_rate: float,
        rolling_profit_factor: float,
        rolling_drawdown: float,
        slippage: float,
        signal_frequency: float,
    ) -> Dict[str, Any]:
        performance_gap = abs(expected_performance - realized_performance)
        performance_score = max(0.0, 1.0 - (performance_gap / max(abs(expected_performance), 0.01)))
        sharpe_score = max(0.0, min(1.0, (rolling_sharpe + 1.0) / 3.0))
        win_score = max(0.0, min(1.0, rolling_win_rate / 0.6))
        pf_score = max(0.0, min(1.0, rolling_profit_factor / 2.0))
        drawdown_penalty = max(0.0, min(1.0, rolling_drawdown / 0.25))
        slippage_penalty = min(1.0, slippage / 0.01)
        freq_score = max(0.0, min(1.0, signal_frequency))

        health_score = (
            performance_score * 25
            + sharpe_score * 20
            + win_score * 15
            + pf_score * 15
            + (1.0 - drawdown_penalty) * 10
            + (1.0 - slippage_penalty) * 10
            + freq_score * 5
        )

        state = "HEALTHY"
        if health_score < 55:
            state = "DEGRADED"
        elif health_score < 70:
            state = "WATCH"

        return {
            "edge_health_score": round(max(0.0, min(100.0, health_score)), 2),
            "state": state,
            "expected_performance": expected_performance,
            "realized_performance": realized_performance,
            "rolling_sharpe": rolling_sharpe,
            "rolling_win_rate": rolling_win_rate,
            "rolling_profit_factor": rolling_profit_factor,
            "rolling_drawdown": rolling_drawdown,
            "slippage": slippage,
            "signal_frequency": signal_frequency,
        }


class StrategyLifecycleManager:
    """Paper-to-live promotion gating for discovered strategies."""

    def __init__(self, auto_live_promotion: bool = False):
        self.auto_live_promotion = auto_live_promotion

    def evaluate_promotion(
        self,
        *,
        strategy_id: str,
        paper_trades: int,
        sharpe: float,
        drawdown: float,
        sample_size: int,
        expected_net: float = 0.0,
    ) -> Dict[str, Any]:
        gates = []
        gates.append(("minimum_paper_trades", paper_trades >= 30))
        gates.append(("minimum_sample_size", sample_size >= 50))
        gates.append(("minimum_sharpe", sharpe >= 0.5))
        gates.append(("drawdown_limit", drawdown <= 0.20))
        gates.append(("positive_expectancy", expected_net > 0.0))

        failed = [name for name, passed in gates if not passed]
        eligible = not failed and self.auto_live_promotion

        return {
            "strategy_id": strategy_id,
            "eligible_for_live": eligible,
            "gates": {name: passed for name, passed in gates},
            "reason": "All promotion gates passed and auto-live is enabled." if eligible else "Promotion blocked by required gates; live trading requires human approval.",
        }


class StrategyGraveyard:
    """Remember rejected or retired strategies to prevent duplicates."""

    def __init__(self):
        self._db: Dict[str, Dict[str, Any]] = {}

    def register(self, strategy_id: str, reason: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        self._db[strategy_id] = {
            "strategy_id": strategy_id,
            "reason": reason,
            "metrics": metrics or {},
            "registered_at": _utcnow(),
        }

    def is_known(self, strategy_id: str) -> bool:
        return strategy_id in self._db

    def lookup(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        return self._db.get(strategy_id)


class OpportunityBoard:
    """Rank opportunities by score and current lifecycle status."""

    def __init__(self):
        self._items: List[Dict[str, Any]] = []

    def push(self, item: Dict[str, Any]) -> None:
        self._items.append(item)

    def rank(self) -> List[Dict[str, Any]]:
        return sorted(self._items, key=lambda item: float(item.get("score", 0.0)), reverse=True)


def _build_research_report() -> Dict[str, Any]:
    return {
        "date": _utcnow(),
        "universe_scanned": 0,
        "hypotheses_tested": 0,
        "new_anomalies": 0,
        "passed_preliminary_validation": 0,
        "passed_statistical_validation": 0,
        "new_paper_strategies": 0,
        "rejected": 0,
        "top_discovery": None,
        "existing_edge_health": [],
    }


__all__ = [
    "PaperTradingExperiment",
    "EdgeMonitor",
    "StrategyLifecycleManager",
    "StrategyGraveyard",
    "OpportunityBoard",
    "_build_research_report",
]
