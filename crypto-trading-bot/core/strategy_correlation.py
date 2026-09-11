"""Strategy correlation & evidence-aware quality scoring.

Rolling correlation between realized strategy P&L streams — momentum, EMA
trend and breakout are NOT assumed independent just because they are separate
classes. Highly redundant strategies are penalized at allocation time.

Also provides the evidence-aware strategy quality score that replaces the old
``expectancy * win_rate * sqrt(n)`` heuristic: it uses the LOWER confidence
bound of net expectancy scaled by regime fit, edge health, execution quality
and correlation penalties.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from core.validation_stats import effective_sample_size

logger = logging.getLogger(__name__)

# Strategy families for diversification limits (spec section 22)
STRATEGY_FAMILIES: Dict[str, str] = {
    "momentum": "MOMENTUM",
    "crypto_momentum": "MOMENTUM",
    "ema_trend_follow": "TREND",
    "trend": "TREND",
    "breakout": "TREND",           # breakout rides the same trend factor
    "volume_profile_breakout": "TREND",
    "mean_reversion": "MEAN_REVERSION",
    "mean_reversion_zscore": "MEAN_REVERSION",
    "overnight": "TEMPORAL",
    "overnight_edge": "TEMPORAL",
    "funding": "STRUCTURAL_CRYPTO",
    "basis": "STRUCTURAL_CRYPTO",
    "pairs": "RELATIVE_VALUE",
    "volatility": "VOLATILITY",
}
DEFAULT_FAMILY = "MOMENTUM"


def family_of(strategy: str) -> str:
    return STRATEGY_FAMILIES.get(strategy, DEFAULT_FAMILY)


class StrategyCorrelationTracker:
    """Rolling correlation matrix between per-strategy daily P&L streams."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite",
                 lookback_days: int = 60) -> None:
        self.db_path = db_path
        self.lookback_days = lookback_days

    def daily_pnl_streams(self) -> Dict[str, Dict[str, float]]:
        """strategy -> {date -> summed net pnl}."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT strategy, substr(exit_time, 1, 10) AS day, SUM(net_pnl) "
                    "FROM trade_memory WHERE strategy IS NOT NULL AND exit_time IS NOT NULL "
                    "GROUP BY strategy, day ORDER BY day DESC LIMIT 2000"
                ).fetchall()
        except sqlite3.OperationalError:
            return {}
        streams: Dict[str, Dict[str, float]] = defaultdict(dict)
        for strategy, day, pnl in rows:
            if day:
                streams[strategy][day] = float(pnl or 0.0)
        return dict(streams)

    def correlation_matrix(self) -> Dict[Tuple[str, str], float]:
        """Pairwise Pearson correlation of aligned daily P&L streams."""
        streams = self.daily_pnl_streams()
        names = sorted(streams)
        matrix: Dict[Tuple[str, str], float] = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                days = sorted(set(streams[a]) & set(streams[b]))[-self.lookback_days:]
                if len(days) < 10:
                    continue
                xs = [streams[a][d] for d in days]
                ys = [streams[b][d] for d in days]
                corr = _pearson(xs, ys)
                if corr is not None:
                    matrix[(a, b)] = corr
                    matrix[(b, a)] = corr
        return matrix

    def correlation_penalty(self, strategy: str,
                            active_strategies: Sequence[str],
                            matrix: Optional[Dict[Tuple[str, str], float]] = None) -> float:
        """Multiplier in (0, 1]: 1 = independent, lower = redundant.

        Uses realized P&L correlation when available; falls back to a family
        penalty (same family = assumed 0.6 correlation).
        """
        if matrix is None:
            matrix = self.correlation_matrix()
        worst = 0.0
        for other in active_strategies:
            if other == strategy:
                continue
            corr = matrix.get((strategy, other))
            if corr is None:
                corr = 0.6 if family_of(strategy) == family_of(other) else 0.1
            worst = max(worst, corr)
        return max(0.25, 1.0 - 0.75 * max(0.0, worst))

    def family_exposure(self, open_positions: Sequence[Dict]) -> Dict[str, float]:
        """Total open notional per strategy family."""
        exposure: Dict[str, float] = defaultdict(float)
        for pos in open_positions:
            strategy = pos.get("strategy") or "unknown"
            exposure[family_of(strategy)] += float(pos.get("size") or 0.0)
        return dict(exposure)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def evidence_aware_strategy_score(
    pnls: Sequence[float],
    *,
    regime_fit: float = 1.0,
    edge_health: float = 1.0,
    execution_quality: float = 1.0,
    correlation_penalty: float = 1.0,
    max_drawdown_frac: float = 0.0,
    z: float = 1.645,
) -> float:
    """Evidence-aware strategy quality score (replaces exp*wr*sqrt(n)).

    Uses the LOWER confidence bound of net expectancy (uncertainty-penalized,
    autocorrelation-adjusted sample size) multiplied by regime fit, edge
    health, execution quality and correlation penalty, with a drawdown
    penalty. Returns >= 0; 0 means "no evidence of positive edge".
    """
    n = len(pnls)
    if n < 5:
        return 0.0
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / max(n - 1, 1)
    ess = effective_sample_size(list(pnls))
    stderr = math.sqrt(var / max(ess, 1.0))
    lcb = mean - z * stderr          # lower confidence bound of expectancy
    if lcb <= 0:
        return 0.0
    dd_penalty = 1.0 / (1.0 + 5.0 * max(0.0, max_drawdown_frac))
    score = (lcb
             * _clamp01(regime_fit)
             * _clamp01(edge_health)
             * _clamp01(execution_quality)
             * _clamp01(correlation_penalty)
             * dd_penalty)
    return max(0.0, score)


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))
