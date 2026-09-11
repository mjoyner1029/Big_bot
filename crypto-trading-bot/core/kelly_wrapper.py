"""
Kelly Criterion position sizing — fully unified with PositionManager schema.

Reads ONLY from the canonical `positions` table (status='CLOSED').
Provides:
    • Overall Kelly (all closed trades)
    • Strategy-specific Kelly
    • Regime-specific Kelly (when >= MIN_SAMPLE trades in that regime)

Never silently falls back without logging a reason.
Falls back to DEFAULT_SIZE_PCT only when sample size < MIN_SAMPLE.
"""
import sqlite3
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from core.kelly_sizer import calculate_kelly_size

logger = logging.getLogger(__name__)

MIN_SAMPLE    = 10      # Minimum closed trades before trusting Kelly
REGIME_SAMPLE = 8       # Minimum trades in a regime for regime-specific sizing
DEFAULT_PCT   = 0.05    # Conservative 5% fallback when sample is insufficient
MAX_PCT       = 0.15    # Hard cap: never bet more than 15% of capital


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_pnls(
    db_path: str,
    lookback_days: int,
    strategy: Optional[str] = None,
    regime: Optional[str] = None,
) -> List[float]:
    """Fetch net_pnl for closed positions matching filters."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    conditions = ["status='CLOSED'", "exit_time >= ?", "net_pnl IS NOT NULL"]
    params: list = [cutoff.isoformat()]

    if strategy:
        conditions.append("strategy = ?")
        params.append(strategy)
    if regime:
        conditions.append("regime = ?")
        params.append(regime)

    sql = f"SELECT net_pnl FROM positions WHERE {' AND '.join(conditions)}"

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [float(r[0]) for r in rows if r[0] is not None]
    except sqlite3.Error as e:
        logger.warning(f"KellySizer DB error: {e}")
        return []


def _stats_from_pnls(pnls: List[float]) -> Optional[Dict]:
    """Compute Kelly stats from a list of P&L values."""
    if not pnls:
        return None
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "trades":   len(pnls),
        "win_rate": len(wins) / len(pnls),
        "avg_win":  sum(wins)   / len(wins)   if wins   else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "expectancy": sum(pnls) / len(pnls),
    }


def _kelly_from_stats(stats: Dict, capital: float, fractional: float = 0.5) -> float:
    """Convert stats dict into a dollar position size."""
    return calculate_kelly_size(
        win_rate=stats["win_rate"] * 100,   # kelly_sizer expects 0-100
        avg_win=stats["avg_win"],
        avg_loss=stats["avg_loss"],
        capital=capital,
        fractional=fractional,
    )


# ─────────────────────────────────────────────────────────────────────────────
# KellySizer public class
# ─────────────────────────────────────────────────────────────────────────────

class KellySizer:
    """
    Position sizer backed by real closed-trade statistics from the positions DB.

    Sizing hierarchy (most to least specific):
        1. Regime + Strategy   — if >= REGIME_SAMPLE trades
        2. Strategy only       — if >= MIN_SAMPLE trades
        3. Regime only         — if >= REGIME_SAMPLE trades
        4. All trades          — if >= MIN_SAMPLE trades
        5. Conservative default (DEFAULT_PCT of capital)

    Half-Kelly is applied at every level for safety.
    Position size is capped at MAX_PCT of capital.
    """

    def __init__(
        self,
        db_path: str = "data/trade_memory.sqlite",
        lookback_days: int = 30,
        fractional: float = 0.5,
    ):
        self.db_path      = db_path
        self.lookback_days = lookback_days
        self.fractional   = fractional

    # ── Main entry point ──────────────────────────────────────────────────────

    def get_position_size(
        self,
        symbol: str,
        capital: float,
        leverage: float = 1.0,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> float:
        """Return dollar position size for this symbol/strategy/regime.

        Args:
            symbol:   Asset symbol (used for logging only)
            capital:  Total trading capital
            leverage: Leverage multiplier (applied after Kelly)
            strategy: Strategy name for strategy-specific sizing
            regime:   Current market regime label

        Returns:
            Dollar position size, already leverage-adjusted and capped.
        """
        size, source = self._compute_kelly(capital, strategy, regime)
        adjusted = min(size * leverage, capital * MAX_PCT)

        logger.debug(
            f"Kelly [{symbol}] source={source} raw=${size:.2f} "
            f"lev={leverage}x capped=${adjusted:.2f} "
            f"({adjusted/capital*100:.1f}% of ${capital:,.0f})"
        )
        return adjusted

    # ── Sizing logic ──────────────────────────────────────────────────────────

    def _compute_kelly(
        self,
        capital: float,
        strategy: Optional[str],
        regime: Optional[str],
    ) -> Tuple[float, str]:
        """
        Returns (dollar_size, source_description).

        Walks down the hierarchy until a sample with >= MIN_SAMPLE is found.
        """
        default = (capital * DEFAULT_PCT, "default_insufficient_data")

        # 1. Regime + Strategy
        if strategy and regime:
            pnls = _fetch_pnls(self.db_path, self.lookback_days, strategy=strategy, regime=regime)
            stats = _stats_from_pnls(pnls)
            if stats and stats["trades"] >= REGIME_SAMPLE:
                size = _kelly_from_stats(stats, capital, self.fractional)
                return size, f"regime+strategy({strategy}/{regime} n={stats['trades']})"

        # 2. Strategy only
        if strategy:
            pnls = _fetch_pnls(self.db_path, self.lookback_days, strategy=strategy)
            stats = _stats_from_pnls(pnls)
            if stats and stats["trades"] >= MIN_SAMPLE:
                size = _kelly_from_stats(stats, capital, self.fractional)
                return size, f"strategy({strategy} n={stats['trades']})"

        # 3. Regime only
        if regime:
            pnls = _fetch_pnls(self.db_path, self.lookback_days, regime=regime)
            stats = _stats_from_pnls(pnls)
            if stats and stats["trades"] >= REGIME_SAMPLE:
                size = _kelly_from_stats(stats, capital, self.fractional)
                return size, f"regime({regime} n={stats['trades']})"

        # 4. All trades
        pnls = _fetch_pnls(self.db_path, self.lookback_days)
        stats = _stats_from_pnls(pnls)
        if stats and stats["trades"] >= MIN_SAMPLE:
            size = _kelly_from_stats(stats, capital, self.fractional)
            return size, f"all_trades(n={stats['trades']})"

        # 5. Fallback
        n = len(pnls)
        logger.info(
            f"Kelly fallback: only {n} closed trades (need {MIN_SAMPLE}), "
            f"using {DEFAULT_PCT:.0%} default"
        )
        return default

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_strategy_report(self, capital: float) -> Dict[str, Dict]:
        """
        Return per-strategy Kelly stats for all strategies with closed trades.
        Useful for strategy weighting and reporting.
        """
        sql = """
            SELECT strategy, net_pnl
            FROM positions
            WHERE status='CLOSED'
              AND exit_time >= ?
              AND net_pnl IS NOT NULL
              AND strategy IS NOT NULL
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        report = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(sql, [cutoff.isoformat()]).fetchall()
        except sqlite3.Error as e:
            logger.warning(f"Kelly report DB error: {e}")
            return {}

        by_strategy: Dict[str, List[float]] = defaultdict(list)
        for strat, pnl in rows:
            by_strategy[strat].append(float(pnl))

        for strat, pnls in by_strategy.items():
            stats = _stats_from_pnls(pnls)
            if stats:
                if stats["trades"] >= MIN_SAMPLE:
                    size = _kelly_from_stats(stats, capital, self.fractional)
                    size = min(size, capital * MAX_PCT)
                else:
                    size = capital * DEFAULT_PCT
                report[strat] = {
                    **stats,
                    "kelly_size":  round(size, 2),
                    "kelly_pct":   round(size / capital, 4),
                    "has_edge":    stats["trades"] >= MIN_SAMPLE,
                }

        return report
