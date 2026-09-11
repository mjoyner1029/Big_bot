"""
Strategy Health Monitor — continuous per-strategy health tracking.

Every strategy is monitored on rolling windows for:
    rolling_expectancy      rolling_win_rate      rolling_sharpe
    rolling_sortino         profit_factor         rolling_drawdown
    sample_size             regime_performance    avg_hold_time
    mfe_avg                 mae_avg

When health degrades below thresholds:
    1. Strategy is automatically paused
    2. An alert is registered for the Research Engine
    3. An experiment is proposed to investigate

Degradation is defined as:
    - Rolling win rate < MIN_WIN_RATE for MIN_SAMPLE trades
    - Rolling expectancy < MIN_EXPECTANCY
    - Drawdown > MAX_DRAWDOWN_PCT of capital
    - Sharpe < MIN_SHARPE for 20+ trades
"""
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

# ── Degradation thresholds ─────────────────────────────────────────────────
MIN_WIN_RATE    = 0.42
MIN_EXPECTANCY  = -8.0      # avg net PnL per trade
MAX_DRAWDOWN    = 0.05      # 5% of capital
MIN_SHARPE      = 0.3
MIN_SAMPLE      = 15        # minimum trades before judging health
RETIRE_MIN_TRADES = 30      # sample needed before retirement is considered
RETIRE_CONSECUTIVE_DEGRADED = 3  # consecutive degraded evaluations → retire


class EdgeState(str, Enum):
    HEALTHY  = "HEALTHY"
    WATCH    = "WATCH"       # statistically below expectation
    DEGRADED = "DEGRADED"    # significant persistent deterioration
    PAUSED   = "PAUSED"      # risk threshold violated
    RETIRED  = "RETIRED"     # edge no longer economically meaningful (permanent)


_CREATE_EDGE_STATE = """
CREATE TABLE IF NOT EXISTS edge_state (
    strategy            TEXT PRIMARY KEY,
    state               TEXT NOT NULL,
    reason              TEXT,
    consecutive_degraded INTEGER DEFAULT 0,
    updated_at          TEXT NOT NULL
)
"""


@dataclass
class StrategyHealth:
    strategy:         str
    window_days:      int
    trades:           int
    win_rate:         float
    expectancy:       float
    avg_win:          float
    avg_loss:         float
    profit_factor:    float
    sharpe:           float
    sortino:          float
    max_drawdown:     float
    avg_hold_hours:   float
    mfe_avg:          float
    mae_avg:          float
    healthy:          bool
    alerts:           List[str] = field(default_factory=list)
    computed_at:      str = field(default_factory=_utcnow)


@dataclass
class HealthAlert:
    strategy:    str
    metric:      str
    value:       float
    threshold:   float
    description: str
    severity:    str    # 'WARNING' | 'CRITICAL'
    created_at:  str = field(default_factory=_utcnow)


class StrategyHealthMonitor:
    """
    Monitors per-strategy performance health and generates alerts.

    Reads from TradeMemory / positions table. Fully read-only —
    does not modify any trading state.
    """

    def __init__(
        self,
        db_path: str = "data/trade_memory.sqlite",
        window_days: int = 14,
        capital: float = 10_000.0,
    ):
        self.db_path     = db_path
        self.window_days = window_days
        self.capital     = capital
        self._alerts: List[HealthAlert] = []
        self._paused_strategies: set = set()
        self._init_state_table()
        self._load_persisted_states()

    def _init_state_table(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(_CREATE_EDGE_STATE)
                conn.commit()
        except Exception as e:
            logger.warning(f"StrategyHealthMonitor: state table init failed: {e}")

    def _load_persisted_states(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT strategy, state FROM edge_state "
                    "WHERE state IN ('PAUSED','RETIRED')"
                ).fetchall()
            for strategy, _state in rows:
                self._paused_strategies.add(strategy)
        except Exception as e:
            logger.warning(f"StrategyHealthMonitor: state load failed: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, StrategyHealth]:
        """
        Compute health for all active strategies.

        Returns dict of strategy_name → StrategyHealth.
        """
        strategies = self._get_active_strategies()
        health_map: Dict[str, StrategyHealth] = {}
        self._alerts.clear()

        for strategy in strategies:
            health = self._compute_health(strategy)
            health_map[strategy] = health

            if not health.healthy:
                self._handle_degradation(health)
            elif health.trades >= MIN_SAMPLE and self.state_of(strategy) in (
                    EdgeState.HEALTHY, EdgeState.WATCH, EdgeState.DEGRADED):
                # PAUSED needs manual unpause; RETIRED is permanent
                self._paused_strategies.discard(strategy)
                self._set_state(strategy, EdgeState.HEALTHY, "metrics recovered",
                                reset_streak=True)

        return health_map

    def get_alerts(self) -> List[Dict]:
        """Return current alerts as dicts (for Research Engine)."""
        return [
            {
                'strategy':    a.strategy,
                'metric':      a.metric,
                'value':       a.value,
                'threshold':   a.threshold,
                'severity':    a.severity,
                'description': a.description,
                'created_at':  a.created_at,
            }
            for a in self._alerts
        ]

    def is_paused(self, strategy: str) -> bool:
        return strategy in self._paused_strategies

    def unpause(self, strategy: str) -> None:
        if self.state_of(strategy) == EdgeState.RETIRED:
            logger.warning(
                f"StrategyHealthMonitor: '{strategy}' is RETIRED — cannot unpause "
                "(retirement is permanent)"
            )
            return
        self._paused_strategies.discard(strategy)
        self._set_state(strategy, EdgeState.HEALTHY, "manual unpause", reset_streak=True)
        logger.info(f"StrategyHealthMonitor: unpaused '{strategy}'")

    def state_of(self, strategy: str) -> EdgeState:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT state FROM edge_state WHERE strategy=?", (strategy,)
                ).fetchone()
            return EdgeState(row[0]) if row else EdgeState.HEALTHY
        except Exception:
            return EdgeState.HEALTHY

    def retire(self, strategy: str, reason: str) -> None:
        """Permanently retire an edge (graveyard). Never auto-reactivated."""
        self._paused_strategies.add(strategy)
        self._set_state(strategy, EdgeState.RETIRED, reason)
        logger.warning(f"StrategyHealthMonitor: ⚰️ RETIRED '{strategy}': {reason}")

    def graveyard(self) -> List[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM edge_state WHERE state='RETIRED'"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def _set_state(self, strategy: str, state: EdgeState, reason: str,
                   reset_streak: bool = False, increment_streak: bool = False) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT consecutive_degraded FROM edge_state WHERE strategy=?",
                    (strategy,),
                ).fetchone()
                streak = row[0] if row else 0
                if reset_streak:
                    streak = 0
                elif increment_streak:
                    streak += 1
                conn.execute(
                    "INSERT INTO edge_state (strategy, state, reason, consecutive_degraded, updated_at) "
                    "VALUES (?,?,?,?,?) ON CONFLICT(strategy) DO UPDATE SET "
                    "state=excluded.state, reason=excluded.reason, "
                    "consecutive_degraded=excluded.consecutive_degraded, "
                    "updated_at=excluded.updated_at",
                    (strategy, state.value, reason, streak, _utcnow()),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"StrategyHealthMonitor: state persist failed: {e}")

    def _degraded_streak(self, strategy: str) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT consecutive_degraded FROM edge_state WHERE strategy=?",
                    (strategy,),
                ).fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    def summary(self) -> Dict:
        health_map = self.run()
        return {
            'healthy':   [s for s, h in health_map.items() if h.healthy],
            'degraded':  [s for s, h in health_map.items() if not h.healthy],
            'paused':    list(self._paused_strategies),
            'retired':   [g['strategy'] for g in self.graveyard()],
            'alerts':    self.get_alerts(),
        }

    # ── Health computation ────────────────────────────────────────────────────

    def _compute_health(self, strategy: str) -> StrategyHealth:
        pnls, holds, mfes, maes = self._load_strategy_trades(strategy)

        if len(pnls) < MIN_SAMPLE:
            # Insufficient data — mark as healthy to avoid false positives
            return StrategyHealth(
                strategy=strategy, window_days=self.window_days,
                trades=len(pnls), win_rate=0.5, expectancy=0.0,
                avg_win=0.0, avg_loss=0.0, profit_factor=1.0,
                sharpe=0.0, sortino=0.0, max_drawdown=0.0,
                avg_hold_hours=0.0, mfe_avg=0.0, mae_avg=0.0,
                healthy=True, alerts=['Insufficient data'],
            )

        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr     = len(wins) / len(pnls)
        exp    = sum(pnls) / len(pnls)
        avg_w  = sum(wins) / len(wins) if wins else 0.0
        avg_l  = sum(losses) / len(losses) if losses else 0.0
        pf     = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

        sharpe  = self._sharpe(pnls)
        sortino = self._sortino(pnls)
        drawdown = self._max_drawdown(pnls)
        hold_avg = sum(holds) / len(holds) if holds else 0.0
        mfe_avg  = sum(mfes) / len(mfes) if mfes else 0.0
        mae_avg  = sum(maes) / len(maes) if maes else 0.0

        alerts = []
        healthy = True

        if wr < MIN_WIN_RATE:
            alerts.append(f"win_rate={wr:.1%} < {MIN_WIN_RATE:.0%}")
            healthy = False
        if exp < MIN_EXPECTANCY:
            alerts.append(f"expectancy=${exp:.2f} < ${MIN_EXPECTANCY:.2f}")
            healthy = False
        if drawdown > MAX_DRAWDOWN * self.capital:
            alerts.append(f"drawdown=${drawdown:.2f} > {MAX_DRAWDOWN:.0%} limit")
            healthy = False
        if len(pnls) >= 20 and sharpe < MIN_SHARPE:
            alerts.append(f"sharpe={sharpe:.2f} < {MIN_SHARPE:.1f}")
            healthy = False

        return StrategyHealth(
            strategy=strategy, window_days=self.window_days,
            trades=len(pnls), win_rate=wr, expectancy=exp,
            avg_win=avg_w, avg_loss=avg_l, profit_factor=pf,
            sharpe=sharpe, sortino=sortino, max_drawdown=drawdown,
            avg_hold_hours=hold_avg, mfe_avg=mfe_avg, mae_avg=mae_avg,
            healthy=healthy, alerts=alerts,
        )

    def _handle_degradation(self, health: StrategyHealth) -> None:
        """Escalate WATCH → DEGRADED → PAUSED → RETIRED based on severity
        and persistence. Never retrains/modifies the strategy to erase losses."""
        logger.warning(
            f"StrategyHealthMonitor: '{health.strategy}' degraded — "
            f"{'; '.join(health.alerts)}"
        )
        n_breaches = len(health.alerts)
        dd_breach = health.max_drawdown > MAX_DRAWDOWN * self.capital
        streak = self._degraded_streak(health.strategy) + 1

        # RETIRED: persistent deterioration with enough evidence, edge no
        # longer economically meaningful
        if (streak >= RETIRE_CONSECUTIVE_DEGRADED
                and health.trades >= RETIRE_MIN_TRADES
                and health.expectancy < 0):
            self.retire(
                health.strategy,
                f"{streak} consecutive degraded evaluations, "
                f"expectancy=${health.expectancy:.2f} over {health.trades} trades",
            )
        elif dd_breach:
            # PAUSED: hard risk threshold violated
            self._paused_strategies.add(health.strategy)
            self._set_state(health.strategy, EdgeState.PAUSED,
                            '; '.join(health.alerts), increment_streak=True)
        elif n_breaches >= 2:
            # DEGRADED: significant deterioration — pause allocation
            self._paused_strategies.add(health.strategy)
            self._set_state(health.strategy, EdgeState.DEGRADED,
                            '; '.join(health.alerts), increment_streak=True)
        else:
            # WATCH: below expectation but not yet actionable — keep trading,
            # flag for review
            self._set_state(health.strategy, EdgeState.WATCH,
                            '; '.join(health.alerts), increment_streak=True)

        for alert_msg in health.alerts:
            metric = alert_msg.split('=')[0]
            self._alerts.append(HealthAlert(
                strategy=health.strategy,
                metric=metric,
                value=getattr(health, metric.replace('%', '_rate'), 0.0),
                threshold=0.0,
                description=f"Strategy '{health.strategy}': {alert_msg}",
                severity='CRITICAL' if health.win_rate < 0.35 else 'WARNING',
            ))

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _get_active_strategies(self) -> List[str]:
        """Get distinct strategy names from recent trades."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.window_days * 2)).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT strategy FROM positions WHERE status='CLOSED' "
                    "AND strategy IS NOT NULL AND exit_time >= ?",
                    (cutoff,),
                ).fetchall()
            return [r[0] for r in rows if r[0]]
        except Exception as e:
            logger.warning(f"StrategyHealthMonitor: DB error: {e}")
            return []

    def _load_strategy_trades(self, strategy: str):
        """Load PnL, hold times, MFE, MAE for a strategy in the window."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.window_days)).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT net_pnl, holding_hours, mfe_pct, mae_pct "
                    "FROM positions WHERE status='CLOSED' AND strategy=? AND exit_time>=?",
                    (strategy, cutoff),
                ).fetchall()
            pnls  = [r[0] for r in rows if r[0] is not None]
            holds = [r[1] for r in rows if r[1] is not None]
            mfes  = [r[2] for r in rows if r[2] is not None]
            maes  = [r[3] for r in rows if r[3] is not None]
            return pnls, holds, mfes, maes
        except Exception as e:
            logger.warning(f"StrategyHealthMonitor: load error for {strategy}: {e}")
            return [], [], [], []

    # ── Stats ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _sharpe(pnls: List[float]) -> float:
        if len(pnls) < 2:
            return 0.0
        avg = sum(pnls) / len(pnls)
        std = (sum((p - avg)**2 for p in pnls) / len(pnls)) ** 0.5
        return avg / std if std > 0 else 0.0

    @staticmethod
    def _sortino(pnls: List[float]) -> float:
        if len(pnls) < 2:
            return 0.0
        avg    = sum(pnls) / len(pnls)
        losses = [p for p in pnls if p < 0]
        if not losses:
            return float('inf')
        downside_std = (sum(p**2 for p in losses) / len(losses)) ** 0.5
        return avg / downside_std if downside_std > 0 else 0.0

    @staticmethod
    def _max_drawdown(pnls: List[float]) -> float:
        equity, peak, max_dd = 0.0, 0.0, 0.0
        for p in pnls:
            equity += p
            peak    = max(peak, equity)
            max_dd  = max(max_dd, peak - equity)
        return max_dd
