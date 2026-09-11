"""
Trade Memory — permanent enriched record of every completed trade.

ARCHITECTURE:
    This is a read-optimised view over the `positions` table, enriched with:
        • Strategy version at time of trade
        • Full feature vector (from FeatureStore, if available)
        • Broker execution details (actual fill prices, fees, slippage)
        • Market regime at entry
        • Claude reasoning
        • Kronos prediction
        • ML meta-model prediction
        • Entry/exit quality scores
        • MFE / MAE
        • Holding time
        • All-in net return

    This database becomes the foundation for:
        • Strategy retraining
        • Experiment backtesting
        • Performance attribution

Usage:
    memory = TradeMemory()
    memory.record(position_dict, opportunity_id=opp_id)
    history = memory.get_history(days=30)
    report  = memory.performance_report()
"""
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

_CREATE_TRADE_MEMORY = """
CREATE TABLE IF NOT EXISTS trade_memory (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id         INTEGER UNIQUE,
    -- Identity
    symbol              TEXT    NOT NULL,
    asset_class         TEXT    DEFAULT 'crypto',
    strategy            TEXT,
    strategy_version    TEXT,
    -- Entry
    entry_time          TEXT,
    entry_price         REAL,
    entry_fill_price    REAL,
    entry_fees          REAL    DEFAULT 0.0,
    entry_slippage_pct  REAL    DEFAULT 0.0,
    signal              TEXT,
    direction           TEXT,
    -- Exit
    exit_time           TEXT,
    exit_price          REAL,
    exit_fill_price     REAL,
    exit_fees           REAL    DEFAULT 0.0,
    exit_slippage_pct   REAL    DEFAULT 0.0,
    close_reason        TEXT,
    -- Size
    size_dollars        REAL,
    quantity            REAL,
    -- Performance
    gross_pnl           REAL,
    total_fees          REAL    DEFAULT 0.0,
    net_pnl             REAL,
    net_return_pct      REAL,
    holding_hours       REAL,
    mfe_pct             REAL,
    mae_pct             REAL,
    -- Quality scores
    entry_quality       REAL    DEFAULT 0.5,    -- 0=bad, 1=perfect
    exit_quality        REAL    DEFAULT 0.5,
    -- Context
    market_regime       TEXT,
    regime_score        REAL,
    -- Predictions at entry time
    kronos_signal       TEXT,
    kronos_confidence   REAL,
    llm_decision        TEXT,
    llm_confidence      REAL,
    ml_prediction       TEXT,
    ml_confidence       REAL,
    -- Strategy signals (JSON)
    strategy_signals    TEXT,
    -- Features (JSON) — from FeatureStore
    features_json       TEXT,
    -- Experiment tracking
    experiment_id       TEXT,
    -- Metadata
    recorded_at         TEXT    NOT NULL,
    broker_entry_order  TEXT,
    broker_exit_order   TEXT
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_tm_symbol    ON trade_memory(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_tm_strategy  ON trade_memory(strategy)",
    "CREATE INDEX IF NOT EXISTS idx_tm_entry     ON trade_memory(entry_time)",
    "CREATE INDEX IF NOT EXISTS idx_tm_pos       ON trade_memory(position_id)",
    "CREATE INDEX IF NOT EXISTS idx_tm_regime    ON trade_memory(market_regime)",
    "CREATE INDEX IF NOT EXISTS idx_tm_strat_ver ON trade_memory(strategy_version)",
]


class TradeMemory:
    """
    Permanent, enriched, immutable trade log.

    Every closed trade is written here exactly once.
    Records are never modified after writing (append-only).
    """

    def __init__(self, db_path: str = "data/trade_memory.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TRADE_MEMORY)
            for idx in _INDEXES:
                conn.execute(idx)
            conn.commit()
        logger.info(f"TradeMemory ready: {self.db_path}")

    # ── Write API ──────────────────────────────────────────────────────────────

    def record(
        self,
        position: Dict,
        opportunity_id: Optional[int] = None,
        features: Optional[Dict] = None,
        experiment_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Write a closed position to permanent trade memory.

        Should be called once per position, immediately after it closes.
        Returns the new trade_memory row ID, or None if the position is
        already recorded.

        Args:
            position:       Closed position dict from PositionManager
            opportunity_id: FeatureStore opportunity ID for feature linkage
            features:       Feature dict at entry time (from FeatureStore)
            experiment_id:  If this trade was part of a canary experiment
        """
        if position.get('status') != 'CLOSED':
            logger.warning(f"TradeMemory: position {position.get('id')} is not CLOSED — skipping")
            return None

        pos_id = position.get('id')

        # Check for duplicate
        with sqlite3.connect(self.db_path) as conn:
            exists = conn.execute(
                "SELECT id FROM trade_memory WHERE position_id=?", (pos_id,)
            ).fetchone()
            if exists:
                logger.debug(f"TradeMemory: position {pos_id} already recorded")
                return exists[0]

        # Calculate derived fields
        entry_price      = position.get('entry_fill_price') or position.get('entry_price', 0)
        exit_price       = position.get('exit_fill_price') or position.get('exit_price', 0)
        entry_fees       = position.get('entry_fees', 0.0) or 0.0
        exit_fees_val    = position.get('exit_fees', 0.0) or 0.0
        total_fees       = entry_fees + exit_fees_val
        gross_pnl        = position.get('pnl', 0.0) or 0.0
        net_pnl          = position.get('net_pnl', gross_pnl - total_fees) or 0.0
        size             = position.get('size', 0.0)
        net_return_pct   = (net_pnl / size) if size > 0 else 0.0

        # Entry quality: how close was actual fill to signal price?
        entry_slippage   = 0.0
        if entry_price and position.get('entry_price'):
            entry_slippage = abs(entry_price - position['entry_price']) / position['entry_price']

        # Exit quality: how close was exit to optimal (target or stop)?
        exit_quality = self._estimate_exit_quality(position)
        entry_quality = max(0.0, 1.0 - entry_slippage * 100)

        # MFE / MAE from positions table
        mfe = position.get('mfe_pct', 0.0) or 0.0
        mae = position.get('mae_pct', 0.0) or 0.0

        # Holding time
        holding_hours = position.get('holding_hours')
        if holding_hours is None:
            try:
                entry_dt = datetime.fromisoformat(str(position.get('entry_time', '')))
                exit_dt  = datetime.fromisoformat(str(position.get('exit_time', '')))
                holding_hours = (exit_dt - entry_dt).total_seconds() / 3600
            except Exception:
                holding_hours = None

        # Features JSON
        features_json = json.dumps(features) if features else position.get('features_json')

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO trade_memory
                   (position_id, symbol, asset_class, strategy, strategy_version,
                    entry_time, entry_price, entry_fill_price, entry_fees,
                    entry_slippage_pct, signal, direction,
                    exit_time, exit_price, exit_fill_price, exit_fees,
                    exit_slippage_pct, close_reason,
                    size_dollars, gross_pnl, total_fees, net_pnl, net_return_pct,
                    holding_hours, mfe_pct, mae_pct,
                    entry_quality, exit_quality,
                    market_regime, kronos_signal, kronos_confidence,
                    llm_decision, strategy_signals, features_json,
                    experiment_id, recorded_at,
                    broker_entry_order, broker_exit_order)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    pos_id,
                    position.get('symbol', ''),
                    position.get('asset_class', 'crypto'),
                    position.get('strategy', ''),
                    position.get('strategy_version', ''),
                    str(position.get('entry_time', '')),
                    position.get('entry_price'),
                    entry_price,
                    entry_fees,
                    entry_slippage,
                    position.get('side', ''),
                    position.get('direction', ''),
                    str(position.get('exit_time', '')),
                    position.get('exit_price'),
                    exit_price,
                    exit_fees_val,
                    0.0,    # exit slippage (could compute if we stored signal price)
                    position.get('close_reason', ''),
                    size,
                    gross_pnl,
                    total_fees,
                    net_pnl,
                    net_return_pct,
                    holding_hours,
                    mfe,
                    mae,
                    entry_quality,
                    exit_quality,
                    position.get('regime', ''),
                    position.get('kronos_confidence', 0.0),   # signal not stored separately
                    position.get('kronos_confidence', 0.0),
                    position.get('llm_decision', ''),
                    position.get('strategies_used', '[]'),
                    features_json,
                    experiment_id,
                    _utcnow(),
                    position.get('broker_order_id', ''),
                    position.get('broker_exit_order_id', ''),
                ),
            )
            conn.commit()
            row_id = cur.lastrowid

        logger.info(
            f"TradeMemory: recorded position #{pos_id} {position.get('symbol')} "
            f"net_pnl=${net_pnl:+.2f} ({net_return_pct:+.1%})"
        )
        return row_id

    # ── Read API ───────────────────────────────────────────────────────────────

    def get_history(self, days: int = 90, strategy: str = None) -> List[Dict]:
        """Closed trades in the last N days, optionally filtered by strategy."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        sql    = "SELECT * FROM trade_memory WHERE entry_time >= ?"
        params = [cutoff]
        if strategy:
            sql    += " AND strategy = ?"
            params.append(strategy)
        sql += " ORDER BY entry_time DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def performance_report(self, days: int = 30) -> Dict:
        """Aggregated performance statistics for the last N days."""
        trades = self.get_history(days=days)
        if not trades:
            return {"status": "no_data", "days": days}

        pnls    = [t['net_pnl'] for t in trades if t['net_pnl'] is not None]
        wins    = [p for p in pnls if p > 0]
        losses  = [p for p in pnls if p <= 0]

        # Per-strategy breakdown
        strat_pnl: Dict[str, List[float]] = {}
        for t in trades:
            s = t.get('strategy', 'unknown') or 'unknown'
            strat_pnl.setdefault(s, []).append(t.get('net_pnl', 0.0) or 0.0)

        by_strategy = {}
        for s, s_pnls in strat_pnl.items():
            s_wins = [p for p in s_pnls if p > 0]
            by_strategy[s] = {
                'trades':     len(s_pnls),
                'win_rate':   len(s_wins) / len(s_pnls) if s_pnls else 0.0,
                'total_pnl':  sum(s_pnls),
                'avg_pnl':    sum(s_pnls) / len(s_pnls) if s_pnls else 0.0,
            }

        return {
            'days':           days,
            'total_trades':   len(pnls),
            'win_rate':       len(wins) / len(pnls) if pnls else 0.0,
            'total_pnl':      round(sum(pnls), 2),
            'avg_pnl':        round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
            'avg_win':        round(sum(wins) / len(wins), 2) if wins else 0.0,
            'avg_loss':       round(sum(losses) / len(losses), 2) if losses else 0.0,
            'expectancy':     round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
            'profit_factor':  round(sum(wins) / abs(sum(losses)), 2) if losses else float('inf'),
            'avg_holding_h':  round(
                sum(t['holding_hours'] for t in trades if t['holding_hours']) /
                max(sum(1 for t in trades if t['holding_hours']), 1), 1
            ),
            'by_strategy':    by_strategy,
        }

    @staticmethod
    def _estimate_exit_quality(position: Dict) -> float:
        """
        Estimate exit quality [0, 1] based on reason and P&L.

        Perfect exit (1.0) = take-profit hit
        Good exit (0.7)    = trailing stop after profit
        Neutral (0.5)      = time limit or signal reversal
        Poor exit (0.3)    = stop loss triggered
        Bad exit (0.1)     = emergency or circuit breaker exit
        """
        reason = (position.get('close_reason') or '').lower()
        pnl    = position.get('net_pnl', 0.0) or 0.0

        if 'take_profit' in reason or 'broker_take_profit' in reason:
            return 1.0
        if 'trailing' in reason and pnl > 0:
            return 0.75
        if 'partial' in reason:
            return 0.70
        if 'time_limit' in reason or 'signal_reversal' in reason:
            return 0.50
        if 'stop_loss' in reason or 'broker_stop_loss' in reason:
            return 0.30 if pnl < 0 else 0.60   # stopped for profit = good
        if 'circuit_breaker' in reason or 'emergency' in reason:
            return 0.10
        return 0.50
