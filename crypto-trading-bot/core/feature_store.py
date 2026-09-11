"""
Feature Store — persistent ML training dataset.

Every market opportunity evaluated by the bot is recorded here, regardless
of whether the bot decides to trade it. Outcomes are appended later once
the future candles are available (5m, 15m, 1h, 4h).

This database becomes the training set for the MetaModel (Phase 8).

Schema:
    opportunities   — one row per candidate evaluated
    outcomes        — forward-price outcomes appended asynchronously

Usage:
    store = FeatureStore()

    # Record before making trade decision
    opp_id = store.record_opportunity(
        symbol='BTC-USD',
        features={...},
        strategy_outputs={...},
        decision='TRADE',   # or 'NO_TRADE'
    )

    # Append outcome when candles are available (run in a background job)
    store.update_outcome(opp_id, {...})
"""
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

_CREATE_OPPORTUNITIES = """
CREATE TABLE IF NOT EXISTS opportunities (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    symbol              TEXT    NOT NULL,
    market              TEXT    DEFAULT 'crypto',
    price               REAL,
    volume_24h          REAL,
    spread_pct          REAL,
    atr_pct             REAL,
    adx                 REAL,
    rsi_14              REAL,
    vwap                REAL,
    trend               TEXT,
    volatility_regime   TEXT,
    market_regime       TEXT,
    -- strategy votes (JSON)
    strategy_outputs    TEXT,
    -- Kronos prediction
    kronos_signal       TEXT,
    kronos_confidence   REAL,
    -- Claude analysis
    llm_analysis        TEXT,
    llm_confidence      REAL,
    -- Composite scores
    opportunity_score   REAL,
    signal_confidence   REAL,
    -- Decision
    decision            TEXT    NOT NULL,   -- 'TRADE' | 'NO_TRADE'
    reject_reason       TEXT,
    -- Link to trade (if executed)
    position_id         INTEGER,
    experiment_id       TEXT,
    strategy_version    TEXT
)
"""

_CREATE_OUTCOMES = """
CREATE TABLE IF NOT EXISTS opportunity_outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id  INTEGER NOT NULL REFERENCES opportunities(id),
    updated_at      TEXT    NOT NULL,
    -- Forward returns
    return_5m       REAL,
    return_15m      REAL,
    return_1h       REAL,
    return_4h       REAL,
    -- Extremes
    mfe_pct         REAL,   -- Maximum Favorable Excursion (best return in window)
    mae_pct         REAL,   -- Maximum Adverse Excursion (worst return in window)
    -- Hit flags
    target_hit      INTEGER,    -- 1 = take-profit triggered
    stop_hit        INTEGER,    -- 1 = stop-loss triggered
    -- Realized outcome (if trade was taken)
    realized_pnl    REAL,
    realized_return REAL,
    holding_hours   REAL
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_opp_symbol    ON opportunities(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_opp_timestamp ON opportunities(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_opp_decision  ON opportunities(decision)",
    "CREATE INDEX IF NOT EXISTS idx_out_opp       ON opportunity_outcomes(opportunity_id)",
]


@dataclass
class OpportunityFeatures:
    """All features recorded for a single market opportunity."""
    symbol:             str
    market:             str       = 'crypto'
    price:              float     = 0.0
    volume_24h:         float     = 0.0
    spread_pct:         float     = 0.0
    atr_pct:            float     = 0.0
    adx:                float     = 0.0
    rsi_14:             float     = 50.0
    vwap:               float     = 0.0
    trend:              str       = 'neutral'
    volatility_regime:  str       = 'normal'
    market_regime:      str       = 'unknown'
    strategy_outputs:   Dict      = field(default_factory=dict)
    kronos_signal:      str       = 'NEUTRAL'
    kronos_confidence:  float     = 0.0
    llm_analysis:       str       = ''
    llm_confidence:     float     = 0.0
    opportunity_score:  float     = 0.0
    signal_confidence:  float     = 0.0


@dataclass
class TradeOutcome:
    """Forward-price outcomes appended once candles are available."""
    return_5m:      Optional[float] = None
    return_15m:     Optional[float] = None
    return_1h:      Optional[float] = None
    return_4h:      Optional[float] = None
    mfe_pct:        Optional[float] = None
    mae_pct:        Optional[float] = None
    target_hit:     Optional[bool]  = None
    stop_hit:       Optional[bool]  = None
    realized_pnl:   Optional[float] = None
    realized_return: Optional[float]= None
    holding_hours:  Optional[float] = None


class FeatureStore:
    """
    Persistent store for all market opportunities and their outcomes.

    Thread-safe via SQLite's WAL mode.
    """

    def __init__(self, db_path: str = "data/feature_store.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_OPPORTUNITIES)
            conn.execute(_CREATE_OUTCOMES)
            self._migrate(conn)  # Run migration BEFORE creating indexes
            for idx in _INDEXES:
                conn.execute(idx)
            conn.commit()
        logger.info(f"FeatureStore ready: {self.db_path}")

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add any missing columns for forward compatibility."""
        cursor = conn.execute("PRAGMA table_info(opportunities)")
        existing = {row[1] for row in cursor.fetchall()}
        additions = {
            'symbol': "TEXT NOT NULL DEFAULT ''",
            'timestamp': "TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z'",
            'decision': "TEXT NOT NULL DEFAULT 'NO_TRADE'",
            'market': "TEXT DEFAULT 'crypto'",
            'ml_prediction': "TEXT",
            'experiment_id': "TEXT",
            'strategy_version': "TEXT",
        }
        for col, defn in additions.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE opportunities ADD COLUMN {col} {defn}")

    # ── Write API ──────────────────────────────────────────────────────────────

    def record_opportunity(
        self,
        features: OpportunityFeatures,
        decision: str,                   # 'TRADE' | 'NO_TRADE'
        reject_reason: Optional[str] = None,
        position_id: Optional[int] = None,
        experiment_id: Optional[str] = None,
        strategy_version: Optional[str] = None,
        ml_prediction: Optional[str] = None,
    ) -> int:
        """
        Record a market opportunity that the bot evaluated.

        Returns the new row ID (use this to call update_outcome later).
        """
        if decision not in ('TRADE', 'NO_TRADE'):
            raise ValueError(f"decision must be 'TRADE' or 'NO_TRADE', got '{decision}'")

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO opportunities
                   (timestamp, symbol, market, price, volume_24h, spread_pct,
                    atr_pct, adx, rsi_14, vwap, trend, volatility_regime,
                    market_regime, strategy_outputs, kronos_signal,
                    kronos_confidence, llm_analysis, llm_confidence,
                    opportunity_score, signal_confidence, decision,
                    reject_reason, position_id, experiment_id,
                    strategy_version, ml_prediction)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    _utcnow(), features.symbol, features.market,
                    features.price, features.volume_24h, features.spread_pct,
                    features.atr_pct, features.adx, features.rsi_14,
                    features.vwap, features.trend, features.volatility_regime,
                    features.market_regime,
                    json.dumps(features.strategy_outputs),
                    features.kronos_signal, features.kronos_confidence,
                    features.llm_analysis, features.llm_confidence,
                    features.opportunity_score, features.signal_confidence,
                    decision, reject_reason, position_id,
                    experiment_id, strategy_version, ml_prediction,
                ),
            )
            conn.commit()
            opp_id = cur.lastrowid

        logger.debug(
            f"FeatureStore: recorded {decision} for {features.symbol} "
            f"(id={opp_id}, score={features.opportunity_score:.3f})"
        )
        return opp_id

    def update_outcome(self, opportunity_id: int, outcome: TradeOutcome) -> bool:
        """
        Append forward-price outcomes to a previously recorded opportunity.

        Call this asynchronously once the required candles are available.
        Returns True on success.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO opportunity_outcomes
                   (opportunity_id, updated_at, return_5m, return_15m,
                    return_1h, return_4h, mfe_pct, mae_pct,
                    target_hit, stop_hit, realized_pnl,
                    realized_return, holding_hours)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    opportunity_id, _utcnow(),
                    outcome.return_5m, outcome.return_15m,
                    outcome.return_1h, outcome.return_4h,
                    outcome.mfe_pct, outcome.mae_pct,
                    int(outcome.target_hit) if outcome.target_hit is not None else None,
                    int(outcome.stop_hit) if outcome.stop_hit is not None else None,
                    outcome.realized_pnl, outcome.realized_return,
                    outcome.holding_hours,
                ),
            )
            conn.commit()
        return True

    def link_position(self, opportunity_id: int, position_id: int) -> None:
        """Link a feature store row to a position that was executed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE opportunities SET position_id=? WHERE id=?",
                (position_id, opportunity_id),
            )
            conn.commit()

    # ── Read API ───────────────────────────────────────────────────────────────

    def get_training_dataset(
        self,
        min_rows: int = 50,
        with_outcomes_only: bool = True,
    ) -> List[Dict]:
        """
        Return the full training dataset for the MetaModel.

        Each row combines opportunity features and their outcomes.
        Only rows with at least return_1h are included by default
        (outcomes must be appended before training).
        """
        sql = """
            SELECT o.*, oo.return_5m, oo.return_15m, oo.return_1h, oo.return_4h,
                   oo.mfe_pct, oo.mae_pct, oo.target_hit, oo.stop_hit,
                   oo.realized_pnl, oo.realized_return, oo.holding_hours
            FROM opportunities o
        """
        if with_outcomes_only:
            sql += " INNER JOIN opportunity_outcomes oo ON o.id = oo.opportunity_id"
        else:
            sql += " LEFT JOIN opportunity_outcomes oo ON o.id = oo.opportunity_id"
        sql += " ORDER BY o.timestamp DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql).fetchall()

        result = [dict(r) for r in rows]

        if len(result) < min_rows:
            logger.warning(
                f"FeatureStore: only {len(result)} rows with outcomes "
                f"(need {min_rows} for reliable training)"
            )

        return result

    def get_recent_opportunities(self, limit: int = 100) -> List[Dict]:
        """Most recent opportunities regardless of outcome."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM opportunities ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict:
        """Summary stats for monitoring."""
        with sqlite3.connect(self.db_path) as conn:
            total   = conn.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0]
            trades  = conn.execute(
                "SELECT COUNT(*) FROM opportunities WHERE decision='TRADE'"
            ).fetchone()[0]
            with_outcomes = conn.execute(
                "SELECT COUNT(DISTINCT opportunity_id) FROM opportunity_outcomes"
            ).fetchone()[0]
        return {
            "total_opportunities": total,
            "trade_decisions":     trades,
            "no_trade_decisions":  total - trades,
            "with_outcomes":       with_outcomes,
            "outcome_coverage":    with_outcomes / total if total else 0.0,
        }
