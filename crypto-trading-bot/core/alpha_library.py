"""Alpha Library — persistent registry of validated edges.

The single source of truth for WHICH edges may generate trading signals.
Research-only hypotheses never reach execution: only alphas in an eligible
lifecycle state (PAPER for paper trading, LIVE_* for real capital) may
contribute candidate signals to the trading loop.

Lifecycle:
    DISCOVERED -> VALIDATING -> {REJECTED | PAPER}
    PAPER -> {LIVE_ELIGIBLE | REJECTED}
    LIVE_ELIGIBLE -> LIVE_LIMITED -> LIVE_SCALED
    any live state -> DEGRADED -> PAUSED -> RETIRED
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


class AlphaState(str, Enum):
    DISCOVERED    = "DISCOVERED"
    VALIDATING    = "VALIDATING"
    REJECTED      = "REJECTED"
    PAPER         = "PAPER"
    LIVE_ELIGIBLE = "LIVE_ELIGIBLE"
    LIVE_LIMITED  = "LIVE_LIMITED"
    LIVE_SCALED   = "LIVE_SCALED"
    DEGRADED      = "DEGRADED"
    PAUSED        = "PAUSED"
    RETIRED       = "RETIRED"


# States allowed to produce candidate signals at all
SIGNAL_ELIGIBLE_STATES = frozenset({
    AlphaState.PAPER, AlphaState.LIVE_ELIGIBLE,
    AlphaState.LIVE_LIMITED, AlphaState.LIVE_SCALED,
})

# States allowed to route real capital
LIVE_ELIGIBLE_STATES = frozenset({
    AlphaState.LIVE_ELIGIBLE, AlphaState.LIVE_LIMITED, AlphaState.LIVE_SCALED,
})

_VALID_TRANSITIONS = {
    AlphaState.DISCOVERED:    {AlphaState.VALIDATING, AlphaState.REJECTED, AlphaState.RETIRED},
    AlphaState.VALIDATING:    {AlphaState.PAPER, AlphaState.REJECTED, AlphaState.RETIRED},
    AlphaState.REJECTED:      {AlphaState.VALIDATING, AlphaState.RETIRED},
    AlphaState.PAPER:         {AlphaState.LIVE_ELIGIBLE, AlphaState.REJECTED,
                               AlphaState.PAUSED, AlphaState.RETIRED},
    AlphaState.LIVE_ELIGIBLE: {AlphaState.LIVE_LIMITED, AlphaState.DEGRADED,
                               AlphaState.PAUSED, AlphaState.RETIRED},
    AlphaState.LIVE_LIMITED:  {AlphaState.LIVE_SCALED, AlphaState.DEGRADED,
                               AlphaState.PAUSED, AlphaState.RETIRED},
    AlphaState.LIVE_SCALED:   {AlphaState.DEGRADED, AlphaState.PAUSED, AlphaState.RETIRED},
    AlphaState.DEGRADED:      {AlphaState.LIVE_LIMITED, AlphaState.PAUSED,
                               AlphaState.RETIRED, AlphaState.PAPER},
    AlphaState.PAUSED:        {AlphaState.PAPER, AlphaState.LIVE_LIMITED, AlphaState.RETIRED},
    AlphaState.RETIRED:       set(),   # graveyard is permanent
}

_CREATE = """
CREATE TABLE IF NOT EXISTS alpha_library (
    alpha_id            TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    strategy_family     TEXT,
    strategy_id         TEXT,
    market              TEXT,
    universe            TEXT,
    conditions          TEXT,
    entry_logic         TEXT,
    exit_logic          TEXT,
    holding_period      TEXT,
    valid_regimes       TEXT,
    params              TEXT,
    -- evidence
    sample_size         INTEGER,
    effective_sample_size REAL,
    gross_expectancy    REAL,
    net_expectancy      REAL,
    median_return       REAL,
    win_rate            REAL,
    profit_factor       REAL,
    sharpe              REAL,
    sortino             REAL,
    max_drawdown        REAL,
    calmar              REAL,
    ci_lower            REAL,
    ci_upper            REAL,
    p_value             REAL,
    adjusted_significance INTEGER,
    deflated_sharpe     REAL,
    backtest_overfit_probability REAL,
    transaction_cost_break_even REAL,
    cost_stress_results TEXT,
    oos_metrics         TEXT,
    walk_forward_metrics TEXT,
    parameter_robustness_score REAL,
    event_concentration REAL,
    regime_stability    REAL,
    paper_metrics       TEXT,
    canary_metrics      TEXT,
    edge_health_score   REAL,
    lifecycle_state     TEXT NOT NULL DEFAULT 'DISCOVERED',
    state_reason        TEXT,
    experiment_id       TEXT,
    created_at          TEXT NOT NULL,
    last_validated_at   TEXT,
    last_signal_at      TEXT,
    updated_at          TEXT
)
"""

_CREATE_HISTORY = """
CREATE TABLE IF NOT EXISTS alpha_state_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    alpha_id    TEXT NOT NULL,
    from_state  TEXT,
    to_state    TEXT NOT NULL,
    reason      TEXT,
    changed_at  TEXT NOT NULL
)
"""

_JSON_FIELDS = {
    "universe", "conditions", "valid_regimes", "params", "cost_stress_results",
    "oos_metrics", "walk_forward_metrics", "paper_metrics", "canary_metrics",
    "entry_conditions", "exit_conditions", "invalidation_conditions",
    "invalid_regimes", "stop_policy", "profit_policy",
}

# Execution-contract columns added by migration (spec: machine-readable alphas)
_MIGRATION_COLUMNS = {
    "alpha_version": "TEXT DEFAULT '1'",
    "subfamily": "TEXT",
    "asset_class": "TEXT",
    "direction": "TEXT",                  # 'long' | 'short' | 'both'
    "entry_conditions": "TEXT",           # condition-DSL JSON (ANDed list)
    "entry_window": "TEXT",
    "exit_conditions": "TEXT",
    "exit_window": "TEXT",
    "stop_policy": "TEXT",                # JSON e.g. {"type":"pct","value":0.02}
    "profit_policy": "TEXT",
    "time_stop_bars": "INTEGER",
    "invalidation_conditions": "TEXT",
    "invalid_regimes": "TEXT",
    "min_liquidity_usd": "REAL",
}


@dataclass
class AlphaRecord:
    alpha_id: str
    name: str
    strategy_family: Optional[str] = None
    strategy_id: Optional[str] = None
    market: Optional[str] = None
    universe: List[str] = field(default_factory=list)
    lifecycle_state: AlphaState = AlphaState.DISCOVERED
    params: Dict[str, Any] = field(default_factory=dict)
    valid_regimes: List[str] = field(default_factory=list)
    net_expectancy: Optional[float] = None
    edge_health_score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class AlphaLibrary:
    """SQLite-backed registry of edges with enforced lifecycle transitions."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite") -> None:
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE)
            conn.execute(_CREATE_HISTORY)
            self._migrate(conn)
            conn.commit()

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Additive-only migration: add execution-contract columns if missing."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(alpha_library)")}
        for col, decl in _MIGRATION_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE alpha_library ADD COLUMN {col} {decl}")

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def register(self, alpha_id: str, name: str, **fields: Any) -> None:
        """Create a new alpha in DISCOVERED state (idempotent)."""
        cols = {"alpha_id": alpha_id, "name": name,
                "lifecycle_state": AlphaState.DISCOVERED.value,
                "created_at": _utcnow(), "updated_at": _utcnow()}
        for k, v in fields.items():
            cols[k] = json.dumps(v, default=str) if k in _JSON_FIELDS else v
        placeholders = ",".join("?" for _ in cols)
        names = ",".join(cols)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO alpha_library ({names}) VALUES ({placeholders})",
                list(cols.values()),
            )
            conn.commit()

    def update_evidence(self, alpha_id: str, **fields: Any) -> None:
        """Update validation/paper/canary evidence fields."""
        if not fields:
            return
        sets, vals = [], []
        for k, v in fields.items():
            sets.append(f"{k}=?")
            vals.append(json.dumps(v, default=str) if k in _JSON_FIELDS else v)
        sets.append("updated_at=?")
        vals.append(_utcnow())
        vals.append(alpha_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE alpha_library SET {', '.join(sets)} WHERE alpha_id=?", vals)
            conn.commit()

    def get(self, alpha_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM alpha_library WHERE alpha_id=?", (alpha_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        for k in _JSON_FIELDS:
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def transition(self, alpha_id: str, to_state: AlphaState, reason: str = "") -> bool:
        """Enforced state transition. Returns False (and logs) if illegal."""
        record = self.get(alpha_id)
        if record is None:
            logger.warning(f"AlphaLibrary: unknown alpha {alpha_id}")
            return False
        current = AlphaState(record["lifecycle_state"])
        if to_state == current:
            return True
        if to_state not in _VALID_TRANSITIONS.get(current, set()):
            logger.warning(
                f"AlphaLibrary: illegal transition {current.value} -> {to_state.value} "
                f"for {alpha_id} ({reason})"
            )
            return False
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE alpha_library SET lifecycle_state=?, state_reason=?, updated_at=? "
                "WHERE alpha_id=?",
                (to_state.value, reason, _utcnow(), alpha_id),
            )
            conn.execute(
                "INSERT INTO alpha_state_history (alpha_id, from_state, to_state, reason, changed_at) "
                "VALUES (?,?,?,?,?)",
                (alpha_id, current.value, to_state.value, reason, _utcnow()),
            )
            conn.commit()
        logger.info(f"AlphaLibrary: {alpha_id} {current.value} -> {to_state.value} ({reason})")
        return True

    def state_of(self, alpha_id: str) -> Optional[AlphaState]:
        record = self.get(alpha_id)
        return AlphaState(record["lifecycle_state"]) if record else None

    def mark_signal(self, alpha_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE alpha_library SET last_signal_at=? WHERE alpha_id=?",
                (_utcnow(), alpha_id),
            )
            conn.commit()

    # ── Eligibility queries (source of truth for the trading loop) ────────────

    def eligible_alphas(
        self,
        market: Optional[str] = None,
        regime: Optional[str] = None,
        live_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Alphas allowed to generate signals, optionally filtered by asset
        market and current regime."""
        states = LIVE_ELIGIBLE_STATES if live_only else SIGNAL_ELIGIBLE_STATES
        placeholders = ",".join("?" for _ in states)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM alpha_library WHERE lifecycle_state IN ({placeholders})",
                [s.value for s in states],
            ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            for k in _JSON_FIELDS:
                if d.get(k):
                    try:
                        d[k] = json.loads(d[k])
                    except (json.JSONDecodeError, TypeError):
                        pass
            if market and d.get("market") and d["market"] != market:
                continue
            regimes = d.get("valid_regimes") or []
            if regime and regimes and regime not in regimes:
                continue
            out.append(d)
        return out

    def is_live_approved(self, alpha_id: str) -> bool:
        state = self.state_of(alpha_id)
        return state in LIVE_ELIGIBLE_STATES if state else False

    def graveyard(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT alpha_id, name, strategy_family, state_reason, updated_at "
                "FROM alpha_library WHERE lifecycle_state='RETIRED'"
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> Dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT lifecycle_state, COUNT(*) FROM alpha_library GROUP BY lifecycle_state"
            ).fetchall()
        return {state: count for state, count in rows}
