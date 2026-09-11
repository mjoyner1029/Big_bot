"""
Model & Strategy Governance — permanent versioned records for all models and strategies.

PHASE 12 & 13

Every production prediction must be traceable back to the exact model that
generated it.  Every trade must reference exact strategy versions.

ModelGovernance
---------------
Records for every model:
    model_id            model_version         model_type
    training_timestamp  training_data_range   feature_version
    feature_list        hyperparameters       training_metrics
    validation_metrics  OOS_metrics           paper_metrics
    deployment_ts       retirement_ts         retirement_reason

StrategyGovernance
------------------
Records for every strategy:
    strategy_id         version               parameters
    code_version        creation_date         experiment_source
    validation_results  paper_results         live_results
    deployment_status

ConfigSnapshot
--------------
Immutable snapshot of every trading session config.
Every trade references: session_id + config_hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


# ─── Schemas ─────────────────────────────────────────────────────────────────────

_CREATE_MODELS = """
CREATE TABLE IF NOT EXISTS model_governance (
    model_id              TEXT PRIMARY KEY,
    model_version         TEXT NOT NULL,
    model_type            TEXT NOT NULL,
    training_timestamp    TEXT,
    training_data_start   TEXT,
    training_data_end     TEXT,
    feature_version       TEXT,
    feature_list          TEXT,
    hyperparameters       TEXT,
    training_metrics      TEXT,
    validation_metrics    TEXT,
    oos_metrics           TEXT,
    paper_metrics         TEXT,
    deployment_timestamp  TEXT,
    retirement_timestamp  TEXT,
    retirement_reason     TEXT,
    notes                 TEXT DEFAULT ''
)
"""

_CREATE_STRATEGIES = """
CREATE TABLE IF NOT EXISTS strategy_governance (
    strategy_id           TEXT PRIMARY KEY,
    strategy_name         TEXT NOT NULL,
    version               TEXT NOT NULL,
    parameters            TEXT,
    code_version          TEXT,
    creation_date         TEXT NOT NULL,
    experiment_source     TEXT,
    validation_results    TEXT,
    paper_results         TEXT,
    live_results          TEXT,
    deployment_status     TEXT DEFAULT 'DEVELOPMENT',
    notes                 TEXT DEFAULT ''
)
"""

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS session_configs (
    session_id     TEXT PRIMARY KEY,
    config_hash    TEXT NOT NULL,
    config_json    TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    ended_at       TEXT,
    trading_mode   TEXT DEFAULT 'PAPER',
    notes          TEXT DEFAULT ''
)
"""

_CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS prediction_audit (
    prediction_id  TEXT PRIMARY KEY,
    model_id       TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    position_id    TEXT,
    symbol         TEXT,
    decision       TEXT,
    confidence     REAL,
    expected_value REAL,
    input_features TEXT,
    output_raw     TEXT,
    predicted_at   TEXT NOT NULL
)
"""


# ─── Dataclasses ─────────────────────────────────────────────────────────────────

@dataclass
class ModelRecord:
    model_id:           str
    model_version:      str
    model_type:         str
    training_timestamp: Optional[str] = None
    training_data_start: Optional[str] = None
    training_data_end:  Optional[str] = None
    feature_version:    str = ''
    feature_list:       List[str] = field(default_factory=list)
    hyperparameters:    Dict = field(default_factory=dict)
    training_metrics:   Dict = field(default_factory=dict)
    validation_metrics: Dict = field(default_factory=dict)
    oos_metrics:        Dict = field(default_factory=dict)
    paper_metrics:      Dict = field(default_factory=dict)
    deployment_timestamp: Optional[str] = None
    retirement_timestamp: Optional[str] = None
    retirement_reason:  Optional[str] = None
    notes:              str = ''


@dataclass
class StrategyRecord:
    strategy_id:       str
    strategy_name:     str
    version:           str
    parameters:        Dict = field(default_factory=dict)
    code_version:      str = ''
    creation_date:     str = field(default_factory=_utcnow)
    experiment_source: str = ''
    validation_results: Dict = field(default_factory=dict)
    paper_results:     Dict = field(default_factory=dict)
    live_results:      Dict = field(default_factory=dict)
    deployment_status: str = 'DEVELOPMENT'
    notes:             str = ''


@dataclass
class SessionConfig:
    session_id:   str
    config_hash:  str
    config:       Dict
    started_at:   str = field(default_factory=_utcnow)
    ended_at:     Optional[str] = None
    trading_mode: str = 'PAPER'
    notes:        str = ''


# ─── Governance classes ───────────────────────────────────────────────────────────

class ModelGovernance:
    """Permanent versioned record of every model trained or deployed."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_MODELS)
            conn.execute(_CREATE_PREDICTIONS)
            conn.commit()

    def register(self, record: ModelRecord) -> str:
        """Save a new model record. Returns model_id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO model_governance VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.model_id, record.model_version, record.model_type,
                 record.training_timestamp, record.training_data_start,
                 record.training_data_end, record.feature_version,
                 json.dumps(record.feature_list), json.dumps(record.hyperparameters),
                 json.dumps(record.training_metrics), json.dumps(record.validation_metrics),
                 json.dumps(record.oos_metrics), json.dumps(record.paper_metrics),
                 record.deployment_timestamp, record.retirement_timestamp,
                 record.retirement_reason, record.notes),
            )
            conn.commit()
        logger.info(f"ModelGovernance: registered model {record.model_id[:8]} v{record.model_version}")
        return record.model_id

    def update_metrics(self, model_id: str, stage: str, metrics: Dict) -> None:
        """Update metrics for a given stage (training/validation/oos/paper)."""
        allowed = {'training', 'validation', 'oos', 'paper'}
        if stage not in allowed:
            raise ValueError(f"stage must be one of {allowed}")
        col = f"{stage}_metrics"   # column name in DB
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE model_governance SET {col}=? WHERE model_id=?",
                (json.dumps(metrics), model_id),
            )
            conn.commit()

    def mark_deployed(self, model_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE model_governance SET deployment_timestamp=? WHERE model_id=?",
                (_utcnow(), model_id),
            )
            conn.commit()

    def mark_retired(self, model_id: str, reason: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE model_governance SET retirement_timestamp=?,retirement_reason=? WHERE model_id=?",
                (_utcnow(), reason, model_id),
            )
            conn.commit()

    def record_prediction(
        self,
        model_id:      str,
        model_version: str,
        decision:      str,
        confidence:    float,
        expected_value: float = 0.0,
        position_id:   str = '',
        symbol:        str = '',
        input_features: Dict = None,
        output_raw:    str = '',
    ) -> str:
        """Audit log every model prediction for traceability."""
        pred_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO prediction_audit VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (pred_id, model_id, model_version, position_id or '', symbol,
                 decision, confidence, expected_value,
                 json.dumps(input_features or {}), output_raw, _utcnow()),
            )
            conn.commit()
        return pred_id

    def get(self, model_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM model_governance WHERE model_id=?", (model_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_active(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM model_governance WHERE deployment_timestamp IS NOT NULL "
                "AND retirement_timestamp IS NULL ORDER BY deployment_timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]


class StrategyGovernance:
    """Permanent versioned record of every strategy deployed."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_STRATEGIES)
            conn.commit()

    def register(self, record: StrategyRecord) -> str:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO strategy_governance VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.strategy_id, record.strategy_name, record.version,
                 json.dumps(record.parameters), record.code_version,
                 record.creation_date, record.experiment_source,
                 json.dumps(record.validation_results), json.dumps(record.paper_results),
                 json.dumps(record.live_results), record.deployment_status, record.notes),
            )
            conn.commit()
        logger.info(f"StrategyGovernance: registered {record.strategy_name} v{record.version}")
        return record.strategy_id

    def update_results(self, strategy_id: str, stage: str, results: Dict) -> None:
        col = f"{stage}_results"
        allowed = {'validation', 'paper', 'live'}
        if stage not in allowed:
            raise ValueError(f"stage must be one of {allowed}")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE strategy_governance SET {col}=? WHERE strategy_id=?",
                (json.dumps(results), strategy_id),
            )
            conn.commit()

    def update_status(self, strategy_id: str, status: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE strategy_governance SET deployment_status=? WHERE strategy_id=?",
                (status, strategy_id),
            )
            conn.commit()

    def get(self, strategy_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM strategy_governance WHERE strategy_id=?", (strategy_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_by_name(self, name: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM strategy_governance WHERE strategy_name=? ORDER BY creation_date DESC",
                (name,),
            ).fetchall()
        return [dict(r) for r in rows]


class ConfigSnapshot:
    """
    Saves an immutable configuration snapshot for every trading session.

    Every trade should reference:
        session_id    — unique session identifier
        config_hash   — SHA256 of the full config

    This allows complete historical reproducibility.
    """

    def __init__(self, db_path: str = "data/trade_memory.sqlite"):
        self.db_path      = db_path
        self._session_id  = None
        self._config_hash = None
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_SESSIONS)
            conn.commit()

    def start_session(self, config: Dict, trading_mode: str = 'PAPER', notes: str = '') -> str:
        """
        Create a new session with an immutable config snapshot.

        Returns session_id.
        """
        config_str  = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        session_id  = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO session_configs "
                "(session_id,config_hash,config_json,started_at,trading_mode,notes) VALUES (?,?,?,?,?,?)",
                (session_id, config_hash, config_str, _utcnow(), trading_mode, notes),
            )
            conn.commit()

        self._session_id  = session_id
        self._config_hash = config_hash
        logger.info(f"ConfigSnapshot: session {session_id[:8]} hash={config_hash}")
        return session_id

    def end_session(self, session_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE session_configs SET ended_at=? WHERE session_id=?",
                (_utcnow(), session_id),
            )
            conn.commit()

    @property
    def current_session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def current_config_hash(self) -> Optional[str]:
        return self._config_hash

    def get_session(self, session_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM session_configs WHERE session_id=?", (session_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d['config'] = json.loads(d.pop('config_json', '{}'))
        return d

    def reproduce(self, session_id: str) -> Optional[Dict]:
        """Return the exact config needed to reproduce a historical session."""
        return self.get_session(session_id)
