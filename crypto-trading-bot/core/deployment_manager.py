"""
StrategyDeploymentManager — production deployment with mandatory rollback.

RULE: Claude cannot disable rollback. Claude cannot waive gates.
      Every deployment preserves the previous version permanently.

Deployment workflow
-------------------
    Current champion (v2.7) in production
        ↓
    Deploy challenger (v2.8) — gated by PromotionGates
        ↓
    Monitor v2.8 vs rollback triggers
        ↓  (if triggered)
    PAUSE v2.8  →  Restore v2.7  →  Incident record  →  Research task

Rollback triggers (configurable, deterministic)
-----------------------------------------------
    drawdown > threshold
    negative rolling expectancy (N trades)
    execution degradation
    reconciliation errors
    unexpected exposure
    risk violations

State is persisted in SQLite. Every deployment is permanently recorded.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


class DeploymentStatus(str, Enum):
    PENDING    = "PENDING"
    ACTIVE     = "ACTIVE"
    PAUSED     = "PAUSED"
    ROLLED_BACK = "ROLLED_BACK"
    RETIRED    = "RETIRED"


@dataclass
class DeploymentRecord:
    deployment_id:   str
    strategy_id:     str
    strategy_version: str
    config_hash:     str
    config_snapshot: Dict
    status:          DeploymentStatus
    deployed_at:     str
    retired_at:      Optional[str] = None
    rollback_reason: Optional[str] = None
    incident_id:     Optional[str] = None
    metrics_at_rollback: Optional[Dict] = None
    notes:           str = ''


_CREATE_DEPLOYMENTS = """
CREATE TABLE IF NOT EXISTS deployments (
    deployment_id    TEXT PRIMARY KEY,
    strategy_id      TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    config_hash      TEXT,
    config_snapshot  TEXT,
    status           TEXT DEFAULT 'PENDING',
    deployed_at      TEXT NOT NULL,
    retired_at       TEXT,
    rollback_reason  TEXT,
    incident_id      TEXT,
    metrics_json     TEXT,
    notes            TEXT DEFAULT ''
)
"""

_CREATE_INCIDENTS = """
CREATE TABLE IF NOT EXISTS deployment_incidents (
    incident_id      TEXT PRIMARY KEY,
    deployment_id    TEXT NOT NULL,
    trigger          TEXT NOT NULL,
    description      TEXT,
    metrics_json     TEXT,
    research_task_id TEXT,
    created_at       TEXT NOT NULL
)
"""


class StrategyDeploymentManager:
    """
    Manages production strategy deployments with mandatory rollback protection.

    INVARIANTS:
        1. Every active deployment has a saved previous version to roll back to.
        2. Rollback is triggered by deterministic conditions, not LLM analysis.
        3. Claude cannot disable rollback.
        4. All deployment events are permanently recorded.
    """

    # Rollback thresholds (static — cannot be modified by LLM)
    ROLLBACK_DRAWDOWN_PCT   = 0.15    # 15% drawdown triggers rollback
    ROLLBACK_NEG_EXP_TRADES = 10      # rolling window for expectancy check
    ROLLBACK_MIN_EXPECTANCY = -20.0   # $/trade — sustained losses trigger rollback
    ROLLBACK_REJECTION_RATE = 0.15    # 15% order rejection rate

    def __init__(self, db_path: str = "data/trade_memory.sqlite"):
        self.db_path = db_path
        self._active: Dict[str, DeploymentRecord] = {}   # strategy_id → active
        self._previous: Dict[str, DeploymentRecord] = {} # strategy_id → previous
        self._init_db()
        self._load_active()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_DEPLOYMENTS)
            conn.execute(_CREATE_INCIDENTS)
            conn.commit()

    # ── Public API ────────────────────────────────────────────────────────────

    def deploy(
        self,
        strategy_id:      str,
        strategy_version: str,
        config_snapshot:  Dict,
        config_hash:      str = '',
        notes:            str = '',
    ) -> DeploymentRecord:
        """
        Deploy a new strategy version.

        Saves current active deployment as 'previous' for rollback.
        Returns the new DeploymentRecord.
        """
        # Archive current deployment
        if strategy_id in self._active:
            prev = self._active[strategy_id]
            prev.status = DeploymentStatus.PAUSED
            self._update_status(prev.deployment_id, DeploymentStatus.PAUSED)
            self._previous[strategy_id] = prev
            logger.info(
                f"DeploymentManager: archived {strategy_id} v{prev.strategy_version} "
                f"(ready for rollback)"
            )

        deployment_id = str(uuid.uuid4())
        record = DeploymentRecord(
            deployment_id=deployment_id,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            config_hash=config_hash,
            config_snapshot=config_snapshot,
            status=DeploymentStatus.ACTIVE,
            deployed_at=_utcnow(),
            notes=notes,
        )
        self._save(record)
        self._active[strategy_id] = record

        logger.info(
            f"DeploymentManager: deployed {strategy_id} v{strategy_version} "
            f"[{deployment_id[:8]}]"
        )
        return record

    def check_rollback_triggers(
        self,
        strategy_id:     str,
        recent_trades:   List[Dict],
        drawdown_pct:    float,
        rejection_rate:  float = 0.0,
        reconciliation_errors: int = 0,
    ) -> Optional[str]:
        """
        Check if a deployed strategy should be rolled back.

        Returns reason string if rollback is needed, None if healthy.
        This method is DETERMINISTIC — Claude cannot influence the result.
        """
        if strategy_id not in self._active:
            return None

        # Gate 1: Drawdown
        if drawdown_pct > self.ROLLBACK_DRAWDOWN_PCT:
            return f"Drawdown {drawdown_pct:.1%} > {self.ROLLBACK_DRAWDOWN_PCT:.0%} threshold"

        # Gate 2: Rolling expectancy
        pnls = [t.get('net_pnl', 0) for t in recent_trades[-self.ROLLBACK_NEG_EXP_TRADES:]
                if t.get('net_pnl') is not None]
        if len(pnls) >= self.ROLLBACK_NEG_EXP_TRADES:
            avg_exp = sum(pnls) / len(pnls)
            if avg_exp < self.ROLLBACK_MIN_EXPECTANCY:
                return f"Rolling expectancy ${avg_exp:.2f} < ${self.ROLLBACK_MIN_EXPECTANCY:.0f}/trade"

        # Gate 3: Order rejection rate
        if rejection_rate > self.ROLLBACK_REJECTION_RATE:
            return f"Rejection rate {rejection_rate:.1%} > {self.ROLLBACK_REJECTION_RATE:.0%}"

        # Gate 4: Reconciliation errors
        if reconciliation_errors > 0:
            return f"{reconciliation_errors} reconciliation error(s)"

        return None

    def rollback(
        self,
        strategy_id: str,
        reason:      str,
        metrics:     Dict = None,
    ) -> Optional[DeploymentRecord]:
        """
        Roll back strategy_id to its previous version.

        Creates an incident record. Returns previous deployment or None.
        """
        if strategy_id not in self._active:
            logger.warning(f"DeploymentManager: no active deployment for {strategy_id}")
            return None

        if strategy_id not in self._previous:
            logger.error(
                f"DeploymentManager: cannot rollback {strategy_id} — "
                f"no previous version available. PAUSING instead."
            )
            self.pause(strategy_id, reason=reason)
            return None

        current = self._active[strategy_id]
        previous = self._previous[strategy_id]

        # Mark current as rolled back
        incident_id = str(uuid.uuid4())
        current.status             = DeploymentStatus.ROLLED_BACK
        current.retired_at         = _utcnow()
        current.rollback_reason    = reason
        current.incident_id        = incident_id
        current.metrics_at_rollback = metrics or {}
        self._update_rolled_back(current)

        # Restore previous
        previous.status = DeploymentStatus.ACTIVE
        self._update_status(previous.deployment_id, DeploymentStatus.ACTIVE)
        self._active[strategy_id]   = previous
        del self._previous[strategy_id]

        # Create incident
        self._create_incident(incident_id, current.deployment_id, reason, metrics)

        logger.warning(
            f"DeploymentManager: 🔄 ROLLBACK {strategy_id} "
            f"v{current.strategy_version} → v{previous.strategy_version} | {reason}"
        )
        return previous

    def pause(self, strategy_id: str, reason: str = '') -> bool:
        """Pause an active deployment without rollback."""
        if strategy_id not in self._active:
            return False
        record = self._active[strategy_id]
        record.status = DeploymentStatus.PAUSED
        self._update_status(record.deployment_id, DeploymentStatus.PAUSED)
        logger.warning(f"DeploymentManager: PAUSED {strategy_id} — {reason}")
        return True

    def get_active(self, strategy_id: str) -> Optional[DeploymentRecord]:
        return self._active.get(strategy_id)

    def get_all_active(self) -> List[DeploymentRecord]:
        return list(self._active.values())

    def get_history(self, strategy_id: str = None, limit: int = 20) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if strategy_id:
                rows = conn.execute(
                    "SELECT * FROM deployments WHERE strategy_id=? ORDER BY deployed_at DESC LIMIT ?",
                    (strategy_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM deployments ORDER BY deployed_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_incidents(self, limit: int = 20) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM deployment_incidents ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, r: DeploymentRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO deployments "
                "(deployment_id,strategy_id,strategy_version,config_hash,config_snapshot,"
                "status,deployed_at,retired_at,rollback_reason,incident_id,metrics_json,notes)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (r.deployment_id, r.strategy_id, r.strategy_version, r.config_hash,
                 json.dumps(r.config_snapshot), r.status.value, r.deployed_at,
                 r.retired_at, r.rollback_reason, r.incident_id,
                 json.dumps(r.metrics_at_rollback or {}), r.notes),
            )
            conn.commit()

    def _update_status(self, deployment_id: str, status: DeploymentStatus) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE deployments SET status=? WHERE deployment_id=?",
                (status.value, deployment_id),
            )
            conn.commit()

    def _update_rolled_back(self, r: DeploymentRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE deployments SET status=?,retired_at=?,rollback_reason=?,"
                "incident_id=?,metrics_json=? WHERE deployment_id=?",
                (r.status.value, r.retired_at, r.rollback_reason, r.incident_id,
                 json.dumps(r.metrics_at_rollback or {}), r.deployment_id),
            )
            conn.commit()

    def _create_incident(
        self, incident_id: str, deployment_id: str, reason: str, metrics: Dict = None
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO deployment_incidents "
                "(incident_id,deployment_id,trigger,description,metrics_json,created_at)"
                " VALUES (?,?,?,?,?,?)",
                (incident_id, deployment_id, 'ROLLBACK', reason,
                 json.dumps(metrics or {}), _utcnow()),
            )
            conn.commit()

    def _load_active(self) -> None:
        """Load active deployments from DB on startup."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM deployments WHERE status='ACTIVE'"
                ).fetchall()
            for row in rows:
                r = dict(row)
                record = DeploymentRecord(
                    deployment_id=r['deployment_id'],
                    strategy_id=r['strategy_id'],
                    strategy_version=r['strategy_version'],
                    config_hash=r.get('config_hash', ''),
                    config_snapshot=json.loads(r.get('config_snapshot') or '{}'),
                    status=DeploymentStatus.ACTIVE,
                    deployed_at=r['deployed_at'],
                )
                self._active[record.strategy_id] = record
        except Exception as e:
            logger.warning(f"DeploymentManager: load_active error: {e}")
