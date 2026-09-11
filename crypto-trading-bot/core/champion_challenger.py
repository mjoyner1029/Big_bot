"""
Champion / Challenger Model Framework.

ARCHITECTURE RULE:
    The current production MetaModel is the Champion.
    Any new model trained on fresh data becomes a Challenger.
    The Champion is NEVER overwritten until the Challenger proves superiority.

    Workflow:
        Champion (live)
        ↓
        New data arrives → train Challenger
        ↓
        Historical evaluation  (Challenger must beat Champion)
        ↓
        Out-of-sample evaluation
        ↓
        Paper trading competition (side-by-side)
        ↓
        Statistical significance test (p < 0.05)
        ↓
        Canary (5% capital)
        ↓
        Promote Challenger → new Champion
        ↓
        Archive old Champion (full version history)

Full version history is maintained in SQLite.
"""
import json
import logging
import os
import pickle
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


class ModelRole(str, Enum):
    CHAMPION   = "CHAMPION"
    CHALLENGER = "CHALLENGER"
    ARCHIVED   = "ARCHIVED"
    CANARY     = "CANARY"


@dataclass
class ModelVersion:
    """Metadata for a single trained model version."""
    version_id:    str
    role:          ModelRole
    trained_at:    str
    training_rows: int
    metrics:       Dict[str, Any] = field(default_factory=dict)
    oos_metrics:   Dict[str, Any] = field(default_factory=dict)
    paper_metrics: Dict[str, Any] = field(default_factory=dict)
    promoted_at:   Optional[str] = None
    archived_at:   Optional[str] = None
    model_path:    Optional[str] = None
    notes:         str = ''


_CREATE_VERSIONS = """
CREATE TABLE IF NOT EXISTS model_versions (
    version_id   TEXT PRIMARY KEY,
    role         TEXT DEFAULT 'CHALLENGER',
    trained_at   TEXT NOT NULL,
    training_rows INTEGER DEFAULT 0,
    metrics      TEXT,
    oos_metrics  TEXT,
    paper_metrics TEXT,
    promoted_at  TEXT,
    archived_at  TEXT,
    model_path   TEXT,
    notes        TEXT DEFAULT ''
)
"""


class ChampionChallenger:
    """
    Manages the Champion/Challenger lifecycle for the MetaModel.

    The champion model is the one used in production.
    Only one challenger is evaluated at a time.
    """

    # Statistical gates for promotion
    MIN_IMPROVEMENT_AUC  = 0.02    # challenger must improve AUC by at least 2%
    MIN_SAMPLE_CHALLENGE = 30      # minimum paper trades for significance test
    P_VALUE_THRESHOLD    = 0.05    # statistical significance for promotion

    def __init__(
        self,
        model_dir: str = "models",
        db_path: str = "data/trade_memory.sqlite",
    ):
        self.model_dir  = model_dir
        self.db_path    = db_path
        os.makedirs(model_dir, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_VERSIONS)
            conn.commit()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_champion(self) -> Optional[ModelVersion]:
        """Return the current production champion, or None if none exists."""
        return self._load_by_role(ModelRole.CHAMPION)

    def get_challenger(self) -> Optional[ModelVersion]:
        """Return the current challenger, or None."""
        return self._load_by_role(ModelRole.CHALLENGER)

    def register_challenger(
        self,
        meta_model,
        training_rows: int,
        metrics: Dict,
        notes: str = '',
    ) -> ModelVersion:
        """
        Register a freshly-trained model as the new challenger.

        Saves the model artifact to disk and records metadata.
        Returns the new ModelVersion.
        """
        version_id  = str(uuid.uuid4())
        model_path  = os.path.join(self.model_dir, f"challenger_{version_id[:8]}.pkl")

        # Archive previous challenger if exists
        old_challenger = self.get_challenger()
        if old_challenger:
            self._set_role(old_challenger.version_id, ModelRole.ARCHIVED)
            logger.info(f"ChampionChallenger: archived previous challenger {old_challenger.version_id[:8]}")

        # Save model artifact
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(meta_model, f)
        except Exception as e:
            logger.warning(f"ChampionChallenger: model serialization failed: {e}")
            model_path = None

        version = ModelVersion(
            version_id=version_id,
            role=ModelRole.CHALLENGER,
            trained_at=_utcnow(),
            training_rows=training_rows,
            metrics=metrics,
            model_path=model_path,
            notes=notes,
        )
        self._save(version)
        logger.info(
            f"ChampionChallenger: new challenger registered {version_id[:8]} "
            f"({training_rows} training rows)"
        )
        return version

    def evaluate_challenger(
        self,
        challenger: ModelVersion,
        oos_metrics: Dict,
    ) -> bool:
        """
        Evaluate the challenger against the champion on OOS data.

        Returns True if the challenger is ready for paper competition.
        """
        champion = self.get_champion()

        # If no champion exists, challenger becomes champion immediately
        if champion is None:
            logger.info("ChampionChallenger: no champion — promoting challenger directly")
            self.promote_challenger(challenger)
            return True

        champ_auc = champion.oos_metrics.get('cv_auc', 0.5)
        chal_auc  = oos_metrics.get('cv_auc', 0.5)

        improvement = chal_auc - champ_auc

        if improvement >= self.MIN_IMPROVEMENT_AUC:
            # Update OOS metrics for challenger
            challenger.oos_metrics = oos_metrics
            self._save(challenger)
            logger.info(
                f"ChampionChallenger: challenger beats champion by {improvement:.3f} AUC "
                f"({champ_auc:.3f} → {chal_auc:.3f}) — proceed to paper competition"
            )
            return True
        else:
            self._set_role(challenger.version_id, ModelRole.ARCHIVED)
            logger.info(
                f"ChampionChallenger: challenger rejected — insufficient improvement "
                f"({improvement:.3f} < {self.MIN_IMPROVEMENT_AUC:.2f})"
            )
            return False

    def record_paper_result(self, challenger_id: str, paper_metrics: Dict) -> None:
        """Record paper trading results for a challenger."""
        version = self._load_by_id(challenger_id)
        if version:
            version.paper_metrics = paper_metrics
            self._save(version)

    def should_promote(self, challenger: ModelVersion) -> Tuple[bool, str]:
        """
        Determine if challenger should be promoted to champion.

        Returns (should_promote, reason).
        Uses a simplified two-proportion z-test for significance.
        """
        champion = self.get_champion()
        if champion is None:
            return True, "No existing champion"

        paper = challenger.paper_metrics
        n = paper.get('paper_trades', 0)
        if n < self.MIN_SAMPLE_CHALLENGE:
            return False, f"Insufficient paper trades: {n} < {self.MIN_SAMPLE_CHALLENGE}"

        # Compare paper accuracy: z-test between champion and challenger
        champ_wr = champion.paper_metrics.get('win_rate', 0.5)
        chal_wr  = paper.get('win_rate', 0.5)

        # Pooled proportion z-test (simplified)
        p_pool = (champ_wr * n + chal_wr * n) / (2 * n)
        se     = (2 * p_pool * (1 - p_pool) / n) ** 0.5
        z      = (chal_wr - champ_wr) / se if se > 0 else 0.0

        # z > 1.645 → p < 0.05 one-tailed
        if z > 1.645 and chal_wr > champ_wr:
            return True, f"Statistically significant improvement: z={z:.2f} win_rate={chal_wr:.1%} vs {champ_wr:.1%}"
        else:
            return False, f"Not significant: z={z:.2f} (need >1.645), win_rate delta={chal_wr-champ_wr:.1%}"

    def promote_challenger(self, challenger: ModelVersion) -> ModelVersion:
        """
        Promote the challenger to champion.

        Archives the current champion. Returns the new champion.
        """
        # Archive current champion
        current = self.get_champion()
        if current:
            current.archived_at = _utcnow()
            self._set_role(current.version_id, ModelRole.ARCHIVED)
            logger.info(f"ChampionChallenger: archived champion {current.version_id[:8]}")

        # Promote challenger
        challenger.role        = ModelRole.CHAMPION
        challenger.promoted_at = _utcnow()

        # Copy to canonical champion path
        if challenger.model_path and os.path.exists(challenger.model_path):
            champion_path = os.path.join(self.model_dir, "meta_model", "meta_model.pkl")
            os.makedirs(os.path.dirname(champion_path), exist_ok=True)
            try:
                import shutil
                shutil.copy2(challenger.model_path, champion_path)
                challenger.model_path = champion_path
            except Exception as e:
                logger.warning(f"ChampionChallenger: copy to champion path failed: {e}")

        self._save(challenger)
        logger.info(
            f"ChampionChallenger: 🏆 PROMOTED {challenger.version_id[:8]} to CHAMPION "
            f"(trained on {challenger.training_rows} rows)"
        )
        return challenger

    def get_version_history(self, limit: int = 20) -> List[ModelVersion]:
        """Full version history, newest first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM model_versions ORDER BY trained_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def get_champion_summary(self) -> Dict:
        """Summary suitable for dashboards and daily reviews."""
        champion   = self.get_champion()
        challenger = self.get_challenger()
        history    = self.get_version_history(limit=5)
        return {
            'champion':   self._version_to_dict(champion) if champion else None,
            'challenger': self._version_to_dict(challenger) if challenger else None,
            'total_versions': len(history),
            'history': [self._version_to_dict(v) for v in history],
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, v: ModelVersion) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO model_versions "
                "(version_id,role,trained_at,training_rows,metrics,oos_metrics,"
                "paper_metrics,promoted_at,archived_at,model_path,notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (v.version_id, v.role.value, v.trained_at, v.training_rows,
                 json.dumps(v.metrics), json.dumps(v.oos_metrics),
                 json.dumps(v.paper_metrics), v.promoted_at, v.archived_at,
                 v.model_path, v.notes),
            )
            conn.commit()

    def _set_role(self, version_id: str, role: ModelRole) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE model_versions SET role=? WHERE version_id=?",
                (role.value, version_id),
            )
            conn.commit()

    def _load_by_role(self, role: ModelRole) -> Optional[ModelVersion]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM model_versions WHERE role=? ORDER BY trained_at DESC LIMIT 1",
                (role.value,),
            ).fetchone()
        return self._row_to_version(row) if row else None

    def _load_by_id(self, version_id: str) -> Optional[ModelVersion]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM model_versions WHERE version_id=?", (version_id,)
            ).fetchone()
        return self._row_to_version(row) if row else None

    @staticmethod
    def _row_to_version(row) -> ModelVersion:
        r = dict(row)
        return ModelVersion(
            version_id=r['version_id'],
            role=ModelRole(r['role']),
            trained_at=r['trained_at'],
            training_rows=r.get('training_rows', 0),
            metrics=json.loads(r.get('metrics') or '{}'),
            oos_metrics=json.loads(r.get('oos_metrics') or '{}'),
            paper_metrics=json.loads(r.get('paper_metrics') or '{}'),
            promoted_at=r.get('promoted_at'),
            archived_at=r.get('archived_at'),
            model_path=r.get('model_path'),
            notes=r.get('notes', ''),
        )

    @staticmethod
    def _version_to_dict(v: ModelVersion) -> Optional[Dict]:
        if v is None:
            return None
        return {
            'version_id':   v.version_id[:8],
            'role':         v.role.value,
            'trained_at':   v.trained_at,
            'training_rows': v.training_rows,
            'auc':          v.metrics.get('cv_auc', v.oos_metrics.get('cv_auc', 0.0)),
            'promoted_at':  v.promoted_at,
        }
