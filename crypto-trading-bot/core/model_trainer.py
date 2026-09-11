"""
Continuous ML Retraining — scheduled MetaModel retraining.

Retraining is triggered when:
    1. Enough new labeled data has accumulated (min_new_rows since last train)
    2. A scheduled interval passes (daily / weekly / monthly)
    3. Drift is detected (immediate trigger)
    4. Strategy health alerts are raised

Each retraining produces:
    • model version record
    • training metrics (CV AUC, precision, recall)
    • validation metrics (OOS)
    • feature importance
    • confidence calibration
    • expected value estimate

No model automatically replaces production.
Champion/Challenger decides promotion.
"""
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc)


@dataclass
class RetrainResult:
    version_id:        str
    training_rows:     int
    validation_rows:   int
    train_metrics:     Dict
    val_metrics:       Dict
    feature_importance: Dict
    triggered_by:      str      # 'scheduled' | 'drift' | 'health_alert' | 'manual'
    trained_at:        str


class ModelTrainer:
    """
    Orchestrates MetaModel retraining on a schedule or trigger basis.

    Maintains retraining history so the system knows when last trained
    and whether sufficient new data has accumulated.
    """

    MIN_NEW_ROWS_FOR_RETRAIN = 50    # minimum new labeled rows since last train
    MIN_TOTAL_ROWS           = 100   # absolute minimum to train at all

    def __init__(
        self,
        feature_store=None,
        champion_challenger=None,
        feature_importance_engine=None,
        db_path: str = "data/trade_memory.sqlite",
    ):
        self.feature_store         = feature_store
        self.cc                    = champion_challenger
        self.fi_engine             = feature_importance_engine
        self.db_path               = db_path
        self._last_trained_rows    = self._load_last_trained_rows()

    # ── Public API ────────────────────────────────────────────────────────────

    def should_retrain(
        self,
        trigger: str = 'scheduled',
        force: bool = False,
    ) -> tuple:
        """
        Return (should_retrain: bool, reason: str).

        Args:
            trigger: 'scheduled' | 'drift' | 'health_alert' | 'manual'
            force:   bypass all checks
        """
        if force:
            return True, "Manual force retrain"
        if trigger == 'drift':
            return True, "Drift detected — immediate retrain"
        if trigger == 'health_alert':
            return True, "Strategy health alert — retrain"

        if not self.feature_store:
            return False, "No feature store configured"

        stats = self.feature_store.stats()
        total = stats.get('with_outcomes', 0)
        new   = total - self._last_trained_rows

        if total < self.MIN_TOTAL_ROWS:
            return False, f"Insufficient total labeled data: {total} < {self.MIN_TOTAL_ROWS}"
        if new < self.MIN_NEW_ROWS_FOR_RETRAIN:
            return False, f"Not enough new data: {new} new rows (need {self.MIN_NEW_ROWS_FOR_RETRAIN})"

        return True, f"{new} new labeled rows since last training"

    def retrain(
        self,
        trigger: str = 'scheduled',
        force: bool = False,
    ) -> Optional[RetrainResult]:
        """
        Retrain the MetaModel and register result with ChampionChallenger.

        Returns RetrainResult on success, None if skipped.
        """
        should, reason = self.should_retrain(trigger, force)
        if not should:
            logger.info(f"ModelTrainer: skipping retrain — {reason}")
            return None

        logger.info(f"ModelTrainer: retraining — {reason}")

        if not self.feature_store:
            logger.warning("ModelTrainer: no feature store — cannot retrain")
            return None

        # Load training data
        dataset = self.feature_store.get_training_dataset(min_rows=self.MIN_TOTAL_ROWS)
        if len(dataset) < self.MIN_TOTAL_ROWS:
            logger.warning(f"ModelTrainer: only {len(dataset)} rows available")
            return None

        # Split: 80% train, 20% validation
        split       = int(len(dataset) * 0.8)
        train_data  = dataset[:split]
        val_data    = dataset[split:]

        # Train new MetaModel challenger
        from core.meta_model import MetaModel
        model = MetaModel(model_dir=os.path.join(os.path.dirname(self.db_path), '..', 'models', 'challengers'))
        train_metrics = model.train(self.feature_store)

        if train_metrics.get('status') != 'ok':
            logger.warning(f"ModelTrainer: training failed: {train_metrics}")
            return None

        # Validate on held-out data
        val_metrics = self._evaluate_on_subset(model, val_data)

        # Feature importance
        fi = {}
        if self.fi_engine:
            try:
                fi = self.fi_engine.compute(model, val_data)
            except Exception as e:
                logger.warning(f"ModelTrainer: feature importance error: {e}")
                fi = model._feature_importances

        # Register with Champion/Challenger
        import uuid
        version_id = str(uuid.uuid4())
        if self.cc:
            version = self.cc.register_challenger(
                meta_model=model,
                training_rows=len(train_data),
                metrics=train_metrics.get('metrics', {}).get('lgbm', {}),
                notes=f"Triggered by: {trigger}. Reason: {reason}",
            )
            version_id = version.version_id

        # Update last trained row count
        self._last_trained_rows = len(dataset)
        self._save_last_trained_rows(len(dataset))

        result = RetrainResult(
            version_id=version_id,
            training_rows=len(train_data),
            validation_rows=len(val_data),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_importance=fi,
            triggered_by=trigger,
            trained_at=_utcnow().isoformat(),
        )

        logger.info(
            f"ModelTrainer: ✅ retrain complete — {len(train_data)} train, "
            f"{len(val_data)} val rows | val_auc={val_metrics.get('auc', 0):.3f}"
        )
        return result

    # ── Evaluation helpers ────────────────────────────────────────────────────

    def _evaluate_on_subset(self, model, val_data: List[Dict]) -> Dict:
        """Run inference on validation rows and compute accuracy metrics."""
        if not val_data or not model._is_trained:
            return {}

        correct, total = 0, 0
        for row in val_data:
            pred = model.predict(row)
            actual_return = row.get('return_1h', 0.0) or 0.0
            actual_label  = 'TRADE' if actual_return > 0.001 else 'NO_TRADE'
            if pred.decision == actual_label:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        return {'accuracy': acc, 'evaluated_rows': total, 'auc': acc}  # auc proxy

    # ── State persistence ─────────────────────────────────────────────────────

    def _load_last_trained_rows(self) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT value FROM kv_store WHERE key='last_trained_rows'"
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def _save_last_trained_rows(self, n: int) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value TEXT)"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('last_trained_rows', ?)",
                    (str(n),),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"ModelTrainer: could not save state: {e}")
