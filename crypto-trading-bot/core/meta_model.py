"""
Meta Model — Signal Filter using Gradient Boosting.

ARCHITECTURE:
    This model does NOT predict price.
    It predicts whether an existing strategy signal should be traded.

    Input:  feature store rows (all features available at decision time)
    Output: TRADE | NO_TRADE + expected_value + confidence

    The model is an ensemble of:
        1. LightGBM (primary)
        2. XGBoost  (secondary)
        3. CatBoost (secondary, optional)

    Predictions are averaged. The final confidence is calibrated
    against historical accuracy in the feature store.

Usage:
    model = MetaModel()
    model.train(feature_store)              # retrain from scratch
    pred = model.predict(features_dict)     # inference
    model.save() / model.load()             # persistence
"""
import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature columns used for training ──────────────────────────────────────────
FEATURE_COLS = [
    "price", "volume_24h", "spread_pct", "atr_pct", "adx",
    "rsi_14", "kronos_confidence", "llm_confidence",
    "opportunity_score", "signal_confidence",
]

CATEGORICAL_COLS = ["trend", "volatility_regime", "market_regime", "kronos_signal"]

# ── Minimum training rows required before model is used ────────────────────────
MIN_TRAINING_ROWS = 50


@dataclass
class MetaPrediction:
    """Prediction result from MetaModel.predict()."""
    decision:       str             # 'TRADE' | 'NO_TRADE'
    expected_value: float           # estimated net return (positive = good)
    confidence:     float           # [0, 1] — calibrated probability
    model_votes:    Dict[str, str]  # individual model predictions
    explanation:    str             # human-readable summary


class MetaModel:
    """
    Ensemble meta-model that decides whether to take or skip a signal.

    Not a price predictor. A signal quality filter.
    """

    def __init__(self, model_dir: str = "models/meta_model"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self._lgbm   = None
        self._xgb    = None
        self._catboost = None
        self._label_encoders: Dict[str, Any] = {}
        self._is_trained = False
        self._training_rows = 0
        self._feature_importances: Dict[str, float] = {}

        # Try to load existing models
        self._try_load()

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        feature_store,          # FeatureStore instance
        target_col: str = "return_1h",   # what we're predicting
        positive_threshold: float = 0.001,  # >0.1% return = positive label
    ) -> Dict:
        """
        Train all ensemble members on the feature store dataset.

        Returns training metrics dict.
        """
        from core.feature_store import FeatureStore

        rows = feature_store.get_training_dataset(min_rows=MIN_TRAINING_ROWS)
        if len(rows) < MIN_TRAINING_ROWS:
            logger.warning(
                f"MetaModel: only {len(rows)} training rows "
                f"(need {MIN_TRAINING_ROWS}) — skipping training"
            )
            return {"status": "insufficient_data", "rows": len(rows)}

        # get_training_dataset returns newest-first; time-series CV requires
        # ascending chronological order
        rows = list(reversed(rows))
        X, y, weights = self._prepare_features(rows, target_col, positive_threshold)
        self._training_rows = len(y)

        metrics = {}
        metrics["lgbm"]     = self._train_lgbm(X, y, weights)
        metrics["xgboost"]  = self._train_xgb(X, y, weights)
        metrics["catboost"] = self._train_catboost(X, y, weights)

        self._is_trained = True
        self.save()

        pos_rate = float(np.mean(y))
        logger.info(
            f"MetaModel trained on {len(y)} rows "
            f"(positive rate: {pos_rate:.1%}) | "
            f"LightGBM: {metrics['lgbm']} "
            f"XGBoost: {metrics['xgboost']}"
        )
        return {"status": "ok", "rows": len(y), "metrics": metrics}

    @staticmethod
    def _ts_cv_auc(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                   embargo: int = 5) -> np.ndarray:
        """Time-series cross-validation AUC with an embargo gap.

        Financial observations are serially correlated: naive shuffled KFold
        leaks future information into training. Rows must be in ascending
        time order. The ``gap`` purges observations adjacent to each
        validation block.
        """
        from sklearn.base import clone
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=embargo)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            m = clone(model)
            m.fit(X[train_idx], y[train_idx])
            proba = m.predict_proba(X[test_idx])[:, 1]
            scores.append(roc_auc_score(y[test_idx], proba))
        return np.array(scores) if scores else np.array([0.5])

    def _train_lgbm(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Dict:
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("lightgbm not installed — skipping LightGBM")
            return {"status": "not_installed"}

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbosity=-1,
        )
        scores = self._ts_cv_auc(model, X, y)
        model.fit(X, y, sample_weight=weights)
        self._lgbm = model
        importances = dict(zip(
            FEATURE_COLS + CATEGORICAL_COLS,
            model.feature_importances_.tolist(),
        ))
        self._feature_importances = importances
        return {"cv_auc": float(np.mean(scores)), "std": float(np.std(scores))}

    def _train_xgb(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Dict:
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("xgboost not installed — skipping XGBoost")
            return {"status": "not_installed"}

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )
        scores = self._ts_cv_auc(model, X, y)
        model.fit(X, y, sample_weight=weights)
        self._xgb = model
        return {"cv_auc": float(np.mean(scores)), "std": float(np.std(scores))}

    def _train_catboost(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Dict:
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            logger.info("catboost not installed — skipping CatBoost")
            return {"status": "not_installed"}

        model = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=0,
        )
        scores = self._ts_cv_auc(model, X, y)
        model.fit(X, y, sample_weight=weights)
        self._catboost = model
        return {"cv_auc": float(np.mean(scores)), "std": float(np.std(scores))}

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, features: Dict) -> MetaPrediction:
        """
        Predict TRADE / NO_TRADE for a single opportunity.

        Returns MetaPrediction. If the model is not yet trained,
        returns a neutral 'NO_TRADE' prediction with 0.0 confidence
        so the bot can still operate on strategy signals alone.
        """
        if not self._is_trained:
            return MetaPrediction(
                decision='NO_TRADE',
                expected_value=0.0,
                confidence=0.0,
                model_votes={},
                explanation="MetaModel not yet trained — insufficient data",
            )

        x = self._features_to_array(features)
        probs: List[float] = []
        votes: Dict[str, str] = {}

        for name, model in [("lgbm", self._lgbm), ("xgboost", self._xgb),
                             ("catboost", self._catboost)]:
            if model is None:
                continue
            try:
                prob = float(model.predict_proba(x.reshape(1, -1))[0][1])
                probs.append(prob)
                votes[name] = 'TRADE' if prob >= 0.5 else 'NO_TRADE'
            except Exception as e:
                logger.warning(f"MetaModel {name} inference error: {e}")

        if not probs:
            return MetaPrediction(
                decision='NO_TRADE',
                expected_value=0.0,
                confidence=0.0,
                model_votes=votes,
                explanation="All models failed inference",
            )

        avg_prob = float(np.mean(probs))
        # Simple expected-value estimate: prob × avg_win - (1-prob) × avg_loss
        # Calibrate as (prob - 0.5) * 2 as a rough EV proxy
        expected_value = (avg_prob - 0.5) * 2.0

        decision   = 'TRADE' if avg_prob >= 0.55 else 'NO_TRADE'
        confidence = avg_prob if avg_prob >= 0.55 else (1.0 - avg_prob)

        explanation = (
            f"Ensemble probability: {avg_prob:.1%} "
            f"({', '.join(f'{k}={v}' for k, v in votes.items())}) "
            f"trained on {self._training_rows} rows"
        )

        return MetaPrediction(
            decision=decision,
            expected_value=expected_value,
            confidence=confidence,
            model_votes=votes,
            explanation=explanation,
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Pickle all trained models to disk."""
        payload = {
            "lgbm":                self._lgbm,
            "xgb":                 self._xgb,
            "catboost":            self._catboost,
            "label_encoders":      self._label_encoders,
            "is_trained":          self._is_trained,
            "training_rows":       self._training_rows,
            "feature_importances": self._feature_importances,
        }
        path = os.path.join(self.model_dir, "meta_model.pkl")
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        logger.info(f"MetaModel saved to {path}")

    def _try_load(self) -> None:
        path = os.path.join(self.model_dir, "meta_model.pkl")
        if not os.path.exists(path):
            return
        try:
            with open(path, 'rb') as f:
                payload = pickle.load(f)
            self._lgbm                = payload.get("lgbm")
            self._xgb                 = payload.get("xgb")
            self._catboost            = payload.get("catboost")
            self._label_encoders      = payload.get("label_encoders", {})
            self._is_trained          = payload.get("is_trained", False)
            self._training_rows       = payload.get("training_rows", 0)
            self._feature_importances = payload.get("feature_importances", {})
            logger.info(
                f"MetaModel loaded from {path} "
                f"({self._training_rows} training rows)"
            )
        except Exception as e:
            logger.warning(f"MetaModel load failed: {e} — will retrain")

    # ── Feature engineering ────────────────────────────────────────────────────

    def _prepare_features(
        self,
        rows: List[Dict],
        target_col: str,
        positive_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert feature store rows to numpy arrays for sklearn-compatible models.

        Target: 1 if forward return > positive_threshold, else 0.
        Weights: more recent rows are weighted higher (exponential decay).
        """
        n = len(rows)
        X = np.zeros((n, len(FEATURE_COLS) + len(CATEGORICAL_COLS)), dtype=float)
        y = np.zeros(n, dtype=int)
        weights = np.ones(n, dtype=float)

        # Fit label encoders on first call
        for i, col in enumerate(CATEGORICAL_COLS):
            values = [str(r.get(col, 'unknown')) for r in rows]
            unique = sorted(set(values))
            if col not in self._label_encoders:
                self._label_encoders[col] = {v: j for j, v in enumerate(unique)}

        for i, row in enumerate(rows):
            # Numeric features
            for j, col in enumerate(FEATURE_COLS):
                X[i, j] = float(row.get(col) or 0.0)

            # Categorical features (label encoded)
            for j, col in enumerate(CATEGORICAL_COLS):
                enc = self._label_encoders.get(col, {})
                val = str(row.get(col, 'unknown'))
                X[i, len(FEATURE_COLS) + j] = float(enc.get(val, 0))

            # Target
            fwd = row.get(target_col)
            y[i] = 1 if (fwd is not None and fwd > positive_threshold) else 0

            # Recency weight: newest rows get weight 1.0, oldest ~0.1
            recency = (n - i) / n           # 1.0 for newest, 1/n for oldest
            weights[i] = 0.1 + 0.9 * recency

        return X, y, weights

    def _features_to_array(self, features: Dict) -> np.ndarray:
        """Convert a single feature dict to a numpy array for inference."""
        x = np.zeros(len(FEATURE_COLS) + len(CATEGORICAL_COLS), dtype=float)
        for j, col in enumerate(FEATURE_COLS):
            x[j] = float(features.get(col) or 0.0)
        for j, col in enumerate(CATEGORICAL_COLS):
            enc = self._label_encoders.get(col, {})
            val = str(features.get(col, 'unknown'))
            x[len(FEATURE_COLS) + j] = float(enc.get(val, 0))
        return x

    def feature_importance_report(self) -> str:
        """Human-readable feature importance summary."""
        if not self._feature_importances:
            return "No feature importances available (model not trained)"
        sorted_fi = sorted(
            self._feature_importances.items(), key=lambda kv: kv[1], reverse=True
        )
        lines = ["Feature Importances (LightGBM):"]
        for name, imp in sorted_fi[:10]:
            lines.append(f"  {name:<30s} {imp:>8.2f}")
        return "\n".join(lines)
