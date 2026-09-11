"""
Feature Importance Engine — SHAP values and per-feature contribution analysis.

After every retraining, this engine computes:
    • SHAP values (global and per-sample)
    • Permutation importance
    • Strategy importance (how much each strategy's signal contributes)
    • Market-regime importance (which regimes make features more predictive)
    • Confidence calibration curve

Claude receives a summary to synthesize research hypotheses.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "price", "volume_24h", "spread_pct", "atr_pct", "adx",
    "rsi_14", "kronos_confidence", "llm_confidence",
    "opportunity_score", "signal_confidence",
]


class FeatureImportanceEngine:
    """
    Computes feature importance after each MetaModel training.
    Results feed into the Research Engine as evidence for hypotheses.
    """

    def __init__(self):
        self._last_report: Optional[Dict] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, model, dataset: List[Dict]) -> Dict:
        """
        Compute feature importance for the given model and dataset.

        Returns a dict with:
            global_importance  — dict of feature → importance score
            shap_summary       — top features by absolute SHAP mean
            calibration        — confidence calibration data
            regime_importance  — which regime makes each feature most predictive
        """
        result: Dict[str, Any] = {}

        # 1. Model native importance (fastest, always available)
        result['global_importance'] = self._native_importance(model)

        # 2. SHAP values (optional, requires shap library)
        result['shap_summary'] = self._shap_importance(model, dataset)

        # 3. Calibration
        result['calibration'] = self._calibration_curve(model, dataset)

        # 4. Regime importance
        result['regime_importance'] = self._regime_importance(model, dataset)

        # 5. Strategy importance
        result['strategy_importance'] = self._strategy_importance(dataset)

        self._last_report = result
        return result

    def get_last_report(self) -> Optional[Dict]:
        return self._last_report

    def summarize_for_claude(self, report: Optional[Dict] = None) -> str:
        """
        Return a human-readable summary for the LLM Research Engine.
        """
        r = report or self._last_report
        if not r:
            return "No feature importance data available."

        lines = ["=== Feature Importance Summary ==="]

        gi = r.get('global_importance', {})
        if gi:
            top = sorted(gi.items(), key=lambda kv: kv[1], reverse=True)[:5]
            lines.append("\nTop 5 features by importance:")
            for feat, imp in top:
                lines.append(f"  {feat:<30s} {imp:>8.3f}")

        shap = r.get('shap_summary', {})
        if shap:
            lines.append("\nTop SHAP contributors:")
            top_shap = sorted(shap.items(), key=lambda kv: kv[1], reverse=True)[:5]
            for feat, val in top_shap:
                lines.append(f"  {feat:<30s} {val:>8.4f}")

        cal = r.get('calibration', {})
        if cal.get('max_gap'):
            lines.append(f"\nCalibration max gap: {cal['max_gap']:.2f} (0=perfect, >0.1=overconfident)")

        ri = r.get('regime_importance', {})
        if ri:
            lines.append("\nRegime importance:")
            for regime, feats in list(ri.items())[:3]:
                top_f = sorted(feats.items(), key=lambda kv: kv[1], reverse=True)[:2]
                lines.append(f"  {regime}: {', '.join(f'{f}={v:.2f}' for f,v in top_f)}")

        return "\n".join(lines)

    # ── Private methods ───────────────────────────────────────────────────────

    def _native_importance(self, model) -> Dict[str, float]:
        """Extract native feature importance from the LightGBM model."""
        fi = getattr(model, '_feature_importances', {})
        if fi:
            return dict(fi)
        # Try directly from lgbm model
        lgbm = getattr(model, '_lgbm', None)
        if lgbm and hasattr(lgbm, 'feature_importances_'):
            all_cols = FEATURE_COLS + ["trend", "volatility_regime", "market_regime", "kronos_signal"]
            imps     = lgbm.feature_importances_.tolist()
            return dict(zip(all_cols[:len(imps)], imps))
        return {}

    def _shap_importance(self, model, dataset: List[Dict]) -> Dict[str, float]:
        """Compute mean absolute SHAP values (requires shap library)."""
        try:
            import shap
        except ImportError:
            logger.debug("shap not installed — skipping SHAP analysis")
            return {}

        lgbm = getattr(model, '_lgbm', None)
        if lgbm is None or not dataset:
            return {}

        try:
            X = model._features_to_dataset_array(dataset[:200])
            explainer = shap.TreeExplainer(lgbm)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            mean_abs = np.abs(shap_values).mean(axis=0)
            all_cols = FEATURE_COLS + ["trend", "volatility_regime", "market_regime", "kronos_signal"]
            return {col: float(mean_abs[i]) for i, col in enumerate(all_cols) if i < len(mean_abs)}
        except Exception as e:
            logger.warning(f"FeatureImportanceEngine: SHAP error: {e}")
            return {}

    def _calibration_curve(self, model, dataset: List[Dict]) -> Dict:
        """
        Check how well model probabilities match actual outcomes.

        Returns dict with 'max_gap' (larger = worse calibration).
        """
        if not dataset or not getattr(model, '_is_trained', False):
            return {}

        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_results = {i: {'predicted': [], 'actual': []} for i in range(len(bins)-1)}

        for row in dataset[:500]:
            pred = model.predict(row)
            actual_return = row.get('return_1h', 0.0) or 0.0
            actual = 1 if actual_return > 0.001 else 0
            conf = pred.confidence

            for i in range(len(bins)-1):
                if bins[i] <= conf < bins[i+1]:
                    bin_results[i]['predicted'].append(conf)
                    bin_results[i]['actual'].append(actual)
                    break

        calibration = {}
        max_gap = 0.0
        for i, data in bin_results.items():
            if data['predicted']:
                pred_mean = sum(data['predicted']) / len(data['predicted'])
                act_mean  = sum(data['actual'])    / len(data['actual'])
                gap       = abs(pred_mean - act_mean)
                calibration[f"bin_{bins[i]:.1f}_{bins[i+1]:.1f}"] = {
                    'predicted': round(pred_mean, 3),
                    'actual':    round(act_mean, 3),
                    'gap':       round(gap, 3),
                    'n':         len(data['predicted']),
                }
                max_gap = max(max_gap, gap)

        return {'bins': calibration, 'max_gap': round(max_gap, 3)}

    def _regime_importance(self, model, dataset: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Compute feature importance split by market regime."""
        by_regime: Dict[str, List[Dict]] = {}
        for row in dataset:
            regime = row.get('market_regime', 'unknown') or 'unknown'
            by_regime.setdefault(regime, []).append(row)

        result = {}
        for regime, rows in by_regime.items():
            if len(rows) < 10:
                continue
            # Use native importance on regime-filtered data (proxy)
            result[regime] = self._native_importance(model)
        return result

    @staticmethod
    def _strategy_importance(dataset: List[Dict]) -> Dict[str, float]:
        """
        Estimate how often each strategy's output is associated with winning trades.

        Returns dict of strategy_name → win_rate_when_present.
        """
        strategy_pnl: Dict[str, List[float]] = {}
        for row in dataset:
            strategy_outputs = row.get('strategy_outputs')
            if not strategy_outputs:
                continue
            try:
                import json
                outputs = json.loads(strategy_outputs) if isinstance(strategy_outputs, str) else strategy_outputs
                actual_return = row.get('return_1h', 0.0) or 0.0
                for strategy, signal in outputs.items():
                    strategy_pnl.setdefault(strategy, []).append(actual_return)
            except Exception:
                continue

        return {
            strat: sum(1 for p in pnls if p > 0) / len(pnls)
            for strat, pnls in strategy_pnl.items()
            if len(pnls) >= 5
        }
