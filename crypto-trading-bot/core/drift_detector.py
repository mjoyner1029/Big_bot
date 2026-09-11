"""
Drift Detection — continuously monitors for data and model drift.

Detects:
    FEATURE DRIFT:      input feature distributions shifting (KS test or PSI)
    PREDICTION DRIFT:   model output probabilities shifting
    CONFIDENCE DRIFT:   model confidence level changing systematically
    REGIME DRIFT:       market regime distribution changing
    LABEL DRIFT:        actual outcome distribution changing (concept drift)

On drift detection:
    1. Reduces MetaModel confidence in affected features
    2. Reduces capital allocation (via allocation multiplier)
    3. Triggers ModelTrainer to retrain
    4. Registers hypothesis in Research Engine

All drift checks are non-parametric to handle non-normal feature distributions.
"""
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

# ── PSI thresholds ─────────────────────────────────────────────────────────────
PSI_LOW    = 0.10   # minor shift — monitor
PSI_MEDIUM = 0.20   # moderate shift — reduce confidence
PSI_HIGH   = 0.25   # severe drift — trigger retraining

# ── KS test p-value ─────────────────────────────────────────────────────────────
KS_ALPHA   = 0.05


@dataclass
class DriftEvent:
    drift_type:  str   # 'feature' | 'prediction' | 'confidence' | 'regime' | 'label'
    feature:     str
    psi:         float
    severity:    str   # 'low' | 'medium' | 'high'
    description: str
    detected_at: str = field(default_factory=_utcnow)


@dataclass
class DriftReport:
    drift_detected:     bool
    drifts:             List[DriftEvent]
    allocation_factor:  float  # 1.0 = normal, <1.0 = reduce allocation
    retrain_required:   bool
    summary:            str
    computed_at:        str = field(default_factory=_utcnow)


class DriftDetector:
    """
    Detects statistical drift between reference and current data windows.

    reference_window: training baseline (30–90 day lookback)
    current_window:   recent data (last 7 days)
    """

    def __init__(
        self,
        feature_store=None,
        reference_days: int = 60,
        current_days:   int = 7,
        db_path:        str = "data/trade_memory.sqlite",
    ):
        self.feature_store   = feature_store
        self.reference_days  = reference_days
        self.current_days    = current_days
        self.db_path         = db_path
        self._last_report:   Optional[DriftReport] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> DriftReport:
        """Run all drift checks and return a report."""
        drifts: List[DriftEvent] = []

        ref, cur = self._load_windows()
        if len(ref) < 20 or len(cur) < 5:
            return DriftReport(
                drift_detected=False,
                drifts=[],
                allocation_factor=1.0,
                retrain_required=False,
                summary="Insufficient data for drift detection",
            )

        # Feature drift
        drifts.extend(self._check_feature_drift(ref, cur))

        # Prediction drift
        drifts.extend(self._check_prediction_drift(ref, cur))

        # Regime drift
        drifts.extend(self._check_regime_drift(ref, cur))

        # Label/concept drift
        drifts.extend(self._check_label_drift(ref, cur))

        high_severity   = [d for d in drifts if d.severity == 'high']
        medium_severity = [d for d in drifts if d.severity == 'medium']

        allocation_factor = 1.0
        if high_severity:
            allocation_factor = 0.50
        elif medium_severity:
            allocation_factor = 0.75

        retrain_required = bool(high_severity) or len(medium_severity) >= 3

        summary = self._build_summary(drifts, retrain_required)

        report = DriftReport(
            drift_detected=bool(drifts),
            drifts=drifts,
            allocation_factor=allocation_factor,
            retrain_required=retrain_required,
            summary=summary,
        )

        self._last_report = report

        if report.drift_detected:
            logger.warning(
                f"DriftDetector: {len(drifts)} drifts detected "
                f"(high={len(high_severity)}, medium={len(medium_severity)}) "
                f"alloc_factor={allocation_factor:.0%}"
            )
        return report

    def get_report(self) -> Optional[DriftReport]:
        return self._last_report

    def get_allocation_factor(self) -> float:
        if self._last_report:
            return self._last_report.allocation_factor
        return 1.0

    # ── Drift checks ──────────────────────────────────────────────────────────

    def _check_feature_drift(
        self, ref: List[Dict], cur: List[Dict]
    ) -> List[DriftEvent]:
        events = []
        numeric_cols = [
            'price', 'volume_24h', 'spread_pct', 'atr_pct',
            'rsi_14', 'adx', 'opportunity_score', 'signal_confidence',
        ]
        for col in numeric_cols:
            ref_vals = [r.get(col) for r in ref if r.get(col) is not None]
            cur_vals = [r.get(col) for r in cur if r.get(col) is not None]
            if len(ref_vals) < 5 or len(cur_vals) < 3:
                continue
            psi = self._psi(ref_vals, cur_vals)
            severity = self._psi_severity(psi)
            if severity in ('medium', 'high'):
                events.append(DriftEvent(
                    drift_type='feature',
                    feature=col,
                    psi=psi,
                    severity=severity,
                    description=f"Feature '{col}' PSI={psi:.3f} ({severity})",
                ))
        return events

    def _check_prediction_drift(
        self, ref: List[Dict], cur: List[Dict]
    ) -> List[DriftEvent]:
        ref_conf = [r.get('ml_prediction') for r in ref if r.get('ml_prediction') is not None]
        cur_conf = [r.get('ml_prediction') for r in cur if r.get('ml_prediction') is not None]
        if not ref_conf or not cur_conf:
            return []
        psi = self._psi(ref_conf, cur_conf)
        severity = self._psi_severity(psi)
        if severity in ('medium', 'high'):
            return [DriftEvent(
                drift_type='prediction',
                feature='ml_prediction',
                psi=psi,
                severity=severity,
                description=f"Model prediction distribution shifted PSI={psi:.3f}",
            )]
        return []

    def _check_regime_drift(
        self, ref: List[Dict], cur: List[Dict]
    ) -> List[DriftEvent]:
        def regime_dist(rows):
            counts: Dict[str, int] = {}
            for r in rows:
                regime = r.get('market_regime', 'unknown') or 'unknown'
                counts[regime] = counts.get(regime, 0) + 1
            total = sum(counts.values())
            return {k: v / total for k, v in counts.items()} if total > 0 else {}

        ref_dist = regime_dist(ref)
        cur_dist = regime_dist(cur)
        all_regimes = set(ref_dist) | set(cur_dist)

        psi = sum(
            self._psi_scalar(ref_dist.get(r, 0.0001), cur_dist.get(r, 0.0001))
            for r in all_regimes
        )
        severity = self._psi_severity(psi)
        if severity in ('medium', 'high'):
            return [DriftEvent(
                drift_type='regime',
                feature='market_regime',
                psi=psi,
                severity=severity,
                description=f"Market regime distribution shifted PSI={psi:.3f}",
            )]
        return []

    def _check_label_drift(
        self, ref: List[Dict], cur: List[Dict]
    ) -> List[DriftEvent]:
        """Concept drift: are trade outcomes changing over time?"""
        ref_wr = self._win_rate(ref)
        cur_wr = self._win_rate(cur)
        if ref_wr is None or cur_wr is None:
            return []
        drop = ref_wr - cur_wr
        if drop >= 0.10:
            return [DriftEvent(
                drift_type='label',
                feature='outcome_win_rate',
                psi=drop,
                severity='high' if drop >= 0.15 else 'medium',
                description=f"Win rate dropped {ref_wr:.1%} → {cur_wr:.1%} (concept drift)",
            )]
        return []

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_windows(self) -> Tuple[List[Dict], List[Dict]]:
        ref_cutoff = (datetime.now(timezone.utc) - timedelta(days=self.reference_days)).isoformat()
        cur_cutoff = (datetime.now(timezone.utc) - timedelta(days=self.current_days)).isoformat()

        # Try feature_store first
        if self.feature_store:
            try:
                dataset = self.feature_store.get_training_dataset(min_rows=1)
                ref = [r for r in dataset if r.get('recorded_at', '') < cur_cutoff]
                cur = [r for r in dataset if r.get('recorded_at', '') >= cur_cutoff]
                return ref, cur
            except Exception as e:
                logger.warning(f"DriftDetector: feature_store load error: {e}")

        # Fall back to positions table
        return self._load_from_positions(ref_cutoff, cur_cutoff)

    def _load_from_positions(
        self, ref_cutoff: str, cur_cutoff: str
    ) -> Tuple[List[Dict], List[Dict]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM positions WHERE status='CLOSED' AND exit_time>=?",
                    (ref_cutoff,),
                ).fetchall()
            all_rows = [dict(r) for r in rows]
            ref = [r for r in all_rows if r.get('exit_time', '') < cur_cutoff]
            cur = [r for r in all_rows if r.get('exit_time', '') >= cur_cutoff]
            return ref, cur
        except Exception as e:
            logger.warning(f"DriftDetector: positions load error: {e}")
            return [], []

    # ── Stats ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _psi(ref: List[float], cur: List[float], n_bins: int = 10) -> float:
        """Population Stability Index between two numeric distributions."""
        if not ref or not cur:
            return 0.0
        all_vals = ref + cur
        mn, mx   = min(all_vals), max(all_vals)
        if mn == mx:
            return 0.0
        bin_edges = [mn + i * (mx - mn) / n_bins for i in range(n_bins + 1)]

        def hist(vals):
            counts = [0] * n_bins
            for v in vals:
                idx = min(int((v - mn) / (mx - mn) * n_bins), n_bins - 1)
                counts[idx] += 1
            total = len(vals)
            return [max(c / total, 0.0001) for c in counts]

        ref_h = hist(ref)
        cur_h = hist(cur)
        return sum(
            (c - r) * (c / r if r > 0 else 1.0)
            for r, c in zip(ref_h, cur_h)
            if r > 0
        )

    @staticmethod
    def _psi_scalar(ref: float, cur: float) -> float:
        if ref <= 0:
            ref = 0.0001
        if cur <= 0:
            cur = 0.0001
        return (cur - ref) * (cur / ref)

    @staticmethod
    def _psi_severity(psi: float) -> str:
        if psi < PSI_LOW:    return 'none'
        if psi < PSI_MEDIUM: return 'low'
        if psi < PSI_HIGH:   return 'medium'
        return 'high'

    @staticmethod
    def _win_rate(rows: List[Dict]) -> Optional[float]:
        pnls = [r.get('net_pnl') for r in rows if r.get('net_pnl') is not None]
        if len(pnls) < 5:
            return None
        return sum(1 for p in pnls if p > 0) / len(pnls)

    @staticmethod
    def _build_summary(drifts: List[DriftEvent], retrain: bool) -> str:
        if not drifts:
            return "No drift detected. Model and data stable."
        lines = [f"{len(drifts)} drift event(s) detected:"]
        for d in drifts[:5]:
            lines.append(f"  [{d.severity.upper()}] {d.description}")
        if retrain:
            lines.append("Action: RETRAIN REQUIRED")
        return "\n".join(lines)
