"""
CalibrationAnalyzer — validate model confidence vs actual success rates.

PHASE 16

A well-calibrated model that claims 70% confidence should actually win 70% of the time.
Poor calibration → allocations are mis-sized → worse risk-adjusted returns.

Calibration buckets:
    50-55%   55-60%   60-65%   65-70%   70-80%   80-100%

Each bucket tracks:
    predicted_prob     — model's stated confidence (e.g. 0.70)
    actual_win_rate    — fraction of trades that won
    sample_size        — number of trades in bucket
    expectancy         — average PnL in bucket
    calibration_error  — |predicted - actual|
    reliability_factor — multiplier for sizing (0.0 – 1.0)

Poorly calibrated buckets reduce allocation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from validation.engine import Trade


@dataclass
class CalibrationBucket:
    """Calibration data for one confidence bucket."""
    label:            str        # e.g. "60-65%"
    low:              float      # lower bound (inclusive)
    high:             float      # upper bound (exclusive)
    predicted_prob:   float      # bucket midpoint
    sample_size:      int
    actual_win_rate:  float
    expectancy:       float
    avg_confidence:   float
    calibration_error: float     # |predicted - actual|
    reliability_factor: float    # 0.0 – 1.0 sizing multiplier

    @property
    def is_well_calibrated(self) -> bool:
        return self.calibration_error < 0.10

    def summary_row(self) -> str:
        status = "✓" if self.is_well_calibrated else "✗"
        return (
            f"  {status} {self.label:<10} "
            f"n={self.sample_size:>5}  "
            f"pred={self.predicted_prob:.0%}  "
            f"actual={self.actual_win_rate:.0%}  "
            f"err={self.calibration_error:.1%}  "
            f"factor={self.reliability_factor:.2f}  "
            f"E=${self.expectancy:>7.3f}"
        )


@dataclass
class CalibrationReport:
    """Full calibration report for a model."""
    model_name:        str
    n_trades:          int
    buckets:           List[CalibrationBucket] = field(default_factory=list)
    brier_score:       float = 0.0   # lower is better (0 = perfect)
    ece:               float = 0.0   # Expected Calibration Error
    overall_accuracy:  float = 0.0
    overall_error:     float = 0.0
    is_well_calibrated: bool = False

    def summary(self) -> str:
        lines = [
            "=" * 70,
            f"  CALIBRATION REPORT: {self.model_name}",
            "=" * 70,
            f"  Trades:      {self.n_trades}",
            f"  Brier score: {self.brier_score:.4f}  (lower=better, 0=perfect)",
            f"  ECE:         {self.ece:.3f}            (lower=better)",
            f"  Well calibrated: {'✓ YES' if self.is_well_calibrated else '✗ NO'}",
            "",
            "  BUCKET DETAIL  (pred=model confidence, actual=empirical win rate)",
            "─" * 70,
        ]
        for b in self.buckets:
            lines.append(b.summary_row())
        lines.append("=" * 70)
        return "\n".join(lines)


class CalibrationAnalyzer:
    """
    Analyzes model calibration by comparing predicted probabilities to actual outcomes.

    Trades must have a `confidence` attribute (0.50 – 1.00).
    A win is defined as pnl_net > 0.

    Usage:
        # Attach confidence to trades first
        analyzer = CalibrationAnalyzer()
        report   = analyzer.analyze(trades_with_confidence, model_name="MetaModel")
        print(report.summary())

        # Get per-bucket reliability factors for position sizing
        factors = analyzer.get_reliability_factors(report)
    """

    # Default bucket boundaries
    BUCKET_EDGES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01]
    BUCKET_LABELS = ['50-55%', '55-60%', '60-65%', '65-70%', '70-80%', '80%+']

    # Calibration error threshold
    MAX_CALIBRATION_ERROR = 0.10   # >10% error → bucket is unreliable

    def analyze(
        self,
        trades:     List[Trade],
        model_name: str = "Model",
        confidences: List[float] = None,  # parallel to trades; or use trade.session_id hack
    ) -> CalibrationReport:
        """
        Analyze calibration of model confidence predictions.

        Args:
            trades:      List of Trade objects (pnl_net is outcome)
            model_name:  Name for the report
            confidences: Optional list of confidence values parallel to trades.
                        If None, uses placeholder uniform confidence.
        """
        if not trades:
            return CalibrationReport(model_name=model_name, n_trades=0)

        if confidences is None:
            # No confidence data available — neutral calibration
            confidences = [0.60] * len(trades)

        n = min(len(trades), len(confidences))
        trades      = trades[:n]
        confidences = confidences[:n]

        # Assign to buckets
        bucket_data: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(self.BUCKET_EDGES) - 1)}

        for trade, conf in zip(trades, confidences):
            bucket_idx = self._bucket_idx(conf)
            win = 1.0 if trade.pnl_net > 0 else 0.0
            bucket_data[bucket_idx].append((conf, win))

        buckets = []
        brier_terms = []
        ece_terms   = []

        for i, (low, high) in enumerate(zip(self.BUCKET_EDGES[:-1], self.BUCKET_EDGES[1:])):
            data   = bucket_data.get(i, [])
            n_buck = len(data)

            if n_buck == 0:
                continue

            avg_conf   = sum(c for c, _ in data) / n_buck
            actual_wr  = sum(w for _, w in data) / n_buck
            mid        = (low + min(high, 1.0)) / 2
            cal_error  = abs(avg_conf - actual_wr)

            # Brier score terms
            brier_terms.extend((c - w) ** 2 for c, w in data)
            # ECE terms
            ece_terms.append((n_buck / n) * cal_error)

            # Reliability factor: 1.0 if well calibrated, reduced if poorly calibrated
            if cal_error < 0.05:
                factor = 1.0
            elif cal_error < 0.10:
                factor = 0.75
            elif cal_error < 0.15:
                factor = 0.50
            else:
                factor = 0.25

            trade_subset = trades[:]   # all trades for expectancy
            pnls         = [t.pnl_net for t, (c, _) in zip(trades, bucket_data[i].__class__()
                             if False else zip(trades, bucket_data[i])) if True]
            # Simpler expectancy from pnl_net for trades in bucket
            pnl_list = []
            for trade_i, (c, w) in zip(trades, data):
                pass
            # Get expectancy from data
            trade_pnls = []
            start = sum(len(bucket_data[j]) for j in range(i))
            end   = start + n_buck
            # Match trades to bucket by iterating
            bucket_pnls = self._get_bucket_pnls(trades, confidences, low, min(high, 1.0))
            expectancy   = sum(bucket_pnls) / len(bucket_pnls) if bucket_pnls else 0.0

            label = self.BUCKET_LABELS[i] if i < len(self.BUCKET_LABELS) else f"{low:.0%}-{min(high,1):.0%}"

            buckets.append(CalibrationBucket(
                label=label, low=low, high=min(high, 1.0),
                predicted_prob=avg_conf,
                sample_size=n_buck,
                actual_win_rate=actual_wr,
                expectancy=expectancy,
                avg_confidence=avg_conf,
                calibration_error=cal_error,
                reliability_factor=factor,
            ))

        brier_score      = sum(brier_terms) / n if brier_terms else 0.0
        ece              = sum(ece_terms) if ece_terms else 0.0
        overall_accuracy = sum(1 for t in trades if t.pnl_net > 0) / n
        overall_error    = abs(sum(confidences) / n - overall_accuracy) if n > 0 else 0.0
        is_calibrated    = ece < 0.10

        return CalibrationReport(
            model_name=model_name,
            n_trades=n,
            buckets=buckets,
            brier_score=brier_score,
            ece=ece,
            overall_accuracy=overall_accuracy,
            overall_error=overall_error,
            is_well_calibrated=is_calibrated,
        )

    def get_reliability_factors(self, report: CalibrationReport) -> Dict[str, float]:
        """Return a dict of {bucket_label: reliability_factor} for position sizing."""
        return {b.label: b.reliability_factor for b in report.buckets}

    def reliability_for_confidence(self, report: CalibrationReport, confidence: float) -> float:
        """Get the reliability factor for a specific confidence value."""
        idx = self._bucket_idx(confidence)
        for b in report.buckets:
            if b.low <= confidence < b.high:
                return b.reliability_factor
        return 0.5  # conservative default

    # ── Private ────────────────────────────────────────────────────────────

    def _bucket_idx(self, conf: float) -> int:
        for i, edge in enumerate(self.BUCKET_EDGES[1:]):
            if conf < edge:
                return i
        return len(self.BUCKET_EDGES) - 2

    @staticmethod
    def _get_bucket_pnls(
        trades: List[Trade],
        confidences: List[float],
        low: float,
        high: float,
    ) -> List[float]:
        return [
            t.pnl_net for t, c in zip(trades, confidences)
            if low <= c < high
        ]
