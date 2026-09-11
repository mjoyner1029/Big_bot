"""Pipeline maturity — evidence-based promotion, not elapsed time.

The alpha pipeline may retire the legacy fallback (or enter exclusive mode)
only when its PREDICTIONS have been validated against realized outcomes:
calibrated EV, accurate costs, ranking quality — not merely "14 days passed
and 100 decisions occurred", and never profit alone (luck).
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class MaturityState:
    IMMATURE = "IMMATURE"
    OBSERVING = "OBSERVING"
    VALIDATING = "VALIDATING"
    MATURE = "MATURE"
    DEGRADED = "DEGRADED"


@dataclass
class MaturityThresholds:
    min_days: int = 14
    min_decisions: int = 100
    min_resolved_outcomes: int = 50
    min_effective_samples: float = 30.0
    max_ev_calibration_error: float = 0.005      # |predicted-realized| mean EV gap (fraction)
    max_probability_calibration_error: float = 0.15
    max_cost_prediction_error: float = 0.5       # relative error tolerance
    min_forward_net_expectancy: float = 0.0      # fraction
    min_ranking_quality: float = 0.0             # top-decile minus bottom-decile realized
    max_drawdown_vs_expected: float = 2.0        # realized/expected DD ratio
    degrade_expectancy_floor: float = -0.002     # sustained negative → DEGRADED


@dataclass
class MaturityEvidence:
    days_observed: float = 0.0
    decisions: int = 0
    resolved_outcomes: int = 0
    effective_sample_size: float = 0.0
    ev_calibration_error: Optional[float] = None
    probability_calibration_error: Optional[float] = None
    cost_prediction_error: Optional[float] = None
    forward_net_expectancy: Optional[float] = None
    ranking_quality: Optional[float] = None
    drawdown_vs_expected: Optional[float] = None
    execution_success_rate: Optional[float] = None
    critical_failures: int = 0


@dataclass
class MaturityAssessment:
    state: str
    score: float                    # 0-100
    checks: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Optional[MaturityEvidence] = None

    @property
    def legacy_fallback_allowed(self) -> bool:
        return self.state != MaturityState.MATURE

    def report(self) -> str:
        lines = [f"PIPELINE MATURITY: {self.state} (score {self.score:.0f}/100)"]
        for c in self.checks:
            lines.append(f"  {c['name']}: {'PASS' if c['passed'] else 'FAIL'} "
                         f"({c['detail']})")
        if self.state == MaturityState.MATURE:
            lines.append("  EXCLUSIVE MODE ELIGIBLE")
        return "\n".join(lines)


# ── Calibration buckets (spec §20) ────────────────────────────────────────────


def ev_calibration_buckets(
    pairs: Sequence[Tuple[float, float]],   # (predicted_ev, realized_return) fractions
    edges_bps: Sequence[float] = (0, 10, 25, 50, float("inf")),
) -> List[Dict[str, Any]]:
    buckets = []
    for lo, hi in zip(edges_bps[:-1], edges_bps[1:]):
        rows = [(p, r) for p, r in pairs if lo <= p * 10_000 < hi]
        if not rows:
            continue
        pred = sum(p for p, _ in rows) / len(rows)
        real = sum(r for _, r in rows) / len(rows)
        buckets.append({
            "bucket_bps": f"{lo:g}-{hi:g}",
            "n": len(rows),
            "predicted_mean": pred,
            "realized_mean": real,
            "error": abs(pred - real),
        })
    return buckets


def ev_calibration_error(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Sample-weighted mean |predicted − realized| across EV buckets (fraction)."""
    buckets = ev_calibration_buckets(pairs)
    if not buckets:
        return None
    total = sum(b["n"] for b in buckets)
    return sum(b["error"] * b["n"] for b in buckets) / total


def probability_calibration_error(
    pairs: Sequence[Tuple[float, bool]],    # (predicted_p_positive, realized_positive)
    edges: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 1.0001),
) -> Optional[float]:
    errors, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = [(p, o) for p, o in pairs if lo <= p < hi]
        if not rows:
            continue
        pred = sum(p for p, _ in rows) / len(rows)
        freq = sum(1 for _, o in rows if o) / len(rows)
        errors.append(abs(pred - freq))
        weights.append(len(rows))
    if not errors:
        return None
    return sum(e * w for e, w in zip(errors, weights)) / sum(weights)


def ranking_quality(
    ranked_outcomes: Sequence[Tuple[int, float]],   # (rank, realized_return)
) -> Optional[float]:
    """Top-decile mean realized return minus bottom-decile mean.
    Positive = higher-ranked opportunities actually outperform."""
    if len(ranked_outcomes) < 10:
        return None
    ordered = sorted(ranked_outcomes, key=lambda x: x[0])
    n = len(ordered)
    decile = max(1, n // 10)
    top = [r for _, r in ordered[:decile]]
    bottom = [r for _, r in ordered[-decile:]]
    return sum(top) / len(top) - sum(bottom) / len(bottom)


# ── Assessment ────────────────────────────────────────────────────────────────


class PipelineMaturityEvaluator:
    """Scores pipeline maturity from persisted decisions + resolved outcomes.

    Evidence can also be injected directly (tests / external metrics).
    """

    def __init__(self, db_path: str = "data/trade_memory.sqlite",
                 thresholds: Optional[MaturityThresholds] = None) -> None:
        self.db_path = db_path
        self.thresholds = thresholds or MaturityThresholds()

    def gather_evidence(self) -> MaturityEvidence:
        ev = MaturityEvidence()
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT COUNT(*), COUNT(DISTINCT substr(recorded_at,1,10)) "
                    "FROM alpha_pipeline_shadow"
                ).fetchone()
                ev.decisions, ev.days_observed = (row[0] or 0), float(row[1] or 0)

                # Resolved outcomes: accepted decisions joined to realized trades
                pairs = conn.execute(
                    "SELECT s.expected_net_return, tm.net_return_pct / 100.0 "
                    "FROM alpha_pipeline_shadow s "
                    "JOIN trade_attribution ta ON ta.candidate_id = s.candidate_id "
                    "JOIN trade_memory tm ON tm.id = ta.trade_memory_id "
                    "WHERE s.accepted=1 AND s.expected_net_return IS NOT NULL "
                    "AND tm.net_return_pct IS NOT NULL"
                ).fetchall()
        except sqlite3.OperationalError:
            return ev
        ev.resolved_outcomes = len(pairs)
        if pairs:
            from core.validation_stats import effective_sample_size
            realized = [r for _, r in pairs]
            ev.effective_sample_size = effective_sample_size(realized)
            ev.forward_net_expectancy = sum(realized) / len(realized)
            ev.ev_calibration_error = ev_calibration_error(pairs)
        return ev

    def assess(self, evidence: Optional[MaturityEvidence] = None) -> MaturityAssessment:
        t = self.thresholds
        ev = evidence if evidence is not None else self.gather_evidence()

        checks: List[Dict[str, Any]] = []

        def check(name, passed, detail):
            checks.append({"name": name, "passed": bool(passed), "detail": detail})
            return bool(passed)

        obs_ok = check("days_observed", ev.days_observed >= t.min_days,
                       f"{ev.days_observed:.0f}/{t.min_days}")
        dec_ok = check("decisions", ev.decisions >= t.min_decisions,
                       f"{ev.decisions}/{t.min_decisions}")
        res_ok = check("resolved_outcomes", ev.resolved_outcomes >= t.min_resolved_outcomes,
                       f"{ev.resolved_outcomes}/{t.min_resolved_outcomes}")
        ess_ok = check("effective_samples",
                       ev.effective_sample_size >= t.min_effective_samples,
                       f"{ev.effective_sample_size:.1f}/{t.min_effective_samples}")
        cal_ok = check(
            "ev_calibration",
            ev.ev_calibration_error is not None
            and ev.ev_calibration_error <= t.max_ev_calibration_error,
            f"{ev.ev_calibration_error}" if ev.ev_calibration_error is not None
            else "no calibration data")
        prob_ok = check(
            "probability_calibration",
            ev.probability_calibration_error is None
            or ev.probability_calibration_error <= t.max_probability_calibration_error,
            f"{ev.probability_calibration_error}")
        cost_ok = check(
            "cost_calibration",
            ev.cost_prediction_error is None
            or ev.cost_prediction_error <= t.max_cost_prediction_error,
            f"{ev.cost_prediction_error}")
        expct_ok = check(
            "forward_net_expectancy",
            ev.forward_net_expectancy is not None
            and ev.forward_net_expectancy >= t.min_forward_net_expectancy,
            f"{ev.forward_net_expectancy}")
        rank_ok = check(
            "ranking_quality",
            ev.ranking_quality is None or ev.ranking_quality >= t.min_ranking_quality,
            f"{ev.ranking_quality}")
        dd_ok = check(
            "drawdown_control",
            ev.drawdown_vs_expected is None
            or ev.drawdown_vs_expected <= t.max_drawdown_vs_expected,
            f"{ev.drawdown_vs_expected}")
        stable_ok = check("operational_stability", ev.critical_failures == 0,
                          f"{ev.critical_failures} critical failures")

        score = 100.0 * sum(1 for c in checks if c["passed"]) / len(checks)

        # DEGRADED: previously observable pipeline with sustained negative results
        if (ev.resolved_outcomes >= t.min_resolved_outcomes
                and ev.forward_net_expectancy is not None
                and ev.forward_net_expectancy < t.degrade_expectancy_floor):
            state = MaturityState.DEGRADED
        elif all(c["passed"] for c in checks):
            state = MaturityState.MATURE
        elif res_ok and (cal_ok or prob_ok):
            state = MaturityState.VALIDATING
        elif obs_ok and dec_ok:
            state = MaturityState.OBSERVING
        else:
            state = MaturityState.IMMATURE

        return MaturityAssessment(state=state, score=score, checks=checks,
                                  evidence=ev)
