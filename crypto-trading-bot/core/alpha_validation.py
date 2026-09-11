"""AlphaValidationPipeline — the ONLY road from hypothesis to PAPER alpha.

Every stage produces a standardized StageResult (PASS/FAIL/SKIP/UNAVAILABLE/
ERROR) with metrics, thresholds, reason codes and timing. Gate policy is
central: mandatory stages cannot be bypassed, UNAVAILABLE never silently
counts as PASS, and the 0-100 validation score can never override a hard
failure.

Stages (reusing existing validators):
    independent OOS                 (chronological segments)
    walk-forward folds              (validation_stats.walk_forward-style)
    purge / embargo                 (overlap gaps between folds)
    parameter robustness            (validation_stats.parameter_robustness_score)
    profit / event concentration    (validation_stats.profit_concentration)
    regime stability                (volatility-regime segmentation)
    transaction-cost stress         (0x/1x/1.5x/2x/3x + break-even)
    Monte Carlo block bootstrap     (serial-dependence-preserving)
    deflated Sharpe / overfit       (validation_stats.deflated_sharpe_ratio)
    extreme-return sanity           (return_units conventions)
    final holdout                   (HoldoutManager, single access)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
UNAVAILABLE = "UNAVAILABLE"
ERROR = "ERROR"

MANDATORY = "MANDATORY"
OPTIONAL = "OPTIONAL"
CONDITIONAL = "CONDITIONAL"


# Structured failure reason codes (spec §24, §66)
class ValidationReason:
    FAILED_OOS = "FAILED_OOS"
    FAILED_TEMPORAL_STABILITY = "FAILED_TEMPORAL_STABILITY"
    FAILED_WALK_FORWARD = "FAILED_WALK_FORWARD"
    INSUFFICIENT_WF_FOLDS = "INSUFFICIENT_WF_FOLDS"
    FAILED_PARAMETER_ROBUSTNESS = "FAILED_PARAMETER_ROBUSTNESS"
    PARAMETER_CLIFF = "PARAMETER_CLIFF"
    FAILED_COST_STRESS = "FAILED_COST_STRESS"
    OUTLIER_DEPENDENT = "OUTLIER_DEPENDENT"
    FAILED_REGIME_STABILITY = "FAILED_REGIME_STABILITY"
    FAILED_MONTE_CARLO = "FAILED_MONTE_CARLO"
    TAIL_RISK_TOO_HIGH = "TAIL_RISK_TOO_HIGH"
    FAILED_DEFLATED_SHARPE = "FAILED_DEFLATED_SHARPE"
    HIGH_PBO = "HIGH_PBO"
    PBO_UNAVAILABLE_INSUFFICIENT_VARIANTS = "PBO_UNAVAILABLE_INSUFFICIENT_VARIANTS"
    FAILED_HOLDOUT = "FAILED_HOLDOUT"
    HOLDOUT_UNAVAILABLE = "HOLDOUT_UNAVAILABLE"
    HOLDOUT_ALREADY_CONSUMED = "HOLDOUT_ALREADY_CONSUMED"
    HOLDOUT_INSUFFICIENT_SAMPLE = "HOLDOUT_INSUFFICIENT_SAMPLE"
    EXTREME_RETURN_REVIEW = "EXTREME_RETURN_REVIEW"
    POSSIBLE_DATA_ERROR = "POSSIBLE_DATA_ERROR"


@dataclass
class StageResult:
    stage_name: str
    status: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    reason_codes: List[str] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    duration_ms: float = 0.0
    data_period: str = ""
    data_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ValidationConfig:
    """Central gate policy — stage requirements live HERE, nowhere else."""
    stage_policy: Dict[str, str] = field(default_factory=lambda: {
        "oos": MANDATORY,
        "temporal_stability": MANDATORY,     # chunk consistency of realized returns
        "true_walk_forward": MANDATORY,      # per-fold train→freeze→forward test
        "parameter_robustness": MANDATORY,   # SKIP allowed for non-numeric hypotheses
        "profit_concentration": MANDATORY,
        "cost_stress": MANDATORY,
        "monte_carlo": MANDATORY,
        "deflated_sharpe": MANDATORY,
        "pbo": CONDITIONAL,                  # mandatory when statistically available
        "extreme_return_sanity": MANDATORY,
        "regime_stability": CONDITIONAL,     # REGIME_SPECIFIC pass allowed
        "event_concentration": CONDITIONAL,  # UNAVAILABLE tolerated
        "holdout": CONDITIONAL,              # forward-paper substitution policy
    })
    # OOS
    min_oos_sample: int = 10
    min_oos_retention: float = 0.2
    # Temporal stability (chunked realized-return consistency — NOT walk-forward)
    wf_folds: int = 4
    min_positive_fold_fraction: float = 0.6
    min_wf_folds_with_data: int = 3
    # TRUE walk-forward (train → freeze → forward test per fold)
    walk_forward_min_folds: int = 3
    walk_forward_min_total_forward_obs: int = 15
    walk_forward_min_profitable_fold_pct: float = 0.6
    walk_forward_min_median_ev: float = 0.0
    walk_forward_min_train_obs: int = 10
    walk_forward_min_test_obs: int = 4
    # Robustness (real parameter-space neighborhoods)
    min_parameter_robustness: float = 40.0    # 0-100 score
    min_neighbor_variants: int = 2
    max_parameter_neighbors: int = 8
    # Concentration
    max_concentration_top5: float = 0.85
    # Cost stress
    cost_frac_1x: float = 0.002
    required_stress_multiple: float = 1.5    # must stay positive at 1.5x costs
    # Monte Carlo
    mc_sims: int = 500
    mc_block_size: int = 5
    min_p_profitable: float = 0.7
    max_p_severe_drawdown: float = 0.2
    severe_drawdown_frac: float = 0.15       # of cumulative sample return path
    # Overfit diagnostics
    min_deflated_sharpe_prob: float = 0.5
    max_pbo: float = 0.5                     # IS winner usually below OOS median → reject
    pbo_min_configurations: int = 4
    pbo_min_observations: int = 40
    pbo_partitions: int = 8
    # Extreme returns
    extreme_mean_return: float = 0.05        # >5% mean per-event needs review
    # Holdout
    min_holdout_sample: int = 8
    holdout_unavailable_policy: str = "paper_with_forward_substitution"
    seed: int = 42


@dataclass
class AlphaValidationResult:
    family_signature: str
    parameter_signature: str
    campaign_id: str
    decision: str                       # PASS | REJECT
    validation_score: float             # 0-100; never overrides hard failures
    stages: List[StageResult] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    valid_regimes: List[str] = field(default_factory=list)
    conservative_prior: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: str = field(default_factory=_utcnow)

    @property
    def passed(self) -> bool:
        return self.decision == "PASS"

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["stages"] = [s.to_dict() for s in self.stages]
        return d


# ── Signatures (spec §8) ──────────────────────────────────────────────────────


def family_signature(h) -> str:
    """WHAT is being tested: family + feature set + symbol + direction +
    holding — numeric parameter values excluded."""
    features = sorted({c.get("feature", "") for c in (h.entry_conditions or [])})
    base_subfamily = "".join(ch for ch in h.subfamily if not ch.isdigit()).strip("_.x")
    return (f"{h.family}:{base_subfamily}:{h.symbol}:{h.direction}:"
            f"h{h.holding_bars}:{'|'.join(features)}")


def parameter_signature(h) -> str:
    """The EXACT parameter version: family signature + full condition values."""
    conds = json.dumps(sorted(
        [(c.get("feature"), c.get("op"), str(c.get("value")))
         for c in (h.entry_conditions or [])]), sort_keys=True)
    digest = hashlib.md5(conds.encode()).hexdigest()[:8]
    return f"{family_signature(h)}:{h.subfamily}:{digest}"


# ── Real parameter vectors + typed, normalized distances (spec §28-31) ───────


def parameter_vector(h) -> Dict[str, Any]:
    """Explicit parameter vector from ACTUAL condition values (never ordinal
    sibling indexes). BETWEEN uses the interval midpoint; holding period is a
    parameter too."""
    vec: Dict[str, Any] = {"holding_bars": float(h.holding_bars)}
    for c in (h.entry_conditions or []):
        key = f"{c.get('feature')}{c.get('op')}"
        value = c.get("value")
        if isinstance(value, bool):
            vec[key] = bool(value)
        elif isinstance(value, (int, float)):
            vec[key] = float(value)
        elif isinstance(value, (list, tuple)) and len(value) == 2 and all(
                isinstance(v, (int, float)) for v in value):
            vec[key] = (float(value[0]) + float(value[1])) / 2
        else:
            vec[key] = str(value)          # categorical
    return vec


def parameter_distance(a: Dict[str, Any], b: Dict[str, Any],
                       scales: Optional[Dict[str, float]] = None) -> float:
    """Type-aware normalized distance between parameter vectors.

    numeric  → |a-b| / dimension scale (range across tested variants)
    boolean  → 0 / 1
    category → 0 / 1 (Hamming)
    Missing dimensions count as maximal (1.0). Euclidean over dimensions.
    """
    scales = scales or {}
    dims = set(a) | set(b)
    total = 0.0
    for d in dims:
        va, vb = a.get(d), b.get(d)
        if va is None or vb is None:
            total += 1.0
            continue
        if isinstance(va, bool) or isinstance(vb, bool):
            total += 0.0 if bool(va) == bool(vb) else 1.0
        elif isinstance(va, float) and isinstance(vb, float):
            scale = scales.get(d) or max(abs(va), abs(vb), 1e-9)
            total += ((va - vb) / scale) ** 2
        else:
            total += 0.0 if str(va) == str(vb) else 1.0
    return math.sqrt(total)


def dimension_scales(vectors: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Per-dimension scale = value range across tested variants (min 1e-9),
    so e.g. holding minutes cannot dominate an RVOL threshold."""
    scales: Dict[str, float] = {}
    dims = {d for v in vectors for d in v}
    for d in dims:
        vals = [v[d] for v in vectors
                if isinstance(v.get(d), float) and not isinstance(v.get(d), bool)]
        if len(vals) >= 2:
            scales[d] = max(max(vals) - min(vals), 1e-9)
    return scales


# ── Pipeline ──────────────────────────────────────────────────────────────────


class AlphaValidationPipeline:
    """Runs every applicable validation stage; promotion requires PASS on all
    mandatory stages. UNAVAILABLE never counts as PASS."""

    def __init__(self, config: Optional[ValidationConfig] = None,
                 db_path: str = "data/trade_memory.sqlite",
                 holdout_manager=None,
                 neighbor_backtest_fn=None) -> None:
        self.config = config or ValidationConfig()
        self.db_path = db_path
        self._holdout_manager = holdout_manager
        # Optional hook: actively backtest additional nearby parameter vectors
        # (discovery/validation data only — never holdout)
        self.neighbor_backtest_fn = neighbor_backtest_fn

    def validate(
        self,
        hypothesis,                           # DiscoveredHypothesis (discovery)
        oos_match=None,                       # matching hypothesis on OOS segment
        holdout_match=None,                   # matching hypothesis on holdout segment
        holdout_available: bool = True,
        sibling_variants: Optional[List] = None,   # same family, other params
        family_trials: int = 1,
        campaign_id: str = "",
        regime_series=None,                   # pd.Series ts->regime label (optional)
        asset_class: str = "crypto",
        holdout_period: Optional[Tuple[str, str]] = None,
    ) -> AlphaValidationResult:
        cfg = self.config
        stages: List[StageResult] = []
        reasons: List[str] = []
        valid_regimes: List[str] = []

        sample = list(hypothesis.metadata.get("returns_sample") or [])
        oos_sample = list((oos_match.metadata.get("returns_sample") or [])
                          if oos_match else [])

        def run_stage(name, fn, *args, **kwargs) -> StageResult:
            start = datetime.now(timezone.utc)
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                logger.error(f"Validation stage {name} ERROR: {e}")
                result = StageResult(name, ERROR, reason_codes=[str(e)[:120]])
            result.stage_name = name
            result.started_at = start.isoformat()
            result.completed_at = _utcnow()
            result.duration_ms = (datetime.now(timezone.utc) - start
                                  ).total_seconds() * 1000
            stages.append(result)
            reasons.extend(result.reason_codes)
            return result

        run_stage("oos", self._stage_oos, hypothesis, oos_match)
        # Chunk consistency of realized returns — useful, but NOT walk-forward
        run_stage("temporal_stability", self._stage_temporal_stability,
                  sample + oos_sample, hypothesis.holding_bars)
        # TRUE walk-forward: per-fold parameter selection on training data
        # only, freeze, then forward test (spec §3-8)
        run_stage("true_walk_forward", self._stage_true_walk_forward,
                  hypothesis, sibling_variants or [])
        run_stage("parameter_robustness", self._stage_robustness,
                  hypothesis, sibling_variants or [])
        # Concentration on the hypothesis's own discovery evidence — combining
        # with OOS would dilute a single dominating outlier
        run_stage("profit_concentration", self._stage_concentration, sample)
        run_stage("cost_stress", self._stage_cost_stress,
                  oos_sample if oos_sample else sample)
        # MC on the discovery evidence — mixing in OOS would dilute tail events
        run_stage("monte_carlo", self._stage_monte_carlo, sample)
        run_stage("deflated_sharpe", self._stage_deflated_sharpe,
                  sample + oos_sample, family_trials)
        # PBO across competing parameter variants (CSCV) — selection overfit
        run_stage("pbo", self._stage_pbo, hypothesis, sibling_variants or [])
        run_stage("extreme_return_sanity", self._stage_extreme_returns,
                  hypothesis, asset_class)
        regime_stage = run_stage("regime_stability", self._stage_regime,
                                 hypothesis, regime_series)
        valid_regimes = regime_stage.metrics.get("valid_regimes", [])
        run_stage("event_concentration", self._stage_event_concentration,
                  hypothesis)
        run_stage("holdout", self._stage_holdout, hypothesis, holdout_match,
                  holdout_available, holdout_period)

        # ── Gate policy: hard gates stay hard ─────────────────────────────────
        decision = "PASS"
        for stage in stages:
            policy = cfg.stage_policy.get(stage.stage_name, OPTIONAL)
            if policy == MANDATORY:
                if stage.status in (FAIL, ERROR, UNAVAILABLE):
                    decision = "REJECT"
            elif policy == CONDITIONAL:
                if stage.status in (FAIL, ERROR):
                    decision = "REJECT"
                # UNAVAILABLE tolerated per configured policy (documented)

        n_scored = sum(1 for s in stages if s.status in (PASS, FAIL, ERROR))
        n_passed = sum(1 for s in stages if s.status == PASS)
        score = 100.0 * n_passed / n_scored if n_scored else 0.0

        prior = self._conservative_prior(stages, oos_sample, sample)

        return AlphaValidationResult(
            family_signature=family_signature(hypothesis),
            parameter_signature=parameter_signature(hypothesis),
            campaign_id=campaign_id,
            decision=decision,
            validation_score=score,
            stages=stages,
            reason_codes=sorted(set(reasons)),
            valid_regimes=valid_regimes,
            conservative_prior=prior,
            metadata={
                "hypothesis_id": hypothesis.hypothesis_id,
                "symbol": hypothesis.symbol,
                "family": hypothesis.family,
                "subfamily": hypothesis.subfamily,
                "direction": hypothesis.direction,
                "conditions": hypothesis.entry_conditions,
                "holding_bars": hypothesis.holding_bars,
                "family_trials": family_trials,
            },
        )

    # ── Stages ────────────────────────────────────────────────────────────────

    def _stage_oos(self, h, oos_match) -> StageResult:
        cfg = self.config
        if oos_match is None:
            return StageResult("oos", FAIL,
                               reason_codes=[ValidationReason.FAILED_OOS],
                               metrics={"note": "no matching OOS effect"})
        retention = (oos_match.mean_return / h.mean_return
                     if h.mean_return > 1e-12 else 0.0)
        ok = (oos_match.sample_size >= cfg.min_oos_sample
              and oos_match.mean_return > 0
              and retention >= cfg.min_oos_retention)
        return StageResult(
            "oos", PASS if ok else FAIL,
            metrics={"discovery_mean": h.mean_return,
                     "oos_mean": oos_match.mean_return,
                     "oos_sample": oos_match.sample_size,
                     "oos_retention": retention},
            thresholds={"min_oos_sample": cfg.min_oos_sample,
                        "min_oos_retention": cfg.min_oos_retention},
            reason_codes=[] if ok else [ValidationReason.FAILED_OOS],
        )

    def _stage_temporal_stability(self, sample: List[float], holding_bars: int) -> StageResult:
        """Chronological chunk consistency of REALIZED returns with purge gaps
        of ``holding_bars`` between chunks. Useful, but explicitly NOT true
        walk-forward — no parameter selection is exercised here."""
        cfg = self.config
        purge = max(1, holding_bars)
        n = len(sample)
        if n < cfg.wf_folds * (cfg.min_oos_sample // 2 + purge):
            return StageResult(
                "temporal_stability", FAIL,
                metrics={"n": n},
                reason_codes=[ValidationReason.INSUFFICIENT_WF_FOLDS])
        fold_size = n // cfg.wf_folds
        folds = []
        for i in range(cfg.wf_folds):
            start = i * fold_size + (purge if i > 0 else 0)   # embargo boundary
            end = min((i + 1) * fold_size, n)
            chunk = sample[start:end]
            if len(chunk) < 3:
                continue
            mean = sum(chunk) / len(chunk)
            folds.append({"fold": i + 1, "n": len(chunk), "expectancy": mean,
                          "win_rate": sum(1 for r in chunk if r > 0) / len(chunk)})
        if len(folds) < cfg.min_wf_folds_with_data:
            return StageResult(
                "temporal_stability", FAIL, metrics={"folds": folds},
                reason_codes=[ValidationReason.INSUFFICIENT_WF_FOLDS])
        expectancies = [f["expectancy"] for f in folds]
        positive_frac = sum(1 for e in expectancies if e > 0) / len(expectancies)
        mean_e = sum(expectancies) / len(expectancies)
        worst = min(expectancies)
        dispersion = (max(expectancies) - worst)
        stability = positive_frac * (1.0 if mean_e > 0 else 0.0)
        ok = (positive_frac >= cfg.min_positive_fold_fraction and mean_e > 0)
        return StageResult(
            "temporal_stability", PASS if ok else FAIL,
            metrics={"folds": folds, "positive_fold_fraction": positive_frac,
                     "mean_expectancy": mean_e, "worst_fold": worst,
                     "dispersion": dispersion,
                     "temporal_stability_score": stability,
                     "purge_window": purge, "embargo_window": purge,
                     "effective_observations": sum(f["n"] for f in folds)},
            thresholds={"min_positive_fold_fraction": cfg.min_positive_fold_fraction},
            reason_codes=[] if ok else [ValidationReason.FAILED_TEMPORAL_STABILITY],
        )

    def _stage_true_walk_forward(self, h, siblings: List) -> StageResult:
        """TRUE walk-forward: per fold, select the best parameter variant using
        TRAINING observations only, FREEZE it, then evaluate that frozen
        variant on the subsequent forward window (purged by holding period).

        Selection uses the per-variant timestamped outcome streams — the
        realized executions of each exact parameter version on those windows.
        Test windows never influence selection; folds roll forward anchored.
        """
        cfg = self.config
        variants = {parameter_signature(h): h}
        for s in siblings:
            variants.setdefault(parameter_signature(s), s)

        series: Dict[str, Dict[str, float]] = {}
        for sig, v in variants.items():
            times = v.metadata.get("sample_times") or []
            rets = v.metadata.get("returns_sample") or []
            if times and len(times) == len(rets):
                series[sig] = dict(zip(times, rets))
        if parameter_signature(h) not in series:
            return StageResult(
                "true_walk_forward", FAIL,
                metrics={"note": "candidate lacks timestamped observations"},
                reason_codes=[ValidationReason.INSUFFICIENT_WF_FOLDS])

        all_times = sorted({t for s in series.values() for t in s})
        n_folds = cfg.walk_forward_min_folds
        segments = n_folds + 1
        if len(all_times) < segments * max(cfg.walk_forward_min_test_obs, 3):
            return StageResult(
                "true_walk_forward", FAIL,
                metrics={"observations": len(all_times)},
                reason_codes=[ValidationReason.INSUFFICIENT_WF_FOLDS])
        seg_size = len(all_times) // segments
        purge = max(1, h.holding_bars)

        folds = []
        total_forward = 0
        for i in range(n_folds):
            train_times = set(all_times[: (i + 1) * seg_size - purge])  # purge tail
            test_slice = all_times[(i + 1) * seg_size:
                                   (i + 2) * seg_size if i < n_folds - 1 else len(all_times)]
            test_times = set(test_slice[purge:])                        # embargo head
            # Parameter selection on TRAINING data only, then freeze
            best_sig, best_train_ev, best_train_n = None, None, 0
            for sig, s in series.items():
                train_rets = [r for t, r in s.items() if t in train_times]
                if len(train_rets) < cfg.walk_forward_min_train_obs:
                    continue
                ev = sum(train_rets) / len(train_rets)
                if best_train_ev is None or ev > best_train_ev:
                    best_sig, best_train_ev, best_train_n = sig, ev, len(train_rets)
            if best_sig is None:
                continue
            frozen = series[best_sig]
            test_rets = [r for t, r in frozen.items() if t in test_times]
            if len(test_rets) < cfg.walk_forward_min_test_obs:
                continue
            from core.validation_stats import max_drawdown, sharpe_ratio
            ev = sum(test_rets) / len(test_rets)
            total_forward += len(test_rets)
            folds.append({
                "fold_id": i + 1,
                "train_start": min(train_times) if train_times else None,
                "train_end": max(train_times) if train_times else None,
                "test_start": min(test_times) if test_times else None,
                "test_end": max(test_times) if test_times else None,
                "training_sample_size": best_train_n,
                "test_sample_size": len(test_rets),
                "frozen_parameter_signature": best_sig,
                "frozen_was_candidate": best_sig == parameter_signature(h),
                "net_expectancy": ev,
                "sharpe": sharpe_ratio(test_rets),
                "max_drawdown": max_drawdown(test_rets),
                "win_rate": sum(1 for r in test_rets if r > 0) / len(test_rets),
                "purge_window": purge, "embargo_window": purge,
            })

        if len(folds) < cfg.walk_forward_min_folds:
            return StageResult(
                "true_walk_forward", FAIL,
                metrics={"folds": folds, "total_forward_trades": total_forward},
                thresholds={"walk_forward_min_folds": cfg.walk_forward_min_folds},
                reason_codes=[ValidationReason.INSUFFICIENT_WF_FOLDS])

        evs = sorted(f["net_expectancy"] for f in folds)
        profitable_pct = sum(1 for e in evs if e > 0) / len(evs)
        median_ev = evs[len(evs) // 2]
        mean_ev = sum(evs) / len(evs)
        checks_ok = (
            total_forward >= cfg.walk_forward_min_total_forward_obs
            and profitable_pct >= cfg.walk_forward_min_profitable_fold_pct
            and median_ev > cfg.walk_forward_min_median_ev
        )
        return StageResult(
            "true_walk_forward", PASS if checks_ok else FAIL,
            metrics={"folds": folds,
                     "profitable_fold_pct": profitable_pct,
                     "positive_EV_fold_pct": profitable_pct,
                     "mean_fold_EV": mean_ev, "median_fold_EV": median_ev,
                     "worst_fold_EV": evs[0], "best_fold_EV": evs[-1],
                     "fold_EV_std": (sum((e - mean_ev) ** 2 for e in evs)
                                     / max(len(evs) - 1, 1)) ** 0.5,
                     "walk_forward_stability_score": profitable_pct
                     * (1.0 if median_ev > 0 else 0.0),
                     "total_forward_trades": total_forward,
                     "n_variants_available": len(series)},
            thresholds={"min_folds": cfg.walk_forward_min_folds,
                        "min_total_forward_obs": cfg.walk_forward_min_total_forward_obs,
                        "min_profitable_fold_pct": cfg.walk_forward_min_profitable_fold_pct,
                        "min_median_ev": cfg.walk_forward_min_median_ev},
            reason_codes=[] if checks_ok else [ValidationReason.FAILED_WALK_FORWARD],
        )

    def _stage_pbo(self, h, siblings: List) -> StageResult:
        """Probability of Backtest Overfitting via CSCV over the family's
        competing parameter variants (never a single return stream)."""
        from core.validation_stats import probability_of_backtest_overfitting
        cfg = self.config
        variants = {parameter_signature(h): h}
        for s in siblings:
            variants.setdefault(parameter_signature(s), s)
        matrix = [v.metadata.get("returns_sample") or [] for v in variants.values()]
        matrix = [m for m in matrix if m]
        result = probability_of_backtest_overfitting(
            matrix, n_partitions=cfg.pbo_partitions,
            min_configurations=cfg.pbo_min_configurations,
            min_observations=cfg.pbo_min_observations)
        if result.get("status") == "UNAVAILABLE":
            return StageResult(
                "pbo", UNAVAILABLE,
                metrics=result,
                reason_codes=[result.get("reason",
                              ValidationReason.PBO_UNAVAILABLE_INSUFFICIENT_VARIANTS)])
        pbo = result["pbo"]
        ok = pbo <= cfg.max_pbo
        return StageResult(
            "pbo", PASS if ok else FAIL,
            metrics={"pbo": pbo,
                     "number_configurations": result["n_configurations"],
                     "number_partitions": result["n_partitions"],
                     "number_cscv_splits": result["n_cscv_splits"],
                     "median_oos_rank": result["median_oos_rank"],
                     "distribution_of_rank_logits": result["rank_logits"][:20]},
            thresholds={"max_pbo": cfg.max_pbo},
            reason_codes=[] if ok else [ValidationReason.HIGH_PBO],
        )

    def _stage_robustness(self, h, siblings: List) -> StageResult:
        """Neighborhood robustness in REAL parameter space: distances are
        computed between actual parameter vectors (numeric dims normalized by
        their tested range; categorical dims Hamming) — never between ordinal
        sibling indexes. Categorical-only hypotheses SKIP.

        Optional active neighbor testing: when ``neighbor_backtest_fn`` is
        configured on the pipeline, it is invoked to test additional nearby
        parameter vectors (discovery/validation data only, never holdout).
        """
        cfg = self.config
        has_numeric = any(not isinstance(c.get("value"), str)
                          for c in (h.entry_conditions or []))
        if not has_numeric:
            return StageResult("parameter_robustness", SKIP,
                               metrics={"note": "no numeric parameters"})

        candidate_vec = parameter_vector(h)
        neighbors = []   # (distance, vector, mean_return, signature)
        vectors = [candidate_vec]
        for s in siblings:
            vec = parameter_vector(s)
            vectors.append(vec)
        scales = dimension_scales(vectors)
        for s in siblings:
            vec = parameter_vector(s)
            dist = parameter_distance(candidate_vec, vec, scales)
            neighbors.append((dist, vec, s.mean_return, parameter_signature(s)))
        neighbors.sort(key=lambda x: x[0])

        if (len(neighbors) < cfg.min_neighbor_variants
                and self.neighbor_backtest_fn is not None):
            # Active neighborhood testing on discovery/validation data ONLY
            try:
                extra = self.neighbor_backtest_fn(h) or []
                for s in extra:
                    vec = parameter_vector(s)
                    dist = parameter_distance(candidate_vec, vec, scales)
                    neighbors.append((dist, vec, s.mean_return,
                                      parameter_signature(s)))
                neighbors.sort(key=lambda x: x[0])
            except Exception as e:
                logger.warning(f"Active neighbor testing failed: {e}")

        if len(neighbors) < cfg.min_neighbor_variants:
            return StageResult(
                "parameter_robustness", FAIL,
                metrics={"variants_tested": len(neighbors) + 1},
                thresholds={"min_neighbor_variants": cfg.min_neighbor_variants},
                reason_codes=[ValidationReason.FAILED_PARAMETER_ROBUSTNESS])

        nearest = neighbors[: cfg.max_parameter_neighbors]
        best = max([h.mean_return] + [ev for _, _, ev, _ in nearest])
        n_pass = sum(1 for _, _, ev, _ in nearest if ev > 0)
        neighbor_pass_rate = n_pass / len(nearest)
        evs = sorted(ev for _, _, ev, _ in nearest)
        neighbor_median_ev = evs[len(evs) // 2]
        mean_n = sum(evs) / len(evs)
        neighbor_ev_std = (sum((e - mean_n) ** 2 for e in evs)
                           / max(len(evs) - 1, 1)) ** 0.5
        plateau_width = sum(1 for _, _, ev, _ in nearest if best > 0 and ev > best * 0.5)

        # Distance-weighted robustness: nearby neighbors matter most
        w_total = w_pos = 0.0
        for dist, _, ev, _ in nearest:
            w = 1.0 / (1.0 + dist)
            w_total += w
            w_pos += w * max(min(ev / best, 1.0), 0.0) if best > 0 else 0.0
        distance_weighted = (w_pos / w_total) if w_total > 0 else 0.0

        # Cliff: candidate strongly positive while its NEAREST neighbors are not
        nearest_k = nearest[: max(2, len(nearest) // 2)]
        near_mean = sum(ev for _, _, ev, _ in nearest_k) / len(nearest_k)
        cliff_score = 0.0
        if h.mean_return > 0:
            cliff_score = max(0.0, 1.0 - max(near_mean, 0.0) / h.mean_return)
        is_cliff = cliff_score > 0.75 and near_mean <= 0

        score = 100.0 * (0.4 * distance_weighted + 0.3 * neighbor_pass_rate
                         + 0.3 * (1.0 - cliff_score))
        ok = score >= cfg.min_parameter_robustness and not is_cliff
        codes = []
        if is_cliff:
            codes.append(ValidationReason.PARAMETER_CLIFF)
        elif not ok:
            codes.append(ValidationReason.FAILED_PARAMETER_ROBUSTNESS)
        return StageResult(
            "parameter_robustness", PASS if ok else FAIL,
            metrics={"parameter_robustness_score": score,
                     "neighbor_pass_rate": neighbor_pass_rate,
                     "neighbor_median_EV": neighbor_median_ev,
                     "neighbor_EV_std": neighbor_ev_std,
                     "parameter_plateau_width": plateau_width,
                     "parameter_cliff_score": cliff_score,
                     "distance_weighted_robustness": distance_weighted,
                     "candidate_vector": candidate_vec,
                     "dimension_scales": scales,
                     "neighborhood": [
                         {"distance": round(d, 4), "vector": v,
                          "mean_return": ev, "parameter_signature": sig[-16:]}
                         for d, v, ev, sig in nearest]},
            thresholds={"min_parameter_robustness": cfg.min_parameter_robustness},
            reason_codes=codes,
        )

    def _stage_concentration(self, sample: List[float]) -> StageResult:
        from core.validation_stats import profit_concentration
        cfg = self.config
        conc = profit_concentration(sample)
        losses = [-r for r in sample if r < 0]
        loss_conc = profit_concentration(losses) if losses else {}
        ok = conc["concentration_score"] <= cfg.max_concentration_top5
        return StageResult(
            "profit_concentration", PASS if ok else FAIL,
            metrics={**conc, "loss_concentration_top5":
                     loss_conc.get("top_5_trades_pct")},
            thresholds={"max_concentration_top5": cfg.max_concentration_top5},
            reason_codes=[] if ok else [ValidationReason.OUTLIER_DEPENDENT],
        )

    def _stage_cost_stress(self, sample: List[float]) -> StageResult:
        cfg = self.config
        if not sample:
            return StageResult("cost_stress", FAIL,
                               reason_codes=[ValidationReason.FAILED_COST_STRESS])
        gross = sum(sample) / len(sample)
        c = cfg.cost_frac_1x
        levels = {f"net_{m:g}x": gross - m * c for m in (0.0, 1.0, 1.5, 2.0, 3.0)}
        break_even_bps = gross * 10_000
        survival = gross / c if c > 0 else float("inf")
        ok = levels[f"net_{cfg.required_stress_multiple:g}x"] > 0
        return StageResult(
            "cost_stress", PASS if ok else FAIL,
            metrics={"gross_expectancy": gross, **levels,
                     "break_even_cost_bps": break_even_bps,
                     "cost_survival_ratio": survival},
            thresholds={"required_stress_multiple": cfg.required_stress_multiple,
                        "cost_frac_1x": c},
            reason_codes=[] if ok else [ValidationReason.FAILED_COST_STRESS],
        )

    def _stage_monte_carlo(self, sample: List[float]) -> StageResult:
        """Block bootstrap (serial dependence preserved) of the return path."""
        cfg = self.config
        n = len(sample)
        if n < 20:
            return StageResult("monte_carlo", FAIL,
                               metrics={"n": n},
                               reason_codes=[ValidationReason.FAILED_MONTE_CARLO])
        rng = random.Random(cfg.seed)
        block = max(2, min(cfg.mc_block_size, n // 4))
        endings, max_dds = [], []
        for _ in range(cfg.mc_sims):
            path: List[float] = []
            while len(path) < n:
                start = rng.randrange(0, n - block + 1)
                path.extend(sample[start:start + block])
            path = path[:n]
            cum = peak = dd = 0.0
            for r in path:
                cum += r
                peak = max(peak, cum)
                dd = max(dd, peak - cum)
            endings.append(cum)
            max_dds.append(dd)
        endings.sort()
        max_dds.sort()
        p_profitable = sum(1 for e in endings if e > 0) / len(endings)
        p5_return = endings[int(0.05 * len(endings))]
        p95_dd = max_dds[int(0.95 * len(max_dds)) - 1]
        expected_total = sum(sample)
        p_severe_dd = sum(1 for d in max_dds
                          if d > max(abs(expected_total), 1e-9) * (
                              1 + cfg.severe_drawdown_frac * 10)) / len(max_dds)
        codes = []
        if p_profitable < cfg.min_p_profitable:
            codes.append(ValidationReason.FAILED_MONTE_CARLO)
        if p_severe_dd > cfg.max_p_severe_drawdown:
            codes.append(ValidationReason.TAIL_RISK_TOO_HIGH)
        ok = not codes
        return StageResult(
            "monte_carlo", PASS if ok else FAIL,
            metrics={"p_profitable": p_profitable,
                     "p5_ending_return": p5_return,
                     "median_ending_return": endings[len(endings) // 2],
                     "p95_ending_return": endings[int(0.95 * len(endings)) - 1],
                     "p95_max_drawdown": p95_dd,
                     "p_severe_drawdown": p_severe_dd,
                     "simulations": cfg.mc_sims, "block_size": block,
                     "method": "block_bootstrap"},
            thresholds={"min_p_profitable": cfg.min_p_profitable,
                        "max_p_severe_drawdown": cfg.max_p_severe_drawdown},
            reason_codes=codes,
        )

    def _stage_deflated_sharpe(self, sample: List[float],
                               family_trials: int) -> StageResult:
        from core.validation_stats import deflated_sharpe_ratio, sharpe_ratio
        cfg = self.config
        if len(sample) < 20:
            return StageResult("deflated_sharpe", FAIL,
                               reason_codes=[ValidationReason.FAILED_DEFLATED_SHARPE])
        sr = sharpe_ratio(sample)
        dsr = deflated_sharpe_ratio(sr, n_trials=max(family_trials, 1),
                                    n_obs=len(sample))
        ok = dsr >= cfg.min_deflated_sharpe_prob
        return StageResult(
            "deflated_sharpe", PASS if ok else FAIL,
            metrics={"sharpe": sr, "deflated_sharpe": dsr,
                     "n_trials_in_family": family_trials,
                     "pbo": None, "pbo_status": UNAVAILABLE},
            thresholds={"min_deflated_sharpe_prob": cfg.min_deflated_sharpe_prob},
            reason_codes=[] if ok else [ValidationReason.FAILED_DEFLATED_SHARPE],
        )

    def _stage_extreme_returns(self, h, asset_class: str) -> StageResult:
        from core.return_units import validate_return_units
        cfg = self.config
        codes = []
        if h.mean_return > cfg.extreme_mean_return:
            codes.append(ValidationReason.EXTREME_RETURN_REVIEW)
        if not validate_return_units(h.mean_return, "hypothesis_mean_return",
                                     asset_class, context=h.hypothesis_id):
            codes.append(ValidationReason.POSSIBLE_DATA_ERROR)
        sample = h.metadata.get("returns_sample") or []
        if any(abs(r) > 1.0 for r in sample):
            codes.append(ValidationReason.POSSIBLE_DATA_ERROR)
        ok = not codes
        return StageResult(
            "extreme_return_sanity", PASS if ok else FAIL,
            metrics={"mean_return": h.mean_return,
                     "max_abs_observation": max((abs(r) for r in sample), default=0)},
            thresholds={"extreme_mean_return": cfg.extreme_mean_return},
            reason_codes=codes,
        )

    def _stage_regime(self, h, regime_series) -> StageResult:
        """Per-regime expectancy. Positive in ≥1 regime with sample →
        REGIME_SPECIFIC pass with a recorded regime constraint."""
        sample = h.metadata.get("returns_sample") or []
        times = h.metadata.get("sample_times") or []
        if regime_series is None or not times or len(times) != len(sample):
            return StageResult("regime_stability", UNAVAILABLE,
                               metrics={"note": "no regime labels for sample"})
        by_regime: Dict[str, List[float]] = {}
        for t, r in zip(times, sample):
            label = regime_series.get(t)
            if label is None:
                continue
            by_regime.setdefault(str(label), []).append(r)
        regime_stats = {}
        valid, invalid = [], []
        for regime, rets in by_regime.items():
            if len(rets) < 5:
                continue
            mean = sum(rets) / len(rets)
            regime_stats[regime] = {"n": len(rets), "expectancy": mean}
            (valid if mean > 0 else invalid).append(regime)
        if not regime_stats:
            return StageResult("regime_stability", UNAVAILABLE,
                               metrics={"note": "insufficient per-regime samples"})
        ok = bool(valid)
        return StageResult(
            "regime_stability",
            PASS if ok else FAIL,
            metrics={"regime_stats": regime_stats, "valid_regimes": valid,
                     "invalid_regimes": invalid,
                     "regime_specific": ok and bool(invalid)},
            reason_codes=[] if ok else [ValidationReason.FAILED_REGIME_STABILITY],
        )

    def _stage_event_concentration(self, h) -> StageResult:
        # Event labels (earnings/macro/crisis) are not yet available in the
        # research data — honest UNAVAILABLE, never silently PASS.
        return StageResult("event_concentration", UNAVAILABLE,
                           metrics={"note": "no event labels in dataset"})

    def _stage_holdout(self, h, holdout_match, holdout_available: bool,
                       holdout_period: Optional[Tuple[str, str]] = None) -> StageResult:
        """Final holdout governed by HoldoutManager (enforced, spec §13-20):

        - access keyed by FAMILY + holdout period — a family gets exactly ONE
          look; sibling/retuned parameter versions cannot re-test the same
          holdout (retuning attack → HOLDOUT_ALREADY_CONSUMED)
        - same exact parameter version replays its RECORDED result
          (idempotent under crashes/retries)
        - registration happens before access
        """
        cfg = self.config
        if not holdout_available:
            return StageResult(
                "holdout", UNAVAILABLE,
                metrics={"policy": cfg.holdout_unavailable_policy,
                         "note": "insufficient untouched history — forward "
                                 "paper evidence substitutes (documented policy)"},
                reason_codes=[ValidationReason.HOLDOUT_UNAVAILABLE])

        fam_sig = family_signature(h)
        param_sig = parameter_signature(h)
        start, end = holdout_period or ("campaign_holdout", "campaign_holdout_end")

        hm = self._holdout_manager
        if hm is not None:
            hm.register(fam_sig, start, end, param_sig)
            ok_access, why = hm.can_access(fam_sig, start, end)
            if not ok_access:
                # Idempotent replay for the SAME exact parameter version
                for record in hm.status(fam_sig):
                    if (record.get("holdout_start") == start
                            and record.get("version_at_access") == param_sig
                            and record.get("result")):
                        prior = json.loads(record["result"])
                        prior["replayed"] = True
                        passed = prior.get("holdout_mean", -1) is not None \
                            and prior.get("holdout_mean", -1) > 0 \
                            and not prior.get("failed", False)
                        return StageResult(
                            "holdout", PASS if passed else FAIL, metrics=prior,
                            reason_codes=[] if passed else
                            [ValidationReason.FAILED_HOLDOUT])
                # Different parameter version — retuning attack blocked
                return StageResult(
                    "holdout", FAIL,
                    metrics={"access_denied": why,
                             "consumed_by_other_version": True},
                    reason_codes=[ValidationReason.HOLDOUT_ALREADY_CONSUMED])

        if holdout_match is None or holdout_match.sample_size < cfg.min_holdout_sample:
            result = StageResult(
                "holdout", FAIL,
                metrics={"note": "effect absent or sample too small on final holdout",
                         "holdout_sample": getattr(holdout_match, "sample_size", 0),
                         "failed": True},
                reason_codes=[ValidationReason.FAILED_HOLDOUT
                              if holdout_match is None else
                              ValidationReason.HOLDOUT_INSUFFICIENT_SAMPLE])
        else:
            retention = (holdout_match.mean_return / h.mean_return
                         if h.mean_return > 1e-12 else 0.0)
            ok = holdout_match.mean_return > 0
            result = StageResult(
                "holdout", PASS if ok else FAIL,
                metrics={"holdout_mean": holdout_match.mean_return,
                         "holdout_sample": holdout_match.sample_size,
                         "holdout_period": [start, end],
                         "effect_retention": retention,
                         "failed": not ok},
                reason_codes=[] if ok else [ValidationReason.FAILED_HOLDOUT])
        if hm is not None:
            hm.record_access(fam_sig, start, end, param_sig, result.metrics)
        return result

    # ── Conservative paper prior (spec §47) ───────────────────────────────────

    @staticmethod
    def _conservative_prior(stages: List[StageResult], oos_sample: List[float],
                            discovery_sample: List[float]) -> Dict[str, Any]:
        """Most conservative reasonable estimate from OOS / walk-forward /
        holdout — NEVER from in-sample discovery metrics."""
        from core.validation_stats import uncertainty_summary
        candidates = []
        for s in stages:
            if s.stage_name == "oos" and "oos_mean" in s.metrics:
                candidates.append(s.metrics["oos_mean"])
            if s.stage_name == "true_walk_forward" and "median_fold_EV" in s.metrics:
                candidates.append(s.metrics["median_fold_EV"])
            if s.stage_name == "holdout" and "holdout_mean" in s.metrics:
                candidates.append(s.metrics["holdout_mean"])
        if not candidates:
            return {}
        stats = uncertainty_summary(oos_sample or discovery_sample)
        return {
            "mean_net_return": min(candidates),   # most conservative estimate
            "standard_error_return": stats["std_error"],
            "median_net_return": stats["median"],
            "confidence_lower_return": stats["ci_lower"],
            "confidence_upper_return": stats["ci_upper"],
            "win_rate": stats["p_net_positive"],
            "trades": stats["n"],
            "effective_sample_size": stats["effective_sample_size"],
            "source": "min(OOS, walk_forward, holdout) — never in-sample",
        }


# ── Artifact persistence (spec §26) ───────────────────────────────────────────

_CREATE_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS alpha_validation_artifacts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    parameter_signature TEXT NOT NULL,
    family_signature    TEXT NOT NULL,
    alpha_id            TEXT,
    campaign_id         TEXT,
    decision            TEXT NOT NULL,
    validation_score    REAL,
    artifact            TEXT NOT NULL,
    created_at          TEXT NOT NULL
)
"""


class ValidationArtifactStore:
    def __init__(self, db_path: str = "data/trade_memory.sqlite") -> None:
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_ARTIFACTS)
            conn.commit()

    def save(self, result: AlphaValidationResult,
             alpha_id: Optional[str] = None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO alpha_validation_artifacts "
                "(parameter_signature, family_signature, alpha_id, campaign_id, "
                " decision, validation_score, artifact, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (result.parameter_signature, result.family_signature, alpha_id,
                 result.campaign_id, result.decision, result.validation_score,
                 json.dumps(result.to_dict(), default=str), _utcnow()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def for_alpha(self, alpha_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM alpha_validation_artifacts WHERE alpha_id=?",
                (alpha_id,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["artifact"] = json.loads(d["artifact"])
            out.append(d)
        return out
