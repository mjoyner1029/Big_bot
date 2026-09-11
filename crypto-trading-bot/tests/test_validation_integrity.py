"""Validation-integrity tests (spec §41-51): true walk-forward, leakage,
purging, holdout governance, CSCV PBO, and real-parameter robustness."""
import json
import uuid

import numpy as np
import pandas as pd
import pytest

from core.alpha_validation import (
    AlphaValidationPipeline,
    ValidationConfig,
    ValidationReason,
    dimension_scales,
    family_signature,
    parameter_distance,
    parameter_signature,
    parameter_vector,
)
from core.discovery_detectors import DiscoveredHypothesis
from core.holdout_manager import HoldoutManager
from core.validation_stats import probability_of_backtest_overfitting


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / f"i_{uuid.uuid4().hex}.sqlite")


def dates(n, start="2025-01-01"):
    return [str(t) for t in pd.date_range(start, periods=n, freq="1D")]


def mk_variant(returns, rvol=3.0, exit_minutes=5.0, symbol="SYN-USD",
               subfamily="rvol", times=None, holding_bars=1):
    returns = list(returns)
    n = len(returns)
    conds = [
        {"feature": "relative_volume_20d", "op": ">", "value": float(rvol)},
        {"feature": "time_of_day", "op": "<", "value": float(exit_minutes)},
    ]
    return DiscoveredHypothesis(
        family="MOMENTUM", subfamily=subfamily, symbol=symbol, direction="long",
        entry_conditions=conds, holding_bars=holding_bars,
        sample_size=n, effective_sample=float(n),
        mean_return=sum(returns) / n, p_value=0.001,
        description=f"rvol={rvol} exit={exit_minutes}",
        hypothesis_id=f"MOMENTUM:{subfamily}:{symbol}:long:rvol{rvol}",
        metadata={"returns_sample": returns,
                  "sample_times": times or dates(n)},
    )


# ── §41 True walk-forward cases ───────────────────────────────────────────────


class TestTrueWalkForward:
    def test_case_a_true_stable_edge_passes(self, db):
        """Training discovers the correct parameter; future folds stay
        profitable."""
        rng = np.random.default_rng(1)
        n = 120
        best = mk_variant(rng.normal(0.02, 0.004, n), rvol=3.0)
        weak1 = mk_variant(rng.normal(0.005, 0.004, n), rvol=2.0)
        weak2 = mk_variant(rng.normal(0.004, 0.004, n), rvol=4.0)
        p = AlphaValidationPipeline(ValidationConfig(), db)
        stage = p._stage_true_walk_forward(best, [weak1, weak2])
        assert stage.status == "PASS", stage.metrics
        # per-fold audit trail exists
        for fold in stage.metrics["folds"]:
            assert fold["frozen_parameter_signature"]
            assert fold["train_end"] < fold["test_start"]   # chronology

    def test_case_b_full_sample_overfit_rejected(self, db):
        """A variant that only looks best when selected using ALL history:
        it dominates early, collapses later. Fold-wise selection keeps picking
        it from training data and it fails forward."""
        rng = np.random.default_rng(2)
        n = 120
        # Overfit variant: stellar first half, negative second half
        overfit = list(rng.normal(0.03, 0.004, n // 2)) + \
            list(rng.normal(-0.01, 0.004, n // 2))
        steady_weak = list(rng.normal(0.0005, 0.004, n))
        cand = mk_variant(overfit, rvol=3.0)
        sib = mk_variant(steady_weak, rvol=2.0)
        p = AlphaValidationPipeline(ValidationConfig(), db)
        stage = p._stage_true_walk_forward(cand, [sib])
        assert stage.status == "FAIL"
        assert ValidationReason.FAILED_WALK_FORWARD in stage.reason_codes \
            or ValidationReason.INSUFFICIENT_WF_FOLDS in stage.reason_codes

    def test_case_c_temporal_stability_passes_but_selection_leaks(self, db):
        """CRITICAL: the full-sample-chosen candidate's return stream looks
        chunk-stable, but honest fold-wise selection keeps choosing whichever
        variant led during training — and that choice fails forward."""
        rng = np.random.default_rng(3)
        n = 120
        # Two anti-phased variants: A leads in odd thirds, B in even thirds.
        a, b = [], []
        for i in range(n):
            phase = (i // (n // 3)) % 2
            a.append(float(rng.normal(0.02 if phase == 0 else -0.015, 0.003)))
            b.append(float(rng.normal(-0.015 if phase == 0 else 0.02, 0.003)))
        # The "blend" candidate (what full-sample selection would produce):
        # mildly positive everywhere → chunk-stable.
        blend = [(x + y) / 2 + 0.003 for x, y in zip(a, b)]
        cand = mk_variant(blend, rvol=3.0)
        var_a = mk_variant(a, rvol=2.0)
        var_b = mk_variant(b, rvol=4.0)

        p = AlphaValidationPipeline(ValidationConfig(), db)
        ts = p._stage_temporal_stability(blend, 1)
        assert ts.status == "PASS"          # chunk stability is fooled...

        wf = p._stage_true_walk_forward(cand, [var_a, var_b])
        # ...but fold-wise training keeps selecting the anti-phased leader
        # (A or B outperform the blend in-sample), which then flips sign
        # in the forward window.
        assert wf.status == "FAIL", wf.metrics

    def test_leakage_test_data_never_enters_training(self, db):
        """Every fold's training window ends before its test window begins,
        with a purge gap of the holding period."""
        rng = np.random.default_rng(4)
        cand = mk_variant(rng.normal(0.02, 0.004, 100), holding_bars=5)
        sib = mk_variant(rng.normal(0.01, 0.004, 100), rvol=2.0, holding_bars=5)
        p = AlphaValidationPipeline(ValidationConfig(), db)
        stage = p._stage_true_walk_forward(cand, [sib])
        for fold in stage.metrics.get("folds", []):
            assert fold["train_end"] < fold["test_start"]
            assert fold["purge_window"] == 5 and fold["embargo_window"] == 5

    def test_purging_removes_boundary_overlap(self, db):
        """5-day holding: observations within the purge gap appear in neither
        training nor test sets of a fold."""
        n = 100
        times = dates(n)
        cand = mk_variant([0.01] * n, holding_bars=5, times=times)
        p = AlphaValidationPipeline(ValidationConfig(), db)
        stage = p._stage_true_walk_forward(cand, [])
        for fold in stage.metrics.get("folds", []):
            # gap between train_end and test_start spans ≥ purge observations
            train_end_idx = times.index(fold["train_end"])
            test_start_idx = times.index(fold["test_start"])
            assert test_start_idx - train_end_idx > fold["purge_window"]


# ── §44-46 Holdout governance ─────────────────────────────────────────────────


def strong_variant(seed=1, rvol=3.0):
    rng = np.random.default_rng(seed)
    return mk_variant(rng.normal(0.02, 0.004, 80), rvol=rvol)


def holdout_match_for(h, seed=9):
    rng = np.random.default_rng(seed)
    return mk_variant(rng.normal(0.018, 0.004, 30), rvol=3.0,
                      times=dates(30, "2025-06-01"))


class TestHoldoutGovernance:
    def _pipeline(self, db):
        return AlphaValidationPipeline(ValidationConfig(), db,
                                       holdout_manager=HoldoutManager(db))

    def test_single_access_and_idempotent_replay(self, db):
        p = self._pipeline(db)
        h = strong_variant()
        period = ("2025-06-01", "2025-06-30")
        first = p._stage_holdout(h, holdout_match_for(h), True, period)
        assert first.status == "PASS"
        # SAME parameter version retried (crash/retry) → recorded result replayed
        replay = p._stage_holdout(h, holdout_match_for(h), True, period)
        assert replay.status == "PASS"
        assert replay.metrics.get("replayed") is True

    def test_case_44_second_version_denied(self, db):
        p = self._pipeline(db)
        v1 = strong_variant(rvol=3.0)
        period = ("2025-06-01", "2025-06-30")
        assert p._stage_holdout(v1, holdout_match_for(v1), True, period).status == "PASS"
        # different parameter version, same family, same holdout → blocked
        v2 = strong_variant(rvol=3.5)
        assert family_signature(v1) == family_signature(v2)
        assert parameter_signature(v1) != parameter_signature(v2)
        denied = p._stage_holdout(v2, holdout_match_for(v2), True, period)
        assert denied.status == "FAIL"
        assert ValidationReason.HOLDOUT_ALREADY_CONSUMED in denied.reason_codes

    def test_case_45_mutation_invalidates(self, db):
        """v1 consumes holdout; mutated v2 cannot inherit v1's pass and cannot
        re-access — it must produce new forward evidence."""
        p = self._pipeline(db)
        v1 = strong_variant(rvol=3.0)
        period = ("2025-06-01", "2025-06-30")
        p._stage_holdout(v1, holdout_match_for(v1), True, period)
        v2 = strong_variant(rvol=3.1)   # mutation → new parameter version
        result = p._stage_holdout(v2, holdout_match_for(v2), True, period)
        assert result.status == "FAIL"
        assert result.metrics.get("consumed_by_other_version")

    def test_case_46_retuning_attack_blocked(self, db):
        """v1 FAILS holdout → tweak parameter → same holdout reused → blocked."""
        p = self._pipeline(db)
        v1 = strong_variant(rvol=3.0)
        period = ("2025-06-01", "2025-06-30")
        failed = p._stage_holdout(v1, None, True, period)   # effect absent
        assert failed.status == "FAIL"
        # attacker tweaks the threshold and tries the SAME holdout again
        for tweak in (2.9, 3.1, 3.2, 2.8):
            vx = strong_variant(rvol=tweak)
            blocked = p._stage_holdout(vx, holdout_match_for(vx), True, period)
            assert blocked.status == "FAIL"
            assert ValidationReason.HOLDOUT_ALREADY_CONSUMED in blocked.reason_codes

    def test_insufficient_sample_reason(self, db):
        p = self._pipeline(db)
        h = strong_variant()
        tiny = mk_variant([0.01] * 3, times=dates(3, "2025-06-01"))
        result = p._stage_holdout(h, tiny, True, ("2025-06-01", "2025-06-30"))
        assert result.status == "FAIL"
        assert ValidationReason.HOLDOUT_INSUFFICIENT_SAMPLE in result.reason_codes

    def test_campaign_wires_real_holdout_manager(self, db):
        from core.research_campaign import ResearchCampaignRunner
        runner = ResearchCampaignRunner(db_path=db)
        assert runner.holdout_manager is not None
        assert runner.pipeline._holdout_manager is runner.holdout_manager


# ── §47-48 PBO (CSCV) ─────────────────────────────────────────────────────────


class TestPBO:
    def test_case_47_lucky_random_family_has_high_pbo(self):
        rng = np.random.default_rng(5)
        # 12 random configurations; any IS winner is pure luck
        matrix = [list(rng.normal(0.0, 0.01, 96)) for _ in range(12)]
        result = probability_of_backtest_overfitting(matrix)
        assert result["status"] == "OK"
        assert result["pbo"] > 0.4   # IS winner ~ random OOS rank

    def test_case_47_genuine_family_has_low_pbo(self):
        rng = np.random.default_rng(6)
        # 6 neighboring genuinely profitable configurations (shared real edge)
        base = rng.normal(0.01, 0.004, 96)
        matrix = [list(base + rng.normal(0, 0.001, 96)) for _ in range(6)]
        result = probability_of_backtest_overfitting(matrix)
        assert result["status"] == "OK"
        assert result["pbo"] < 0.4

    def test_case_48_insufficient_variants_unavailable(self):
        result = probability_of_backtest_overfitting([[0.01] * 100])
        assert result["status"] == "UNAVAILABLE"
        assert result["reason"] == "PBO_UNAVAILABLE_INSUFFICIENT_VARIANTS"
        assert "pbo" not in result   # no invented number

    def test_insufficient_observations_unavailable(self):
        matrix = [[0.01] * 10 for _ in range(6)]
        result = probability_of_backtest_overfitting(matrix)
        assert result["status"] == "UNAVAILABLE"
        assert "OBSERVATIONS" in result["reason"]

    def test_pbo_stage_wiring_and_dsr_separate(self, db):
        rng = np.random.default_rng(7)
        cand = mk_variant(rng.normal(0.02, 0.004, 96), rvol=3.0)
        sibs = [mk_variant(rng.normal(0.018, 0.004, 96), rvol=v)
                for v in (2.0, 2.5, 3.5, 4.0)]
        p = AlphaValidationPipeline(ValidationConfig(), db)
        result = p.validate(cand, oos_match=mk_variant(
            rng.normal(0.018, 0.004, 30), times=dates(30, "2025-07-01")),
            holdout_available=False, sibling_variants=sibs, family_trials=20)
        stage_names = {s.stage_name for s in result.stages}
        assert "pbo" in stage_names and "deflated_sharpe" in stage_names
        pbo_stage = next(s for s in result.stages if s.stage_name == "pbo")
        dsr_stage = next(s for s in result.stages if s.stage_name == "deflated_sharpe")
        assert pbo_stage.status == "PASS"
        assert "pbo" in pbo_stage.metrics and "deflated_sharpe" in dsr_stage.metrics


# ── §49-51 Real parameter values ──────────────────────────────────────────────


class TestRealParameterValues:
    def test_case_49_actual_values_and_distances(self):
        v2 = parameter_vector(mk_variant([0.01] * 10, rvol=2.0))
        v3 = parameter_vector(mk_variant([0.01] * 10, rvol=3.0))
        v5 = parameter_vector(mk_variant([0.01] * 10, rvol=5.0))
        assert v2["relative_volume_20d>"] == 2.0
        assert v3["relative_volume_20d>"] == 3.0
        assert v5["relative_volume_20d>"] == 5.0
        scales = dimension_scales([v2, v3, v5])
        assert parameter_distance(v2, v3, scales) < parameter_distance(v2, v5, scales)

    def test_case_50_multidimensional_normalized_distance(self):
        a = parameter_vector(mk_variant([0.01] * 10, rvol=3.0, exit_minutes=5))
        b = parameter_vector(mk_variant([0.01] * 10, rvol=3.5, exit_minutes=5))
        c = parameter_vector(mk_variant([0.01] * 10, rvol=5.0, exit_minutes=60))
        scales = dimension_scales([a, b, c])
        # exit-minutes range (55) must not drown the RVOL range (2.0)
        assert parameter_distance(a, b, scales) < parameter_distance(a, c, scales)

    def test_categorical_dimensions_hamming(self):
        thu = DiscoveredHypothesis(
            family="TEMPORAL", subfamily="dow", symbol="X", direction="long",
            entry_conditions=[{"feature": "day_of_week", "op": "==", "value": "THURSDAY"}],
            holding_bars=1, sample_size=10, effective_sample=10.0,
            mean_return=0.01, p_value=0.01, hypothesis_id="t")
        mon = DiscoveredHypothesis(
            family="TEMPORAL", subfamily="dow", symbol="X", direction="long",
            entry_conditions=[{"feature": "day_of_week", "op": "==", "value": "MONDAY"}],
            holding_bars=1, sample_size=10, effective_sample=10.0,
            mean_return=0.01, p_value=0.01, hypothesis_id="m")
        va, vb = parameter_vector(thu), parameter_vector(mon)
        assert parameter_distance(va, va) == 0.0
        assert parameter_distance(va, vb) == 1.0   # Hamming, not numeric

    def test_robustness_report_shows_actual_vectors(self, db):
        rng = np.random.default_rng(8)
        cand = mk_variant(rng.normal(0.018, 0.003, 60), rvol=3.0)
        sibs = [mk_variant(rng.normal(ev, 0.003, 60), rvol=v)
                for v, ev in ((2.0, 0.010), (2.5, 0.015), (3.5, 0.017), (4.0, 0.012))]
        p = AlphaValidationPipeline(ValidationConfig(), db)
        stage = p._stage_robustness(cand, sibs)
        assert stage.status == "PASS"
        neighborhood = stage.metrics["neighborhood"]
        tested_rvols = {n["vector"]["relative_volume_20d>"] for n in neighborhood}
        assert tested_rvols == {2.0, 2.5, 3.5, 4.0}   # real values, not 0,1,2,3
        # nearest neighbors (2.5/3.5) sorted before far ones (2.0/4.0)
        assert neighborhood[0]["vector"]["relative_volume_20d>"] in (2.5, 3.5)

    def test_parameter_cliff_rejected_plateau_passes(self, db):
        p = AlphaValidationPipeline(ValidationConfig(), db)
        cand = mk_variant([0.0042] * 60, rvol=3.0)
        cliff_sibs = [mk_variant([ev] * 60, rvol=v)
                      for v, ev in ((2.9, -0.0003), (3.1, -0.0004), (2.0, -0.001))]
        cliff = p._stage_robustness(cand, cliff_sibs)
        assert cliff.status == "FAIL"
        assert ValidationReason.PARAMETER_CLIFF in cliff.reason_codes

        plateau_sibs = [mk_variant([ev] * 60, rvol=v)
                        for v, ev in ((2.0, 0.0010), (2.5, 0.0016), (3.5, 0.0017),
                                      (4.0, 0.0012))]
        plateau = p._stage_robustness(mk_variant([0.0018] * 60, rvol=3.0),
                                      plateau_sibs)
        assert plateau.status == "PASS"

    def test_case_51_active_neighbor_backtesting(self, db):
        """Too few nearby variants → pipeline requests targeted extra runs
        (via the hook, on discovery data — holdout is never touched)."""
        calls = []

        def neighbor_fn(h):
            calls.append(parameter_vector(h))
            return [mk_variant([0.0015] * 60, rvol=2.5),
                    mk_variant([0.0016] * 60, rvol=3.5)]

        p = AlphaValidationPipeline(ValidationConfig(), db,
                                    neighbor_backtest_fn=neighbor_fn)
        cand = mk_variant([0.0018] * 60, rvol=3.0)
        distant_only = [mk_variant([0.001] * 60, rvol=10.0)]
        stage = p._stage_robustness(cand, distant_only)
        assert calls, "active neighbor testing not invoked"
        assert stage.status == "PASS"
        tested = {n["vector"]["relative_volume_20d>"]
                  for n in stage.metrics["neighborhood"]}
        assert {2.5, 3.5} <= tested
