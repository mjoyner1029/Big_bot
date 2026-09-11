"""Full-validation funnel tests (spec §52-56): cases A-K, parameter
signatures, campaign batching with global FDR, checkpoint/resume, and
promotion safety."""
import json
import sqlite3
import uuid

import numpy as np
import pandas as pd
import pytest

from core.alpha_library import AlphaLibrary, AlphaState
from core.alpha_validation import (
    AlphaValidationPipeline,
    ValidationConfig,
    ValidationReason,
    family_signature,
    parameter_signature,
)
from core.discovery_detectors import DiscoveredHypothesis
from core.research_campaign import CampaignConfig, ResearchCampaignRunner


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / f"v_{uuid.uuid4().hex}.sqlite")


def mk_hyp(returns, symbol="SYN-USD", direction="long", family="TEMPORAL",
           subfamily="day_of_week", conditions=None, holding_bars=1,
           times=None, mean=None, p=0.001):
    returns = list(returns)
    n = len(returns)
    if times is None:
        times = [str(ts) for ts in
                 pd.date_range("2025-01-01", periods=n, freq="1D")]
    return DiscoveredHypothesis(
        family=family, subfamily=subfamily, symbol=symbol, direction=direction,
        entry_conditions=conditions or [
            {"feature": "day_of_week", "op": "==", "value": "THURSDAY"}],
        holding_bars=holding_bars, sample_size=n, effective_sample=float(n),
        mean_return=(mean if mean is not None else sum(returns) / n),
        p_value=p,
        description="synthetic", hypothesis_id=f"{family}:{subfamily}:{symbol}:{direction}:",
        metadata={"returns_sample": returns, "sample_times": times},
    )


def good_sample(n=80, mean=0.02, seed=1):
    rng = np.random.default_rng(seed)
    return list(rng.normal(mean, 0.004, n))


def pipeline(db, **overrides):
    cfg = ValidationConfig(**overrides) if overrides else ValidationConfig()
    return AlphaValidationPipeline(cfg, db)


GOOD_SIBLINGS = None  # filled per test


def good_siblings():
    return [mk_hyp(good_sample(60, 0.018, s), subfamily=f"var{s}",
                   conditions=[{"feature": "relative_volume_20d", "op": ">",
                                "value": 2.0 + s}])
            for s in range(3)]


NUMERIC_CONDS = [{"feature": "relative_volume_20d", "op": ">", "value": 3.0}]


class TestCaseATrueRobustEdge:
    def test_full_pass_and_paper_promotion(self, db):
        h = mk_hyp(good_sample(), conditions=NUMERIC_CONDS)
        oos = mk_hyp(good_sample(30, 0.018, 2), conditions=NUMERIC_CONDS)
        hold = mk_hyp(good_sample(20, 0.017, 3), conditions=NUMERIC_CONDS)
        result = pipeline(db).validate(
            h, oos_match=oos, holdout_match=hold, holdout_available=True,
            sibling_variants=good_siblings(), family_trials=10,
            campaign_id="caseA")
        assert result.passed, result.reason_codes
        assert result.validation_score > 80

        runner = ResearchCampaignRunner(db_path=db)
        alpha_id = runner.promote_validated("caseA", h, result)
        lib = AlphaLibrary(db)
        assert lib.state_of(alpha_id) == AlphaState.PAPER
        assert not lib.is_live_approved(alpha_id)
        # conservative prior uses min(OOS, WF, holdout), never discovery
        record = lib.get(alpha_id)
        assert record["oos_metrics"]["mean_net_return"] <= oos.mean_return
        # full audit artifact persisted
        artifacts = runner.artifacts.for_alpha(alpha_id)
        assert artifacts and artifacts[0]["decision"] == "PASS"
        stage_names = {s["stage_name"] for s in artifacts[0]["artifact"]["stages"]}
        assert {"oos", "temporal_stability", "true_walk_forward",
                "parameter_robustness", "cost_stress",
                "monte_carlo", "deflated_sharpe", "pbo", "holdout"} <= stage_names


class TestCaseCOOSFailure:
    def test_missing_oos_rejects(self, db):
        h = mk_hyp(good_sample(), conditions=NUMERIC_CONDS)
        result = pipeline(db).validate(h, oos_match=None, holdout_available=False,
                                       sibling_variants=good_siblings())
        assert not result.passed
        assert ValidationReason.FAILED_OOS in result.reason_codes

    def test_low_oos_retention_rejects(self, db):
        h = mk_hyp(good_sample(80, 0.02))
        weak_oos = mk_hyp(good_sample(30, 0.001, 2))   # 5% retention
        result = pipeline(db).validate(h, oos_match=weak_oos, holdout_available=False,
                                       sibling_variants=good_siblings())
        oos_stage = next(s for s in result.stages if s.stage_name == "oos")
        assert oos_stage.status == "FAIL"
        assert not result.passed


class TestCaseDOneFoldWonder:
    def test_single_good_fold_rejected(self, db):
        # First quarter strongly positive; the rest slightly negative
        sample = good_sample(25, 0.05, 1) + list(
            np.random.default_rng(2).normal(-0.002, 0.003, 75))
        h = mk_hyp(sample, conditions=NUMERIC_CONDS)
        oos = mk_hyp(good_sample(30, 0.01, 3), conditions=NUMERIC_CONDS)
        result = pipeline(db).validate(h, oos_match=oos, holdout_available=False,
                                       sibling_variants=good_siblings())
        wf = next(s for s in result.stages if s.stage_name == "temporal_stability")
        assert wf.status == "FAIL"
        assert ValidationReason.FAILED_TEMPORAL_STABILITY in result.reason_codes
        assert not result.passed


class TestCaseEParameterCliff:
    def test_cliff_rejected(self, db):
        h = mk_hyp(good_sample(), conditions=NUMERIC_CONDS)
        oos = mk_hyp(good_sample(30, 0.018, 2), conditions=NUMERIC_CONDS)
        cliff_siblings = [
            mk_hyp([- 0.002] * 60, subfamily=f"var{s}", mean=-0.002,
                   conditions=[{"feature": "relative_volume_20d", "op": ">",
                                "value": 2.0 + s}])
            for s in range(3)
        ]
        result = pipeline(db).validate(h, oos_match=oos, holdout_available=False,
                                       sibling_variants=cliff_siblings)
        assert not result.passed
        assert ValidationReason.PARAMETER_CLIFF in result.reason_codes

    def test_too_few_variants_fails_robustness(self, db):
        h = mk_hyp(good_sample(), conditions=NUMERIC_CONDS)
        oos = mk_hyp(good_sample(30, 0.018, 2), conditions=NUMERIC_CONDS)
        result = pipeline(db).validate(h, oos_match=oos, holdout_available=False,
                                       sibling_variants=[])
        assert ValidationReason.FAILED_PARAMETER_ROBUSTNESS in result.reason_codes

    def test_categorical_only_skips_robustness(self, db):
        h = mk_hyp(good_sample(),
                   conditions=[{"feature": "day_of_week", "op": "==",
                                "value": "THURSDAY"}])
        oos = mk_hyp(good_sample(30, 0.018, 2))
        result = pipeline(db).validate(h, oos_match=oos, holdout_available=False,
                                       sibling_variants=[])
        rob = next(s for s in result.stages
                   if s.stage_name == "parameter_robustness")
        assert rob.status == "SKIP"


class TestCaseFCostSensitive:
    def test_fails_at_configured_stress_multiple(self, db):
        # 25 bps gross: positive at 1x (20 bps) but negative at 1.5x
        h = mk_hyp(good_sample(80, 0.0025), conditions=NUMERIC_CONDS)
        oos = mk_hyp(good_sample(30, 0.0025, 2), conditions=NUMERIC_CONDS)
        result = pipeline(db).validate(h, oos_match=oos, holdout_available=False,
                                       sibling_variants=good_siblings())
        cost = next(s for s in result.stages if s.stage_name == "cost_stress")
        assert cost.metrics["net_1x"] > 0
        assert cost.metrics["net_1.5x"] < 0
        assert cost.status == "FAIL"
        assert not result.passed
        assert ValidationReason.FAILED_COST_STRESS in result.reason_codes


class TestCaseGOutlierEdge:
    def test_concentrated_profit_rejected(self, db):
        sample = [0.0002] * 79 + [0.5]   # one trade is nearly all profit
        h = mk_hyp(sample, conditions=NUMERIC_CONDS)
        oos = mk_hyp(good_sample(30, 0.01, 2), conditions=NUMERIC_CONDS)
        result = pipeline(db).validate(h, oos_match=oos, holdout_available=False,
                                       sibling_variants=good_siblings())
        assert ValidationReason.OUTLIER_DEPENDENT in result.reason_codes
        assert not result.passed


class TestCaseHRegimeSpecific:
    def test_regime_specific_pass_with_constraint(self, db):
        rng = np.random.default_rng(7)
        n = 80
        times = [str(ts) for ts in pd.date_range("2025-01-01", periods=n, freq="1D")]
        regimes = {t: ("HIGH_VOL" if i % 2 == 0 else "LOW_VOL")
                   for i, t in enumerate(times)}
        sample = [float(rng.normal(0.04, 0.004)) if i % 2 == 0
                  else float(rng.normal(0.001, 0.002)) for i in range(n)]
        h = mk_hyp(sample, conditions=NUMERIC_CONDS, times=times)
        oos = mk_hyp(good_sample(30, 0.018, 2), conditions=NUMERIC_CONDS)
        result = pipeline(db).validate(
            h, oos_match=oos, sibling_variants=good_siblings(),
            holdout_available=False, regime_series=regimes)
        regime = next(s for s in result.stages
                      if s.stage_name == "regime_stability")
        assert regime.status == "PASS"
        assert "HIGH_VOL" in regime.metrics["valid_regimes"]
        assert result.passed
        assert "HIGH_VOL" in result.valid_regimes


class TestCaseIMonteCarloTail:
    def test_unacceptable_tail_rejected(self, db):
        # Positive mean but bootstrap paths frequently unprofitable
        rng = np.random.default_rng(9)
        sample = list(rng.normal(0.005, 0.001, 60)) + [-0.035] * 8
        h = mk_hyp(sample, conditions=NUMERIC_CONDS)
        assert h.mean_return > 0   # gross mean is positive...
        oos = mk_hyp(good_sample(30, 0.018, 2), conditions=NUMERIC_CONDS)
        result = pipeline(db).validate(h, oos_match=oos, holdout_available=False,
                                       sibling_variants=good_siblings())
        mc = next(s for s in result.stages if s.stage_name == "monte_carlo")
        assert mc.status == "FAIL"
        assert (ValidationReason.FAILED_MONTE_CARLO in result.reason_codes
                or ValidationReason.TAIL_RISK_TOO_HIGH in result.reason_codes)
        assert not result.passed


class TestCaseJHoldoutFailure:
    def test_holdout_failure_rejects_despite_everything_else(self, db):
        h = mk_hyp(good_sample(), conditions=NUMERIC_CONDS)
        oos = mk_hyp(good_sample(30, 0.018, 2), conditions=NUMERIC_CONDS)
        result = pipeline(db).validate(
            h, oos_match=oos, holdout_match=None, holdout_available=True,
            sibling_variants=good_siblings())
        holdout = next(s for s in result.stages if s.stage_name == "holdout")
        assert holdout.status == "FAIL"
        assert ValidationReason.FAILED_HOLDOUT in result.reason_codes
        assert not result.passed


class TestCaseKHoldoutUnavailable:
    def test_unavailable_holdout_follows_substitution_policy(self, db):
        h = mk_hyp(good_sample(), conditions=NUMERIC_CONDS)
        oos = mk_hyp(good_sample(30, 0.018, 2), conditions=NUMERIC_CONDS)
        result = pipeline(db).validate(
            h, oos_match=oos, holdout_available=False,
            sibling_variants=good_siblings())
        holdout = next(s for s in result.stages if s.stage_name == "holdout")
        assert holdout.status == "UNAVAILABLE"
        assert holdout.metrics["policy"] == "paper_with_forward_substitution"
        # CONDITIONAL stage: UNAVAILABLE tolerated → candidate may still pass
        assert result.passed
        assert ValidationReason.HOLDOUT_UNAVAILABLE in result.reason_codes


class TestHardGatesStayHard:
    def test_high_score_cannot_override_mandatory_failure(self, db):
        # Everything passes except cost stress
        h = mk_hyp(good_sample(80, 0.0025), conditions=NUMERIC_CONDS)
        oos = mk_hyp(good_sample(30, 0.0025, 2), conditions=NUMERIC_CONDS)
        result = pipeline(db).validate(h, oos_match=oos, holdout_available=False,
                                       sibling_variants=good_siblings())
        assert result.validation_score > 60   # most stages passed...
        assert not result.passed              # ...but the hard gate holds

    def test_unavailable_mandatory_stage_is_not_pass(self, db):
        cfg = ValidationConfig()
        cfg.stage_policy["regime_stability"] = "MANDATORY"
        h = mk_hyp(good_sample(), conditions=NUMERIC_CONDS)
        oos = mk_hyp(good_sample(30, 0.018, 2), conditions=NUMERIC_CONDS)
        result = AlphaValidationPipeline(cfg, db).validate(
            h, oos_match=oos, sibling_variants=good_siblings(),
            holdout_available=False,
            regime_series=None)   # regime UNAVAILABLE
        assert not result.passed   # UNAVAILABLE ≠ PASS for mandatory stages


class TestParameterSignatures:
    def test_family_same_parameters_differ(self):
        h2 = mk_hyp(good_sample(), subfamily="rvol_2x",
                    conditions=[{"feature": "relative_volume_20d", "op": ">",
                                 "value": 2.0}])
        h5 = mk_hyp(good_sample(), subfamily="rvol_5x",
                    conditions=[{"feature": "relative_volume_20d", "op": ">",
                                 "value": 5.0}])
        assert family_signature(h2) == family_signature(h5)
        assert parameter_signature(h2) != parameter_signature(h5)

    def test_different_categorical_values_differ_in_parameters(self):
        thu = mk_hyp(good_sample())
        mon = mk_hyp(good_sample(),
                     conditions=[{"feature": "day_of_week", "op": "==",
                                  "value": "MONDAY"}])
        assert family_signature(thu) == family_signature(mon)
        assert parameter_signature(thu) != parameter_signature(mon)


class TestPromotionSafety:
    def test_failed_result_cannot_promote(self, db):
        h = mk_hyp(good_sample(), conditions=NUMERIC_CONDS)
        result = pipeline(db).validate(h, oos_match=None)   # fails OOS
        runner = ResearchCampaignRunner(db_path=db)
        with pytest.raises(PermissionError, match="PASSING"):
            runner.promote_validated("x", h, result)
        assert AlphaLibrary(db).eligible_alphas() == []


# ── Campaign batching / global FDR / resume (spec §54-55) ─────────────────────


def synthetic_universe(n_instruments=40, bars=300, planted=("SYM00",)):
    rng = np.random.default_rng(11)
    idx = pd.date_range("2025-01-01", periods=bars, freq="1D")
    weekdays = idx.strftime("%A").str.upper()
    data = {}
    for i in range(n_instruments):
        name = f"SYM{i:02d}"
        rets = rng.normal(0.0, 0.004, bars)
        if name in planted:
            rets[weekdays == "THURSDAY"] += 0.02
        close = 100 * np.exp(np.cumsum(rets))
        data[name] = pd.DataFrame(
            {"open": close, "high": close * 1.005, "low": close * 0.995,
             "close": close, "volume": rng.uniform(9e5, 1.1e6, bars)}, index=idx)
    return data


class TestCampaignBatching:
    def test_batches_share_global_context(self, db):
        data = synthetic_universe(40)
        cfg = CampaignConfig(instrument_batch_size=4)   # 10 batches
        runner = ResearchCampaignRunner(db_path=db, config=cfg)
        report = runner.run(data, campaign_id="batchtest")
        assert report["batches"] == 10
        # One campaign-global registry: hypotheses from all batches counted once
        assert report["hypotheses_tested"] > 40
        # Global FDR: with 40 mostly-noise instruments in ONE family context,
        # random effects must be crushed (per-batch FDR would let some through)
        assert report["fdr_survivors"] <= 6

    def test_resume_equals_uninterrupted(self, tmp_path):
        data = synthetic_universe(20, bars=250)
        cfg = CampaignConfig(instrument_batch_size=4)   # 5 batches

        # Uninterrupted reference
        db_a = str(tmp_path / "a.sqlite")
        ref = ResearchCampaignRunner(db_path=db_a, config=cfg).run(
            data, campaign_id="ref")

        # Interrupted after 2 batches
        db_b = str(tmp_path / "b.sqlite")
        calls = {"n": 0}
        from core.discovery_detectors import default_detectors as real_factory

        def exploding_factory(**kwargs):
            calls["n"] += 1
            if calls["n"] > 2:
                raise KeyboardInterrupt("simulated crash")
            return real_factory(**kwargs)

        runner_b = ResearchCampaignRunner(db_path=db_b, config=cfg,
                                          detectors_factory=exploding_factory)
        with pytest.raises(KeyboardInterrupt):
            runner_b.run(data, campaign_id="crashy")

        # Checkpoints for the completed batches exist
        with sqlite3.connect(db_b) as conn:
            n_ckpt = conn.execute(
                "SELECT COUNT(*) FROM campaign_checkpoints WHERE campaign_id='crashy'"
            ).fetchone()[0]
        assert n_ckpt == 2

        # Resume with a working factory — completed work not repeated
        runner_b2 = ResearchCampaignRunner(db_path=db_b, config=cfg)
        resumed = runner_b2.run(data, campaign_id="crashy", resume=True)
        assert resumed["hypotheses_tested"] == ref["hypotheses_tested"]
        assert resumed["hypotheses_emitted"] == ref["hypotheses_emitted"]
        assert resumed["fdr_survivors"] == ref["fdr_survivors"]
        assert resumed["new_alphas"] == ref["new_alphas"]


class TestFullFunnelCampaign:
    def test_planted_edge_survives_noise_rejected(self, db):
        data = synthetic_universe(8, bars=700, planted=("SYM00",))
        runner = ResearchCampaignRunner(db_path=db)
        report = runner.run(data, campaign_id="funnel")
        # true edge on SYM00 survives the FULL pipeline into PAPER
        assert any("SYM00" in a for a in report["new_alphas"]), report["new_alphas"]
        # every promoted alpha has a persisted validation artifact
        for alpha_id in report["new_alphas"]:
            artifacts = runner.artifacts.for_alpha(alpha_id)
            assert artifacts and artifacts[0]["decision"] == "PASS"
            assert AlphaLibrary(db).state_of(alpha_id) == AlphaState.PAPER
        # noise instruments produce no alphas
        assert not any(f"SYM0{i}" in a for i in range(1, 8)
                       for a in report["new_alphas"])
        assert report["stage_pass_counts"]   # funnel metrics recorded
