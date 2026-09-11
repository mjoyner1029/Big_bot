"""End-to-end tests for the alpha-to-execution pipeline (spec §57-58).

Covers required cases:
  A valid live alpha → candidate reaches allocation
  B paper alpha → paper signal only, never live
  C paused alpha → no candidate
  D regime mismatch → rejected
  E negative EV → NO TRADE
  F high win-probability / negative EV → reject
  G low win-probability / positive EV → can pass
  H high execution cost → reject
  I correlated opportunities → allocation reduced
  J conflicting alphas → deterministic resolution / abstain
  K missing credentials during import → module imports
  L live mode without credentials → clean startup error
  M all-NaN feature → no candidate (abstain)
  N stale market data → data-quality rejection
  O Claude says buy but alpha invalid → NO TRADE
"""
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd
import pytest

from core.alpha_conditions import (
    ConditionValidationError,
    evaluate_conditions,
    validate_condition,
    validate_conditions,
)
from core.alpha_library import AlphaLibrary, AlphaState
from core.alpha_signal_engine import AlphaSignalEngine, ReasonCode
from core.data_quality import DataQualityMonitor
from core.ev_model import EconomicEVModel
from core.feature_registry import FeatureRegistry
from core.portfolio_allocator import AllocatorConfig, PortfolioAllocator


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / f"pipe_{uuid.uuid4().hex}.sqlite")


def make_market_df(n=250, seed=7, trending=True, start="2026-06-01"):
    rng = np.random.default_rng(seed)
    drift = 0.002 if trending else 0.0
    rets = rng.normal(drift, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range(start, periods=n, freq="1D")
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.uniform(9e5, 1.1e6, n),
    }, index=idx)


def register_alpha(lib, alpha_id, states, **fields):
    defaults = dict(
        strategy_id="momentum", market="crypto", asset_class="crypto",
        direction="long", universe=["TEST-USD"], alpha_version="1",
        edge_health_score=0.8,
    )
    defaults.update(fields)
    lib.register(alpha_id, alpha_id, **defaults)
    for s in states:
        assert lib.transition(alpha_id, s, reason="test"), f"transition to {s} failed"


LIVE_PATH = [AlphaState.VALIDATING, AlphaState.PAPER, AlphaState.LIVE_ELIGIBLE]
PAPER_PATH = [AlphaState.VALIDATING, AlphaState.PAPER]


def engine_for(db):
    return AlphaSignalEngine(AlphaLibrary(db), FeatureRegistry())


# ── Condition DSL safety ──────────────────────────────────────────────────────


class TestConditionDSL:
    def test_valid_conditions(self):
        conds = validate_conditions([
            {"feature": "day_of_week", "op": "==", "value": "THURSDAY"},
            {"feature": "relative_volume_20d", "op": ">", "value": 3},
            {"feature": "market_regime", "op": "IN", "value": ["BULL", "HIGH_VOL"]},
            {"feature": "gap_pct", "op": "BETWEEN", "value": [-0.08, -0.03]},
        ])
        assert len(conds) == 4

    def test_unknown_feature_rejected(self):
        with pytest.raises(ConditionValidationError, match="Unknown feature"):
            validate_condition({"feature": "__import__('os')", "op": ">", "value": 1})

    def test_unknown_operator_rejected(self):
        with pytest.raises(ConditionValidationError, match="Unknown operator"):
            validate_condition({"feature": "price", "op": "EXEC", "value": 1})

    def test_no_arbitrary_code_paths(self):
        # values are data, never evaluated
        with pytest.raises(ConditionValidationError):
            validate_condition({"feature": "price", "op": ">", "value": {"code": "x"}})

    def test_missing_feature_fails_condition(self):
        conds = validate_conditions([{"feature": "funding_rate", "op": ">", "value": 0}])
        ok, details = evaluate_conditions(conds, {"funding_rate": None})
        assert not ok and "unavailable" in details[0]


# ── Case A: valid live alpha ──────────────────────────────────────────────────


class TestCaseAValidLiveAlpha:
    def test_live_alpha_generates_live_candidate(self, db):
        lib = AlphaLibrary(db)
        register_alpha(lib, "live_alpha", LIVE_PATH)
        engine = AlphaSignalEngine(lib)
        cands = engine.generate_candidates({"TEST-USD": make_market_df()},
                                           regime="bullish", live_mode=True)
        assert len(cands) == 1
        assert cands[0].execution_mode == "live"
        assert cands[0].alpha_id == "live_alpha"

    def test_candidate_reaches_allocation_with_positive_ev(self, db):
        lib = AlphaLibrary(db)
        register_alpha(lib, "live_alpha", LIVE_PATH)
        engine = AlphaSignalEngine(lib)
        cands = engine.generate_candidates({"TEST-USD": make_market_df()},
                                           regime="bullish")
        c = cands[0]
        # positive, confident EV (as the EV model would attach)
        c.expected_net_return = 0.004
        c.ev_lower_bound = 0.002
        c.execution_feasibility = 90.0
        decisions = PortfolioAllocator(AllocatorConfig(capital=10_000)).allocate(cands)
        assert any(d.accepted and d.allocation_usd > 0 for d in decisions)


# ── Case B: paper alpha never live ────────────────────────────────────────────


class TestCaseBPaperAlpha:
    def test_paper_alpha_paper_candidate_only(self, db):
        lib = AlphaLibrary(db)
        register_alpha(lib, "paper_alpha", PAPER_PATH)
        engine = AlphaSignalEngine(lib)
        cands = engine.generate_candidates({"TEST-USD": make_market_df()},
                                           regime="bullish", live_mode=True)
        assert len(cands) == 1
        assert cands[0].execution_mode == "paper"  # NEVER live


# ── Case C: paused alpha ──────────────────────────────────────────────────────


class TestCaseCPausedAlpha:
    def test_paused_alpha_no_candidate(self, db):
        lib = AlphaLibrary(db)
        register_alpha(lib, "paused_alpha", PAPER_PATH + [AlphaState.PAUSED])
        engine = AlphaSignalEngine(lib)
        cands = engine.generate_candidates({"TEST-USD": make_market_df()})
        assert cands == []

    def test_retired_and_rejected_no_candidates(self, db):
        lib = AlphaLibrary(db)
        register_alpha(lib, "retired", [AlphaState.VALIDATING, AlphaState.RETIRED])
        register_alpha(lib, "rejected", [AlphaState.VALIDATING, AlphaState.REJECTED])
        register_alpha(lib, "discovered", [])
        engine = AlphaSignalEngine(lib)
        assert engine.generate_candidates({"TEST-USD": make_market_df()}) == []


# ── Case D: regime mismatch ───────────────────────────────────────────────────


class TestCaseDRegimeMismatch:
    def test_wrong_regime_rejected(self, db):
        lib = AlphaLibrary(db)
        register_alpha(lib, "bull_only", LIVE_PATH, valid_regimes=["bullish"])
        engine = AlphaSignalEngine(lib)
        cands = engine.generate_candidates({"TEST-USD": make_market_df()},
                                           regime="bearish")
        assert cands == []
        assert any(r.reason_code == ReasonCode.REGIME_MISMATCH
                   for r in engine.last_rejections)


# ── Case E/F/G: economic EV gating ────────────────────────────────────────────


class TestEconomicEV:
    def test_ev_formula_uses_magnitudes_not_probability_proxy(self, db):
        model = EconomicEVModel(db, min_samples=5)
        # 70% win rate, +1% wins, -5% losses
        returns = [0.01] * 70 + [-0.05] * 30
        est = model.estimate("x", expected_costs=0.0, returns=returns)
        expected = 0.7 * 0.01 + 0.3 * (-0.05)
        assert est.expected_net_return == pytest.approx(expected, abs=1e-9)
        # NOT the (p-0.5)*2 proxy
        assert est.expected_net_return != pytest.approx((0.7 - 0.5) * 2, abs=0.01)

    def test_case_f_high_probability_negative_ev_rejected(self, db):
        model = EconomicEVModel(db, min_samples=5)
        returns = [0.005] * 70 + [-0.04] * 30   # 70% wins, ruinous losses
        est = model.estimate("f", returns=returns)
        assert est.probability_positive > 0.65
        assert est.expected_net_return < 0
        cands = _candidate_with_ev(db, est.expected_net_return, est.ev_lower_bound)
        decisions = PortfolioAllocator().allocate(cands)
        assert not any(d.accepted for d in decisions)
        assert any(ReasonCode.EXPECTED_EV_TOO_LOW in d.reason_codes for d in decisions)

    def test_case_g_low_probability_positive_ev_passes(self, db):
        model = EconomicEVModel(db, min_samples=5)
        # 42% win rate, +3% wins vs -1% losses → strong positive EV.
        # Interleaved like a real trade stream (block ordering would fake
        # autocorrelation and crush the effective sample size).
        import random
        returns = ([0.03] * 42) + ([-0.01] * 58)
        random.Random(5).shuffle(returns)
        est = model.estimate("g", returns=returns)
        assert est.probability_positive < 0.5
        assert est.expected_net_return > 0
        assert est.ev_lower_bound > 0
        cands = _candidate_with_ev(db, est.expected_net_return, est.ev_lower_bound)
        decisions = PortfolioAllocator().allocate(cands)
        assert any(d.accepted for d in decisions)

    def test_case_e_negative_ev_no_trade(self, db):
        cands = _candidate_with_ev(db, -0.002, -0.004)
        decisions = PortfolioAllocator().allocate(cands)
        assert not any(d.accepted for d in decisions)

    def test_uncertain_ev_rejected_by_lower_bound(self, db):
        # +0.60% expected but lower bound negative → cash wins
        cands = _candidate_with_ev(db, 0.006, -0.003)
        decisions = PortfolioAllocator().allocate(cands)
        assert not any(d.accepted for d in decisions)
        assert any(ReasonCode.LOWER_BOUND_NEGATIVE in d.reason_codes for d in decisions)

    def test_case_h_execution_costs_flip_ev(self, db):
        model = EconomicEVModel(db, min_samples=5)
        returns = [0.002] * 60 + [-0.001] * 40   # small gross edge
        no_cost = model.estimate("h", expected_costs=0.0, returns=returns)
        with_cost = model.estimate("h", expected_costs=0.005, returns=returns)
        assert no_cost.expected_net_return > 0
        assert with_cost.expected_net_return < 0
        cands = _candidate_with_ev(db, with_cost.expected_net_return,
                                   with_cost.ev_lower_bound)
        assert not any(d.accepted for d in PortfolioAllocator().allocate(cands))

    def test_insufficient_evidence_abstains(self, db):
        model = EconomicEVModel(db, min_samples=8)
        assert model.estimate("tiny", returns=[0.01] * 3) is None


def _candidate_with_ev(db, ev, lcb, symbol="TEST-USD", direction="long",
                       alpha_id=None):
    lib = AlphaLibrary(db)
    aid = alpha_id or f"a_{uuid.uuid4().hex[:6]}"
    register_alpha(lib, aid, LIVE_PATH, universe=[symbol], direction=direction)
    engine = AlphaSignalEngine(lib)
    cands = engine.generate_candidates({symbol: make_market_df()}, regime="bullish")
    assert cands, "expected candidate"
    for c in cands:
        c.expected_net_return = ev
        c.ev_lower_bound = lcb
        c.execution_feasibility = 90.0
    return cands


# ── Case I: correlated opportunities ─────────────────────────────────────────


class TestCaseICorrelatedOpportunities:
    def test_correlated_candidates_get_reduced_allocation(self, db):
        lib = AlphaLibrary(db)
        register_alpha(lib, "alpha_a", LIVE_PATH, universe=["AAA-USD"])
        register_alpha(lib, "alpha_b", LIVE_PATH, universe=["BBB-USD"])
        engine = AlphaSignalEngine(lib)
        cands = engine.generate_candidates(
            {"AAA-USD": make_market_df(seed=1), "BBB-USD": make_market_df(seed=2)},
            regime="bullish")
        assert len(cands) == 2
        for c in cands:
            c.expected_net_return = 0.004
            c.ev_lower_bound = 0.002
            c.execution_feasibility = 90.0
        allocator = PortfolioAllocator(
            AllocatorConfig(capital=10_000),
            alpha_correlations={("alpha_a", "alpha_b"): 0.95,
                                ("alpha_b", "alpha_a"): 0.95})
        decisions = [d for d in allocator.allocate(cands) if d.accepted]
        assert len(decisions) == 2
        sizes = sorted(d.allocation_usd for d in decisions)
        assert sizes[0] == pytest.approx(sizes[1] * 0.5)  # not independent


# ── Case J: conflicting alphas ────────────────────────────────────────────────


class TestCaseJConflictingAlphas:
    def _conflicting(self, db, long_lcb, short_lcb):
        lib = AlphaLibrary(db)
        register_alpha(lib, "long_alpha", LIVE_PATH, direction="long")
        register_alpha(lib, "short_alpha", LIVE_PATH, direction="short")
        engine = AlphaSignalEngine(lib)
        cands = engine.generate_candidates({"TEST-USD": make_market_df()},
                                           regime="bullish")
        assert len(cands) == 2
        for c in cands:
            c.expected_net_return = 0.004
            c.execution_feasibility = 90.0
            c.ev_lower_bound = long_lcb if c.direction == "long" else short_lcb
        return PortfolioAllocator().allocate(cands)

    def test_clear_winner_selected(self, db):
        decisions = self._conflicting(db, long_lcb=0.006, short_lcb=0.001)
        accepted = [d for d in decisions if d.accepted]
        assert len(accepted) == 1 and accepted[0].candidate.direction == "long"
        assert any(ReasonCode.SIGNAL_CONFLICT in d.reason_codes
                   for d in decisions if not d.accepted)

    def test_ambiguous_conflict_abstains(self, db):
        decisions = self._conflicting(db, long_lcb=0.003, short_lcb=0.0029)
        assert not any(d.accepted for d in decisions)
        assert all(ReasonCode.SIGNAL_CONFLICT in d.reason_codes for d in decisions)


# ── Case K/L: credentials ─────────────────────────────────────────────────────


class TestCredentialHandling:
    def test_case_k_import_without_credentials(self):
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[1]
        code = (
            "import os\n"
            "for var in ('ANTHROPIC_API_KEY','ALPACA_API_KEY','ALPACA_API_SECRET'):\n"
            "    os.environ.pop(var, None)\n"
            "import ultimate_bot_v3_llm\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=str(repo_root), timeout=120,
        )
        assert "IMPORT_OK" in result.stdout, (
            f"import failed: rc={result.returncode}\n{result.stderr[-2000:]}")

    def test_case_l_live_mode_without_credentials_clean_error(self, monkeypatch):
        from config.capabilities import check_startup_requirements
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
        errors = check_startup_requirements("LIVE")
        assert errors and "LIVE" in errors[0]

    def test_paper_and_research_need_no_credentials(self, monkeypatch):
        from config.capabilities import check_startup_requirements
        for var in ("ANTHROPIC_API_KEY", "ALPACA_API_KEY", "ALPACA_API_SECRET"):
            monkeypatch.delenv(var, raising=False)
        assert check_startup_requirements("PAPER") == []
        assert check_startup_requirements("RESEARCH") == []


# ── Case M/N: data quality ────────────────────────────────────────────────────


class TestDataQualityCases:
    def test_case_m_all_nan_feature_no_candidate(self, db):
        lib = AlphaLibrary(db)
        import json
        register_alpha(lib, "needs_rvol", LIVE_PATH,
                       entry_conditions=json.dumps(
                           [{"feature": "relative_volume_20d", "op": ">", "value": 2}]))
        engine = AlphaSignalEngine(lib)
        df = make_market_df()
        df["volume"] = np.nan   # feature becomes unavailable
        cands = engine.generate_candidates({"TEST-USD": df}, regime="bullish")
        assert cands == []

    def test_case_n_stale_data_rejected(self, db):
        lib = AlphaLibrary(db)
        register_alpha(lib, "any_alpha", LIVE_PATH)
        engine = AlphaSignalEngine(lib, data_quality_monitor=DataQualityMonitor())
        df = make_market_df()
        # Stale: identical close for the last 10 bars
        df.iloc[-10:, df.columns.get_loc("close")] = float(df["close"].iloc[-11])
        cands = engine.generate_candidates({"TEST-USD": df}, regime="bullish")
        assert cands == []
        assert any(r.reason_code == ReasonCode.DATA_QUALITY_FAILURE
                   for r in engine.last_rejections)


# ── Case O: Claude cannot create candidates ───────────────────────────────────


class TestCaseOClaudeCannotOverride:
    def test_no_eligible_alpha_means_no_trade_regardless_of_llm(self, db):
        """LLM opinions cannot create candidates: with no eligible alpha,
        the pipeline emits nothing no matter what any LLM says."""
        lib = AlphaLibrary(db)
        register_alpha(lib, "research_only", [AlphaState.VALIDATING])
        engine = AlphaSignalEngine(lib)
        cands = engine.generate_candidates({"TEST-USD": make_market_df()},
                                           regime="bullish")
        assert cands == []  # Claude has no code path to inject candidates


# ── Conditions actually gate matching (Thursday/RVOL example) ────────────────


class TestCurrentMarketMatching:
    def test_rvol_condition_gates_candidates_per_symbol(self, db):
        import json
        lib = AlphaLibrary(db)
        register_alpha(lib, "rvol_alpha", LIVE_PATH,
                       universe=["HOT-USD", "COLD-USD"],
                       entry_conditions=json.dumps(
                           [{"feature": "relative_volume_20d", "op": ">", "value": 3}]))
        hot = make_market_df(seed=3)
        hot.iloc[-1, hot.columns.get_loc("volume")] = 5e6    # RVOL ~5x
        cold = make_market_df(seed=4)                        # RVOL ~1x
        engine = AlphaSignalEngine(lib)
        cands = engine.generate_candidates({"HOT-USD": hot, "COLD-USD": cold},
                                           regime="bullish")
        symbols = {c.symbol for c in cands}
        assert symbols == {"HOT-USD"}
        assert any(r.reason_code == ReasonCode.CONDITIONS_NOT_MET
                   and r.symbol == "COLD-USD" for r in engine.last_rejections)

    def test_one_alpha_multiple_candidates(self, db):
        lib = AlphaLibrary(db)
        register_alpha(lib, "broad_alpha", LIVE_PATH, universe=["A-USD", "B-USD"])
        engine = AlphaSignalEngine(lib)
        cands = engine.generate_candidates(
            {"A-USD": make_market_df(seed=5), "B-USD": make_market_df(seed=6)},
            regime="bullish")
        assert len(cands) == 2

    def test_same_symbol_multiple_opportunities(self, db):
        lib = AlphaLibrary(db)
        register_alpha(lib, "overnight_alpha", LIVE_PATH)
        register_alpha(lib, "momentum_alpha", LIVE_PATH)
        engine = AlphaSignalEngine(lib)
        cands = engine.generate_candidates({"TEST-USD": make_market_df()},
                                           regime="bullish")
        assert len(cands) == 2
        assert len({c.alpha_id for c in cands}) == 2


# ── Discovery detectors ───────────────────────────────────────────────────────


class TestDiscoveryDetectors:
    def test_planted_weekday_effect_discovered(self):
        from core.discovery_detectors import TemporalAnomalyDetector
        rng = np.random.default_rng(11)
        n = 400
        idx = pd.date_range("2025-01-01", periods=n, freq="1D")
        rets = rng.normal(0, 0.004, n)
        thursdays = idx.strftime("%A").str.upper() == "THURSDAY"
        rets[thursdays] += 0.02   # strong planted Thursday effect
        close = 100 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({"open": close, "high": close * 1.01,
                           "low": close * 0.99, "close": close,
                           "volume": np.full(n, 1e6)}, index=idx)
        hyps = TemporalAnomalyDetector().scan({"SYN-USD": df})
        thursday_hyps = [h for h in hyps if any(
            c.get("value") == "THURSDAY" for c in h.entry_conditions)]
        assert thursday_hyps, "planted Thursday effect not discovered"
        assert all(h.p_value < 0.05 for h in thursday_hyps if h.direction == "long")

    def test_random_data_survivors_are_controlled_by_fdr(self):
        from core.discovery_detectors import run_discovery_scan
        rng = np.random.default_rng(13)
        data = {}
        for i in range(3):
            n = 300
            close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
            idx = pd.date_range("2025-01-01", periods=n, freq="1D")
            data[f"RW{i}-USD"] = pd.DataFrame(
                {"open": close, "high": close * 1.01, "low": close * 0.99,
                 "close": close, "volume": rng.uniform(9e5, 1.1e6, n)}, index=idx)
        report = run_discovery_scan(data)
        # On pure noise, FDR must keep discoveries near zero
        assert len(report["fdr_survivors"]) <= 2
        assert report["hypotheses_tested"] > 0
        assert report["detectors_skipped_missing_data"]  # honest about missing data

    def test_detectors_requiring_data_do_not_fabricate(self):
        from core.discovery_detectors import CryptoFundingDetector, EarningsEffectDetector
        assert EarningsEffectDetector().scan({}) == []
        assert CryptoFundingDetector().scan({}) == []
        assert EarningsEffectDetector.requires_data


# ── Attribution + learning loop ───────────────────────────────────────────────


class TestAttributionAndLearning:
    def test_trade_fully_attributed(self, db):
        from core.trade_attribution import TradeAttributionStore
        store = TradeAttributionStore(db)
        store.record("alpha_x", alpha_version="2", candidate_id="c1",
                     signal_id="s1", meta_model_version="1.0.0",
                     feature_version="1.0.0", execution_model_version="cost_v2")
        rows = store.for_alpha("alpha_x")
        assert len(rows) == 1
        r = rows[0]
        assert r["alpha_version"] == "2" and r["meta_model_version"] == "1.0.0"
        assert r["feature_version"] == "1.0.0"

    def test_retraining_is_threshold_gated(self, db):
        from core.trade_attribution import LearningLoop
        loop = LearningLoop(db, retrain_min_new_samples=5)
        for i in range(4):
            decision = loop.on_trade_closed("alpha_x", 0.01)
            assert not decision["retrain"]
        decision = loop.on_trade_closed("alpha_x", 0.01)
        assert decision["retrain"]  # threshold reached
        loop.mark_retrained()
        assert not loop.on_trade_closed("alpha_x", 0.01)["retrain"]

    def test_drift_triggers_retrain_flag(self, db):
        from core.trade_attribution import LearningLoop
        loop = LearningLoop(db, retrain_min_new_samples=100)
        decision = loop.on_trade_closed("alpha_x", -0.01, drift_detected=True)
        assert decision["retrain"] and "drift_detected" in decision["reasons"]


# ── Feature registry safety ───────────────────────────────────────────────────


class TestFeatureRegistry:
    def test_unknown_feature_rejected(self):
        reg = FeatureRegistry()
        with pytest.raises(ValueError, match="Unknown feature"):
            reg.compute("X", make_market_df(), ["not_a_feature"])

    def test_features_computed_from_decision_bar_only(self):
        reg = FeatureRegistry()
        df = make_market_df(seed=9)
        upto_t = reg.compute("X", df.iloc[:100], ["price", "returns_5d", "rsi"])
        # Recomputing the same window later gives identical values —
        # features never depend on anything beyond the provided window
        again = FeatureRegistry().compute("X", df.iloc[:100], ["price", "returns_5d", "rsi"])
        assert upto_t == again

    def test_context_features_never_fabricated(self):
        reg = FeatureRegistry()
        out = reg.compute("X", make_market_df(), ["funding_rate", "market_regime"],
                          context={"market_regime": "bullish"})
        assert out["market_regime"] == "bullish"
        assert out["funding_rate"] is None
