"""Full-wiring tests (spec §95-114): the autonomous research loop actually
invokes every discovery component, and the new profitability systems
(execution alpha, capacity, meta-alpha, survival, crowding, ruin gates) work.
"""
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core.execution_optimizer import (
    BorrowChecker,
    ExecutionLearner,
    ExecutionOptimizer,
    SlippageModel,
    VenueQuote,
    market_impact_bps,
    select_venue,
)
from core.meta_alpha import CrowdingMonitor, EdgeSurvivalModel, MetaAlphaModel
from core.opportunity_economics import (
    OpportunityHalfLifeEstimator,
    capital_time_efficiency,
    expected_dollar_alpha,
    opportunity_economics,
    signal_frequency,
)
from core.portfolio_robustness import (
    ResearchGapDirector,
    regime_coverage_matrix,
    regime_coverage_score,
)
from core.research_orchestrator import (
    AutonomousResearchOrchestrator,
    GeneratedHypothesisDetector,
    OrchestratorConfig,
)
from core.search_ledger import (
    SearchLedger,
    allocate_research_budget,
    information_value_test,
    research_roi_scores,
    source_roi_report,
)


def synth_df(n=350, drift=0.0003, vol=0.01, seed=0, overnight_share=0.5,
             start="2024-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1D")   # gap-free for quality gate
    daily = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(daily))
    prev = np.concatenate([[100.0], close[:-1]])
    opens = prev * np.exp(daily * overnight_share)
    return pd.DataFrame({
        "open": opens, "close": close,
        "high": np.maximum(opens, close) * 1.004,
        "low": np.minimum(opens, close) * 0.996,
        "volume": rng.integers(1_000_000, 2_000_000, n).astype(float),
    }, index=idx)


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / f"wiring_{uuid.uuid4().hex}.sqlite")


# ── §95: one autonomous campaign invokes EVERYTHING ───────────────────────────


class TestFullWiring:
    def test_campaign_invokes_all_components(self, db):
        from data.external_intelligence import ExternalSourceRegistry, RawEventStore
        from data.external_sources import CongressionalTradesSource

        registry = ExternalSourceRegistry(event_store=RawEventStore(db))
        registry.register(CongressionalTradesSource(fetch_fn=lambda since: [{
            "member": "Rep. Wired", "ticker": "S001",
            "transaction_type": "buy", "transaction_date": "2024-06-01",
            "disclosure_date": "2024-06-20"}]))

        data = {f"S{i:03d}": synth_df(seed=i) for i in range(10)}
        orch = AutonomousResearchOrchestrator(db_path=db, source_registry=registry)
        report = orch.run_campaign(data)   # ONE call — no manual component use

        inv = report["component_invocations"]
        for component in ("external_refresh", "external_features",
                          "cross_sectional_scan", "return_decomposition",
                          "outlier_detection", "change_point_detection",
                          "hypothesis_generation", "campaign_validation"):
            assert inv.get(component, 0) >= 1, f"{component} was not invoked"
        assert report["universal_research_report"]
        assert report["campaign"]["campaign_id"]
        assert report["external_events_refreshed"].get("congressional_trades") == 1
        assert report["search_breadth_total"] >= 1
        assert "research_roi" in report

    def test_external_failure_does_not_stop_research(self, db):
        from data.external_intelligence import (
            AVAILABLE,
            ExternalSourceRegistry,
            RawEventStore,
        )
        from data.external_sources import USASpendingSource

        class ExplodingSource(USASpendingSource):
            source_name = "exploding"

            def availability(self):
                return AVAILABLE

            def fetch_since(self, since):
                raise RuntimeError("provider down")

        registry = ExternalSourceRegistry(event_store=RawEventStore(db))
        registry.register(ExplodingSource())
        data = {f"S{i}": synth_df(seed=i) for i in range(6)}
        orch = AutonomousResearchOrchestrator(db_path=db, source_registry=registry)
        report = orch.run_campaign(data)     # must not raise
        assert report["campaign"]["campaign_id"]


# ── §96: Micron-like synthetic anomaly, end to end ────────────────────────────


class TestMicronLikeEndToEnd:
    def test_overnight_anomaly_discovered_and_validated(self, db):
        # one asset earns ~all appreciation overnight; nothing names it
        data = {f"S{i:03d}": synth_df(seed=i) for i in range(9)}
        data["S777"] = synth_df(n=350, drift=0.004, vol=0.004, seed=777,
                                overnight_share=0.97)
        orch = AutonomousResearchOrchestrator(db_path=db)
        report = orch.run_campaign(data)

        # discovered: decomposition flagged it and generated an overnight spec
        campaign = report["campaign"]
        assert campaign["hypotheses_emitted"] >= 1
        assert report["hypotheses_generated_universal"] >= 1
        # the overnight hypothesis went through FULL validation and survived
        promoted = report["new_paper_alphas"] or []
        assert any("overnight" in str(a).lower() for a in promoted), (
            f"expected a validated overnight alpha; got {promoted}; "
            f"rejections={report['rejections']}, "
            f"stages={campaign.get('stage_pass_counts')}")

    def test_generated_detector_computes_real_overnight_sample(self):
        df = synth_df(n=300, drift=0.003, vol=0.004, seed=7, overnight_share=0.95)
        spec = {"family": "TEMPORAL", "subfamily": "overnight", "symbol": "X",
                "direction": "long", "entry_conditions": [], "holding_bars": 1,
                "rationale": "overnight"}
        det = GeneratedHypothesisDetector([spec])
        hyps = det.scan({"X": df})
        assert len(hyps) == 1
        assert hyps[0].mean_return > 0
        assert hyps[0].sample_size > 200      # real conditioned sample, not stub


# ── §97: CAKE-like outperformer surfaced without being told ───────────────────


class TestOutperformerSurfaced:
    def test_scanner_flags_planted_hero_via_orchestrator(self, db):
        data = {f"S{i:03d}": synth_df(seed=i) for i in range(40)}
        data["S999"] = synth_df(drift=0.009, seed=999)
        orch = AutonomousResearchOrchestrator(db_path=db)
        discovery = orch.discover(data)
        assert "S999" in discovery["anomaly_flags"]
        assert any("OUTPERFORMANCE" in f
                   for f in discovery["anomaly_flags"]["S999"])
        # and hypothesis specs use percentile families, never observed values
        mom = [s for s in discovery["generated_specs"]
               if s["subfamily"] == "relative_strength"]
        assert mom, "no relative-strength specs generated"
        thresholds = {c["value"] for s in mom for c in s["entry_conditions"]}
        assert len(thresholds) >= 2   # a GRID, not one mined threshold


# ── §98: lucky-jump asset — outlier found, hypothesis dies in validation ──────


class TestOverfitOutperformerRejected:
    def test_one_lucky_jump_fails_concentration(self):
        from core.validation_stats import profit_concentration
        rets = [0.001] * 40 + [0.50] + [0.0005] * 20   # one huge lucky win
        conc = profit_concentration(rets)
        top_share = conc.get("top5_share") or conc.get("top_5_share") or \
            max(v for k, v in conc.items() if isinstance(v, float))
        assert top_share > 0.85    # exceeds campaign concentration gate

    def test_jump_is_outlier_but_not_alpha(self, db):
        from core.universal_discovery import MarketOutlierDetector
        rng = np.random.default_rng(1)
        matrix = {f"S{i}": {"returns_20d": float(rng.normal(0, 0.02))}
                  for i in range(30)}
        matrix["LUCKY"] = {"returns_20d": 1.2}
        outliers = MarketOutlierDetector().detect(matrix)
        assert any(o.symbol == "LUCKY" for o in outliers)


# ── §99-100: execution destroys / improves edge ───────────────────────────────


class TestExecutionAlpha:
    def test_execution_destroys_edge(self):
        opt = ExecutionOptimizer(fee_bps=2.0)
        res = opt.net_execution_ev(
            gross_alpha_ev=0.0015,             # +15 bps gross
            spread_pct=0.0012, order_notional=50_000, adv_usd=2_000_000,
            volatility_daily=0.03)
        assert res["execution_cost_bps"] >= 15
        assert not res["viable"]               # 15 bps gross < ~20 bps cost

    def test_cheaper_method_preferred_same_alpha(self):
        methods = {"market": {"cost_bps": 12.0, "fill_prob": 1.0},
                   "limit_mid": {"cost_bps": 5.0, "fill_prob": 1.0}}
        opt = ExecutionOptimizer(method_costs=methods, fee_bps=0.0)
        plan = opt.best_entry(gross_alpha_bps=30.0, spread_pct=0.001,
                              order_notional=10_000, adv_usd=1e7,
                              methods=("market", "limit_mid"))
        assert plan.method == "limit_mid"

    def test_passive_method_penalized_for_missed_fills(self):
        methods = {"market": {"cost_bps": 10.0, "fill_prob": 1.0},
                   "limit_passive": {"cost_bps": 1.0, "fill_prob": 0.2}}
        opt = ExecutionOptimizer(method_costs=methods, fee_bps=0.0)
        plan = opt.best_entry(gross_alpha_bps=60.0, spread_pct=0.001,
                              order_notional=10_000, adv_usd=1e7,
                              methods=("market", "limit_passive"))
        assert plan.method == "market"   # 0.2*(60-1) < 1.0*(60-10)

    def test_impact_grows_with_participation(self):
        small = market_impact_bps(order_notional=1_000, adv_usd=1e7)
        big = market_impact_bps(order_notional=1_000_000, adv_usd=1e7)
        assert big > small * 5

    def test_venue_selection_net_cost(self):
        quotes = [VenueQuote("cheap_fees_wide", fee_bps=1.0, spread_pct=0.004),
                  VenueQuote("fair_fees_tight", fee_bps=4.0, spread_pct=0.0004)]
        best = select_venue(quotes, order_notional=20_000)
        assert best["venue"] == "fair_fees_tight"

    def test_execution_learner_damped(self, db):
        learner = ExecutionLearner(db_path=db, min_samples=5)
        assert learner.calibration_multiplier() == 1.0   # no data → neutral
        for _ in range(10):     # realized slippage 3x predicted
            learner.record(symbol="X", method="market",
                           predicted_slippage_bps=2.0, actual_slippage_bps=6.0)
        m = learner.calibration_multiplier()
        assert 1.0 < m < 1.5    # damped — never jumps straight to 3x


# ── §101: capacity-aware dollar economics ─────────────────────────────────────


class TestCapacityEconomics:
    def test_dollar_alpha_reflects_capacity(self):
        a = expected_dollar_alpha(expected_net_return=0.008,
                                  practical_capacity_usd=5_000,
                                  signals_per_year=50)
        b = expected_dollar_alpha(expected_net_return=0.002,
                                  practical_capacity_usd=2_000_000,
                                  signals_per_year=50)
        assert b["dollar_alpha_per_year"] > a["dollar_alpha_per_year"] * 10

    def test_opportunity_economics_bundle(self):
        econ = opportunity_economics(expected_net_return=0.004, adv_usd=5e7,
                                     holding_days=5, signals_per_year=40)
        assert econ["estimated_alpha_capacity_usd"] > 0
        assert econ["dollar_alpha_per_year"] > 0
        assert econ["capital_time_efficiency"] == pytest.approx(0.004 / 5)

    def test_half_life_estimator(self):
        prof = OpportunityHalfLifeEstimator().estimate({
            1: [0.002] * 30, 5: [0.006] * 30, 20: [0.007] * 30,
            60: [0.003] * 30, 120: [0.001] * 30})
        assert prof.peak_horizon == 20
        assert prof.half_life == 60          # 0.003 ≤ half of 0.007 peak
        assert prof.optimal_exit_horizon == 20

    def test_signal_frequency(self):
        f = signal_frequency(["t"] * 42, observation_days=252)
        assert f["per_year"] == pytest.approx(42, rel=0.01)


# ── §102: bear market — market-neutral wins, no forced longs ──────────────────


class TestMarketNeutralAllocation:
    def _cand(self, alpha_id, symbol, direction, ev, lcb):
        from core.alpha_signal_engine import OpportunityCandidate
        return OpportunityCandidate(
            candidate_id=f"c_{alpha_id}", alpha_id=alpha_id, alpha_version="1",
            symbol=symbol, asset_class="stock", direction=direction,
            signal_time="2026-01-01T00:00:00", execution_mode="paper",
            expected_net_return=ev, conservative_ev=lcb, ev_lower_bound=lcb,
            regime_fit=1.0, execution_feasibility=90.0)

    def test_bear_market_allocates_to_relative_value(self):
        from core.portfolio_allocator import PortfolioAllocator
        # in the synthetic bear regime the momentum alpha's measured EV is
        # negative while the market-neutral pair alpha stays positive
        momentum = self._cand("mom_long", "AAA", "long", -0.004, -0.006)
        rel_value = self._cand("pair_rv", "BBB/CCC", "long", 0.003, 0.001)
        decisions = PortfolioAllocator().allocate([momentum, rel_value])
        accepted = {d.candidate.alpha_id for d in decisions if d.accepted}
        assert accepted == {"pair_rv"}     # no forced long exposure

    def test_pairs_detector_finds_planted_cointegrated_pair(self):
        rng = np.random.default_rng(3)
        n = 400
        common = np.cumsum(rng.normal(0.0004, 0.01, n))
        idx = pd.bdate_range("2024-01-01", periods=n)
        a = 100 * np.exp(common + rng.normal(0, 0.002, n))
        # cointegrated partner: same stochastic trend + mean-reverting spread
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = 0.85 * spread[i - 1] + rng.normal(0, 0.004)
        b = 50 * np.exp(common + spread)
        data = {"AAA": pd.DataFrame({"close": a}, index=idx),
                "BBB": pd.DataFrame({"close": b}, index=idx)}
        # plus unrelated noise symbols
        for i in range(4):
            data[f"N{i}"] = synth_df(n=n, seed=50 + i)[["close"]]
        from core.discovery_detectors import SearchLimits
        from core.market_neutral import PairsRelativeValueDetector
        hyps = PairsRelativeValueDetector(
            limits=SearchLimits(minimum_effective_sample=15)).scan(data)
        assert any({"AAA", "BBB"} == set(h.metadata["pair"]) for h in hyps)


# ── §103: regime coverage / gap direction ─────────────────────────────────────


class TestRegimeCoverageAndGaps:
    def test_five_identical_bull_alphas_less_diversified(self):
        bull_only = {f"bull_{i}": [("BULL", 0.01)] * 20 + [("BEAR", -0.01)] * 20
                     for i in range(5)}
        diversified = {
            "trend": [("BULL", 0.01)] * 20 + [("BEAR", -0.002)] * 20,
            "rel_value": [("BULL", 0.002)] * 20 + [("BEAR", 0.004)] * 20
                         + [("SIDEWAYS", 0.003)] * 20,
            "temporal": [("SIDEWAYS", 0.004)] * 20 + [("HIGH_VOL", 0.002)] * 20,
        }
        regimes = ("BULL", "BEAR", "SIDEWAYS", "HIGH_VOL")
        s_bull = regime_coverage_score(regime_coverage_matrix(bull_only), regimes)
        s_div = regime_coverage_score(regime_coverage_matrix(diversified), regimes)
        assert s_div > s_bull
        assert s_bull == 0.25       # bull alphas cover only BULL

    def test_gap_director_flags_bear_gap_and_directs_research(self):
        matrix = regime_coverage_matrix(
            {"bull_mom": [("BULL", 0.01)] * 20 + [("BEAR", -0.01)] * 20})
        result = ResearchGapDirector().analyze(
            matrix, alpha_families={"bull_mom": "MOMENTUM"},
            alpha_directions={"bull_mom": "long"})
        assert any(g.startswith("REGIME_COVERAGE_GAP:BEAR") for g in result["gaps"])
        assert any(g == "DIRECTION_GAP:short" for g in result["gaps"])
        assert any("market-neutral" in p["research_directive"]
                   for p in result["priorities"])
        assert "cash is acceptable" in result["note"]


# ── §104: edge decay HEALTHY → WATCH → DEGRADED → PAUSED ──────────────────────


class TestEdgeDecay:
    def test_decay_walks_down_the_states(self):
        model = EdgeSurvivalModel()
        older = [0.01] * 30
        states = []
        for recent_ev in (0.010, 0.005, 0.002, -0.004):
            recent = [recent_ev] * 30
            states.append(model.assess("a1", recent_returns=recent,
                                       older_returns=older).recommended_state)
        assert states == ["HEALTHY", "WATCH", "DEGRADED", "PAUSED"]

    def test_capital_multiplier_shrinks_with_survival(self):
        model = EdgeSurvivalModel()
        healthy = model.assess("a", recent_returns=[0.01] * 30,
                               older_returns=[0.01] * 30)
        paused = model.assess("a", recent_returns=[-0.005] * 30,
                              older_returns=[0.01] * 30)
        assert healthy.capital_multiplier == 1.0
        assert paused.capital_multiplier == 0.0

    def test_thin_recent_evidence_does_not_kill_edge(self):
        model = EdgeSurvivalModel()
        a = model.assess("a", recent_returns=[-0.01] * 3,
                         older_returns=[0.01] * 30)
        assert a.recommended_state in ("HEALTHY", "WATCH")


# ── §105: crowding / timing shift ─────────────────────────────────────────────


class TestCrowding:
    def test_edge_timing_shift_flagged(self):
        result = CrowdingMonitor().assess(
            "a1",
            post_signal_returns_early=[0.010] * 20,
            post_signal_returns_recent=[0.003] * 20,   # post-signal shrinking
            pre_signal_moves_early=[0.002] * 20,
            pre_signal_moves_recent=[0.006] * 20)      # pre-signal growing
        assert "EDGE_TIMING_SHIFT" in result["flags"]
        assert result["recommended_action"] == "re_research_timing"

    def test_healthy_edge_not_flagged(self):
        result = CrowdingMonitor().assess(
            "a1",
            post_signal_returns_early=[0.01] * 20,
            post_signal_returns_recent=[0.009] * 20,
            pre_signal_moves_early=[0.002] * 20,
            pre_signal_moves_recent=[0.002] * 20)
        assert not result["crowded"]


# ── §106: short borrow constraints ────────────────────────────────────────────


class TestBorrowConstraints:
    def test_equity_short_without_borrow_data_fails_closed(self):
        check = BorrowChecker().check("XYZ", "stock", "short")
        assert not check["executable"]
        assert check["reason"] == "BORROW_UNAVAILABLE"

    def test_crypto_short_and_longs_unaffected(self):
        checker = BorrowChecker()
        assert checker.check("BTC-USD", "crypto", "short")["executable"]
        assert checker.check("XYZ", "stock", "long")["executable"]

    def test_borrow_provider_enables_shorts(self):
        from core.execution_optimizer import BorrowInfo

        class Provider:
            def borrow_info(self, symbol):
                return BorrowInfo(available=True, borrow_rate_annual=0.02)
        check = BorrowChecker(Provider()).check("XYZ", "stock", "short")
        assert check["executable"]
        assert check["borrow_rate_annual"] == 0.02


# ── §108-109: source incremental value ────────────────────────────────────────


class TestSourceIncrementalValue:
    def test_useless_source_gets_no_authority(self):
        rng = np.random.default_rng(4)
        base = list(rng.normal(0.002, 0.01, 200))
        # augmented model = same returns, no improvement
        augmented = list(rng.normal(0.002, 0.01, 200))
        res = information_value_test(base, augmented)
        assert not res["significant"]
        assert res["incremental_value"] == 0.0

    def test_genuinely_predictive_source_detected(self):
        rng = np.random.default_rng(5)
        base = list(rng.normal(0.001, 0.01, 300))
        augmented = [r + 0.004 for r in rng.normal(0.001, 0.01, 300)]
        res = information_value_test(base, augmented)
        assert res["significant"]
        assert res["incremental_ev"] > 0.003


# ── §110: point-in-time — 13F filing-date rule ────────────────────────────────


class TestInstitutionalPIT:
    def test_13f_available_at_filing_not_portfolio_date(self, db):
        from data.external_intelligence import RawEventStore
        from data.research_interfaces import InstitutionalFilingSource
        source = InstitutionalFilingSource(fetch_fn=lambda since: [{
            "symbol": "ABC", "institution": "Big Fund LP",
            "period_end": "2026-03-31",        # portfolio as-of
            "filing_date": "2026-05-10",       # public availability
            "shares": 1_000_000, "value_usd": 5e7, "change_shares": 250_000}])
        store = RawEventStore(db)
        store.ingest(source.fetch_since("2026-01-01"))
        # April: quarter ended but not filed — invisible
        assert store.events_available_at("2026-04-15T00:00:00") == []
        vis = store.events_available_at("2026-05-11T00:00:00")
        assert len(vis) == 1 and vis[0].payload["period_end"] == "2026-03-31"


# ── §111-112: cash + risk-of-ruin hard gate ───────────────────────────────────


class TestCashAndRuin:
    def _cand(self, alpha_id, ev, lcb, symbol="AAA"):
        from core.alpha_signal_engine import OpportunityCandidate
        return OpportunityCandidate(
            candidate_id=f"c_{alpha_id}", alpha_id=alpha_id, alpha_version="1",
            symbol=symbol, asset_class="stock", direction="long",
            signal_time="2026-01-01T00:00:00", execution_mode="paper",
            expected_net_return=ev, conservative_ev=lcb, ev_lower_bound=lcb,
            regime_fit=1.0, execution_feasibility=90.0)

    def test_all_below_hurdle_means_cash(self):
        from core.portfolio_allocator import PortfolioAllocator
        cands = [self._cand("a", -0.001, -0.002),
                 self._cand("b", 0.0, -0.001, symbol="BBB")]
        decisions = PortfolioAllocator().allocate(cands)
        assert not any(d.accepted for d in decisions)

    def test_ruinous_portfolio_rejected_by_hard_gate(self):
        from core.alpha_signal_engine import ReasonCode
        from core.portfolio_allocator import (
            AllocatorConfig,
            PortfolioAllocator,
        )
        rng = np.random.default_rng(6)
        wild = list(rng.normal(0.05, 0.9, 120))     # huge return, ruinous vol
        safe = list(rng.normal(0.003, 0.01, 120))
        cands = [self._cand("wild", 0.05, 0.02, symbol="WLD"),
                 self._cand("safe", 0.003, 0.001, symbol="SAF")]
        alloc = PortfolioAllocator(AllocatorConfig(
            base_position_frac=0.30, max_asset_exposure_frac=0.5,
            max_total_exposure_frac=1.0, max_family_exposure_frac=1.0,
            max_correlated_cluster_frac=1.0, max_risk_of_ruin=0.05))
        decisions = alloc.allocate(
            cands, alpha_return_samples={"wild": wild, "safe": safe})
        by_id = {d.candidate.alpha_id: d for d in decisions}
        assert not by_id["wild"].accepted
        assert ReasonCode.RISK_OF_RUIN_LIMIT in by_id["wild"].reason_codes
        assert by_id["safe"].accepted


# ── §113: meta-alpha learns regime-conditioned allocation ─────────────────────


class TestMetaAlpha:
    def _train(self, model):
        rng = np.random.default_rng(7)
        for _ in range(60):
            model.observe("momentum", "TREND", float(rng.normal(0.008, 0.004)))
            model.observe("momentum", "SIDEWAYS", float(rng.normal(-0.004, 0.004)))
            model.observe("mean_rev", "TREND", float(rng.normal(-0.003, 0.004)))
            model.observe("mean_rev", "SIDEWAYS", float(rng.normal(0.007, 0.004)))

    def test_regime_dependent_ranking_learned(self):
        model = MetaAlphaModel(mode="active")
        self._train(model)
        trend_rank = model.rank_alphas(["momentum", "mean_rev"], "TREND")
        side_rank = model.rank_alphas(["momentum", "mean_rev"], "SIDEWAYS")
        assert trend_rank[0][0] == "momentum"
        assert side_rank[0][0] == "mean_rev"
        assert model.regime_fit_for_allocation("momentum", "TREND") > \
            model.regime_fit_for_allocation("momentum", "SIDEWAYS")

    def test_shadow_mode_never_changes_sizing(self):
        model = MetaAlphaModel(mode="shadow")
        self._train(model)
        # shadow: logs comparison, returns neutral multiplier
        assert model.regime_fit_for_allocation("momentum", "SIDEWAYS") == 1.0
        entry = model.shadow_compare(["momentum", "mean_rev"], "SIDEWAYS")
        assert entry["meta_ranking"][0] == "mean_rev"
        assert entry["deterministic"] == ["momentum", "mean_rev"]

    def test_validation_never_bypassed(self):
        """Meta-alpha only modulates capital among validated alphas — it has
        no promotion authority (structural: no library access at all)."""
        import inspect

        from core import meta_alpha
        src = inspect.getsource(meta_alpha)
        assert "AlphaLibrary" not in src
        assert "promote_validated" not in src
        assert ".transition(" not in src


# ── §114: end-to-end forward learning chain ───────────────────────────────────


class TestForwardLearningChain:
    def test_result_flows_to_health_and_next_allocation(self):
        model = MetaAlphaModel(mode="active")
        survival = EdgeSurvivalModel()
        # month 1: alpha works
        for _ in range(30):
            model.observe("a1", "BULL", 0.01)
        ev_before = model.predict("a1", "BULL")["expected_alpha_ev"]
        s1 = survival.assess("a1", recent_returns=[0.01] * 30,
                             older_returns=[0.01] * 30)
        # month 2: realized results decay — same attribution stream updates all
        for _ in range(30):
            model.observe("a1", "BULL", -0.006)
        s2 = survival.assess("a1", recent_returns=[-0.006] * 30,
                             older_returns=[0.01] * 30)
        ev_after = model.predict("a1", "BULL")["expected_alpha_ev"]
        assert ev_after < ev_before
        assert s1.capital_multiplier > s2.capital_multiplier
        assert s2.recommended_state in ("DEGRADED", "PAUSED")


# ── Search ledger + research ROI + source ROI ─────────────────────────────────


class TestLedgerAndROI:
    def test_breadth_accumulates_across_campaigns_and_sources(self, db):
        ledger = SearchLedger(db)
        ledger.record_search("PRICE", 82_000)
        ledger.record_search("TEMPORAL", 22_000)
        ledger.record_search("GOVERNMENT", 9_000)
        ledger.record_search("CRYPTO_SMART_MONEY", 18_000)
        assert ledger.total_breadth() == 131_000
        # new source does NOT reset history
        ledger.record_search("CROSS_SOURCE", 31_000)
        assert ledger.total_breadth() == 162_000
        assert ledger.breadth_by_source()["PRICE"] == 82_000

    def test_low_frequency_family_not_abandoned_early(self, db):
        ledger = SearchLedger(db)
        ledger.record_search("PRICE", 50_000)
        ledger.record_outcome("PRICE", "validated", 25)
        ledger.record_search("GOVERNMENT", 40)          # tiny sample, 0 hits
        roi = research_roi_scores(ledger, min_observations=200)
        assert roi["GOVERNMENT"]["verdict"] == "insufficient_sample"
        assert roi["GOVERNMENT"]["shrunk_validation_rate"] > 0   # shrinkage
        budget = allocate_research_budget(roi, total_budget=10_000)
        assert budget["GOVERNMENT"] > 0                 # floor: never starves

    def test_source_roi_reports_poor_economics_without_cancelling(self, db):
        ledger = SearchLedger(db)
        ledger.record_search("OPTIONS_FEED", 5_000)
        ledger.record_outcome("OPTIONS_FEED", "validated", 1, forward_pnl=300.0)
        ledger.record_search("GOVERNMENT", 5_000)
        ledger.record_outcome("GOVERNMENT", "validated", 2, forward_pnl=900.0)
        report = source_roi_report(ledger, {"OPTIONS_FEED": 2000.0,
                                            "GOVERNMENT": 0.0})
        assert report["OPTIONS_FEED"]["economics"] == "poor_source_economics"
        assert report["GOVERNMENT"]["economics"] == "free_source"


# ── Interfaces: unavailable by default, dedup, features ───────────────────────


class TestResearchInterfaces:
    def test_all_interfaces_unavailable_without_providers(self):
        from data.external_intelligence import UNAVAILABLE
        from data.research_interfaces import default_interface_sources
        for src in default_interface_sources():
            assert src.availability() == UNAVAILABLE
            assert src.fetch_since("2026-01-01") == []   # never fabricates

    def test_canonical_event_dedup(self):
        from data.external_intelligence import ExternalEvent
        from data.research_interfaces import deduplicate_events
        mk = lambda src, first_seen: ExternalEvent(
            source_name=src, event_type="congress_trade",
            event_time="2026-02-01", publication_time=first_seen,
            first_seen_time=first_seen, symbols=["ABC"], payload={})
        events = [mk("aggregator_a", "2026-02-10"),
                  mk("official_filings", "2026-02-05")]
        deduped = deduplicate_events(events)
        assert len(deduped) == 1
        assert deduped[0].source_name == "official_filings"   # earliest available
        assert deduped[0].payload["corroborating_sources"] == ["aggregator_a"]

    def test_revision_acceleration_and_insider_clusters(self):
        from data.research_interfaces import (
            insider_cluster_features,
            revision_acceleration_score,
        )
        score = revision_acceleration_score([
            {"direction": "up", "magnitude_pct": 0.05},
            {"direction": "up", "magnitude_pct": 0.03},
            {"direction": "down", "magnitude_pct": 0.01}])
        assert score["net_revision_count"] == 1
        assert score["revision_acceleration_score"] > 0
        feats = insider_cluster_features([
            {"transaction_type": "open_market_purchase", "role": "CEO",
             "insider_name": "A", "value_usd": 1e6},
            {"transaction_type": "open_market_purchase", "role": "CFO",
             "insider_name": "B", "value_usd": 5e5},
            {"transaction_type": "open_market_purchase", "role": "Director",
             "insider_name": "C", "value_usd": 2e5},
            {"transaction_type": "option_exercise", "role": "CEO",
             "insider_name": "A", "value_usd": 9e9}])   # exercises don't count
        assert feats["insider_cluster"]
        assert feats["ceo_cfo_cluster"]
        assert feats["open_market_buy_count"] == 3

    def test_cross_asset_relationship_structural_change(self):
        from data.research_interfaces import rolling_cross_asset_relationship
        rng = np.random.default_rng(8)
        x = rng.normal(0, 0.01, 300)
        y = np.concatenate([2.0 * x[:150], -0.5 * x[150:]]) \
            + rng.normal(0, 0.001, 300)
        rel = rolling_cross_asset_relationship(list(x), list(y), window=60)
        assert rel["structural_change"]
        assert rel["beta_first_half"] > 1.5 and rel["beta_second_half"] < 0

    def test_volatility_risk_premium(self):
        from data.research_interfaces import volatility_risk_premium
        assert volatility_risk_premium(0.30, 0.22) == pytest.approx(0.08)
        assert volatility_risk_premium(None, 0.2) is None
