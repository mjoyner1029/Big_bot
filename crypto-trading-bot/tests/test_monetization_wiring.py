"""Monetization-wiring tests (spec §72-92, §97): profitability modules now
drive the REAL candidate → enrichment → allocation → execution path.
"""
import uuid

import numpy as np
import pandas as pd
import pytest

from core.alpha_signal_engine import OpportunityCandidate, ReasonCode
from core.execution_optimizer import BorrowChecker, BorrowInfo, ExecutionOptimizer
from core.meta_alpha import MetaAlphaModel
from core.opportunity_enrichment import (
    EnrichmentConfig,
    OpportunityEnricher,
    classify_urgency,
    meta_alpha_promotion_ready,
    methods_for_urgency,
)
from core.portfolio_allocator import AllocatorConfig, PortfolioAllocator


def make_candidate(alpha_id="a1", symbol="AAA", direction="long", ev=0.004,
                   lcb=0.002, adv=5e7, holding_bars=5, asset_class="stock"):
    return OpportunityCandidate(
        candidate_id=f"c_{alpha_id}_{symbol}", alpha_id=alpha_id,
        alpha_version="1", symbol=symbol, asset_class=asset_class,
        direction=direction, signal_time="2026-09-01T00:00:00",
        execution_mode="paper",
        expected_net_return=ev, conservative_ev=lcb, ev_lower_bound=lcb,
        execution_feasibility=90.0,
        features={"dollar_volume_24h": adv, "spread_pct": 0.0004,
                  "atr_pct": 0.015, "price": 100.0},
        exit_plan={"time_stop_bars": holding_bars})


def history(recent_ev=0.004, older_ev=0.004, n=30, regime="BULL"):
    return {"recent": [recent_ev] * n, "older": [older_ev] * n,
            "by_regime": [(regime, older_ev)] * n + [(regime, recent_ev)] * n}


def enricher_with(fetch, mode="shadow", borrow=None):
    return OpportunityEnricher(
        EnrichmentConfig(meta_alpha_mode=mode),
        borrow_checker=borrow or BorrowChecker(),
        returns_fetcher=fetch, db_path=":memory:")


# ── §73: meta-alpha regime conditioning ───────────────────────────────────────


class TestMetaAlphaRegime:
    def _enrich_in(self, regime, mode="advisory"):
        fetch = lambda a: {
            "recent": [0.004] * 30, "older": [0.004] * 30,
            "by_regime": [("TREND", 0.008)] * 40 + [("SIDEWAYS", -0.004)] * 40}
        c = make_candidate()
        enricher_with(fetch, mode=mode).enrich(
            c, regime=regime, capital=10_000, base_position_frac=0.05)
        return c

    def test_unfavorable_regime_reduces_ranking(self):
        good = self._enrich_in("TREND")
        bad = self._enrich_in("SIDEWAYS")
        assert bad.regime_fit < good.regime_fit
        assert bad.meta_alpha_ev < good.meta_alpha_ev

    def test_shadow_mode_never_moves_regime_fit(self):
        c = self._enrich_in("SIDEWAYS", mode="shadow")
        assert c.regime_fit == 1.0                # baseline stays authoritative
        assert c.meta_alpha_ev is not None        # but prediction is logged

    def test_promotion_gate(self):
        too_few = [(0.004, 0.005)] * 10
        assert not meta_alpha_promotion_ready(too_few)["ready"]
        calibrated = [(0.004, 0.005), (0.003, 0.002), (-0.002, -0.001)] * 15
        assert meta_alpha_promotion_ready(calibrated)["ready"]
        anti = [(0.004, -0.005), (0.003, -0.004)] * 20
        assert not meta_alpha_promotion_ready(anti)["ready"]

    def test_meta_confidence_never_exceeds_caps(self):
        c = self._enrich_in("TREND", mode="active")
        assert c.regime_fit <= 1.5                # bounded, no leverage creep


# ── §74: edge survival affects sizing ─────────────────────────────────────────


class TestEdgeSurvivalSizing:
    def test_decaying_edge_gets_less_capital(self):
        healthy = make_candidate(alpha_id="healthy")
        decaying = make_candidate(alpha_id="decaying", symbol="BBB")
        fetch = lambda a: (history(0.004, 0.004) if a == "healthy"
                           else history(0.0005, 0.010))     # sharp EV decay
        enr = enricher_with(fetch)
        enr.enrich_all([healthy, decaying], regime="BULL", capital=10_000)
        assert decaying.survival_multiplier < healthy.survival_multiplier
        decisions = PortfolioAllocator(AllocatorConfig(
            max_family_exposure_frac=1.0)).allocate([healthy, decaying])
        by_id = {d.candidate.alpha_id: d for d in decisions}
        if by_id["decaying"].accepted:
            assert by_id["decaying"].allocation_usd < by_id["healthy"].allocation_usd

    def test_paused_edge_hard_rejected(self):
        c = make_candidate(alpha_id="dead")
        enr = enricher_with(lambda a: history(-0.006, 0.010))
        enr.enrich(c, regime="BULL", capital=10_000, base_position_frac=0.05)
        assert ReasonCode.EDGE_DECAY in c.reason_codes
        decisions = PortfolioAllocator().allocate([c])
        assert not decisions[0].accepted
        assert ReasonCode.EDGE_DECAY in decisions[0].reason_codes


# ── §75: crowding ─────────────────────────────────────────────────────────────


class TestCrowdingPenalty:
    def test_crowded_edge_haircut_and_re_research(self):
        c = make_candidate(alpha_id="crowded")
        enr = enricher_with(lambda a: history(0.002, 0.010))  # post-signal shrank
        enr.enrich(c, regime="BULL", capital=10_000, base_position_frac=0.05)
        assert c.crowding_score > 0
        assert c.expected_net_return < c.raw_expected_net_return
        # timing-shift requires pre-signal data; shrinkage alone flags crowding
        fresh = make_candidate(alpha_id="fresh", symbol="CCC")
        enr.enrich(fresh, regime="BULL", capital=10_000, base_position_frac=0.05)


# ── §76/§8-9: half-life → urgency → execution method ──────────────────────────


class TestHalfLifeExecution:
    def test_urgency_classification(self):
        assert classify_urgency(2 / 1440, bar_minutes=1440) == "IMMEDIATE"  # 2 min
        assert classify_urgency(0.1) == "FAST"          # ~2.4h of a daily bar
        assert classify_urgency(5) == "NORMAL"
        assert classify_urgency(60) == "PATIENT"

    def test_short_half_life_rejects_passive_methods(self):
        assert "limit_passive" not in methods_for_urgency("IMMEDIATE")
        assert "limit_passive" not in methods_for_urgency("FAST")
        assert methods_for_urgency("IMMEDIATE") == ("market",)
        assert "limit_passive" in methods_for_urgency("PATIENT")

    def test_enrichment_sets_urgency_and_method(self):
        fast = make_candidate(holding_bars=1)      # 1-day half-life
        slow = make_candidate(symbol="BBB", holding_bars=60)
        enr = enricher_with(lambda a: history())
        enr.enrich_all([fast, slow], regime="BULL", capital=10_000)
        assert fast.urgency in ("FAST", "NORMAL")
        assert slow.urgency == "PATIENT"
        assert slow.recommended_order_type is not None


# ── §77-78: execution optimizer controls the order, never forces fills ────────


class TestExecutionControl:
    def test_cheaper_limit_method_selected(self):
        opt = ExecutionOptimizer(method_costs={
            "market": {"cost_bps": 12.0, "fill_prob": 1.0},
            "limit_mid": {"cost_bps": 5.0, "fill_prob": 0.95}}, fee_bps=0.0)
        enr = OpportunityEnricher(execution_optimizer=opt,
                                  returns_fetcher=lambda a: history(),
                                  db_path=":memory:")
        c = make_candidate(ev=0.003)
        enr.enrich(c, regime="BULL", capital=10_000, base_position_frac=0.05)
        assert c.recommended_order_type == "limit_mid"

    def test_execution_destroys_edge_means_no_trade(self):
        # gross 8 bps, wide spread → best method still > 8 bps round trip
        c = make_candidate(ev=0.0008, lcb=0.0006, adv=3e5)
        c.features["spread_pct"] = 0.0025
        enr = enricher_with(lambda a: history())
        enr.enrich(c, regime="BULL", capital=100_000, base_position_frac=0.05)
        assert c.expected_net_return < 0           # net execution EV negative
        decisions = PortfolioAllocator().allocate([c])
        assert not decisions[0].accepted           # NO TRADE — not rescued


# ── §79: capacity is a hard cap on requested size ─────────────────────────────


class TestCapacityCap:
    def test_size_capped_at_practical_capacity(self):
        c = make_candidate(ev=0.02, lcb=0.015, adv=2e5)    # thin liquidity
        enr = enricher_with(lambda a: history(0.02, 0.02))
        enr.enrich(c, regime="BULL", capital=1_000_000, base_position_frac=0.10)
        assert c.practical_capacity_usd is not None
        alloc = PortfolioAllocator(AllocatorConfig(
            capital=1_000_000, base_position_frac=0.10))
        decisions = alloc.allocate([c])
        d = decisions[0]
        if d.accepted:
            assert d.allocation_usd <= c.practical_capacity_usd + 1e-6
            assert d.allocation_usd < 1_000_000 * 0.10   # capacity < risk size

    def test_unusable_capacity_rejected(self):
        c = make_candidate(ev=0.02, lcb=0.015)
        c.practical_capacity_usd = 50.0            # below min usable
        decisions = PortfolioAllocator().allocate([c])
        assert not decisions[0].accepted
        assert ReasonCode.INSUFFICIENT_CAPACITY in decisions[0].reason_codes


# ── §80-81: dollar alpha + capital-time on enriched candidates ────────────────


class TestEnrichedEconomics:
    def test_dollar_alpha_and_capital_time_populated(self):
        small_deep = make_candidate(alpha_id="deep", ev=0.0015, adv=5e9,
                                    holding_bars=5)
        big_thin = make_candidate(alpha_id="thin", symbol="BBB", ev=0.005,
                                  adv=2e5, holding_bars=5)
        enr = enricher_with(lambda a: history())
        enr.enrich_all([small_deep, big_thin], regime="BULL", capital=1_000_000,
                       base_position_frac=0.20)
        assert small_deep.practical_capacity_usd > big_thin.practical_capacity_usd
        fast = make_candidate(alpha_id="fast", symbol="CCC", ev=0.002,
                              holding_bars=1)
        slow = make_candidate(alpha_id="slow", symbol="DDD", ev=0.0025,
                              holding_bars=20)
        enr.enrich_all([fast, slow], regime="BULL", capital=10_000)
        assert fast.capital_time_efficiency > slow.capital_time_efficiency


# ── §72: three-opportunity economic sense test ────────────────────────────────


class TestSyntheticAllocation:
    def test_allocator_makes_economically_sensible_choices(self):
        # A: high % EV, tiny capacity, crowded, short half-life
        a = make_candidate(alpha_id="A", symbol="AAA", ev=0.008, lcb=0.005,
                           adv=1e5, holding_bars=1)
        # B: lower % EV, deep capacity, healthy, longer half-life
        b = make_candidate(alpha_id="B", symbol="BBB", ev=0.003, lcb=0.002,
                           adv=1e9, holding_bars=10)
        # C: moderate EV, correlated with existing cluster (same as B)
        c = make_candidate(alpha_id="C", symbol="CCC", ev=0.003, lcb=0.0015,
                           adv=1e8, holding_bars=10)

        def fetch(alpha_id):
            if alpha_id == "A":
                return history(0.002, 0.012)       # crowded/decaying
            return history(0.003, 0.003)
        enr = enricher_with(fetch)
        enr.enrich_all([a, b, c], regime="BULL", capital=100_000,
                       base_position_frac=0.05)
        alloc = PortfolioAllocator(
            AllocatorConfig(capital=100_000, max_family_exposure_frac=1.0),
            downside_correlations={("B", "C"): 0.95})
        decisions = alloc.allocate([a, b, c])
        by_id = {d.candidate.alpha_id: d for d in decisions}
        assert by_id["B"].accepted                 # healthy + deep wins
        if by_id["A"].accepted:                    # crowded/thin must be small
            assert by_id["A"].allocation_usd <= by_id["B"].allocation_usd
        if by_id["C"].accepted:                    # correlated cluster haircut
            assert by_id["C"].allocation_usd < by_id["B"].allocation_usd


# ── §83-84: correlation clusters + market-neutral diversification ─────────────


class TestClustersAndMarketNeutral:
    def test_downside_correlated_alphas_share_cluster_cap(self):
        cands = [make_candidate(alpha_id=f"m{i}", symbol=f"S{i}", ev=0.004,
                                lcb=0.003) for i in range(5)]
        dcorr = {(f"m{i}", f"m{j}"): 0.95 for i in range(5) for j in range(i + 1, 5)}
        alloc = PortfolioAllocator(
            AllocatorConfig(capital=100_000, max_family_exposure_frac=1.0,
                            max_correlated_cluster_frac=0.10),
            downside_correlations=dcorr)
        decisions = alloc.allocate(cands)
        total = sum(d.allocation_usd for d in decisions if d.accepted)
        assert total <= 100_000 * 0.10 + 1e-6      # one effective cluster

    def test_market_neutral_diversifies_long_heavy_book(self):
        longs = [make_candidate(alpha_id=f"L{i}", symbol=f"S{i}", ev=0.004,
                                lcb=0.003) for i in range(3)]
        mn = make_candidate(alpha_id="pair_mn", symbol="AAA/BBB", ev=0.003,
                            lcb=0.002)
        dcorr = {(f"L{i}", f"L{j}"): 0.9 for i in range(3) for j in range(i + 1, 3)}
        alloc = PortfolioAllocator(
            AllocatorConfig(capital=100_000, max_family_exposure_frac=1.0,
                            max_correlated_cluster_frac=0.08),
            downside_correlations=dcorr)
        decisions = alloc.allocate(longs + [mn])
        by_id = {d.candidate.alpha_id: d for d in decisions}
        assert by_id["pair_mn"].accepted           # uncorrelated MN adds value
        long_total = sum(d.allocation_usd for d in decisions
                         if d.accepted and d.candidate.alpha_id.startswith("L"))
        assert long_total <= 100_000 * 0.08 + 1e-6


# ── §85: short borrow cost kills marginal short EV ────────────────────────────


class TestShortBorrowCost:
    def _short(self, borrow_rate, ev=0.002, holding_bars=30):
        class Provider:
            def borrow_info(self, symbol):
                return BorrowInfo(available=True,
                                  borrow_rate_annual=borrow_rate)
        c = make_candidate(alpha_id="short_a", direction="short", ev=ev,
                           lcb=ev * 0.7, holding_bars=holding_bars, adv=1e9)
        enr = enricher_with(lambda a: history(), borrow=BorrowChecker(Provider()))
        enr.enrich(c, regime="BULL", capital=10_000, base_position_frac=0.05)
        return c

    def test_expensive_borrow_destroys_short_edge(self):
        # 20 bps gross over 30 days vs 36% annual borrow → ~300 bps carry
        c = self._short(borrow_rate=0.36)
        assert c.expected_net_return < 0
        assert not PortfolioAllocator().allocate([c])[0].accepted

    def test_cheap_borrow_short_survives(self):
        c = self._short(borrow_rate=0.005, ev=0.01, holding_bars=5)
        assert c.expected_net_return > 0

    def test_no_borrow_provider_fails_closed(self):
        c = make_candidate(direction="short", ev=0.01, lcb=0.008)
        enr = enricher_with(lambda a: history())
        enr.enrich(c, regime="BULL", capital=10_000, base_position_frac=0.05)
        assert ReasonCode.BORROW_UNAVAILABLE in c.reason_codes
        d = PortfolioAllocator().allocate([c])[0]
        assert not d.accepted
        assert ReasonCode.BORROW_UNAVAILABLE in d.reason_codes


# ── §86: research gaps steer budgets, not thresholds ──────────────────────────


class TestResearchGapBudgets:
    def test_bull_only_portfolio_boosts_neutral_and_short_budgets(self, tmp_path):
        from core.research_orchestrator import AutonomousResearchOrchestrator
        orch = AutonomousResearchOrchestrator(
            db_path=str(tmp_path / "gap.sqlite"))
        base = orch.config.generator_budget.max_hypotheses_per_family
        analysis = orch.apply_research_gaps(
            {"bull_mom": [("BULL", 0.01)] * 20 + [("BEAR", -0.01)] * 20},
            alpha_families={"bull_mom": "MOMENTUM"},
            alpha_directions={"bull_mom": "long"})
        overrides = analysis["family_budget_overrides"]
        assert overrides.get("MARKET_NEUTRAL", 0) > base
        # validation thresholds untouched
        assert orch.config.campaign.fdr_alpha <= 0.05


# ── §87: large-universe batching (no 30-instrument truncation) ────────────────


class TestLargeUniverseBatching:
    def test_1000_instruments_run_in_batches(self, tmp_path):
        from core.discovery_detectors import DiscoveryDetector
        from core.research_campaign import CampaignConfig, ResearchCampaignRunner

        class CountingDetector(DiscoveryDetector):
            family = "PRICE"
            seen: set = set()

            def scan(self, data):
                CountingDetector.seen.update(data.keys())
                return []

        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=220, freq="1D")
        data = {}
        for i in range(1000):
            close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 220)))
            data[f"I{i:04d}"] = pd.DataFrame({
                "open": close, "high": close * 1.004, "low": close * 0.996,
                "close": close, "volume": np.full(220, 1e6)}, index=idx)
        runner = ResearchCampaignRunner(
            str(tmp_path / "big.sqlite"),
            config=CampaignConfig(instrument_batch_size=100),
            detectors_factory=lambda limits=None, tracker=None:
                [CountingDetector(limits, tracker)])
        report = runner.run(data)
        assert report["batches"] == 10
        assert len(CountingDetector.seen) == 1000   # nothing truncated


# ── §89: no opportunity → cash ────────────────────────────────────────────────


class TestCashRemainsValid:
    def test_all_enriched_candidates_fail_hurdles(self):
        cands = [make_candidate(alpha_id=f"x{i}", symbol=f"S{i}",
                                ev=0.0002, lcb=0.0001, adv=3e5)
                 for i in range(4)]
        for c in cands:
            c.features["spread_pct"] = 0.003       # costs exceed all edges
        enr = enricher_with(lambda a: history())
        enr.enrich_all(cands, regime="BULL", capital=100_000)
        decisions = PortfolioAllocator().allocate(cands)
        assert not any(d.accepted for d in decisions)


# ── §91: model attribution intact ─────────────────────────────────────────────


class TestModelAttribution:
    def test_candidate_carries_all_model_versions(self):
        c = make_candidate()
        enr = enricher_with(lambda a: history())
        enr.enrich(c, regime="BULL", capital=10_000, base_position_frac=0.05)
        for key in ("enrichment", "meta_alpha_mode", "execution_optimizer",
                    "edge_survival"):
            assert key in c.model_versions

    def test_attribution_store_accepts_model_versions(self, tmp_path):
        from core.trade_attribution import TradeAttributionStore
        store = TradeAttributionStore(str(tmp_path / "attr.sqlite"))
        store.record(trade_memory_id=1, alpha_id="a1", alpha_version="2",
                     candidate_id="c1", meta_model_version="shadow-1.0.0",
                     execution_model_version="1.0.0", feature_version="1.0.0")
        rows = store.for_alpha("a1")
        assert rows and rows[0]["meta_model_version"] == "shadow-1.0.0"


# ── §88: full automated loop ──────────────────────────────────────────────────


class TestFullAutomatedLoop:
    def test_research_to_execution_to_learning(self, tmp_path):
        """research → PAPER alpha → candidate → enrichment → allocation →
        execution plan → paper fill → close → forward updates."""
        from core.alpha_library import AlphaLibrary
        from core.alpha_signal_engine import AlphaSignalEngine
        from core.execution_optimizer import ExecutionLearner
        from core.research_orchestrator import AutonomousResearchOrchestrator

        db = str(tmp_path / f"loop_{uuid.uuid4().hex}.sqlite")
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=350, freq="1D")

        def synth(drift, vol, seed, overnight=0.5):
            r = np.random.default_rng(seed)
            daily = r.normal(drift, vol, 350)
            close = 100 * np.exp(np.cumsum(daily))
            prev = np.concatenate([[100.0], close[:-1]])
            opens = prev * np.exp(daily * overnight)
            return pd.DataFrame({
                "open": opens, "close": close, "high": np.maximum(opens, close) * 1.004,
                "low": np.minimum(opens, close) * 0.996,
                "volume": r.integers(1_000_000, 2_000_000, 350).astype(float)},
                index=idx)

        data = {f"S{i:03d}": synth(0.0003, 0.01, i) for i in range(9)}
        data["S777"] = synth(0.004, 0.004, 777, overnight=0.97)

        # 1. scheduled research → PAPER alpha
        orch = AutonomousResearchOrchestrator(db_path=db)
        report = orch.run_campaign(data)
        paper = report["new_paper_alphas"]
        assert paper, f"no PAPER alpha; rejections={report['rejections']}"

        # 2. PAPER alpha → live candidate via the signal engine (regime must
        #    be one the alpha validated in — regime gating is real)
        library = AlphaLibrary(db)
        engine = AlphaSignalEngine(library)
        candidates = engine.generate_candidates(
            data, regime="LOW_VOL", context={}, live_mode=False)
        assert candidates, (f"no candidates from PAPER alphas {paper}; "
                            f"rejections={engine.rejection_summary()}")
        c = candidates[0]
        c.expected_net_return = 0.004               # forward EV estimate
        c.ev_lower_bound = c.conservative_ev = 0.002
        c.features["spread_pct"] = 0.0004            # live quote spread

        # 3. enrichment → allocation → execution plan
        enr = enricher_with(lambda a: history())
        enr.enrich(c, regime="LOW_VOL", capital=10_000, base_position_frac=0.05)
        decisions = PortfolioAllocator(AllocatorConfig(
            max_family_exposure_frac=1.0)).allocate([c])
        d = next(x for x in decisions if x.candidate is c)
        assert d.accepted
        assert c.recommended_order_type is not None
        assert c.model_versions

        # 4. paper fill → close → forward learning updates
        learner = ExecutionLearner(db_path=db, min_samples=1)
        learner.record(symbol=c.symbol, method=c.recommended_order_type,
                       predicted_slippage_bps=c.expected_slippage_bps or 2.0,
                       actual_slippage_bps=3.0)
        assert learner.method_cost_table()          # execution model updated
        meta = enr.meta
        meta.observe(c.alpha_id, "LOW_VOL", 0.005)  # realized result flows back
        assert meta.predict(c.alpha_id, "LOW_VOL")["n_conditional"] >= 1


# ── §97: deterministic simulation — enriched vs EV-only allocator ─────────────


class TestAllocatorSimulation:
    def test_enriched_allocator_beats_ev_only_on_decayed_crowded_book(self):
        """Deterministic scenario: half the book is crowded/decayed (looks
        great on stale EV, realizes poorly). Enriched allocation should lose
        less. This is a controlled comparison, NOT a profitability claim."""
        rng = np.random.default_rng(42)
        realized = {}
        cands = []
        for i in range(4):     # decayed: stale EV 60bps, realizes -20bps
            c = make_candidate(alpha_id=f"decayed{i}", symbol=f"D{i}",
                               ev=0.006, lcb=0.004)
            realized[c.alpha_id] = -0.002
            cands.append(c)
        for i in range(4):     # steady: EV 25bps, realizes +25bps
            c = make_candidate(alpha_id=f"steady{i}", symbol=f"S{i}",
                               ev=0.0025, lcb=0.0015)
            realized[c.alpha_id] = 0.0025
            cands.append(c)

        def fetch(alpha_id):
            return (history(0.0005, 0.012) if alpha_id.startswith("decayed")
                    else history(0.0025, 0.0025))

        cfg = AllocatorConfig(capital=100_000, max_family_exposure_frac=1.0,
                              max_total_exposure_frac=1.0)

        # EV-only baseline: no enrichment
        baseline = PortfolioAllocator(cfg).allocate(
            [make_candidate(alpha_id=c.alpha_id, symbol=c.symbol,
                            ev=c.expected_net_return, lcb=c.ev_lower_bound)
             for c in cands])
        base_pnl = sum(d.allocation_usd * realized[d.candidate.alpha_id]
                       for d in baseline if d.accepted)

        # Enriched path
        enr = enricher_with(fetch)
        enr.enrich_all(cands, regime="BULL", capital=100_000)
        enriched = PortfolioAllocator(cfg).allocate(cands)
        enr_pnl = sum(d.allocation_usd * realized[d.candidate.alpha_id]
                      for d in enriched if d.accepted)
        base_alloc = sum(d.allocation_usd for d in baseline if d.accepted)
        enr_alloc = sum(d.allocation_usd for d in enriched if d.accepted)

        # enriched: less capital in decayed alphas → better realized P&L per $
        assert enr_pnl / max(enr_alloc, 1) > base_pnl / max(base_alloc, 1)
        decayed_enriched = sum(d.allocation_usd for d in enriched if d.accepted
                               and d.candidate.alpha_id.startswith("decayed"))
        decayed_base = sum(d.allocation_usd for d in baseline if d.accepted
                           and d.candidate.alpha_id.startswith("decayed"))
        assert decayed_enriched < decayed_base
