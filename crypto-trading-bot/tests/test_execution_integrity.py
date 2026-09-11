"""Execution-integrity tests (spec §67-90): the optimizer's order reaches the
broker, costs are honestly attributed, portfolio risk uses synchronized paths,
and research scale/budgets behave as configured.
"""
import math
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from core.execution_decision import (
    NO_TRADE_EV_GONE,
    NO_TRADE_EXECUTION_COST,
    NO_TRADE_SIGNAL_EXPIRED,
    USE_AS_IS,
    USE_SUPPORTED_ALTERNATIVE,
    BrokerCapabilities,
    ExecutionDecision,
    ExecutionDecisionStore,
    build_execution_decision,
    deterministic_limit_price,
    pre_submission_recheck,
    realized_execution_attribution,
    remaining_ev_sufficient,
    resolve_for_broker,
)
from core.portfolio_paths import (
    ProductionCorrelationService,
    build_return_matrix,
    downside_correlation_matrix,
    effective_cluster_count,
    portfolio_return_series,
    simulate_portfolio_paths,
    stress_correlation_matrix,
)

_now = lambda: datetime.now(timezone.utc)


def make_candidate(**over):
    from core.alpha_signal_engine import OpportunityCandidate
    base = dict(
        candidate_id="c1", alpha_id="a1", alpha_version="1", symbol="AAA",
        asset_class="stock", direction="long",
        signal_time=_now().isoformat(), execution_mode="paper",
        expected_net_return=0.003, conservative_ev=0.002, ev_lower_bound=0.002,
        raw_expected_net_return=0.004,
        recommended_order_type="limit_mid", urgency="NORMAL",
        half_life_bars=5.0, expected_slippage_bps=3.0,
        expected_total_cost_bps=8.0,
        features={"spread_pct": 0.0006, "price": 100.0,
                  "expected_cost_components": {
                      "spread_bps": 3.0, "slippage_bps": 1.0, "impact_bps": 0.5,
                      "fee_bps": 2.0, "entry_cost_bps": 4.0}},
        exit_plan={"time_stop_bars": 5},
    )
    base.update(over)
    return OpportunityCandidate(**base)


# ── §67: order type + limit price reach the broker ────────────────────────────


class TestOrderPropagation:
    def test_limit_order_reaches_paper_broker(self):
        from core.broker import PaperBroker
        broker = PaperBroker(starting_cash=100_000)
        captured = {}
        original = broker.submit_order

        def spy(order):
            captured["order_type"] = order.order_type
            captured["limit_price"] = order.limit_price
            return original(order)
        broker.submit_order = spy

        c = make_candidate()
        ed = build_execution_decision(c, allocation_usd=1_000, price=100.0)
        assert ed.order_type == "LIMIT"
        assert ed.limit_price == pytest.approx(100.0)   # limit_mid → mid
        ed = resolve_for_broker(ed, BrokerCapabilities.detect(broker))
        assert ed.resolution == USE_AS_IS
        broker.submit_and_wait(symbol=ed.symbol, side=ed.side,
                               quantity=ed.quantity, current_price=100.0,
                               order_type=ed.order_type,
                               limit_price=ed.limit_price, timeout_seconds=5)
        assert captured["order_type"] == "LIMIT"
        assert captured["limit_price"] == pytest.approx(100.0)

    def test_deterministic_limit_prices(self):
        # passive buy rests at bid; marketable buy capped past the ask
        p = deterministic_limit_price(side="BUY", bid=99.9, ask=100.1, mid=100.0,
                                      spread_pct=0.002, urgency="NORMAL",
                                      method="limit_passive")
        assert p == pytest.approx(99.9)
        m = deterministic_limit_price(side="BUY", bid=99.9, ask=100.1, mid=100.0,
                                      spread_pct=0.002, urgency="FAST",
                                      method="marketable_limit")
        assert 100.1 < m <= 100.3          # price protection: never unbounded
        assert deterministic_limit_price(
            side="SELL", bid=99.9, ask=100.1, mid=100.0, spread_pct=0.002,
            urgency="NORMAL", method="limit_passive") == pytest.approx(100.1)


# ── §68: unsupported method → re-cost, never silent MARKET ────────────────────


class TestUnsupportedFallback:
    def test_fallback_recosts_and_uses_supported(self):
        c = make_candidate(recommended_order_type="limit_passive",
                           expected_net_return=0.004)
        ed = build_execution_decision(c, allocation_usd=1_000, price=100.0)
        ed.expected_slippage_bps = 1.0
        caps = BrokerCapabilities(supports_limit_orders=False)
        ed = resolve_for_broker(ed, caps)
        assert ed.resolution == USE_SUPPORTED_ALTERNATIVE
        assert ed.method == "market"
        assert ed.detail["fallback_from"] == "limit_passive"
        assert ed.expected_slippage_bps > 1.0      # re-costed, not copied
        assert ed.net_ev < 0.004                    # cost honestly deducted

    def test_fallback_rejects_when_ev_dies(self):
        c = make_candidate(recommended_order_type="limit_passive",
                           expected_net_return=0.00005)
        ed = build_execution_decision(c, allocation_usd=1_000, price=100.0)
        ed.expected_slippage_bps = 1.5
        ed = resolve_for_broker(ed, BrokerCapabilities(supports_limit_orders=False))
        assert ed.resolution == NO_TRADE_EXECUTION_COST


# ── §69: signal expiration enforced ───────────────────────────────────────────


class TestSignalExpiration:
    def test_expired_signal_never_fills(self):
        # 2-minute half-life on minute bars; resolution 5 minutes later
        c = make_candidate(half_life_bars=2.0,
                           signal_time=(_now() - timedelta(minutes=5)).isoformat())
        ed = build_execution_decision(c, allocation_usd=1_000, price=100.0,
                                      bar_minutes=1.0)
        assert ed.signal_expiration_time is not None
        ed = resolve_for_broker(ed, BrokerCapabilities())
        assert ed.resolution == NO_TRADE_SIGNAL_EXPIRED

    def test_passive_slower_than_half_life_rejected(self):
        # 3-minute half-life; passive expected fill ~12 min → not an option
        c = make_candidate(recommended_order_type="limit_passive",
                           half_life_bars=3.0, urgency="PATIENT",
                           expected_net_return=0.004)
        ed = build_execution_decision(c, allocation_usd=1_000, price=100.0,
                                      bar_minutes=1.0)
        ed = resolve_for_broker(ed, BrokerCapabilities())
        assert ed.resolution in (USE_SUPPORTED_ALTERNATIVE,
                                 NO_TRADE_EXECUTION_COST)
        if ed.resolution == USE_SUPPORTED_ALTERNATIVE:
            assert ed.method != "limit_passive"


# ── §70: partial fills ────────────────────────────────────────────────────────


class TestPartialFills:
    def _decision(self, net_ev=0.003):
        c = make_candidate(expected_net_return=net_ev)
        return build_execution_decision(c, allocation_usd=1_000, price=100.0)

    def test_remainder_chased_only_if_ev_survives(self):
        ed = self._decision(net_ev=0.003)
        ok = remaining_ev_sufficient(ed, filled_qty=4, requested_qty=10,
                                     current_spread_pct=0.0006,
                                     reference_spread_pct=0.0006)
        assert ok
        # market worsens sharply → cancel remainder
        bad = remaining_ev_sufficient(ed, filled_qty=4, requested_qty=10,
                                      current_spread_pct=0.02,
                                      reference_spread_pct=0.0006)
        assert not bad

    def test_fully_filled_needs_no_chase(self):
        ed = self._decision()
        assert not remaining_ev_sufficient(ed, filled_qty=10, requested_qty=10,
                                           current_spread_pct=0.0006,
                                           reference_spread_pct=0.0006)


# ── §71: execution cost components stay separate ──────────────────────────────


class TestCostComponents:
    def test_predicted_components_distinct(self):
        from core.execution_optimizer import ExecutionOptimizer
        plan = ExecutionOptimizer(fee_bps=2.0).cost_of_method(
            "market", spread_pct=0.001, order_notional=50_000, adv_usd=1e7,
            volatility_daily=0.02)
        comps = plan.detail["components"]
        assert set(comps) == {"spread_bps", "slippage_bps", "impact_bps", "fee_bps"}
        assert comps["spread_bps"] > 0 and comps["impact_bps"] > 0
        assert comps["fee_bps"] == 2.0
        total = sum(comps.values())
        assert total == pytest.approx(plan.expected_cost_bps, rel=0.01)

    def test_realized_attribution_separates_components(self):
        c = make_candidate()
        ed = build_execution_decision(c, allocation_usd=10_000, price=100.0)
        realized = realized_execution_attribution(
            ed, fill_price=100.06, mid_at_decision=100.0,
            fees_usd=2.0, notional_usd=10_000)
        assert realized["realized_spread_bps"] <= ed.expected_spread_bps + 1e-9
        assert realized["realized_fee_bps"] == pytest.approx(2.0)
        assert "realized_slippage_bps" in realized
        assert "realized_impact_bps" in realized
        total = realized["realized_total_cost_bps"]
        assert total == pytest.approx(6.0 + 2.0, rel=0.01)   # 6bps adverse + fee

    def test_decision_persisted_with_realized(self, tmp_path):
        store = ExecutionDecisionStore(str(tmp_path / "ed.sqlite"))
        c = make_candidate()
        ed = build_execution_decision(c, allocation_usd=1_000, price=100.0)
        rid = store.record(ed, realized={"realized_total_cost_bps": 7.5})
        assert rid > 0


# ── §72-73: synchronized portfolio vs concatenation ───────────────────────────


class TestSynchronizedPortfolio:
    def test_simultaneous_losers_show_higher_ruin(self):
        rng = np.random.default_rng(0)
        shock = rng.normal(0, 0.03, 120)
        days = [f"2026-01-{i % 28 + 1:02d}x{i}" for i in range(120)]
        a = {days[i]: 0.001 + shock[i] for i in range(120)}
        b = {days[i]: 0.001 + shock[i] for i in range(120)}     # loses TOGETHER
        _, matrix = build_return_matrix({"A": a, "B": b})
        sync_series = portfolio_return_series(matrix, {"A": 0.5, "B": 0.5})
        sync = simulate_portfolio_paths(sync_series, seed=1)
        # concatenation (the OLD wrong way) halves apparent volatility
        concat = simulate_portfolio_paths(
            [0.5 * r for r in list(a.values()) + list(b.values())], seed=1)
        assert sync["max_drawdown_p95"] > concat["max_drawdown_p95"]
        assert sync["risk_of_ruin"]["0.5"] >= concat["risk_of_ruin"]["0.5"]

    def test_negatively_correlated_alphas_reduce_ruin(self):
        rng = np.random.default_rng(1)
        shock = rng.normal(0, 0.04, 120)
        days = [f"t{i}" for i in range(120)]
        a = {days[i]: 0.002 + shock[i] for i in range(120)}
        hedge = {days[i]: 0.002 - shock[i] for i in range(120)}
        solo = simulate_portfolio_paths(
            portfolio_return_series(build_return_matrix({"A": a})[1], {"A": 1.0}),
            seed=2)
        _, m2 = build_return_matrix({"A": a, "H": hedge})
        hedged = simulate_portfolio_paths(
            portfolio_return_series(m2, {"A": 0.5, "H": 0.5}), seed=2)
        assert hedged["risk_of_ruin"]["0.5"] < solo["risk_of_ruin"]["0.5"]
        assert hedged["max_drawdown_p95"] < solo["max_drawdown_p95"]

    def test_inactive_alpha_contributes_zero(self):
        _, matrix = build_return_matrix(
            {"A": {"t1": 0.01, "t2": -0.01}, "B": {"t2": 0.005}})
        assert matrix["B"] == [0.0, 0.005]          # documented policy

    def test_time_varying_weights(self):
        _, m = build_return_matrix({"A": {"t1": 0.01, "t2": 0.01}})
        series = portfolio_return_series(
            m, weight_history=[{"A": 1.0}, {"A": 0.5}])
        assert series == [0.01, 0.005]


# ── §74-75: downside + stress correlation ─────────────────────────────────────


class TestDownsideStressCorrelation:
    def test_hidden_downside_correlation_detected(self):
        rng = np.random.default_rng(3)
        n = 300
        crash = rng.random(n) < 0.1                 # shared crash days
        days = [f"t{i}" for i in range(n)]
        mk = lambda seed: {
            days[i]: (-0.03 if crash[i] else float(np.random.default_rng(seed + i)
                                                   .normal(0.002, 0.01)))
            for i in range(n)}
        _, matrix = build_return_matrix({"A": mk(10), "B": mk(9000)})
        down = downside_correlation_matrix(matrix)
        from core.strategy_correlation import _pearson
        overall = _pearson(matrix["A"], matrix["B"])
        assert down[("A", "B")] > overall           # tail dependence exposed
        assert down[("A", "B")] > 0.5

    def test_stress_correlation_with_mask(self):
        rng = np.random.default_rng(4)
        n = 200
        stress = [i % 20 == 0 for i in range(n)]
        days = [f"t{i}" for i in range(n)]
        a = {days[i]: (-0.05 if stress[i] else float(rng.normal(0, 0.01)))
             for i in range(n)}
        b = {days[i]: (-0.04 if stress[i] else float(rng.normal(0, 0.01)))
             for i in range(n)}
        _, matrix = build_return_matrix({"A": a, "B": b})
        sc = stress_correlation_matrix(matrix, stress_mask=stress)
        assert sc.get(("A", "B"), 0) > 0.4

    def test_effective_clusters_vs_nominal(self):
        rng = np.random.default_rng(5)
        base = rng.normal(0.001, 0.01, 150)
        days = [f"t{i}" for i in range(150)]
        alphas = {f"m{k}": {days[i]: float(base[i] + rng.normal(0, 0.001))
                            for i in range(150)} for k in range(4)}
        alphas["indep"] = {days[i]: float(rng.normal(0.001, 0.01))
                           for i in range(150)}
        _, matrix = build_return_matrix(alphas)
        res = effective_cluster_count(matrix)
        assert res["nominal_alphas"] == 5
        assert res["effective_clusters"] == 2       # 4 clones + 1 independent


# ── §76-77: dollar alpha + capital time in ranking ────────────────────────────


class TestEconomicRanking:
    def test_scalable_dollar_alpha_boosts_score(self):
        from core.portfolio_allocator import PortfolioAllocator
        a = make_candidate(candidate_id="cA", alpha_id="A", symbol="AAA",
                           expected_net_return=0.006, ev_lower_bound=0.002,
                           conservative_ev=0.002)
        a.expected_dollar_alpha = 0.006 * 5_000            # tiny capacity
        b = make_candidate(candidate_id="cB", alpha_id="B", symbol="BBB",
                           expected_net_return=0.002, ev_lower_bound=0.002,
                           conservative_ev=0.002)
        b.expected_dollar_alpha = 0.002 * 500_000          # deep capacity
        decisions = PortfolioAllocator().allocate([a, b])
        assert all(d.accepted for d in decisions)
        # same lower bound → B's bounded dollar-alpha boost wins the ranking
        assert b.final_opportunity_score > a.final_opportunity_score
        # …but the boost is bounded: it cannot dominate a big EV gap
        assert b.final_opportunity_score < a.final_opportunity_score * 1.5


# ── §78-79: empirical half-life ───────────────────────────────────────────────


class TestEmpiricalHalfLife:
    def _decay_data(self, n=40):
        # signal peaks at 10m, clearly below half-peak by 30m, gone by 90m
        curve = {1: 0.0004, 5: 0.0009, 10: 0.0012, 30: 0.0005,
                 60: 0.0002, 90: 0.00003}
        rng = np.random.default_rng(6)
        return {h: list(m + rng.normal(0, m * 0.1, n))
                for h, m in curve.items()}

    def test_half_life_estimated_from_decay(self):
        from core.opportunity_economics import empirical_decay_curve
        est = empirical_decay_curve(self._decay_data())
        assert est is not None
        assert est["peak_time"] == 10
        assert est["half_life"] == 30
        assert est["decay_95"] == 90
        assert est["n_signals"] == 40
        assert est["stable"]

    def test_empirical_overrides_planned_holding(self):
        from core.opportunity_enrichment import OpportunityEnricher
        est = {"half_life": 0.5, "stable": True, "ci_low": 0.3, "ci_high": 1.0}
        enr = OpportunityEnricher(
            returns_fetcher=lambda a: {"recent": [0.003] * 20,
                                       "older": [0.003] * 20, "by_regime": []},
            half_life_provider=lambda a: est, db_path=":memory:")
        c = make_candidate(exit_plan={"time_stop_bars": 6 * 60})  # planned 6h
        enr.enrich(c, regime="BULL", capital=10_000, base_position_frac=0.05)
        assert c.half_life_bars == 0.5                  # empirical wins
        assert c.features["half_life_source"] == "EMPIRICAL"

    def test_fallback_is_marked_not_pretended(self):
        from core.opportunity_enrichment import OpportunityEnricher
        enr = OpportunityEnricher(
            returns_fetcher=lambda a: {"recent": [], "older": [], "by_regime": []},
            half_life_provider=lambda a: None, db_path=":memory:")
        c = make_candidate()
        enr.enrich(c, regime="BULL", capital=10_000, base_position_frac=0.05)
        assert c.features["half_life_source"] == "HALF_LIFE_FALLBACK"

    def test_tiny_sample_rejected(self):
        from core.opportunity_economics import empirical_decay_curve
        assert empirical_decay_curve({1: [0.01] * 3, 5: [0.02] * 3}) is None

    def test_half_life_store_roundtrip(self, tmp_path):
        from core.opportunity_economics import HalfLifeStore, empirical_decay_curve
        store = HalfLifeStore(str(tmp_path / "hl.sqlite"))
        est = empirical_decay_curve(self._decay_data())
        store.save("alpha_x", est)
        loaded = store.get("alpha_x")
        assert loaded["half_life"] == 30 and loaded["source"] == "EMPIRICAL"


# ── §80: weekly research has no hidden truncation ─────────────────────────────


class TestResearchScale:
    def test_zero_max_means_full_universe(self):
        import inspect

        from ultimate_bot_v3_llm import LLMTradingBot
        src = inspect.getsource(LLMTradingBot.run_weekly_research_campaign)
        assert "if max_instruments and max_instruments > 0" in src
        assert "[:12]" not in src and "[:8]" not in src   # old hidden slices gone

    def test_budgets_persist_across_campaigns(self, tmp_path):
        from core.research_orchestrator import AutonomousResearchOrchestrator
        db = str(tmp_path / "budget.sqlite")
        orch1 = AutonomousResearchOrchestrator(db_path=db)
        orch1.apply_research_gaps(
            {"bull_mom": [("BULL", 0.01)] * 20 + [("BEAR", -0.01)] * 20},
            alpha_families={"bull_mom": "MOMENTUM"},
            alpha_directions={"bull_mom": "long"})
        # a NEW orchestrator (next campaign) loads the persisted budgets
        orch2 = AutonomousResearchOrchestrator(db_path=db)
        loaded = orch2._load_research_budgets()
        base = orch2.config.generator_budget.max_hypotheses_per_family
        assert loaded.get("MARKET_NEUTRAL", 0) > base


# ── §83-84: meta shadow persistence + promotion ───────────────────────────────


class TestMetaShadow:
    def test_shadow_predictions_persist_and_resolve(self, tmp_path):
        from core.meta_alpha import MetaShadowStore
        store = MetaShadowStore(str(tmp_path / "meta.sqlite"))
        pid = store.record_prediction("a1", "BULL", predicted_ev=0.004,
                                      confidence=0.6, rank=1)
        store.resolve(pid, realized_return=0.005)
        pairs = store.resolved_pairs()
        assert pairs == [(0.004, 0.005)]

    def test_promotion_uses_persisted_evidence(self, tmp_path):
        from core.meta_alpha import MetaShadowStore
        from core.opportunity_enrichment import meta_alpha_promotion_ready
        store = MetaShadowStore(str(tmp_path / "meta.sqlite"))
        for _ in range(40):
            pid = store.record_prediction("a1", "BULL", 0.004, 0.6)
            store.resolve(pid, 0.0045)
        assert meta_alpha_promotion_ready(store.resolved_pairs())["ready"]


# ── §85: realized net return drives edge health ───────────────────────────────


class TestRealizedNetDrivesHealth:
    def test_execution_destroyed_edge_reflected(self):
        from core.meta_alpha import EdgeSurvivalModel
        # predicted +12 bps; realized NET after execution −3 bps
        realized_net = [-0.0003] * 30
        predicted_would_be = [0.0012] * 30
        model = EdgeSurvivalModel()
        honest = model.assess("a", recent_returns=realized_net,
                              older_returns=[0.0012] * 30)
        dishonest = model.assess("a", recent_returns=predicted_would_be,
                                 older_returns=[0.0012] * 30)
        assert honest.recommended_state in ("DEGRADED", "PAUSED")
        assert dishonest.recommended_state == "HEALTHY"
        assert honest.capital_multiplier < dishonest.capital_multiplier


# ── §86: pre-submission recheck cancels vanished EV ───────────────────────────


class TestPreSubmissionRecheck:
    def test_spread_widening_cancels_order(self):
        c = make_candidate(expected_net_return=0.0025)   # +25 bps
        ed = build_execution_decision(c, allocation_usd=1_000, price=100.0)
        ed = pre_submission_recheck(ed, current_spread_pct=0.0035,
                                    reference_spread_pct=0.0006)
        assert ed.resolution == NO_TRADE_EV_GONE        # 29bps widening kills it

    def test_stable_spread_keeps_order(self):
        c = make_candidate(expected_net_return=0.0025)
        ed = build_execution_decision(c, allocation_usd=1_000, price=100.0)
        ed = pre_submission_recheck(ed, current_spread_pct=0.0006,
                                    reference_spread_pct=0.0006)
        assert ed.resolution == USE_AS_IS


# ── §87: full paper end-to-end through the REAL broker ────────────────────────


class TestPaperEndToEnd:
    def test_enriched_candidate_to_paper_fill_and_attribution(self, tmp_path):
        from core.broker import PaperBroker
        from core.execution_optimizer import ExecutionLearner
        db = str(tmp_path / f"e2e_{uuid.uuid4().hex}.sqlite")
        broker = PaperBroker(starting_cash=100_000)
        c = make_candidate()

        ed = build_execution_decision(c, allocation_usd=2_000, price=100.0)
        ed = resolve_for_broker(ed, BrokerCapabilities.detect(broker))
        assert not ed.resolution.startswith("NO_TRADE")
        fill = broker.submit_and_wait(
            symbol=ed.symbol, side=ed.side, quantity=ed.quantity,
            current_price=100.0, order_type=ed.order_type,
            limit_price=ed.limit_price, timeout_seconds=5)
        assert fill.status.value in ("FILLED", "PARTIAL")

        realized = realized_execution_attribution(
            ed, fill_price=fill.fill_price, mid_at_decision=100.0,
            fees_usd=fill.fees, notional_usd=max(fill.notional, 1.0))
        store = ExecutionDecisionStore(db)
        rid = store.record(ed, realized=realized)
        assert rid > 0
        learner = ExecutionLearner(db_path=db, min_samples=1)
        learner.record(symbol=ed.symbol, method=ed.method,
                       predicted_slippage_bps=ed.expected_slippage_bps,
                       actual_slippage_bps=realized.get("realized_slippage_bps", 0))
        assert learner.method_cost_table()

    def test_correlation_service_populates_allocator(self, tmp_path):
        from core.portfolio_allocator import PortfolioAllocator
        db = str(tmp_path / "corr.sqlite")
        svc = ProductionCorrelationService(db)
        alloc = PortfolioAllocator()
        svc.populate_allocator(alloc)      # empty DB → empty matrices, no crash
        assert alloc.downside_correlations == {}


# ── §88: portfolio model comparison (old vs synchronized) ─────────────────────


class TestPortfolioModelComparison:
    def test_synchronized_model_reports_honest_risk(self):
        rng = np.random.default_rng(7)
        shock = rng.normal(0, 0.025, 150)
        days = [f"t{i}" for i in range(150)]
        alphas = {f"a{k}": {days[i]: 0.001 + shock[i] for i in range(150)}
                  for k in range(3)}
        _, matrix = build_return_matrix(alphas)
        w = {a: 1 / 3 for a in alphas}
        new = simulate_portfolio_paths(portfolio_return_series(matrix, w), seed=3)
        pooled = [r / 3 for col in matrix.values() for r in col]   # old way
        old = simulate_portfolio_paths(pooled, seed=3)
        report = {
            "old": {"ruin": old["risk_of_ruin"]["0.5"],
                    "dd_p95": old["max_drawdown_p95"],
                    "es": old["expected_shortfall_5pct"]},
            "new": {"ruin": new["risk_of_ruin"]["0.5"],
                    "dd_p95": new["max_drawdown_p95"],
                    "es": new["expected_shortfall_5pct"],
                    "log_growth": new["expected_log_growth"]},
        }
        # perfectly correlated book: old concatenation UNDERSTATES risk
        assert report["new"]["dd_p95"] > report["old"]["dd_p95"]
        assert new["reproducibility"]["seed"] == 3   # reproducible (spec §94)
