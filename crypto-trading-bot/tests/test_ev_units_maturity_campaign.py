"""Spec-mandated tests: EV units (cases A-G), spread semantics, pipeline
maturity (cases A-D), and the synthetic research campaign end-to-end."""
import json
import sqlite3
import uuid

import numpy as np
import pandas as pd
import pytest

from core.alpha_library import AlphaLibrary, AlphaState
from core.ev_model import EconomicEVModel
from core.pipeline_maturity import (
    MaturityEvidence,
    MaturityState,
    MaturityThresholds,
    PipelineMaturityEvaluator,
    ev_calibration_buckets,
    probability_calibration_error,
    ranking_quality,
)
from core.research_campaign import CampaignConfig, ResearchCampaignRunner
from core.return_units import to_bps, to_fractional_return, to_percent, validate_return_units


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / f"s_{uuid.uuid4().hex}.sqlite")


def _register_alpha_with_oos(db, alpha_id, mean_net_return, se_return,
                             win_rate=0.58, trades=40):
    lib = AlphaLibrary(db)
    lib.register(alpha_id, alpha_id, universe=["TEST-USD"], direction="long")
    lib.update_evidence(alpha_id, oos_metrics={
        "mean_net_return": mean_net_return,
        "standard_error_return": se_return,
        "win_rate": win_rate, "trades": trades,
    })
    for s in (AlphaState.VALIDATING, AlphaState.PAPER):
        lib.transition(alpha_id, s)
    return lib


def _add_forward(db, alpha_id, returns_pct):
    from core.trade_attribution import TradeAttributionStore
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS trade_memory (id INTEGER PRIMARY KEY "
            "AUTOINCREMENT, entry_time TEXT, exit_time TEXT, net_return_pct REAL, "
            "gross_pnl REAL, total_fees REAL, size_dollars REAL, mae_pct REAL, "
            "mfe_pct REAL, holding_hours REAL, close_reason TEXT)"
        )
        store = TradeAttributionStore(db)
        for i, r in enumerate(returns_pct):
            cur = conn.execute(
                "INSERT INTO trade_memory (entry_time, exit_time, net_return_pct, "
                "size_dollars, holding_hours) VALUES (?,?,?,?,?)",
                (f"2026-0{(i % 8) + 1}-{(i % 27) + 1:02d}T00:00:00",
                 f"2026-0{(i % 8) + 1}-{(i % 27) + 1:02d}T04:00:00", r, 1000.0, 4.0),
            )
            conn.commit()
            store.record(alpha_id, trade_memory_id=cur.lastrowid)


# ── §61 Cold-start EV cases ───────────────────────────────────────────────────


class TestColdStartCases:
    def test_case_a_units_consistent(self, db):
        """OOS mean=0.008, SE=0.002, 50% haircut → prior ≈ 0.004 with
        uncertainty in the SAME fractional units (never 0.004 − 1.645×2.0)."""
        lib = _register_alpha_with_oos(db, "a", 0.008, 0.002)
        est = EconomicEVModel(db).estimate_with_prior(lib.get("a"))
        assert est.expected_net_return == pytest.approx(0.004, abs=1e-9)
        # SE = sqrt(0.002² + (0.004/2)²) ≈ 0.00283 — same units as the mean
        assert est.prediction_std == pytest.approx(0.002828, abs=1e-4)
        assert est.ev_lower_bound == pytest.approx(
            0.004 - 1.645 * est.prediction_std, abs=1e-9)
        assert -0.01 < est.ev_lower_bound < 0.004   # sane band, not −3.28

    def test_case_b_dollar_pnl_normalized_by_actual_notional(self):
        """Per-trade returns use each trade's OWN notional."""
        from core.strategy_experiment_runner import BacktestMetrics, SimulatedTrade

        def trade(net_pnl, notional):
            return SimulatedTrade(
                symbol="X", direction="long", entry_index=0, exit_index=1,
                entry_price=100.0, exit_price=101.0, gross_pnl=net_pnl,
                cost=0.0, net_pnl=net_pnl, exit_reason="t", notional=notional)

        trades = [trade(10.0, 1000.0), trade(10.0, 500.0), trade(-20.0, 2000.0)]
        m = BacktestMetrics.from_trades(trades)
        expected_returns = [0.01, 0.02, -0.01]
        assert m.mean_net_return == pytest.approx(sum(expected_returns) / 3)
        assert m.net_expectancy_return == m.mean_net_return
        assert m.standard_error_return > 0

    def test_case_c_three_poor_forward_trades_do_not_flip_prior(self, db):
        lib = _register_alpha_with_oos(db, "c", 0.005, 0.001, trades=500)
        _add_forward(db, "c", [-1.0, -1.2, -0.8])
        est = EconomicEVModel(db, min_samples=3).estimate_with_prior(
            lib.get("c"), prior_strength=20)
        # historical prior dominates: EV must remain positive
        assert est.expected_net_return > 0
        assert est.historical_weight > est.forward_weight

    def test_case_d_100_poor_forward_trades_drive_ev_negative(self, db):
        lib = _register_alpha_with_oos(db, "d", 0.0025, 0.0008, trades=200)
        rng = np.random.default_rng(4)
        _add_forward(db, "d", list(rng.normal(-0.18, 0.3, 100)))  # −0.18% mean
        est = EconomicEVModel(db, min_samples=8).estimate_with_prior(
            lib.get("d"), prior_strength=20)
        assert est.expected_net_return < 0
        assert est.forward_weight > est.historical_weight

    def test_case_e_100_strong_forward_trades_dominate_positively(self, db):
        lib = _register_alpha_with_oos(db, "e", 0.001, 0.0008)
        rng = np.random.default_rng(5)
        _add_forward(db, "e", list(rng.normal(1.5, 0.5, 100)))   # +1.5% mean
        est = EconomicEVModel(db, min_samples=8).estimate_with_prior(
            lib.get("e"), prior_strength=20)
        assert est.expected_net_return > 0.005
        assert est.forward_weight > 0.7

    def test_case_f_negative_lower_bound_means_no_trade(self, db):
        lib = _register_alpha_with_oos(db, "f", 0.002, 0.005)  # huge uncertainty
        est = EconomicEVModel(db).estimate_with_prior(lib.get("f"))
        assert est.expected_net_return > 0
        assert est.ev_lower_bound < 0
        from core.alpha_signal_engine import OpportunityCandidate
        from core.portfolio_allocator import PortfolioAllocator
        c = OpportunityCandidate(
            candidate_id="x", alpha_id="f", alpha_version="1", symbol="TEST-USD",
            asset_class="crypto", direction="long", signal_time="now",
            execution_mode="paper", expected_net_return=est.expected_net_return,
            ev_lower_bound=est.ev_lower_bound)
        decisions = PortfolioAllocator().allocate([c])
        assert not any(d.accepted for d in decisions)

    def test_case_g_positive_ev_and_lower_bound_passes(self, db):
        lib = _register_alpha_with_oos(db, "g", 0.02, 0.001, trades=100)
        est = EconomicEVModel(db).estimate_with_prior(lib.get("g"))
        assert est.expected_net_return > 0 and est.ev_lower_bound > 0
        from core.alpha_signal_engine import OpportunityCandidate
        from core.portfolio_allocator import PortfolioAllocator
        c = OpportunityCandidate(
            candidate_id="y", alpha_id="g", alpha_version="1", symbol="TEST-USD",
            asset_class="crypto", direction="long", signal_time="now",
            execution_mode="paper", expected_net_return=est.expected_net_return,
            ev_lower_bound=est.ev_lower_bound, execution_feasibility=90.0)
        decisions = PortfolioAllocator().allocate([c])
        assert any(d.accepted for d in decisions)

    def test_no_hardcoded_position_size_remains(self):
        import inspect
        import core.ev_model as m
        src = inspect.getsource(m)
        assert "_OOS_POSITION_SIZE_USD" not in src
        assert "/ 1000" not in src

    def test_unit_helpers_and_anomaly_detection(self):
        assert to_fractional_return(1.0, "percent") == 0.01
        assert to_fractional_return(25, "bps") == 0.0025
        assert to_percent(0.01) == 1.0
        assert to_bps(0.0025) == 25
        assert validate_return_units(0.01, "x", "stock")
        # +400% on a liquid equity → RETURN_UNIT_ANOMALY
        assert not validate_return_units(4.0, "expected_return", "stock")
        # short-hold ceiling tightens
        assert not validate_return_units(0.3, "x", "crypto", holding_hours=4)


# ── §62 Spread semantics ──────────────────────────────────────────────────────


class TestSpreadSemantics:
    def test_quote_spread_computed_from_bid_ask(self):
        from core.transaction_costs import TransactionCostModel
        est = TransactionCostModel("binance").estimate_cost(
            "X", "buy", 1.0, 100.05,
            market_state={"bid": 100.0, "ask": 100.10})
        # spread = 0.10 / 100.05; half-spread paid per side
        expected = (0.10 / 100.05) / 2 * 100.05
        assert est["spread_cost"] == pytest.approx(expected, rel=1e-6)
        assert est["spread_source"] == "QUOTE"

    def test_bar_range_never_becomes_spread(self):
        """high=105 low=95 must NOT produce a 10% bid/ask spread anywhere."""
        from core.transaction_costs import TransactionCostModel
        est = TransactionCostModel("binance").estimate_cost(
            "X", "buy", 1.0, 100.0,
            market_state={"high": 105.0, "low": 95.0})  # no quotes given
        assert est["spread_source"] == "FALLBACK"
        assert est["estimated_spread_bps"] == pytest.approx(8.0)  # labeled fallback
        assert est["estimated_spread_bps"] < 100  # not 1000 bps from bar range

    def test_candidate_spread_field_is_none_without_quotes(self):
        """Bot candidates: spread_pct=None, bar range under its own name."""
        import inspect
        import ultimate_bot_v3_llm as m
        src = inspect.getsource(m.LLMTradingBot.analyze_trade_opportunity)
        assert "'spread_pct': None" in src
        assert "'bar_range_pct'" in src

    def test_universe_scanner_has_no_spread_field(self):
        from core.universe_scanner import Candidate, ScannerConfig
        assert "bar_range_pct" in Candidate.__dataclass_fields__
        assert "spread_pct" not in Candidate.__dataclass_fields__
        assert hasattr(ScannerConfig(), "max_bar_range_pct")


# ── §63 Pipeline maturity cases ───────────────────────────────────────────────


class TestPipelineMaturityCases:
    def _assess(self, **kwargs):
        ev = MaturityEvidence(**kwargs)
        return PipelineMaturityEvaluator(":memory:").assess(ev)

    def test_case_a_poor_calibration_not_mature(self):
        a = self._assess(days_observed=14, decisions=100, resolved_outcomes=80,
                         effective_sample_size=60,
                         ev_calibration_error=0.02,   # terrible calibration
                         forward_net_expectancy=0.003)
        assert a.state != MaturityState.MATURE
        assert a.legacy_fallback_allowed

    def test_case_b_few_resolved_outcomes_not_mature(self):
        a = self._assess(days_observed=14, decisions=100, resolved_outcomes=5,
                         effective_sample_size=4)
        assert a.state != MaturityState.MATURE

    def test_case_c_full_evidence_mature(self):
        a = self._assess(days_observed=30, decisions=200, resolved_outcomes=120,
                         effective_sample_size=90,
                         ev_calibration_error=0.001,
                         probability_calibration_error=0.05,
                         cost_prediction_error=0.2,
                         forward_net_expectancy=0.002,
                         ranking_quality=0.001,
                         drawdown_vs_expected=1.2,
                         execution_success_rate=0.98)
        assert a.state == MaturityState.MATURE
        assert not a.legacy_fallback_allowed
        assert "EXCLUSIVE MODE ELIGIBLE" in a.report()

    def test_case_d_degradation_detected(self):
        a = self._assess(days_observed=60, decisions=400, resolved_outcomes=150,
                         effective_sample_size=100,
                         ev_calibration_error=0.001,
                         forward_net_expectancy=-0.004)   # sustained losses
        assert a.state == MaturityState.DEGRADED
        assert a.legacy_fallback_allowed   # fallback/safety behavior restored

    def test_profit_alone_does_not_mature(self):
        # Positive expectancy but no calibration data → not mature
        a = self._assess(days_observed=30, decisions=200, resolved_outcomes=80,
                         effective_sample_size=60,
                         forward_net_expectancy=0.01,
                         ev_calibration_error=None)
        assert a.state != MaturityState.MATURE

    def test_calibration_buckets_and_ranking(self):
        pairs = [(0.0005, 0.0004)] * 30 + [(0.003, 0.0031)] * 30
        buckets = ev_calibration_buckets(pairs)
        assert len(buckets) == 2
        assert all(b["error"] < 0.001 for b in buckets)
        p_err = probability_calibration_error([(0.7, True)] * 7 + [(0.7, False)] * 3)
        assert p_err == pytest.approx(0.0, abs=1e-9)
        rq = ranking_quality([(i, 0.01 - 0.001 * i) for i in range(1, 21)])
        assert rq > 0   # top ranks outperform


# ── §64 Synthetic research campaign end-to-end ────────────────────────────────


def _ohlcv(rets, seed_vol=1e6, start="2025-01-01"):
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range(start, periods=len(rets), freq="1D")
    rng = np.random.default_rng(1)
    return pd.DataFrame({"open": close, "high": close * 1.005,
                         "low": close * 0.995, "close": close,
                         "volume": rng.uniform(0.9, 1.1, len(rets)) * seed_vol},
                        index=idx)


def _make_universe(seed=42, n=700):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1D")
    weekdays = idx.strftime("%A").str.upper()

    # TRUE edge: strong persistent Thursday effect across full history
    true_rets = rng.normal(0.0, 0.004, n)
    true_rets[weekdays == "THURSDAY"] += 0.02

    # Pure noise
    noise_rets = rng.normal(0.0, 0.004, n)

    # Cost-sensitive fake edge: real but tiny Thursday effect (< costs)
    tiny_rets = rng.normal(0.0, 0.0008, n)
    tiny_rets[weekdays == "THURSDAY"] += 0.0012   # 12 bps < 20 bps cost

    # Single-outlier edge: flat, one enormous Thursday win
    outlier_rets = rng.normal(0.0, 0.002, n)
    thursday_idx = np.where(weekdays == "THURSDAY")[0]
    outlier_rets[thursday_idx[5]] += 1.0

    return {
        "TRUE-USD": _ohlcv(true_rets),
        "NOISE-USD": _ohlcv(noise_rets),
        "TINY-USD": _ohlcv(tiny_rets),
        "OUTLIER-USD": _ohlcv(outlier_rets),
    }


class TestResearchCampaign:
    def test_full_campaign_funnel(self, db):
        runner = ResearchCampaignRunner(db_path=db)
        report = runner.run(_make_universe())

        # Funnel integrity
        assert report["hypotheses_tested"] > 0
        assert report["preliminary_survivors"] <= report["after_dedup"]
        assert report["fdr_survivors"] <= report["preliminary_survivors"]
        assert report["full_validation_survivors"] <= report["fdr_survivors"]
        assert report["survivorship_bias_risk"] is True   # honest flag

        # TRUE synthetic edge promoted
        new_alphas = report["new_alphas"]
        assert any("TRUE-USD" in a and "day_of_week" in a and "long" in a
                   for a in new_alphas), f"true edge not promoted: {new_alphas}"

        # Noise / cost-sensitive / outlier edges rejected
        assert not any("NOISE-USD" in a for a in new_alphas)
        assert not any("TINY-USD" in a for a in new_alphas)
        assert not any("OUTLIER-USD" in a and "day_of_week" in a for a in new_alphas)
        rejections = report["rejections"]
        assert rejections.get("FAILED_FDR", 0) + rejections.get("FAILED_OOS", 0) \
            + rejections.get("LOW_EFFECT_SIZE", 0) \
            + rejections.get("COST_SENSITIVE", 0) \
            + rejections.get("OUTLIER_DEPENDENT", 0) \
            + rejections.get("NEAR_DUPLICATE", 0) > 0

    def test_promoted_alpha_is_paper_never_live(self, db):
        runner = ResearchCampaignRunner(db_path=db)
        report = runner.run(_make_universe())
        lib = AlphaLibrary(db)
        for alpha_id in report["new_alphas"]:
            assert lib.state_of(alpha_id) == AlphaState.PAPER
            assert not lib.is_live_approved(alpha_id)

    def test_promoted_alpha_has_normalized_oos_units(self, db):
        runner = ResearchCampaignRunner(db_path=db)
        report = runner.run(_make_universe())
        lib = AlphaLibrary(db)
        assert report["new_alphas"]
        record = lib.get(report["new_alphas"][0])
        oos = record["oos_metrics"]
        assert "mean_net_return" in oos and "standard_error_return" in oos
        assert abs(oos["mean_net_return"]) < 0.5   # fractional, not $ or %
        # cold-start prior now works off this alpha directly
        est = EconomicEVModel(db).estimate_with_prior(record)
        assert est is not None and est.source == "oos_prior"
        assert est.ev_lower_bound > -0.05

    def test_graveyard_prevents_rediscovery(self, db):
        runner = ResearchCampaignRunner(db_path=db)
        first = runner.run(_make_universe())
        second = runner.run(_make_universe())
        # rejected signatures skipped on the second pass
        assert second["rejections"].get("GRAVEYARD_DUPLICATE", 0) > 0

    def test_campaign_report_formatting(self, db):
        runner = ResearchCampaignRunner(db_path=db)
        report = runner.run(_make_universe())
        text = ResearchCampaignRunner.format_report(report)
        assert "RESEARCH CAMPAIGN" in text
        assert "FDR survivors" in text
        assert "SURVIVORSHIP_BIAS_RISK" in text
