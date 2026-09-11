"""
Tests for Phase 7, 8, 15, 16, 25 modules:
    ScenarioEngine, RealisticExecutionSimulator,
    PerformanceAttributionEngine, CalibrationAnalyzer,
    LiveCapitalGraduation
"""
from __future__ import annotations

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone

_utcnow = lambda: datetime.now(timezone.utc)


def make_trades(n=50, win_rate=0.55, seed=0):
    from validation.engine import Trade
    rng  = np.random.default_rng(seed)
    t0   = _utcnow() - timedelta(days=60)
    trades = []
    for i in range(n):
        entry = t0 + timedelta(hours=i * 3)
        exit_ = entry + timedelta(hours=2)
        win   = rng.random() < win_rate
        pg    = float(rng.uniform(20, 80) if win else -rng.uniform(10, 50))
        fee   = abs(pg) * 0.001
        slip  = abs(pg) * 0.0005
        trades.append(Trade(
            entry_time=entry, exit_time=exit_,
            pnl_gross=pg, pnl_net=pg - fee - slip,
            size=100.0, fees=fee, slippage=slip,
            symbol='BTC/USD', strategy='test_strat', direction='LONG',
            regime='bull',
        ))
    return trades


# ══════════════════════════════════════════════════════════════════════════════
# ScenarioEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestScenarioEngine:

    def test_run_returns_report(self):
        from validation.scenario_engine import ScenarioEngine, BUILTIN_SCENARIOS
        trades  = make_trades(60, win_rate=0.60)
        engine  = ScenarioEngine(capital=10_000)
        report  = engine.run(trades, label="test")
        assert report.label == "test"
        assert len(report.results) > 0

    def test_scenario_count(self):
        from validation.scenario_engine import ScenarioEngine
        trades  = make_trades(60)
        engine  = ScenarioEngine(capital=10_000)
        report  = engine.run(trades, label="test")
        assert len(report.results) == len(engine.scenarios)

    def test_zero_edge_scenario(self):
        from validation.scenario_engine import ScenarioEngine, ScenarioType
        trades = make_trades(60, win_rate=0.65)
        engine = ScenarioEngine(capital=10_000)
        result = engine.run_single(trades, ScenarioType.ZERO_EDGE, label="test")
        assert result is not None
        # Zero edge: signal removed, only costs remain → pnl should be near 0 or negative
        assert result.scenario_result.total_pnl_net <= result.original_result.total_pnl_net + 1

    def test_broker_outage_reduces_trades(self):
        from validation.scenario_engine import ScenarioEngine, ScenarioType
        trades = make_trades(100, win_rate=0.60, seed=5)
        engine = ScenarioEngine(capital=10_000)
        result = engine.run_single(trades, ScenarioType.BROKER_OUTAGE_50PCT, label="test")
        # With 50% fill rate, should have fewer trades
        assert result.n_scenario_trades <= result.n_original_trades

    def test_survival_computed(self):
        from validation.scenario_engine import ScenarioEngine
        trades = make_trades(80, win_rate=0.60)
        engine = ScenarioEngine(capital=10_000)
        report = engine.run(trades, label="test")
        for r in report.results:
            assert isinstance(r.survived, bool)

    def test_summary_output(self):
        from validation.scenario_engine import ScenarioEngine
        trades = make_trades(50)
        engine = ScenarioEngine(capital=10_000)
        report = engine.run(trades, label="test")
        s = report.summary()
        assert "SCENARIO STRESS TEST" in s

    def test_scenario_result_summary(self):
        from validation.scenario_engine import ScenarioEngine, ScenarioType
        trades = make_trades(50)
        engine = ScenarioEngine(capital=10_000)
        result = engine.run_single(trades, ScenarioType.VOL_SPIKE_2X, label="test")
        s = result.summary()
        assert isinstance(s, str)

    def test_covid_scenario_runs(self):
        from validation.scenario_engine import ScenarioEngine, ScenarioType
        trades = make_trades(80)
        engine = ScenarioEngine(capital=10_000)
        result = engine.run_single(trades, ScenarioType.COVID_CRASH_2020, label="test")
        assert result is not None

    def test_custom_scenarios(self):
        from validation.scenario_engine import ScenarioEngine, ScenarioSpec, ScenarioType
        custom = [
            ScenarioSpec(name="TestCustom", scenario_type=ScenarioType.VOL_SPIKE_2X,
                         description="Custom test", vol_mult=1.5),
        ]
        trades = make_trades(50)
        engine = ScenarioEngine(capital=10_000, scenarios=custom)
        report = engine.run(trades, label="custom")
        assert len(report.results) == 1


# ══════════════════════════════════════════════════════════════════════════════
# RealisticExecutionSimulator
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutionSimulator:

    def test_simulate_entry_fills(self):
        from validation.execution_simulator import RealisticExecutionSimulator, ExecutionParams
        sim  = RealisticExecutionSimulator(params=ExecutionParams(missed_fill_prob=0, rejection_prob=0))
        fill = sim.simulate_entry(mid_price=50_000.0, qty=0.1, direction='LONG')
        assert fill.filled
        assert fill.fill_price > 0
        assert fill.fee > 0

    def test_long_entry_pays_ask(self):
        """LONG entry should fill above mid (paying spread)."""
        from validation.execution_simulator import RealisticExecutionSimulator, ExecutionParams
        sim  = RealisticExecutionSimulator(params=ExecutionParams(
            spread_bps=10, slippage_bps=5, missed_fill_prob=0, rejection_prob=0, worse_entry_prob=0,
        ))
        fill = sim.simulate_entry(mid_price=50_000.0, qty=1.0, direction='LONG')
        assert fill.filled
        assert fill.fill_price > 50_000.0  # above mid

    def test_simulate_exit_fills(self):
        from validation.execution_simulator import RealisticExecutionSimulator, ExecutionParams
        sim  = RealisticExecutionSimulator(params=ExecutionParams(missed_fill_prob=0, rejection_prob=0))
        fill = sim.simulate_exit(mid_price=51_000.0, qty=0.1, direction='LONG')
        assert fill.filled

    def test_missed_fill(self):
        from validation.execution_simulator import RealisticExecutionSimulator, ExecutionParams
        sim  = RealisticExecutionSimulator(
            params=ExecutionParams(missed_fill_prob=1.0, rejection_prob=0), seed=42,
        )
        fill = sim.simulate_entry(mid_price=50_000.0, qty=0.1)
        assert not fill.filled

    def test_partial_fill(self):
        from validation.execution_simulator import RealisticExecutionSimulator, ExecutionParams
        sim  = RealisticExecutionSimulator(
            params=ExecutionParams(
                missed_fill_prob=0, rejection_prob=0,
                partial_fill_prob=1.0, partial_fill_frac=0.5,
            ), seed=42,
        )
        fill = sim.simulate_entry(mid_price=50_000.0, qty=1.0)
        assert fill.filled and fill.partial
        assert fill.fill_qty < fill.requested_qty

    def test_round_trip(self):
        from validation.execution_simulator import RealisticExecutionSimulator, ExecutionParams
        sim = RealisticExecutionSimulator(
            params=ExecutionParams(missed_fill_prob=0, rejection_prob=0, partial_fill_prob=0)
        )
        entry_fill, exit_fill, net_pnl = sim.simulate_round_trip(
            entry_mid=50_000.0, exit_mid=51_000.0, qty=0.01, direction='LONG',
        )
        assert entry_fill.filled and exit_fill.filled
        assert net_pnl < 10.0  # costs eat into gross PnL (gross = 10.0)

    def test_batch_simulate(self):
        from validation.execution_simulator import RealisticExecutionSimulator, ExecutionParams
        sim = RealisticExecutionSimulator(
            params=ExecutionParams(missed_fill_prob=0.10, rejection_prob=0, partial_fill_prob=0)
        )
        orders = [{'mid_price': 50_000 + i * 100, 'qty': 0.1, 'direction': 'LONG', 'side': 'entry'}
                  for i in range(50)]
        report = sim.batch_simulate(orders)
        assert report.n_orders == 50
        assert report.fill_rate <= 1.0
        assert report.fill_rate > 0.5  # 10% miss rate → >50% fill rate

    def test_execution_report_summary(self):
        from validation.execution_simulator import RealisticExecutionSimulator, ExecutionParams
        sim = RealisticExecutionSimulator(
            params=ExecutionParams(missed_fill_prob=0, rejection_prob=0)
        )
        orders = [{'mid_price': 50_000.0, 'qty': 0.01, 'direction': 'LONG', 'side': 'entry'}
                  for _ in range(10)]
        report = sim.batch_simulate(orders)
        s = report.summary()
        assert "EXECUTION QUALITY" in s

    def test_theoretical_to_net(self):
        from validation.execution_simulator import RealisticExecutionSimulator
        sim    = RealisticExecutionSimulator()
        net    = sim.theoretical_to_net(theoretical_pnl=100.0, notional=10_000.0)
        assert net < 100.0   # costs reduce PnL

    def test_fill_cost_bps(self):
        from validation.execution_simulator import RealisticExecutionSimulator, ExecutionParams
        sim  = RealisticExecutionSimulator(
            params=ExecutionParams(spread_bps=6, slippage_bps=4, missed_fill_prob=0,
                                   rejection_prob=0, worse_entry_prob=0, partial_fill_prob=0)
        )
        fill = sim.simulate_entry(mid_price=50_000.0, qty=1.0, direction='LONG')
        assert fill.cost_bps > 0


# ══════════════════════════════════════════════════════════════════════════════
# PerformanceAttributionEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestAttribution:

    def test_attribute_returns_report(self):
        from validation.attribution import PerformanceAttributionEngine
        trades = make_trades(60)
        eng    = PerformanceAttributionEngine()
        report = eng.attribute(trades, label="test")
        assert report.label == "test"
        assert report.trade_count == 60

    def test_asset_attribution(self):
        from validation.attribution import PerformanceAttributionEngine
        trades = make_trades(60)
        eng    = PerformanceAttributionEngine()
        report = eng.attribute(trades, label="test")
        assert 'BTC/USD' in report.by_asset

    def test_strategy_attribution(self):
        from validation.attribution import PerformanceAttributionEngine
        trades = make_trades(60)
        eng    = PerformanceAttributionEngine()
        report = eng.attribute(trades, label="test")
        assert 'test_strat' in report.by_strategy

    def test_direction_attribution(self):
        from validation.attribution import PerformanceAttributionEngine
        trades = make_trades(60)
        eng    = PerformanceAttributionEngine()
        report = eng.attribute(trades, label="test")
        assert 'LONG' in report.by_direction

    def test_regime_attribution(self):
        from validation.attribution import PerformanceAttributionEngine
        trades = make_trades(60)
        eng    = PerformanceAttributionEngine()
        report = eng.attribute(trades, label="test")
        assert 'bull' in report.by_regime

    def test_time_of_day_buckets(self):
        from validation.attribution import PerformanceAttributionEngine
        trades = make_trades(120, seed=10)
        eng    = PerformanceAttributionEngine()
        report = eng.attribute(trades, label="test")
        assert len(report.by_time_of_day) > 0

    def test_holding_period_buckets(self):
        from validation.attribution import PerformanceAttributionEngine
        trades = make_trades(60)
        eng    = PerformanceAttributionEngine()
        report = eng.attribute(trades, label="test")
        assert len(report.by_holding) > 0

    def test_cost_drag(self):
        from validation.attribution import PerformanceAttributionEngine
        trades = make_trades(60, win_rate=0.70)
        eng    = PerformanceAttributionEngine()
        report = eng.attribute(trades, label="test")
        assert report.total_fees > 0
        assert report.total_slippage > 0

    def test_summary_output(self):
        from validation.attribution import PerformanceAttributionEngine
        trades = make_trades(60)
        eng    = PerformanceAttributionEngine()
        report = eng.attribute(trades, label="test")
        s = report.summary()
        assert "PERFORMANCE ATTRIBUTION" in s
        assert "BTC/USD" in s

    def test_empty_trades(self):
        from validation.attribution import PerformanceAttributionEngine
        eng    = PerformanceAttributionEngine()
        report = eng.attribute([], label="empty")
        assert report.trade_count == 0
        assert report.total_pnl_net == 0

    def test_bucket_win_rate(self):
        from validation.attribution import PerformanceAttributionEngine
        trades = make_trades(60)
        eng    = PerformanceAttributionEngine()
        report = eng.attribute(trades, label="test")
        for bucket in report.by_asset.values():
            assert 0.0 <= bucket.win_rate <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# CalibrationAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibration:

    def test_analyze_returns_report(self):
        from validation.calibration import CalibrationAnalyzer
        trades      = make_trades(100)
        confidences = [0.62] * 100
        analyzer    = CalibrationAnalyzer()
        report      = analyzer.analyze(trades, model_name="TestModel", confidences=confidences)
        assert report.model_name == "TestModel"
        assert report.n_trades == 100

    def test_brier_score_range(self):
        from validation.calibration import CalibrationAnalyzer
        trades      = make_trades(100)
        confidences = [0.65] * 100
        analyzer    = CalibrationAnalyzer()
        report      = analyzer.analyze(trades, model_name="M", confidences=confidences)
        assert 0.0 <= report.brier_score <= 1.0

    def test_ece_range(self):
        from validation.calibration import CalibrationAnalyzer
        trades      = make_trades(100)
        confidences = [0.60] * 100
        analyzer    = CalibrationAnalyzer()
        report      = analyzer.analyze(trades, model_name="M", confidences=confidences)
        assert report.ece >= 0.0

    def test_buckets_populated(self):
        from validation.calibration import CalibrationAnalyzer
        rng         = np.random.default_rng(0)
        trades      = make_trades(200)
        confidences = list(rng.uniform(0.50, 0.90, 200))
        analyzer    = CalibrationAnalyzer()
        report      = analyzer.analyze(trades, model_name="M", confidences=confidences)
        assert len(report.buckets) > 0

    def test_reliability_factors_in_range(self):
        from validation.calibration import CalibrationAnalyzer
        trades      = make_trades(100)
        confidences = [0.62] * 100
        analyzer    = CalibrationAnalyzer()
        report      = analyzer.analyze(trades, model_name="M", confidences=confidences)
        factors     = analyzer.get_reliability_factors(report)
        for f in factors.values():
            assert 0.0 <= f <= 1.0

    def test_summary_output(self):
        from validation.calibration import CalibrationAnalyzer
        trades      = make_trades(80)
        confidences = [0.65] * 80
        analyzer    = CalibrationAnalyzer()
        report      = analyzer.analyze(trades, model_name="MetaModel", confidences=confidences)
        s           = report.summary()
        assert "CALIBRATION REPORT" in s
        assert "MetaModel" in s

    def test_no_confidence_data(self):
        """Without confidence data, should use neutral defaults."""
        from validation.calibration import CalibrationAnalyzer
        trades   = make_trades(50)
        analyzer = CalibrationAnalyzer()
        report   = analyzer.analyze(trades, model_name="M", confidences=None)
        assert report.n_trades == 50

    def test_empty_trades(self):
        from validation.calibration import CalibrationAnalyzer
        analyzer = CalibrationAnalyzer()
        report   = analyzer.analyze([], model_name="M")
        assert report.n_trades == 0

    def test_reliability_for_confidence(self):
        from validation.calibration import CalibrationAnalyzer
        trades      = make_trades(100)
        confidences = [0.65] * 100
        analyzer    = CalibrationAnalyzer()
        report      = analyzer.analyze(trades, model_name="M", confidences=confidences)
        factor      = analyzer.reliability_for_confidence(report, 0.65)
        assert 0.0 <= factor <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# LiveCapitalGraduation
# ══════════════════════════════════════════════════════════════════════════════

class TestLiveGraduation:

    @pytest.fixture
    def tmpdb(self, tmp_path):
        return str(tmp_path / "graduation.sqlite")

    def test_initial_stage_is_shadow(self, tmpdb):
        from validation.live_graduation import LiveCapitalGraduation, GraduationStage
        g = LiveCapitalGraduation(db_path=tmpdb)
        assert g.current_stage('strat-1') == GraduationStage.SHADOW

    def test_graduate_shadow_to_paper(self, tmpdb):
        from validation.live_graduation import LiveCapitalGraduation, GraduationStage
        g = LiveCapitalGraduation(db_path=tmpdb)
        d = g.graduate(
            strategy_id='strat-1',
            target_stage=GraduationStage.PAPER,
            human_token='human-paper-auth-token',
            rationale='30-day shadow campaign complete',
        )
        assert d.approved
        assert g.current_stage('strat-1') == GraduationStage.PAPER

    def test_cannot_skip_stages(self, tmpdb):
        from validation.live_graduation import LiveCapitalGraduation, GraduationStage
        g = LiveCapitalGraduation(db_path=tmpdb)
        with pytest.raises(ValueError):
            g.graduate(
                strategy_id='strat-2',
                target_stage=GraduationStage.CANARY_LIVE,   # skip PAPER
                human_token='HUMAN-AUTHORIZED-token-here',
                rationale='trying to skip stages',
            )

    def test_live_requires_authorized_token(self, tmpdb):
        from validation.live_graduation import LiveCapitalGraduation, GraduationStage
        g = LiveCapitalGraduation(db_path=tmpdb)
        # Graduate to PAPER first
        g.graduate('strat-3', GraduationStage.PAPER, 'human-paper-auth', 'shadow ok')
        # Try canary without proper token
        with pytest.raises(ValueError):
            g.graduate(
                'strat-3', GraduationStage.CANARY_LIVE,
                human_token='short_token',  # missing HUMAN-AUTHORIZED
                rationale='test',
            )

    def test_canary_requires_authorized_token(self, tmpdb):
        from validation.live_graduation import LiveCapitalGraduation, GraduationStage
        g = LiveCapitalGraduation(db_path=tmpdb)
        g.graduate('strat-4', GraduationStage.PAPER, 'human-paper-auth-token', 'shadow complete')
        d = g.graduate(
            'strat-4', GraduationStage.CANARY_LIVE,
            human_token='HUMAN-AUTHORIZED-2024-01-01-canary',
            rationale='90-day paper campaign passed all gates',
        )
        assert d.approved
        assert g.current_stage('strat-4').value == 'CANARY_LIVE'

    def test_capital_pct_by_stage(self, tmpdb):
        from validation.live_graduation import LiveCapitalGraduation, GraduationStage, STAGE_CAPITAL_PCT
        g = LiveCapitalGraduation(db_path=tmpdb)
        assert STAGE_CAPITAL_PCT[GraduationStage.SHADOW] == 0.00
        assert STAGE_CAPITAL_PCT[GraduationStage.CANARY_LIVE] == 0.02
        assert STAGE_CAPITAL_PCT[GraduationStage.LIMITED_LIVE] == 0.25
        assert STAGE_CAPITAL_PCT[GraduationStage.NORMAL_LIVE] == 1.00

    def test_rollback_to_paper(self, tmpdb):
        from validation.live_graduation import LiveCapitalGraduation, GraduationStage
        g = LiveCapitalGraduation(db_path=tmpdb)
        g.graduate('strat-5', GraduationStage.PAPER, 'human-paper-auth-token', 'ok')
        g.graduate('strat-5', GraduationStage.CANARY_LIVE, 'HUMAN-AUTHORIZED-canary-live', 'ok')
        d = g.rollback_to_paper('strat-5', reason='drawdown threshold breached')
        assert g.current_stage('strat-5') == GraduationStage.PAPER

    def test_empty_token_raises(self, tmpdb):
        from validation.live_graduation import LiveCapitalGraduation, GraduationStage
        g = LiveCapitalGraduation(db_path=tmpdb)
        with pytest.raises(ValueError):
            g.graduate('strat-6', GraduationStage.PAPER, '', 'no token')

    def test_short_token_raises(self, tmpdb):
        from validation.live_graduation import LiveCapitalGraduation, GraduationStage
        g = LiveCapitalGraduation(db_path=tmpdb)
        with pytest.raises(ValueError):
            g.graduate('strat-7', GraduationStage.PAPER, 'abc', 'short token')

    def test_history_recorded(self, tmpdb):
        from validation.live_graduation import LiveCapitalGraduation, GraduationStage
        g = LiveCapitalGraduation(db_path=tmpdb)
        g.graduate('strat-8', GraduationStage.PAPER, 'human-paper-auth-ok', 'shadow done')
        h = g.get_history('strat-8')
        assert len(h) == 1

    def test_current_capital_pct(self, tmpdb):
        from validation.live_graduation import LiveCapitalGraduation, GraduationStage
        g = LiveCapitalGraduation(db_path=tmpdb)
        assert g.current_capital_pct('new-strat') == 0.0
        g.graduate('new-strat', GraduationStage.PAPER, 'human-paper-auth-token', 'ok')
        assert g.current_capital_pct('new-strat') == 0.0  # PAPER = 0% live capital

    def test_graduation_status_text(self, tmpdb):
        from validation.live_graduation import LiveCapitalGraduation
        g = LiveCapitalGraduation(db_path=tmpdb)
        s = g.graduation_status('my-strat')
        assert 'my-strat' in s
        assert 'SHADOW' in s
