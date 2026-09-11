import pandas as pd
import pytest

from research.opportunity_framework import (
    MarketAnomaly,
    OpportunityDetector,
    OpportunityScanner,
    TemporalAnomalyDetector,
    MomentumDetector,
    MeanReversionDetector,
    VolumeGapDetector,
    compute_opportunity_score,
    benjamini_hochberg_qvalues,
    HoldoutManager,
)


def _make_price_frame(seed=1.0, n=120):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    base = [seed + i * 0.15 for i in range(n)]
    df = pd.DataFrame(
        {
            "open": base,
            "high": [x * 1.02 for x in base],
            "low": [x * 0.98 for x in base],
            "close": [x * (1 + 0.001 * (i % 6 - 2)) for i, x in enumerate(base)],
            "volume": [1000 + i * 10 for i in range(n)],
        },
        index=idx,
    )
    return df


def test_market_anomaly_has_required_fields():
    anomaly = MarketAnomaly(
        anomaly_id="a-1",
        detector_name="temporal",
        market="equity",
        symbols=["AAPL"],
        hypothesis={"condition": "close_to_open > 0.003"},
        entry_condition="close_to_open > 0.003",
        exit_condition="close_to_open <= 0.00",
        holding_period="1d",
        sample_size=120,
        expected_return=0.012,
        median_return=0.010,
        volatility=0.015,
        sharpe=1.4,
        sortino=1.9,
        max_drawdown=0.12,
        win_rate=0.62,
        profit_factor=1.6,
        statistical_significance=0.04,
        confidence_interval=(0.005, 0.023),
        estimated_costs=0.003,
        net_expected_return=0.009,
        regime_information={"regime": "bull"},
        event_concentration=0.12,
        discovery_score=82.0,
        status="DISCOVERED",
    )
    assert anomaly.anomaly_id == "a-1"
    assert anomaly.status == "DISCOVERED"
    assert anomaly.net_expected_return > 0


def test_temporal_detector_generates_anomaly():
    detector = TemporalAnomalyDetector()
    anomalies = detector.scan({"AAPL": _make_price_frame()})
    assert anomalies
    assert anomalies[0].detector_name == "temporal"
    assert anomalies[0].status in {"DISCOVERED", "VALIDATING"}


def test_momentum_detector_generates_anomaly():
    detector = MomentumDetector()
    anomalies = detector.scan({"AAPL": _make_price_frame()})
    assert anomalies
    assert anomalies[0].detector_name == "momentum"


def test_mean_reversion_detector_generates_anomaly():
    detector = MeanReversionDetector()
    anomalies = detector.scan({"AAPL": _make_price_frame()})
    assert anomalies
    assert anomalies[0].detector_name == "mean_reversion"


def test_volume_gap_detector_generates_anomaly():
    detector = VolumeGapDetector()
    anomalies = detector.scan({"AAPL": _make_price_frame()})
    assert anomalies
    assert anomalies[0].detector_name == "volume_gap"


def test_opportunity_score_prefers_stable_edges():
    robust = compute_opportunity_score(
        net_expectancy=0.012,
        sharpe=1.7,
        sortino=2.2,
        drawdown=0.18,
        profit_factor=1.8,
        sample_size=400,
        significance=0.02,
        oos_sharpe=1.4,
        parameter_robustness=0.9,
        cost_robustness=0.9,
        regime_stability=0.8,
        liquidity=0.85,
        execution_feasibility=0.9,
        event_concentration=0.12,
        recent_performance=0.75,
        multiple_testing_adjusted=True,
    )
    weak = compute_opportunity_score(
        net_expectancy=0.006,
        sharpe=0.4,
        sortino=0.5,
        drawdown=0.50,
        profit_factor=1.1,
        sample_size=80,
        significance=0.15,
        oos_sharpe=0.1,
        parameter_robustness=0.2,
        cost_robustness=0.2,
        regime_stability=0.3,
        liquidity=0.4,
        execution_feasibility=0.5,
        event_concentration=0.60,
        recent_performance=0.2,
        multiple_testing_adjusted=False,
    )
    assert robust > weak
    assert 0 <= robust <= 100
    assert 0 <= weak <= 100


def test_bh_qvalues_are_monotonic_and_non_negative():
    pvals = [0.001, 0.006, 0.02, 0.04, 0.3]
    qvals = benjamini_hochberg_qvalues(pvals)
    assert len(qvals) == len(pvals)
    assert all(q >= 0 for q in qvals)
    assert qvals[0] <= qvals[1] <= qvals[2] <= qvals[3] <= qvals[4]


def test_holdout_manager_blocks_reuse_of_final_holdout():
    mgr = HoldoutManager()
    mgr.freeze_holdout("2024-01-01", "2024-03-31")
    assert mgr.is_final_holdout_accessible("2024-02-15") is False
    assert mgr.is_final_holdout_accessible("2024-04-01") is True


def test_paper_only_gate_is_enforced_for_discovery():
    scanner = OpportunityScanner()
    anomaly = MarketAnomaly(
        anomaly_id="paper-1",
        detector_name="temporal",
        market="crypto",
        symbols=["BTC-USD"],
        hypothesis={"condition": "relative_volume > 3"},
        entry_condition="relative_volume > 3",
        exit_condition="close <= open",
        holding_period="1d",
        sample_size=200,
        expected_return=0.02,
        median_return=0.01,
        volatility=0.04,
        sharpe=1.2,
        sortino=1.5,
        max_drawdown=0.18,
        win_rate=0.58,
        profit_factor=1.7,
        statistical_significance=0.03,
        confidence_interval=(0.01, 0.02),
        estimated_costs=0.004,
        net_expected_return=0.016,
        regime_information={"regime": "risk_on"},
        event_concentration=0.08,
        discovery_score=80.0,
        status="PAPER_CANDIDATE",
    )
    assert scanner.paper_only_gate(anomaly) is True


def test_detector_registry_and_base_class_are_integrated():
    assert issubclass(OpportunityDetector, object)
    assert OpportunityScanner().registry
    assert "temporal" in OpportunityScanner().registry
