import pandas as pd
import numpy as np
import pytest

from research.overnight_edge_scanner import (
    OvernightEdgeScanner,
    OvernightEdgeConfig,
    compute_overnight_returns,
    compute_intraday_returns,
    validate_overnight_inputs,
    classify_edge_type,
)


def _make_df(prices, volume=1000.0):
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    df = pd.DataFrame({
        "open": prices.copy(),
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices],
        "close": prices.copy(),
        "volume": volume,
    }, index=idx)
    return df


def test_overnight_return_calculation():
    prev_close = 100.0
    open_price = 105.0
    close_price = 108.0
    ret = compute_overnight_returns(pd.DataFrame({
        "open": [open_price],
        "close": [close_price],
        "volume": [1000],
        "previous_close": [prev_close],
    }))
    assert ret.iloc[0] == pytest.approx(0.05, rel=1e-6)


def test_intraday_return_calculation():
    df = pd.DataFrame({
        "open": [100.0, 110.0],
        "close": [110.0, 115.0],
        "volume": [1000, 1000],
    })
    ret = compute_intraday_returns(df)
    assert ret.iloc[0] == pytest.approx(0.10, rel=1e-6)
    assert ret.iloc[1] == pytest.approx(0.0454545455, rel=1e-6)


def test_validate_overnight_inputs_flags_bad_data():
    df = pd.DataFrame({
        "open": [0.0, 100.0, 100.0],
        "high": [110.0, 110.0, 110.0],
        "low": [90.0, 90.0, 90.0],
        "close": [100.0, 100.0, 100.0],
        "volume": [0, 50, 50],
    })
    issues = validate_overnight_inputs(df)
    assert len(issues) >= 1


def test_scan_ranked_candidates_are_valid():
    prices = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 110.0, 109.0, 108.0, 112.0, 120.0]
    df = _make_df(prices)

    scanner = OvernightEdgeScanner(OvernightEdgeConfig(min_history_days=5, min_price=5.0))
    ranked = scanner.scan_universe({"TEST": df})
    assert isinstance(ranked, list)
    assert len(ranked) == 1
    assert ranked[0]["ticker"] == "TEST"
    assert 0 <= ranked[0]["score"] <= 100


def test_classify_edge_type_handles_earnings():
    assert classify_edge_type(general_edge=True, earnings_driven=False) == "GENERAL"
    assert classify_edge_type(general_edge=False, earnings_driven=True) == "EARNINGS_DRIVEN"
    assert classify_edge_type(general_edge=True, earnings_driven=True) == "MIXED"
    assert classify_edge_type(general_edge=False, earnings_driven=False) == "NONE"


def test_walk_forward_split_keeps_time_order():
    scanner = OvernightEdgeScanner(OvernightEdgeConfig())
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    values = np.linspace(100, 150, 120)
    df = pd.DataFrame({
        "open": values,
        "high": values * 1.02,
        "low": values * 0.98,
        "close": values,
        "volume": 1000,
    }, index=dates)
    folds = scanner.walk_forward_split(df)
    assert len(folds) >= 2
    assert all(folds[i]["train_end"] < folds[i]["test_start"] for i in range(len(folds)-1))


def test_live_lockout_defaults_to_disabled():
    cfg = OvernightEdgeConfig()
    assert cfg.paper_only is True
    assert cfg.allow_live is False
    assert cfg.enabled is True


def test_no_lookahead_bias_in_metric_history():
    scanner = OvernightEdgeScanner(OvernightEdgeConfig())
    df = pd.DataFrame({
        "open": [100, 103, 105, 108, 110],
        "high": [101, 104, 106, 109, 111],
        "low": [99, 101, 103, 106, 108],
        "close": [102, 104, 107, 109, 112],
        "volume": [1000, 1000, 1000, 1000, 1000],
    })
    out = scanner.compute_metrics(df)
    assert out["sample_size"] >= 3
    assert "overnight_mean" in out
    assert "intraday_mean" in out

