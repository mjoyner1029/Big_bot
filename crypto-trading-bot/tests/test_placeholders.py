"""Smoke test for all modified modules."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports...")
from config.config import CONFIG, get_all_symbols, is_crypto
from strategies.thresholds import get_trade_thresholds, _atr_based_thresholds
from strategies.strategy_engine import _compute_ta_score, _compute_sell_ta_score
from models.price_predictor import predict_price_movement
from models.sentiment_model import analyze_sentiment
from models.llm_analyst import llm_validate_trade
from models.reinforce_trainer import _state_from_row, _macd_sign
from trading.portfolio_manager import PortfolioManager
from backtest.backtester import run_backtest
from backtest.metrics import compute_backtest_metrics, _infer_bars_per_year, _empty_metrics
print("All imports OK")

# Test ATR-based thresholds
t = get_trade_thresholds(100.0, 0.7, side="buy", asset_type="crypto", atr=2.5)
print(f"ATR thresholds: TP={t['take_profit_price']}, SL={t['stop_loss_price']}")

# Grid fallback
t2 = get_trade_thresholds(100.0, 0.7, side="buy", asset_type="crypto", atr=None)
print(f"Grid thresholds: TP={t2['take_profit_price']}, SL={t2['stop_loss_price']}")

# MACD normalisation
print(f"MACD sign (BTC): {_macd_sign(50.0, 60000.0)}")
print(f"MACD sign (SOL): {_macd_sign(0.5, 20.0)}")

# State from row with missing data
import pandas as pd
row_missing = pd.Series({"Close": 100.0})
state = _state_from_row(row_missing)
print(f"State from missing row: {state}")
assert state is None, "Should be None when indicators are missing"

row_full = pd.Series({"rsi": 35.0, "macd_hist": -0.5, "close_to_ema20": 0.02, "Close": 100.0})
state2 = _state_from_row(row_full)
print(f"State from full row: {state2}")
assert state2 is not None, "Should be a tuple when all indicators are present"

# TA score with missing indicators
row_empty = pd.Series({"Close": 100.0})
score = _compute_ta_score(row_empty)
print(f"TA score with no indicators: {score}")
assert score is None, "Should be None with <2 indicators"

row_partial = pd.Series({"rsi": 35.0, "macd_hist": 0.5, "Close": 100.0})
score2 = _compute_ta_score(row_partial)
print(f"TA score with 2 indicators: {score2}")
assert score2 is not None, "Should compute with >= 2 indicators"

# Metrics empty
em = _empty_metrics(500)
print(f"Empty metrics win_rate: {em['win_rate']}")
assert em["win_rate"] is None, "win_rate should be None for empty"
assert em["final_equity"] == 500, "final_equity should be starting capital"

# Portfolio position sizing with thresholds
pm = PortfolioManager.__new__(PortfolioManager)
pm.cash = 500.0
pm.starting_capital = 500.0
pm.open_positions = []
pm.closed_trades = []
pm.state_path = "/tmp/test_state.json"

signal = {
    "symbol": "ETH-USD",
    "side": "buy",
    "entry_price": 2000.0,
    "confidence": 0.7,
    "asset_type": "crypto",
}
qty = pm.compute_position_size(signal)
print(f"Position size (no SL in signal): {qty}")
assert qty > 0, "Should compute non-zero quantity"

print("\nALL TESTS PASSED")
