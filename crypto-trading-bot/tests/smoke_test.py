"""Smoke-test: verify all modules import cleanly and key functions work."""

from data.fetcher import fetch_latest_market_data, fetch_multiple_symbols
from indicators.ta_indicators import add_ta_indicators, get_latest_indicator_snapshot
from models.model_utils import build_features, train_test_split_time, save_model, load_model, FEATURE_COLS
from models.price_predictor import train_model, predict_price_movement
from models.sentiment_model import fetch_headlines, score_text, analyze_sentiment
from models.llm_analyst import llm_sentiment_analysis, llm_interpret_indicators, llm_validate_trade
from models.reinforce_trainer import QLearningTrader, train_reinforcement_model, load_rl_agent, rl_suggest_action
from strategies.strategy_engine import generate_trade_signal
from strategies.thresholds import get_trade_thresholds
from trading.trade_executor import execute_trade, check_open_positions, get_portfolio
from trading.paper_trader import execute_paper_trade
from trading.portfolio_manager import PortfolioManager
from backtest.backtester import run_backtest
from backtest.metrics import compute_backtest_metrics
from logs.trade_logger import log_trade
from config.config import CONFIG, get_all_symbols, is_crypto, get_news_terms_for

print("All imports OK")
print(f"Feature columns: {len(FEATURE_COLS)}")
print(f"Symbols: {get_all_symbols()}")

# Threshold sanity check
t_stock = get_trade_thresholds(100.0, 0.75, side="buy", asset_type="stock")
t_crypto = get_trade_thresholds(50000.0, 0.75, side="buy", asset_type="crypto")
print(f"Stock TP/SL:  TP={t_stock['take_profit_price']}  SL={t_stock['stop_loss_price']}")
print(f"Crypto TP/SL: TP={t_crypto['take_profit_price']}  SL={t_crypto['stop_loss_price']}")

# Portfolio manager
pm = PortfolioManager(state_path="/tmp/_test_bot_state.json")
print(f"Portfolio cash: ${pm.cash}")

# Sentiment score
vader = score_text("Bitcoin surges to new all-time high as institutions buy aggressively")
print(f"VADER score (bullish headline): {vader:.4f}")

print("\n✅  All checks passed")
