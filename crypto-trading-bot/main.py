import logging
from typing import Optional
from strategies.strategy_engine import generate_trade_signal
from trading.trade_executor import execute_trade
from data.fetcher import fetch_latest_market_data
from config.config import CONFIG
import pandas as pd

# Set up logging for better traceability and debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/bot.log"),
        logging.StreamHandler()
    ]
)

def main() -> None:
    """
    Main entry point for the crypto trading bot.
    Fetches market data, generates trade signals, and executes trades.
    """
    try:
        # Use config values for ticker, period, interval for reusability
        ticker = CONFIG.get("symbol", "ETH-USD")
        period = CONFIG.get("period", "1mo")
        interval = CONFIG.get("interval", "1h")
        market_data: Optional[pd.DataFrame] = fetch_latest_market_data(
            ticker=ticker, period=period, interval=interval
        )
        if market_data is None or market_data.empty:
            logging.warning("Market data is empty or could not be fetched.")
            return

        trade_signal = generate_trade_signal(market_data)
        if trade_signal:
            execute_trade(trade_signal)
        else:
            logging.info("No trade signal generated for current market data.")

    except Exception as e:
        logging.exception(f"An error occurred in main(): {e}")

if __name__ == "__main__":
    main()

# Improvements:
# - Added logging for better traceability and debugging.
# - Added type hints and docstring to main().
# - Added error handling for API/data issues.
# - Used config values for symbol, period, and interval for reusability.
# - Checks for empty DataFrame before proceeding.
# - Logging is recommended throughout the codebase for production readiness.
