import yfinance as yf
import pandas as pd
import logging
from typing import Optional
from config.config import CONFIG

def fetch_latest_market_data(
    ticker: Optional[str] = None,
    period: Optional[str] = None,
    interval: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch latest market data from yFinance.
    Args:
        ticker: Symbol to fetch (default from config)
        period: Period string (default from config)
        interval: Interval string (default from config)
    Returns:
        DataFrame with market data or None if fetch fails.
    """
    ticker = ticker or CONFIG.get("symbol", "ETH-USD")
    period = period or CONFIG.get("period", "1mo")
    interval = interval or CONFIG.get("interval", "1h")
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data is None or data.empty:
            logging.warning(f"No data returned for {ticker} ({period}, {interval})")
            return None
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None
