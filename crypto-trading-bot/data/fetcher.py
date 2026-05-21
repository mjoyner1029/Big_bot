import yfinance as yf
import pandas as pd
import logging
import os
from typing import Optional, List
from config.config import CONFIG
from datetime import datetime
from trading.rate_limiter import get_yfinance_limiter


def fetch_latest_market_data(
    ticker: Optional[str] = None,
    period: Optional[str] = None,
    interval: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch latest OHLCV market data from yFinance.

    Args:
        ticker:   Symbol (default from config).
        period:   Look-back period string e.g. '3mo' (default from config).
        interval: Bar interval string e.g. '1h' (default from config).
    Returns:
        DataFrame with OHLCV data, or None on failure.
    """
    ticker = ticker or CONFIG.get("symbol", "ETH-USD")
    period = period or CONFIG.get("period", "3mo")
    interval = interval or CONFIG.get("interval", "1h")
    
    # Apply rate limiting
    rate_limiter = get_yfinance_limiter()
    if CONFIG.get("rate_limiting_enabled", True):
        rate_limiter.wait_if_needed()
    
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data is None or data.empty:
            logging.warning(f"No data returned for {ticker} ({period}, {interval})")
            return None
        # Flatten multi-level columns if present (yfinance sometimes returns multi-index)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        logging.info(f"Fetched {len(data)} bars for {ticker} ({interval})")
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}", exc_info=True)
        return None


def fetch_multiple_symbols(
    symbols: Optional[List[str]] = None,
    period: Optional[str] = None,
    interval: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch market data for multiple symbols.

    Returns:
        Dict mapping symbol -> DataFrame.
    """
    from config.config import get_all_symbols
    symbols = symbols or get_all_symbols()
    period = period or CONFIG.get("period", "3mo")
    interval = interval or CONFIG.get("interval", "1h")

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_latest_market_data(ticker=sym, period=period, interval=interval)
        if df is not None and not df.empty:
            result[sym] = df
    return result


def save_historical_data(
    df: pd.DataFrame,
    symbol: str,
    directory: str = "data/historical",
) -> str:
    """
    Persist a DataFrame to CSV in the historical data directory.

    Returns:
        The file path written.
    """
    os.makedirs(directory, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_sym = symbol.replace("/", "_").replace("-", "_")
    filepath = os.path.join(directory, f"{safe_sym}_{ts}.csv")
    df.to_csv(filepath)
    logging.info(f"Saved historical data to {filepath}")
    return filepath


def load_historical_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load a previously saved CSV into a DataFrame."""
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logging.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return None
