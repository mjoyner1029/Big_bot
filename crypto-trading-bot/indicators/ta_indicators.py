import ta
import pandas as pd
import logging

def add_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TA indicators (RSI, MACD, EMA) to DataFrame.
    Args:
        df: DataFrame with 'Close' column.
    Returns:
        DataFrame with new indicator columns.
    """
    df = df.copy()
    try:
        df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        df["macd"] = ta.trend.MACD(df["Close"]).macd_diff()
        df["ema20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    except Exception as e:
        logging.error(f"TA indicator calculation failed: {e}")
    return df
