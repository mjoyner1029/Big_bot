import ta
import pandas as pd
import numpy as np
import logging
from typing import Optional


def add_ta_indicators(df: pd.DataFrame, close_col: str = "Close",
                      high_col: str = "High", low_col: str = "Low",
                      volume_col: str = "Volume") -> pd.DataFrame:
    """
    Add a comprehensive set of technical-analysis indicators to a DataFrame.

    Indicators added:
        Momentum  – RSI-14, Stochastic %K/%D, Williams %R, ROC-10
        Trend     – MACD line/signal/histogram, EMA-9/20/50, SMA-50/200, ADX, Ichimoku base
        Volatility– Bollinger Bands (upper/mid/lower, %B, bandwidth), ATR-14
        Volume    – OBV, VWAP, MFI, Accumulation/Distribution

    Args:
        df:         DataFrame with OHLCV columns.
        close_col:  Name of the close-price column.
        high_col:   Name of the high-price column.
        low_col:    Name of the low-price column.
        volume_col: Name of the volume column.
    Returns:
        A copy of the DataFrame with new indicator columns appended.
    """
    df = df.copy()

    close = df[close_col]
    high = df[high_col]
    low = df[low_col]
    volume = df[volume_col] if volume_col in df.columns else None

    try:
        # ── Momentum ─────────────────────────────────────────────
        df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        df["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
        df["roc"] = ta.momentum.ROCIndicator(close, window=10).roc()

        # ── Trend ────────────────────────────────────────────────
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        df["ema9"] = ta.trend.EMAIndicator(close, window=9).ema_indicator()
        df["ema20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        df["ema50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        df["sma50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
        df["sma200"] = ta.trend.SMAIndicator(close, window=200).sma_indicator()

        adx = ta.trend.ADXIndicator(high, low, close, window=14)
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

        ichimoku = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
        df["ichimoku_base"] = ichimoku.ichimoku_base_line()

        # ── Volatility ───────────────────────────────────────────
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_pctb"] = bb.bollinger_pband()       # %B
        df["bb_width"] = bb.bollinger_wband()       # bandwidth

        df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

        # ── Volume ───────────────────────────────────────────────
        if volume is not None:
            df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            df["mfi"] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()
            df["adi"] = ta.volume.AccDistIndexIndicator(high, low, close, volume).acc_dist_index()

            # VWAP (intraday approximation using cumulative typical-price * volume)
            typical_price = (high + low + close) / 3
            df["vwap"] = (typical_price * volume).cumsum() / volume.cumsum()

        # ── Derived features useful for ML ───────────────────────
        df["close_pct_change"] = close.pct_change()
        df["close_log_return"] = np.log(close / close.shift(1))
        df["high_low_range"] = (high - low) / close
        df["close_to_ema20"] = (close - df["ema20"]) / df["ema20"]
        df["close_to_sma50"] = (close - df["sma50"]) / df["sma50"]

    except Exception as e:
        logging.error(f"TA indicator calculation failed: {e}", exc_info=True)

    return df


def get_latest_indicator_snapshot(df: pd.DataFrame) -> dict:
    """Return a dict of the most recent row's indicator values (NaNs dropped)."""
    df = add_ta_indicators(df)
    latest = df.iloc[-1].dropna().to_dict()
    return latest
