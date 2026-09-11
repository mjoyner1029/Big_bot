import pandas as pd
import numpy as np
import logging
from typing import Optional

try:
    import ta
except Exception:  # pragma: no cover - optional dependency fallback
    ta = None

# Import Freqtrade-style patterns
try:
    from indicators.freqtrade_patterns import (
        populate_indicators_freqtrade,
        generate_entry_signals,
        generate_stop_loss_atr,
        clear_indicator_cache,
    )
except ImportError:
    # Fallback if module not available
    populate_indicators_freqtrade = None
    generate_entry_signals = None
    generate_stop_loss_atr = None
    clear_indicator_cache = None


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

    if ta is None:
        return _add_ta_indicators_fallback(df, close, high, low, volume)

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


def _add_ta_indicators_fallback(
    df: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: Optional[pd.Series],
) -> pd.DataFrame:
    """Compute a compact indicator set when python-ta is unavailable."""
    out = df.copy()

    def last_valid(series: pd.Series) -> pd.Series:
        return series.replace([np.inf, -np.inf], np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    out["rsi"] = rsi.fillna(50.0)

    rolling_high = high.rolling(window=14)
    rolling_low = low.rolling(window=14)
    highest_high = rolling_high.max()
    lowest_low = rolling_low.min()
    out["stoch_k"] = ((close - lowest_low) / (highest_high - lowest_low)).replace([np.inf, -np.inf], np.nan) * 100
    out["stoch_d"] = out["stoch_k"].rolling(window=3).mean()
    out["williams_r"] = ((highest_high - close) / (highest_high - lowest_low)).replace([np.inf, -np.inf], np.nan) * -100
    out["roc"] = close.pct_change(periods=10) * 100

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["ema9"] = close.ewm(span=9, adjust=False).mean()
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["sma50"] = close.rolling(window=50).mean()
    out["sma200"] = close.rolling(window=200).mean()

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(window=14).mean()

    out["adx"] = np.nan
    out["adx_pos"] = np.nan
    out["adx_neg"] = np.nan
    out["ichimoku_base"] = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2

    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std(ddof=0)
    bb_upper = bb_mid + (2 * bb_std)
    bb_lower = bb_mid - (2 * bb_std)
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    out["bb_pctb"] = (close - bb_lower) / (bb_upper - bb_lower)
    out["bb_width"] = (bb_upper - bb_lower) / bb_mid

    if volume is not None:
        out["obv"] = (np.sign(close.diff().fillna(0)) * volume.fillna(0)).cumsum()
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume.fillna(0)
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0.0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0.0)
        pos_sum = positive_flow.rolling(window=14).sum()
        neg_sum = negative_flow.rolling(window=14).sum().abs().replace(0, np.nan)
        money_ratio = pos_sum / neg_sum
        out["mfi"] = 100 - (100 / (1 + money_ratio))
        out["adi"] = ((2 * close - high - low) / (high - low).replace(0, np.nan) * volume.fillna(0)).cumsum()
        out["vwap"] = (typical_price * volume.fillna(0)).cumsum() / volume.fillna(0).cumsum().replace(0, np.nan)

    out["close_pct_change"] = close.pct_change()
    out["close_log_return"] = np.log(close / close.shift(1))
    out["high_low_range"] = (high - low) / close.replace(0, np.nan)
    out["close_to_ema20"] = (close - out["ema20"]) / out["ema20"]
    out["close_to_sma50"] = (close - out["sma50"]) / out["sma50"]

    return out


# ════════════════════════════════════════════════════════════════════════════
# Freqtrade-compatible Interface
# ════════════════════════════════════════════════════════════════════════════

def add_freqtrade_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Freqtrade-style indicators with caching and fast/slow variants.
    
    Key differences from add_ta_indicators:
    - Lowercase column names by default
    - Includes fast/slow RSI variants (7 and 14)
    - Automatic EMA/SMA family (9, 20, 50, 200)
    - Crossover signals for entries (EMA9/20, EMA20/50, EMA50/200)
    - Price alignment checks (above SMA50, above SMA200)
    - Entry/exit signal generation built-in
    
    Returns DataFrame with ~60+ indicator columns.
    """
    if populate_indicators_freqtrade is None:
        logging.warning("Freqtrade patterns unavailable, falling back to standard TA")
        return add_ta_indicators(df)
    
    try:
        df_processed = populate_indicators_freqtrade(df)
        return df_processed
    except Exception as e:
        logging.error(f"Freqtrade indicator calculation failed: {e}", exc_info=True)
        return add_ta_indicators(df)


def get_freqtrade_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate entry/exit signals using Freqtrade-style composite logic.
    
    Returns DataFrame with additional columns:
    - entry_signal: 1 (bullish), -1 (bearish), 0 (neutral)
    - stop_loss_long/short: ATR-based stop levels
    - take_profit_long/short: Risk-reward adjusted TP levels
    """
    if generate_entry_signals is None or generate_stop_loss_atr is None:
        logging.warning("Freqtrade signal generation unavailable")
        return df
    
    try:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # Generate composite entry signals
        df['entry_signal'] = generate_entry_signals(df)
        
        # Generate stop-loss and take-profit levels
        sl_tp = generate_stop_loss_atr(df, atr_multiplier=2.0)
        for col, series in sl_tp.items():
            df[col] = series
        
        # Add trend alignment score (0-3: how many EMA/SMA layers aligned bullish)
        df['trend_alignment'] = (
            (df['ema9'] > df['ema20']).astype(int) +
            (df['ema20'] > df['ema50']).astype(int) +
            (df['ema50'] > df['sma200']).astype(int)
        )
        
        return df
    except Exception as e:
        logging.error(f"Signal generation failed: {e}", exc_info=True)
        return df


def clear_freqtrade_cache():
    """Clear the indicator cache when new data arrives."""
    if clear_indicator_cache is not None:
        clear_indicator_cache()

