"""
Freqtrade-style indicator patterns for the trading bot.

Freqtrade approach:
  1. Indicators added directly to DataFrame via populate_indicators()
  2. Convention: lowercase names, fast/slow variants (e.g., rsi_fast, rsi_slow)
  3. Efficient caching and vectorized operations
  4. Composable: indicators feed into each other

This module provides a decorator-based interface and preset bundles.
"""
import pandas as pd
import numpy as np
import logging
from typing import Callable, Optional, Dict, List
from functools import wraps

try:
    import ta
except ImportError:
    ta = None

logger = logging.getLogger(__name__)

if ta is None:
    logger.warning(
        "'ta' library not installed — using internal indicator fallbacks. "
        "Install 'ta' for the reference implementations."
    )


# ════════════════════════════════════════════════════════════════════════════
# Internal fallback implementations (used when 'ta' is unavailable).
# These are real computations — NEVER silent all-NaN series.
# ════════════════════════════════════════════════════════════════════════════


def _fb_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0).where(close.notna())


def _fb_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, min_periods=period, adjust=False).mean()


def _fb_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period, min_periods=period).mean()


def _fb_true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    return pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)


def _fb_atr(df: pd.DataFrame, period: int) -> pd.Series:
    return _fb_true_range(df).ewm(alpha=1 / period, min_periods=period).mean()


# ════════════════════════════════════════════════════════════════════════════
# Freqtrade-style Decorator & Caching
# ════════════════════════════════════════════════════════════════════════════

_indicator_cache: Dict = {}


def freqtrade_indicator(func: Callable) -> Callable:
    """
    Decorator for indicator functions.
    Follows Freqtrade pattern: function returns (column_name, series).
    """
    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        cache_key = f"{func.__name__}_{id(df)}_{str(args)}_{str(kwargs)}"
        if cache_key in _indicator_cache:
            return _indicator_cache[cache_key]
        result = func(df, *args, **kwargs)
        _indicator_cache[cache_key] = result
        return result
    return wrapper


def clear_indicator_cache():
    """Clear the indicator cache when data changes."""
    global _indicator_cache
    _indicator_cache.clear()


# ════════════════════════════════════════════════════════════════════════════
# Momentum Indicators (Fast/Slow variants)
# ════════════════════════════════════════════════════════════════════════════

@freqtrade_indicator
def rsi_fast(df: pd.DataFrame, period: int = 7) -> pd.Series:
    """RSI with fast (short-term) period."""
    if ta is None:
        return _fb_rsi(df['close'], period)
    return ta.momentum.RSIIndicator(df['close'], window=period).rsi()


@freqtrade_indicator
def rsi_slow(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI with slow (standard) period."""
    if ta is None:
        return _fb_rsi(df['close'], period)
    return ta.momentum.RSIIndicator(df['close'], window=period).rsi()


@freqtrade_indicator
def stoch_fast(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
    """Stochastic %K and %D (fast)."""
    if ta is None:
        low_min = df['low'].rolling(k_period, min_periods=k_period).min()
        high_max = df['high'].rolling(k_period, min_periods=k_period).max()
        rng = (high_max - low_min).replace(0, np.nan)
        k = 100 * (df['close'] - low_min) / rng
        d = k.rolling(d_period, min_periods=d_period).mean()
        return k, d
    stoch = ta.momentum.StochasticOscillator(
        df['high'], df['low'], df['close'],
        window=k_period, smooth_window=d_period
    )
    return stoch.stoch(), stoch.stoch_signal()


@freqtrade_indicator
def macd_fast(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """MACD line, signal, histogram."""
    if ta is None:
        macd_line = _fb_ema(df['close'], fast) - _fb_ema(df['close'], slow)
        signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
        return macd_line, signal_line, macd_line - signal_line
    macd = ta.trend.MACD(df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
    return macd.macd(), macd.macd_signal(), macd.macd_diff()


# ════════════════════════════════════════════════════════════════════════════
# Trend Indicators
# ════════════════════════════════════════════════════════════════════════════

@freqtrade_indicator
def ema_family(df: pd.DataFrame, periods: List[int] = [9, 20, 50, 200]) -> Dict[str, pd.Series]:
    """EMA family (9, 20, 50, 200) for trend identification."""
    if ta is None:
        return {f"ema{p}": _fb_ema(df['close'], p) for p in periods}
    result = {}
    for period in periods:
        result[f"ema{period}"] = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator()
    return result


@freqtrade_indicator
def sma_family(df: pd.DataFrame, periods: List[int] = [50, 200]) -> Dict[str, pd.Series]:
    """SMA family (50, 200) for support/resistance."""
    if ta is None:
        return {f"sma{p}": _fb_sma(df['close'], p) for p in periods}
    result = {}
    for period in periods:
        result[f"sma{period}"] = ta.trend.SMAIndicator(df['close'], window=period).sma_indicator()
    return result


@freqtrade_indicator
def adx_trend(df: pd.DataFrame, period: int = 14) -> tuple:
    """ADX + DI+ and DI- for trend strength and direction."""
    if ta is None:
        up = df['high'].diff()
        down = -df['low'].diff()
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
        tr = _fb_true_range(df).ewm(alpha=1 / period, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / tr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / tr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx_s = dx.ewm(alpha=1 / period, min_periods=period).mean()
        return adx_s, plus_di, minus_di
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=period)
    return adx.adx(), adx.adx_pos(), adx.adx_neg()


@freqtrade_indicator
def crossover(df: pd.DataFrame, line1: pd.Series, line2: pd.Series) -> pd.Series:
    """
    Detect crossover: 1 when line1 crosses above line2, -1 when below.
    Freqtrade uses this heavily for entry/exit signals.
    """
    cross = pd.Series(0, index=df.index, dtype=int)
    above = (line1 > line2).astype(bool)
    prev_above = above.shift(1).fillna(False).astype(bool)
    
    cross[above & ~prev_above] = 1   # Golden cross
    cross[~above & prev_above] = -1  # Death cross
    return cross


# ════════════════════════════════════════════════════════════════════════════
# Volatility Indicators
# ════════════════════════════════════════════════════════════════════════════

@freqtrade_indicator
def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
    """Bollinger Bands with upper, mid, lower, %B (band position), and bandwidth."""
    if ta is None:
        mid = _fb_sma(df['close'], period)
        std = df['close'].rolling(period, min_periods=period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        rng = (upper - lower).replace(0, np.nan)
        return {
            "bb_upper": upper,
            "bb_mid": mid,
            "bb_lower": lower,
            "bb_pctb": (df['close'] - lower) / rng,
            "bb_width": rng / mid.replace(0, np.nan),
        }
    bb = ta.volatility.BollingerBands(df['close'], window=period, window_dev=std_dev)
    return {
        "bb_upper": bb.bollinger_hband(),
        "bb_mid": bb.bollinger_mavg(),
        "bb_lower": bb.bollinger_lband(),
        "bb_pctb": bb.bollinger_pband(),      # % distance from lower to upper
        "bb_width": bb.bollinger_wband(),     # bandwidth as % of mid
    }


@freqtrade_indicator
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range for volatility measurement and stop-loss sizing."""
    if ta is None:
        return _fb_atr(df, period)
    return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()


# ════════════════════════════════════════════════════════════════════════════
# Volume Indicators
# ════════════════════════════════════════════════════════════════════════════

@freqtrade_indicator
def obv(df: pd.DataFrame) -> pd.Series:
    """On Balance Volume — accumulation/distribution of volume."""
    if 'volume' not in df.columns:
        return pd.Series(np.nan, index=df.index)
    if ta is None:
        direction = np.sign(df['close'].diff()).fillna(0)
        return (direction * df['volume']).cumsum()
    return ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()


@freqtrade_indicator
def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted momentum oscillator."""
    if 'volume' not in df.columns:
        return pd.Series(np.nan, index=df.index)
    if ta is None:
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        pos = mf.where(tp.diff() > 0, 0.0).rolling(period, min_periods=period).sum()
        neg = mf.where(tp.diff() < 0, 0.0).rolling(period, min_periods=period).sum()
        ratio = pos / neg.replace(0, np.nan)
        return 100 - (100 / (1 + ratio))
    return ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=period).money_flow_index()


# ════════════════════════════════════════════════════════════════════════════
# Composite Signal Generation (Freqtrade Pattern)
# ════════════════════════════════════════════════════════════════════════════

def populate_indicators_freqtrade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Freqtrade-style populate_indicators: adds all indicators in one pass.
    Naming convention: all lowercase, fast/slow variants available.
    """
    df = df.copy()

    # Ensure required columns (Freqtrade normalizes to lowercase)
    df.columns = [c.lower() for c in df.columns]
    
    # Momentum
    df['rsi_fast'] = rsi_fast(df, 7)
    df['rsi_slow'] = rsi_slow(df, 14)
    df['rsi'] = rsi_slow(df)  # Default to slow
    
    stoch_k, stoch_d = stoch_fast(df)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    
    df['macd'], df['macd_signal'], df['macd_hist'] = macd_fast(df)

    # Trend
    ema_dict = ema_family(df)
    for col, series in ema_dict.items():
        df[col] = series
    
    sma_dict = sma_family(df)
    for col, series in sma_dict.items():
        df[col] = series
    
    df['adx'], df['adx_pos'], df['adx_neg'] = adx_trend(df)

    # Volatility
    bb_dict = bollinger_bands(df)
    for col, series in bb_dict.items():
        df[col] = series
    
    df['atr'] = atr(df)

    # Volume
    if 'volume' in df.columns:
        df['obv'] = obv(df)
        df['mfi'] = mfi(df)

    # Crossovers (useful for signal generation)
    df['ema9_20_cross'] = crossover(df, df['ema9'], df['ema20'])
    df['ema20_50_cross'] = crossover(df, df['ema20'], df['ema50'])
    df['ema50_200_cross'] = crossover(df, df['ema50'], df['ema200'])

    # EMA/SMA alignment (trend strength)
    df['price_above_sma50'] = (df['close'] > df['sma50']).astype(int)
    df['price_above_sma200'] = (df['close'] > df['sma200']).astype(int)
    df['sma50_above_sma200'] = (df['sma50'] > df['sma200']).astype(int)

    indicator_health_check(df, raise_on_failure=False)
    return df


def indicator_health_check(
    df: pd.DataFrame,
    key_columns: List[str] = ('rsi_slow', 'macd', 'bb_mid', 'atr'),
    recent_rows: int = 50,
    max_nan_frac: float = 0.5,
    raise_on_failure: bool = True,
) -> bool:
    """Fail-fast guard: recent values of critical indicators must not be
    (mostly) NaN. A silent all-NaN feature series must never reach a strategy.

    Returns True when healthy. Raises RuntimeError (or returns False when
    raise_on_failure=False, with a critical log) on failure.
    """
    present = [c for c in key_columns if c in df.columns]
    if not present:
        return True
    recent = df[present].iloc[-recent_rows:]
    nan_frac = recent.isna().mean()
    broken = [c for c in present if nan_frac[c] > max_nan_frac]
    if not broken:
        return True
    msg = (
        f"Indicator health check FAILED: {broken} are "
        f">{max_nan_frac:.0%} NaN in the last {len(recent)} rows — "
        "strategies must abstain from this data"
    )
    if raise_on_failure:
        raise RuntimeError(msg)
    logger.critical(msg)
    return False


def generate_entry_signals(df: pd.DataFrame) -> pd.Series:
    """
    Generate entry signals using Freqtrade-style composite logic.
    Returns: 1 (bullish), -1 (bearish), 0 (neutral)
    """
    signals = pd.Series(0, index=df.index, dtype=int)

    # Bullish setup: EMA alignment + RSI not overbought + Price above Bollinger mid
    bullish = (
        (df['ema9'] > df['ema20']) &
        (df['ema20'] > df['ema50']) &
        (df['ema50'] > df['sma200']) &
        (df['rsi_slow'] < 70) &
        (df['rsi_slow'] > 40) &
        (df['close'] > df['bb_mid']) &
        (df['adx'] > 25)
    )
    signals[bullish] = 1

    # Bearish setup: EMA misalignment + RSI not oversold + Price below Bollinger mid
    bearish = (
        (df['ema9'] < df['ema20']) &
        (df['ema20'] < df['ema50']) &
        (df['ema50'] < df['sma200']) &
        (df['rsi_slow'] > 30) &
        (df['rsi_slow'] < 60) &
        (df['close'] < df['bb_mid']) &
        (df['adx'] > 25)
    )
    signals[bearish] = -1

    return signals


def generate_stop_loss_atr(df: pd.DataFrame, atr_multiplier: float = 2.0) -> Dict[str, pd.Series]:
    """
    Freqtrade pattern: Calculate stop-loss levels using ATR.
    Returns entry price + stop level for risk management.
    """
    return {
        "stop_loss_long": df['close'] - (df['atr'] * atr_multiplier),
        "stop_loss_short": df['close'] + (df['atr'] * atr_multiplier),
        "take_profit_long": df['close'] + (df['atr'] * atr_multiplier * 2),
        "take_profit_short": df['close'] - (df['atr'] * atr_multiplier * 2),
    }
