"""
Indicators module — TA calculations and Freqtrade-style signal generation.

Standard TA:
  from indicators import add_ta_indicators
  
Freqtrade patterns (recommended):
  from indicators import add_freqtrade_indicators, get_freqtrade_signals
"""

from indicators.ta_indicators import (
    add_ta_indicators,
    get_latest_indicator_snapshot,
    add_freqtrade_indicators,
    get_freqtrade_signals,
    clear_freqtrade_cache,
)

from indicators.freqtrade_patterns import (
    populate_indicators_freqtrade,
    generate_entry_signals,
    generate_stop_loss_atr,
    rsi_fast,
    rsi_slow,
    stoch_fast,
    macd_fast,
    ema_family,
    sma_family,
    adx_trend,
    bollinger_bands,
    atr,
    obv,
    mfi,
    crossover,
)

__all__ = [
    # Standard TA interface
    "add_ta_indicators",
    "get_latest_indicator_snapshot",
    
    # Freqtrade interface (recommended)
    "add_freqtrade_indicators",
    "get_freqtrade_signals",
    "clear_freqtrade_cache",
    
    # Individual Freqtrade indicators for advanced use
    "populate_indicators_freqtrade",
    "generate_entry_signals",
    "generate_stop_loss_atr",
    "rsi_fast",
    "rsi_slow",
    "stoch_fast",
    "macd_fast",
    "ema_family",
    "sma_family",
    "adx_trend",
    "bollinger_bands",
    "atr",
    "obv",
    "mfi",
    "crossover",
]
