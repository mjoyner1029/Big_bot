"""Market regime detection — classifies the current market as trending,
ranging, or high-volatility and adapts strategy parameters accordingly.

A professional trader uses different playbooks for different market environments:
  • Trending (ADX > 25, clear EMA alignment)  → wider TP, tighter SL, let winners run
  • Ranging  (ADX < 20, price bouncing between bands) → tighter TP, mean-reversion bias
  • Volatile (ATR spike, VIX-like elevated)   → reduce position size, wider SL
  • Quiet    (ATR compressed, narrow BBands)   → wait or scalp with tight stops

The detector outputs a regime label + parameter adjustments that override defaults.
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.config import CONFIG


# ── Regime labels ─────────────────────────────────────────────────
REGIME_TRENDING_UP = "trending_up"
REGIME_TRENDING_DOWN = "trending_down"
REGIME_RANGING = "ranging"
REGIME_VOLATILE = "volatile"
REGIME_QUIET = "quiet"
REGIME_UNKNOWN = "unknown"


# ── Regime parameter adjustments ──────────────────────────────────
# Each regime returns multipliers for key strategy parameters.
_REGIME_PARAMS: Dict[str, Dict[str, float]] = {
    REGIME_TRENDING_UP: {
        "tp_mult": 1.3,          # wider TP — let trends run
        "sl_mult": 0.9,          # tighter SL — don't give back gains
        "size_mult": 1.0,        # normal size
        "confidence_adj": 0.03,  # slight boost for trend-following signals
        "min_confidence": 0.50,  # lower bar — momentum carries
    },
    REGIME_TRENDING_DOWN: {
        "tp_mult": 1.3,
        "sl_mult": 0.9,
        "size_mult": 1.0,
        "confidence_adj": 0.03,  # boost for short signals
        "min_confidence": 0.50,
    },
    REGIME_RANGING: {
        "tp_mult": 0.7,          # tighter TP — take quick profits
        "sl_mult": 1.0,
        "size_mult": 0.8,        # reduce size — mean-reversion
        "confidence_adj": 0.0,
        "min_confidence": 0.58,  # higher bar in choppy markets
    },
    REGIME_VOLATILE: {
        "tp_mult": 1.1,
        "sl_mult": 1.4,          # wider SL — don't get shaken out by noise
        "size_mult": 0.5,        # HALF size — protect capital
        "confidence_adj": -0.05, # penalise — harder to predict
        "min_confidence": 0.62,  # high bar required
    },
    REGIME_QUIET: {
        "tp_mult": 0.6,          # tight TP — small moves
        "sl_mult": 0.8,
        "size_mult": 0.6,        # smaller size — not much to capture
        "confidence_adj": 0.0,
        "min_confidence": 0.55,
    },
    REGIME_UNKNOWN: {
        "tp_mult": 1.0,
        "sl_mult": 1.0,
        "size_mult": 0.8,
        "confidence_adj": 0.0,
        "min_confidence": 0.55,
    },
}


def detect_regime(df: pd.DataFrame, lookback: int = 48) -> Tuple[str, Dict[str, float]]:
    """Classify the current market regime from recent price data.

    Uses:
      1. ADX (14) for trend strength
      2. ADX +DI / -DI for trend direction
      3. Bollinger Bandwidth for volatility expansion/compression
      4. ATR percentile rank for volatility spike detection
      5. EMA alignment (9 vs 20 vs 50) for trend confirmation

    Args:
        df:       DataFrame with TA indicators already added (must have adx,
                  bb_width, atr, ema9, ema20, ema50, Close columns).
        lookback: Number of recent bars for percentile calculations.
    Returns:
        (regime_label, params_dict) with parameter multipliers.
    """
    if df is None or len(df) < lookback:
        return REGIME_UNKNOWN, _REGIME_PARAMS[REGIME_UNKNOWN]

    recent = df.tail(lookback)
    last = df.iloc[-1]

    # ── Extract indicators ────────────────────────────────────────
    adx = last.get("adx")
    adx_pos = last.get("adx_pos", 0)
    adx_neg = last.get("adx_neg", 0)
    bb_width = last.get("bb_width")
    atr = last.get("atr")
    ema9 = last.get("ema9")
    ema20 = last.get("ema20")
    ema50 = last.get("ema50")
    close = last.get("Close")

    # Missing indicators → unknown
    if any(v is None or (isinstance(v, float) and np.isnan(v))
           for v in [adx, bb_width, atr]):
        return REGIME_UNKNOWN, _REGIME_PARAMS[REGIME_UNKNOWN]

    adx = float(adx)
    bb_width = float(bb_width)
    atr = float(atr)

    # ── ATR percentile (is volatility elevated?) ──────────────────
    atr_series = recent["atr"].dropna()
    if len(atr_series) > 10:
        atr_pctile = (atr_series < atr).mean()  # what % of recent bars had lower ATR
    else:
        atr_pctile = 0.5

    # ── BB width percentile (is bandwidth expanded?) ──────────────
    bbw_series = recent["bb_width"].dropna()
    if len(bbw_series) > 10:
        bbw_pctile = (bbw_series < bb_width).mean()
    else:
        bbw_pctile = 0.5

    # ── EMA alignment ─────────────────────────────────────────────
    ema_bullish = False
    ema_bearish = False
    if all(v is not None and not (isinstance(v, float) and np.isnan(v))
           for v in [ema9, ema20, ema50, close]):
        ema9, ema20, ema50, close = float(ema9), float(ema20), float(ema50), float(close)
        ema_bullish = (close > ema9 > ema20 > ema50)
        ema_bearish = (close < ema9 < ema20 < ema50)

    # ── Classification logic ──────────────────────────────────────
    regime = REGIME_UNKNOWN

    # Volatile: ATR spike + BB expansion
    if atr_pctile > 0.85 and bbw_pctile > 0.80:
        regime = REGIME_VOLATILE

    # Strong trend: ADX > 25 + directional EMA alignment
    elif adx > 25:
        if ema_bullish or adx_pos > adx_neg:
            regime = REGIME_TRENDING_UP
        elif ema_bearish or adx_neg > adx_pos:
            regime = REGIME_TRENDING_DOWN
        else:
            regime = REGIME_TRENDING_UP if adx_pos > adx_neg else REGIME_TRENDING_DOWN

    # Quiet: very low volatility, compressed bands
    elif atr_pctile < 0.20 and bbw_pctile < 0.25:
        regime = REGIME_QUIET

    # Ranging: low ADX, price oscillating
    elif adx < 20:
        regime = REGIME_RANGING

    # Moderate trend (20 < ADX < 25) — classify as ranging-to-trending
    else:
        if ema_bullish:
            regime = REGIME_TRENDING_UP
        elif ema_bearish:
            regime = REGIME_TRENDING_DOWN
        else:
            regime = REGIME_RANGING

    params = _REGIME_PARAMS[regime].copy()

    logging.debug(
        f"[Regime] {regime}  ADX={adx:.1f}  ATR_pctile={atr_pctile:.2f}  "
        f"BBW_pctile={bbw_pctile:.2f}  EMA_bull={ema_bullish}  EMA_bear={ema_bearish}"
    )

    return regime, params


def get_regime_info(df: pd.DataFrame) -> Dict:
    """Return full regime info dict for logging / dashboard consumption."""
    regime, params = detect_regime(df)
    return {
        "regime": regime,
        **params,
    }
