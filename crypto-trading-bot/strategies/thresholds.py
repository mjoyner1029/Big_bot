"""Dynamic take-profit, stop-loss, and trailing-stop thresholds.

Thresholds adapt based on:
  - confidence score
  - asset type (crypto is more volatile -> wider bands)
  - trade side (buy vs. sell / short)
  - ATR (Average True Range) when available for volatility-adjusted sizing
"""
import logging
from typing import Dict, Optional


# -- Base grids -- (min_confidence, tp_pct, sl_pct) ----------------
_CRYPTO_GRID = [
    (0.85, 0.18, 0.04),   # very high confidence
    (0.70, 0.12, 0.035),
    (0.55, 0.07, 0.025),
    (0.00, 0.04, 0.02),   # low confidence
]

_STOCK_GRID = [
    (0.85, 0.10, 0.025),
    (0.70, 0.06, 0.02),
    (0.55, 0.04, 0.015),
    (0.00, 0.025, 0.01),
]


def _lookup(confidence: float, grid: list) -> tuple:
    for min_conf, tp, sl in grid:
        if confidence >= min_conf:
            return tp, sl
    return grid[-1][1], grid[-1][2]


# -- ATR-based thresholds ------------------------------------------

def _atr_based_thresholds(
    entry_price: float,
    confidence: float,
    side: str,
    atr: float,
) -> Dict[str, float]:
    """
    Compute TP/SL/trailing-stop from the 14-period ATR.

    Multipliers scale with confidence:
      - High confidence  -> tighter SL (1.5x ATR), wider TP (3.5x ATR)
      - Low confidence   -> wider SL (2.5x ATR), tighter TP (1.5x ATR)
    """
    sl_mult = 2.5 - confidence * 1.0
    tp_mult = 1.5 + confidence * 2.0

    sl_distance = atr * sl_mult
    tp_distance = atr * tp_mult

    sl_pct = sl_distance / entry_price if entry_price > 0 else 0.02
    tp_pct = tp_distance / entry_price if entry_price > 0 else 0.04
    trail_pct = round((atr * sl_mult * 0.6) / entry_price, 6) if entry_price > 0 else 0.012

    if side == "buy":
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
    else:
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance

    logging.info(
        f"[Thresholds] ATR-based: ATR={atr:.4f}  "
        f"SL_mult={sl_mult:.1f}x  TP_mult={tp_mult:.1f}x  "
        f"TP=${tp_price:.2f}  SL=${sl_price:.2f}"
    )

    return {
        "take_profit_pct": round(tp_pct, 6),
        "stop_loss_pct": round(sl_pct, 6),
        "trailing_stop_pct": trail_pct,
        "take_profit_price": round(tp_price, 4),
        "stop_loss_price": round(sl_price, 4),
    }


# -- Main entry point ----------------------------------------------

def get_trade_thresholds(
    entry_price: float,
    confidence: float,
    side: str = "buy",
    asset_type: str = "crypto",
    atr: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate adaptive TP / SL / trailing-stop levels.

    When ATR (Average True Range) is provided, TP and SL distances are
    computed from real market volatility.  Otherwise falls back to
    percentage-based grids calibrated per asset class.

    Args:
        entry_price: Entry price of the trade.
        confidence:  Confidence score (0-1).
        side:        "buy" or "sell".
        asset_type:  "crypto" or "stock".
        atr:         14-period ATR at the time of entry (optional).
    Returns:
        Dict with TP/SL percentages, prices, and a trailing-stop pct.
    """
    if atr is not None and atr > 0 and entry_price > 0:
        return _atr_based_thresholds(entry_price, confidence, side, atr)

    # Fall back to grid-based thresholds
    grid = _CRYPTO_GRID if asset_type == "crypto" else _STOCK_GRID
    tp_pct, sl_pct = _lookup(confidence, grid)

    # Trailing stop is tighter than the initial SL
    trail_pct = round(sl_pct * 0.6, 4)

    if side == "buy":
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:  # sell / short
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

    return {
        "take_profit_pct": tp_pct,
        "stop_loss_pct": sl_pct,
        "trailing_stop_pct": trail_pct,
        "take_profit_price": round(tp_price, 4),
        "stop_loss_price": round(sl_price, 4),
    }
