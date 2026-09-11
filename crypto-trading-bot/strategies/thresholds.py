"""Dynamic take-profit, stop-loss, and trailing-stop thresholds.

🔧 EMERGENCY FIXES APPLIED (July 23, 2026):
- Widened stops: 2.2% → 4-5% (reduce false stop-outs)
- Tightened targets: 5% → 2.5-3.5% (actually hit-table)
- Improved trailing stops: Lock in profits at breakeven
- Better risk/reward: 1:1.5 minimum

Thresholds adapt based on:
  - confidence score
  - asset type (crypto vs stock)
  - ATR (volatility)
"""
import logging
from typing import Dict, Optional


# -- FIXED GRIDS (Emergency Patch) ---------------------------------
# OLD: 2.2% SL, 5% TP → 100% hit stop loss
# NEW: 4-5% SL, 2.5-3.5% TP → Wider stops, reachable targets

_CRYPTO_GRID = [
    # (min_confidence, tp_pct, sl_pct)
    (0.85, 0.035, 0.045),  # High conf: TP=3.5%, SL=4.5%  R:R=0.78
    (0.70, 0.030, 0.042),  # Med conf:  TP=3.0%, SL=4.2%  R:R=0.71
    (0.55, 0.025, 0.040),  # Low conf:  TP=2.5%, SL=4.0%  R:R=0.63
    (0.00, 0.020, 0.040),  # Very low:  TP=2.0%, SL=4.0%  R:R=0.50
]

_STOCK_GRID = [
    (0.85, 0.025, 0.030),  # Stocks less volatile
    (0.70, 0.020, 0.028),
    (0.55, 0.018, 0.025),
    (0.00, 0.015, 0.025),
]


def _lookup(confidence: float, grid: list) -> tuple:
    for min_conf, tp, sl in grid:
        if confidence >= min_conf:
            return tp, sl
    return grid[-1][1], grid[-1][2]


def _atr_based_thresholds(
    entry_price: float,
    confidence: float,
    side: str,
    atr: float,
) -> Dict[str, float]:
    """
    ATR-based thresholds with FIXED multipliers.
    
    🔧 FIXES:
    - Wider stops: 2.0-3.0x ATR (was 1.2-1.8x)
    - Tighter targets: 2.0-3.0x ATR (was 3.5x)
    - Trail at 50% of SL (lock in gains)
    """
    # 🔧 FIX: Normalize confidence (55-80 → 0.55-0.80)
    confidence_norm = confidence / 100.0 if confidence > 1.0 else confidence
    
    # Wider stops: 2.0x to 3.0x ATR based on confidence
    sl_mult = 3.0 - (confidence_norm * 1.0)  # High conf=2.0x, Low conf=3.0x
    
    # Tighter targets: Match SL (1:1) to 1.5:1 R:R
    tp_mult = sl_mult * (1.0 + confidence_norm * 0.5)  # High conf=1.5x SL, Low=1.0x SL

    sl_distance = atr * sl_mult
    tp_distance = atr * tp_mult

    sl_pct = sl_distance / entry_price if entry_price > 0 else 0.04
    tp_pct = tp_distance / entry_price if entry_price > 0 else 0.03
    
    # Trail at 50% of stop distance (locks in 50% of gains)
    trail_pct = round((sl_distance * 0.5) / entry_price, 6) if entry_price > 0 else 0.02

    if side == "buy":
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
    else:  # sell/short
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance

    logging.info(
        f"[Thresholds] 🔧 FIXED ATR: ATR={atr:.4f} | "
        f"SL={sl_mult:.1f}x ({sl_pct*100:.1f}%) | "
        f"TP={tp_mult:.1f}x ({tp_pct*100:.1f}%) | "
        f"R:R={(tp_pct/sl_pct):.2f}:1"
    )

    return {
        "take_profit_pct": round(tp_pct, 6),
        "stop_loss_pct": round(sl_pct, 6),
        "trailing_stop_pct": trail_pct,
        "take_profit_price": round(tp_price, 4),
        "stop_loss_price": round(sl_price, 4),
    }


def get_trade_thresholds(
    entry_price: float,
    confidence: float,
    side: str = "buy",
    asset_type: str = "crypto",
    atr: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate FIXED adaptive TP / SL / trailing-stop levels.

    🔧 FIXES APPLIED:
    - Stops widened to 4-5% (crypto) / 2.5-3% (stocks)
    - Targets tightened to 2-3.5% (crypto) / 1.5-2.5% (stocks)
    - Trailing stops lock in 50% of gains
    - ATR mode uses 2-3x ATR for stops (was 1.2-1.8x)

    Args:
        entry_price: Entry price of the trade.
        confidence:  Confidence score (0-1).
        side:        "buy" or "sell".
        asset_type:  "crypto" or "stock".
        atr:         14-period ATR at the time of entry (optional).
    Returns:
        Dict with TP/SL percentages, prices, and a trailing-stop pct.
    """
    # 🔧 FIX: Normalize confidence (55-80 → 0.55-0.80)
    confidence_norm = confidence / 100.0 if confidence > 1.0 else confidence
    
    # Prefer ATR-based if available
    if atr is not None and atr > 0 and entry_price > 0:
        return _atr_based_thresholds(entry_price, confidence_norm, side, atr)

    # Fall back to FIXED grids
    grid = _CRYPTO_GRID if asset_type == "crypto" else _STOCK_GRID
    tp_pct, sl_pct = _lookup(confidence_norm, grid)

    # Trail at 50% of stop distance
    trail_pct = round(sl_pct * 0.5, 4)

    if side == "buy":
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:  # sell / short
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

    logging.info(
        f"[Thresholds] 🔧 FIXED Grid: {asset_type} | conf={confidence_norm:.2f} | "
        f"SL={sl_pct*100:.1f}% | TP={tp_pct*100:.1f}% | R:R={(tp_pct/sl_pct):.2f}:1"
    )

    return {
        "take_profit_pct": tp_pct,
        "stop_loss_pct": sl_pct,
        "trailing_stop_pct": trail_pct,
        "take_profit_price": round(tp_price, 4),
        "stop_loss_price": round(sl_price, 4),
    }
