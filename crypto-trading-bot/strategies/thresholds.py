from typing import Dict

def get_trade_thresholds(entry_price: float, confidence: float) -> Dict[str, float]:
    """
    Calculate take-profit and stop-loss thresholds based on confidence.
    Args:
        entry_price: Entry price of the trade.
        confidence: Confidence score (0-1).
    Returns:
        Dict with TP/SL percentages and prices.
    """
    if confidence >= 0.85:
        tp_pct = 0.15
        sl_pct = 0.03
    elif confidence >= 0.6:
        tp_pct = 0.10
        sl_pct = 0.025
    else:
        tp_pct = 0.05
        sl_pct = 0.02

    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)

    return {
        "take_profit_pct": tp_pct,
        "stop_loss_pct": sl_pct,
        "take_profit_price": round(tp_price, 4),
        "stop_loss_price": round(sl_price, 4)
    }
