from typing import List, Dict

def compute_roi(results: List[Dict]) -> float:
    """
    Compute average ROI from backtest results.
    Args:
        results: List of trade dicts.
    Returns:
        Average ROI.
    """
    total = 0
    for trade in results:
        profit = trade["take_profit_price"] - trade["entry_price"]
        total += profit
    return round((total / len(results)) if results else 0, 2)
