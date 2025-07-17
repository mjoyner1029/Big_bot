from typing import Callable, List, Dict, Any
import logging
import pandas as pd

def run_backtest(strategy: Callable[[pd.DataFrame], Dict[str, Any]], historical_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Run backtest on a strategy.
    Args:
        strategy: Function that generates trade signals.
        historical_df: DataFrame with historical data.
    Returns:
        List of trade signals.
    """
    results = []
    for i in range(50, len(historical_df)):
        df_slice = historical_df.iloc[:i]
        try:
            signal = strategy(df_slice)
            if signal:
                results.append(signal)
        except Exception as e:
            logging.error(f"Backtest error at index {i}: {e}")
    return results
