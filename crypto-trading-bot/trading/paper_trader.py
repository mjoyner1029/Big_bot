import logging
from typing import Dict, Any

def record_paper_trade(signal: Dict[str, Any]) -> None:
    """
    Log a simulated trade.
    Args:
        signal: Trade signal dict.
    """
    logging.info(f"Simulated trade logged: {signal}")
