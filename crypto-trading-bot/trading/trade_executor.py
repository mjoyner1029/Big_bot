from config.config import CONFIG
import logging
from typing import Dict, Any

def execute_trade(signal: Dict[str, Any]) -> None:
    """
    Execute a trade (paper or live).
    Args:
        signal: Trade signal dict.
    """
    if CONFIG["use_paper_trading"]:
        logging.info(f"[PAPER TRADE] Buying {signal['symbol']} at ${signal['entry_price']:.2f}")
        logging.info(f"TP: ${signal['take_profit_price']}, SL: ${signal['stop_loss_price']}, Confidence: {signal['confidence']:.2f}")
        # Add logging here or call a trade logger
    else:
        logging.info("[LIVE TRADE] Executing real trade... (not implemented)")
