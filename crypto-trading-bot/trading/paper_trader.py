"""Paper-trading engine — simulates order fills and tracks P&L.

All trades are recorded to the CSV trade log and to the in-memory
portfolio manager so that position limits and risk gates work
identically in paper and live modes.
"""
import logging
from typing import Dict, Any
from datetime import datetime, timezone

from logs.trade_logger import log_trade


def execute_paper_trade(signal: Dict[str, Any], portfolio) -> None:
    """
    Simulate a market fill for *signal* and update *portfolio*.

    Args:
        signal:    Trade signal dict from the strategy engine.
        portfolio: PortfolioManager instance.
    """
    symbol = signal["symbol"]
    side = signal["side"]
    entry_price = signal["entry_price"]
    confidence = signal["confidence"]
    tp = signal.get("take_profit_price", 0)
    sl = signal.get("stop_loss_price", 0)
    trail = signal.get("trailing_stop_pct", 0)

    qty = portfolio.compute_position_size(signal)
    cost = qty * entry_price

    logging.info(
        f"[PAPER] {side.upper()} {qty:.6f} {symbol} @ ${entry_price:.2f}  "
        f"cost=${cost:.2f}  conf={confidence:.3f}  TP=${tp:.2f}  SL=${sl:.2f}"
    )

    # Record in portfolio
    portfolio.record_position(signal, qty=qty)

    # Persist to CSV log
    log_trade(
        timestamp=signal.get("timestamp", datetime.now(timezone.utc).isoformat()),
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        tp_price=tp,
        sl_price=sl,
        confidence=confidence,
        qty=qty,
        result="open",
    )
