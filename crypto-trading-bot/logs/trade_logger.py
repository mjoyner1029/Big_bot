"""Structured CSV trade logger.

Every trade open/close event is appended to the trade_log.csv file
configured in CONFIG.  This provides a persistent audit trail
independent of the JSON state file.
"""
import csv
import logging
import os
from typing import Optional
from datetime import datetime, timezone

from config.config import CONFIG

_FIELDNAMES = [
    "timestamp", "symbol", "side", "entry_price", "tp_price", "sl_price",
    "confidence", "qty", "result", "exit_price", "pnl", "strategy_name",
]


def _ensure_header(path: str) -> None:
    """Write canonical header; rotate incompatible legacy schema files."""
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if not write_header:
        try:
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                first_row = next(reader, [])
            if first_row != _FIELDNAMES:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                legacy_path = f"{path}.legacy_{ts}"
                os.replace(path, legacy_path)
                logging.warning(
                    "[TradeLogger] Rotated legacy trade log schema to %s; starting fresh canonical log",
                    legacy_path,
                )
                write_header = True
        except Exception as e:
            logging.error(f"[TradeLogger] Failed to validate/rotate log header: {e}")

    if write_header:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()


def log_trade(
    timestamp: str,
    symbol: str,
    side: str,
    entry_price: float,
    tp_price: float = 0,
    sl_price: float = 0,
    confidence: float = 0,
    qty: float = 0,
    result: str = "open",
    exit_price: Optional[float] = None,
    pnl: Optional[float] = None,
    strategy_name: str = "",
) -> None:
    """Append one row to the trade log CSV."""
    path = CONFIG.get("trade_log_path", "logs/trade_log.csv")
    _ensure_header(path)

    row = {
        "timestamp": timestamp,
        "symbol": symbol,
        "side": side,
        "entry_price": round(entry_price, 4),
        "tp_price": round(tp_price, 4),
        "sl_price": round(sl_price, 4),
        "confidence": round(confidence, 4),
        "qty": round(qty, 6),
        "result": result,
        "exit_price": round(exit_price, 4) if exit_price is not None else "",
        "pnl": round(pnl, 2) if pnl is not None else "",
        "strategy_name": strategy_name,
    }

    try:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writerow(row)
    except Exception as e:
        logging.error(f"[TradeLogger] Failed to write row: {e}")


def initialize_trade_log(path: Optional[str] = None) -> str:
    """Ensure the canonical trade log file exists with the expected header."""
    resolved = path or CONFIG.get("trade_log_path", "logs/trade_log.csv")
    _ensure_header(resolved)
    return resolved
