"""Position reconciler — syncs local state with broker reality.

At every cycle (or on startup) the reconciler:
  1. Fetches real positions from Alpaca (stocks) and ccxt (crypto).
  2. Compares them with the local PortfolioManager's open_positions list.
  3. Resolves discrepancies:
       • Phantom local positions (we think we hold it, broker says no) → close locally
       • Missing local positions (broker has it, we don't track it)   → adopt into local state
       • Quantity mismatch (broker qty ≠ local qty)                    → align to broker
  4. Logs every adjustment so the trade audit trail stays intact.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from config.config import CONFIG, is_crypto


# ── Broker position fetchers ─────────────────────────────────────

def _fetch_alpaca_positions() -> Dict[str, Dict[str, Any]]:
    """Return {symbol: {qty, side, current_price, avg_entry}} from Alpaca."""
    api_key = CONFIG.get("alpaca_api_key", "")
    api_secret = CONFIG.get("alpaca_api_secret", "")
    if not api_key or not api_secret:
        return {}

    try:
        import requests
        base = CONFIG.get("alpaca_base_url", "https://paper-api.alpaca.markets")
        resp = requests.get(
            f"{base}/v2/positions",
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
            timeout=15,
        )
        if not resp.ok:
            logging.warning(f"[Reconciler] Alpaca positions HTTP {resp.status_code}")
            return {}

        positions = {}
        for p in resp.json():
            sym = p["symbol"]
            qty = abs(float(p["qty"]))
            side = "buy" if float(p["qty"]) > 0 else "sell"
            positions[sym] = {
                "qty": qty,
                "side": side,
                "current_price": float(p.get("current_price", 0)),
                "avg_entry": float(p.get("avg_entry_price", 0)),
                "market_value": float(p.get("market_value", 0)),
            }
        return positions

    except Exception as e:
        logging.warning(f"[Reconciler] Failed to fetch Alpaca positions: {e}")
        return {}


def _fetch_ccxt_positions() -> Dict[str, Dict[str, Any]]:
    """Return {symbol: {qty, side, current_price, avg_entry}} from ccxt exchange."""
    ex_key = CONFIG.get("coinbase_api_key", "")
    ex_secret = CONFIG.get("coinbase_api_secret", "")
    if not ex_key or not ex_secret:
        return {}

    try:
        import ccxt
        exchange_id = CONFIG.get("exchange", "coinbase")
        exchange_cls = getattr(ccxt, exchange_id, None)
        if exchange_cls is None:
            return {}

        ex = exchange_cls({
            "apiKey": ex_key,
            "secret": ex_secret,
            "password": CONFIG.get("coinbase_passphrase", ""),
            "enableRateLimit": True,
        })

        balance = ex.fetch_balance()
        positions = {}

        # For spot exchanges, non-zero balances are "positions"
        for coin, amount in balance.get("total", {}).items():
            if coin in ("USD", "USDT", "USDC", "BUSD") or amount <= 0:
                continue

            sym_yf = f"{coin}-USD"  # normalize to our convention
            try:
                ticker = ex.fetch_ticker(f"{coin}/USD")
                cur_price = float(ticker.get("last", 0))
            except Exception:
                cur_price = 0

            positions[sym_yf] = {
                "qty": float(amount),
                "side": "buy",  # spot is always long
                "current_price": cur_price,
                "avg_entry": 0,  # spot exchanges don't track cost basis well
                "market_value": float(amount) * cur_price,
            }

        return positions

    except Exception as e:
        logging.warning(f"[Reconciler] Failed to fetch ccxt positions: {e}")
        return {}


# ── Core reconciliation ──────────────────────────────────────────

def reconcile(portfolio_manager) -> Dict[str, Any]:
    """Run a full reconciliation cycle.

    Args:
        portfolio_manager:  The running PortfolioManager instance.

    Returns:
        Summary dict: {phantoms_closed, orphans_adopted, qty_adjusted, errors}
    """
    if CONFIG.get("use_paper_trading", True):
        logging.debug("[Reconciler] Paper trading — skipping reconciliation")
        return {"phantoms_closed": 0, "orphans_adopted": 0, "qty_adjusted": 0, "errors": 0}

    logging.info("[Reconciler] Starting position reconciliation …")

    # 1. Fetch combined broker positions
    broker_positions: Dict[str, Dict[str, Any]] = {}
    broker_positions.update(_fetch_alpaca_positions())
    broker_positions.update(_fetch_ccxt_positions())

    local_positions = {p["symbol"]: p for p in portfolio_manager.open_positions}

    summary = {"phantoms_closed": 0, "orphans_adopted": 0, "qty_adjusted": 0, "errors": 0}

    # 2. Phantom positions (local thinks we hold, broker says no)
    for sym, local_pos in list(local_positions.items()):
        if sym not in broker_positions:
            logging.warning(
                f"[Reconciler] PHANTOM position: {sym} — locally tracked but NOT on broker. "
                f"qty={local_pos['qty']}, entry=${local_pos['entry_price']:.2f}. Closing locally."
            )
            try:
                # Close the phantom position at the last known price (entry as fallback)
                exit_price = local_pos.get("current_price", local_pos["entry_price"])
                portfolio_manager.close_position(
                    local_pos, exit_price=exit_price, result="reconciler_phantom"
                )
                summary["phantoms_closed"] += 1
            except Exception as e:
                logging.error(f"[Reconciler] Error closing phantom {sym}: {e}")
                summary["errors"] += 1

    # 3. Orphan positions (broker has it, we don't track it)
    for sym, broker_pos in broker_positions.items():
        if sym not in local_positions:
            # Only adopt if the value is meaningful (> $10)
            if broker_pos["market_value"] < 10:
                logging.debug(f"[Reconciler] Ignoring dust position: {sym} (${broker_pos['market_value']:.2f})")
                continue

            logging.warning(
                f"[Reconciler] ORPHAN position: {sym} — on broker but NOT tracked locally. "
                f"qty={broker_pos['qty']}, price=${broker_pos['current_price']:.2f}. Adopting."
            )
            try:
                # Create a synthetic signal to record the position
                signal = {
                    "symbol": sym,
                    "side": broker_pos["side"],
                    "entry_price": broker_pos["avg_entry"] or broker_pos["current_price"],
                    "asset_type": "crypto" if is_crypto(sym) else "stock",
                    "confidence": 0.5,
                    "take_profit_price": 0,
                    "stop_loss_price": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                portfolio_manager.record_position(
                    signal, order_id="reconciler_adopted", qty=broker_pos["qty"]
                )
                summary["orphans_adopted"] += 1
            except Exception as e:
                logging.error(f"[Reconciler] Error adopting orphan {sym}: {e}")
                summary["errors"] += 1

    # 4. Quantity mismatches
    for sym in set(local_positions) & set(broker_positions):
        local_qty = local_positions[sym]["qty"]
        broker_qty = broker_positions[sym]["qty"]
        tolerance = max(local_qty * 0.01, 0.0001)  # 1% or 0.0001

        if abs(local_qty - broker_qty) > tolerance:
            logging.warning(
                f"[Reconciler] QTY MISMATCH: {sym} — "
                f"local={local_qty:.6f}, broker={broker_qty:.6f}. Adjusting to broker."
            )
            local_positions[sym]["qty"] = broker_qty
            summary["qty_adjusted"] += 1

    if any(v > 0 for k, v in summary.items() if k != "errors"):
        portfolio_manager.save_state()

    logging.info(
        f"[Reconciler] Done — phantoms={summary['phantoms_closed']}, "
        f"orphans={summary['orphans_adopted']}, qty_adjusted={summary['qty_adjusted']}, "
        f"errors={summary['errors']}"
    )
    return summary
