"""Trade executor — routes orders to the right venue.

Supports:
  • Paper trading  (simulated, logged locally)
  • Live crypto    (via ccxt → Coinbase / any ccxt-supported exchange)
  • Live stocks    (via Alpaca REST API — bracket orders with server-side TP/SL)

Server-side stop-loss sync:
  When placing live orders, TP and SL are attached as bracket / OCO orders
  directly on the broker so positions are protected even if the bot crashes.
"""
import logging
from typing import Dict, Any, Optional

from config.config import CONFIG, is_crypto
from trading.paper_trader import execute_paper_trade
from trading.portfolio_manager import PortfolioManager


_portfolio = PortfolioManager()


def execute_trade(signal: Dict[str, Any]) -> bool:
    """
    Execute a trade (paper or live) based on the signal dict.

    Returns True if the order was placed successfully (or simulated).
    """
    symbol = signal["symbol"]

    # ── Risk gate: check portfolio limits ─────────────────────────
    if not _portfolio.can_open_position(signal):
        logging.warning(f"[Executor] Position rejected by portfolio manager for {symbol}")
        return False

    if CONFIG["use_paper_trading"]:
        return _execute_paper(signal)

    # Live trading
    if is_crypto(symbol):
        return _execute_crypto_live(signal)
    return _execute_stock_live(signal)


# ── Paper trading ─────────────────────────────────────────────────

def _execute_paper(signal: Dict[str, Any]) -> bool:
    """Simulate the trade and record it."""
    execute_paper_trade(signal, _portfolio)
    return True


# ── Live crypto via ccxt ──────────────────────────────────────────

def _get_ccxt_exchange():
    """Lazy-init a ccxt exchange instance."""
    import ccxt
    exchange_id = CONFIG.get("exchange", "coinbase")
    exchange_cls = getattr(ccxt, exchange_id, None)
    if exchange_cls is None:
        raise ValueError(f"Unsupported ccxt exchange: {exchange_id}")
    return exchange_cls({
        "apiKey": CONFIG["coinbase_api_key"],
        "secret": CONFIG["coinbase_api_secret"],
        "password": CONFIG.get("coinbase_passphrase", ""),
        "enableRateLimit": True,
    })


def _execute_crypto_live(signal: Dict[str, Any]) -> bool:
    """Place a real crypto order via ccxt with server-side stop-loss.

    After the main market order fills, places a stop-loss and take-profit
    limit order so the position is protected even if the bot goes offline.
    """
    try:
        exchange = _get_ccxt_exchange()
        side = signal["side"]
        symbol = signal["symbol"].replace("-", "/")  # BTC-USD → BTC/USD
        amount = _portfolio.compute_position_size(signal)
        tp = signal.get("take_profit_price")
        sl = signal.get("stop_loss_price")

        order_type = "market"
        logging.info(f"[Crypto LIVE] {side.upper()} {amount:.6f} {symbol} (market)")

        if side == "buy":
            order = exchange.create_market_buy_order(symbol, amount)
        else:
            order = exchange.create_market_sell_order(symbol, amount)

        order_id = order.get("id", "N/A")
        fill_price = float(order.get("average", signal["entry_price"]))
        logging.info(f"[Crypto LIVE] Order filled: {order_id} @ ${fill_price:.2f}")

        # ── Server-side stop-loss & take-profit ───────────────────
        close_side = "sell" if side == "buy" else "buy"

        if sl and sl > 0:
            try:
                sl_order = exchange.create_order(
                    symbol, "stop_loss_limit", close_side, amount,
                    params={"stopPrice": sl, "price": sl * 0.995 if close_side == "sell" else sl * 1.005},
                )
                logging.info(f"[Crypto LIVE] SL order placed: {sl_order.get('id')} @ ${sl:.2f}")
                signal["broker_sl_order_id"] = sl_order.get("id")
            except Exception as e:
                logging.warning(f"[Crypto LIVE] Could not place server-side SL: {e}")

        if tp and tp > 0:
            try:
                tp_order = exchange.create_limit_order(symbol, close_side, amount, tp)
                logging.info(f"[Crypto LIVE] TP order placed: {tp_order.get('id')} @ ${tp:.2f}")
                signal["broker_tp_order_id"] = tp_order.get("id")
            except Exception as e:
                logging.warning(f"[Crypto LIVE] Could not place server-side TP: {e}")

        _portfolio.record_position(signal, order_id=order_id)
        return True

    except Exception as e:
        logging.error(f"[Crypto LIVE] Order failed: {e}", exc_info=True)
        return False


# ── Live stocks via Alpaca ────────────────────────────────────────

def _get_alpaca_api():
    """Lazy-init Alpaca REST client."""
    import alpaca_trade_api as tradeapi
    return tradeapi.REST(
        key_id=CONFIG["alpaca_api_key"],
        secret_key=CONFIG["alpaca_api_secret"],
        base_url=CONFIG["alpaca_base_url"],
        api_version="v2",
    )


def _execute_stock_live(signal: Dict[str, Any]) -> bool:
    """Place a real stock order via Alpaca bracket order (TP + SL attached).

    When take_profit_price and stop_loss_price are present in the signal,
    a bracket order is submitted so the exit orders live on Alpaca's servers.
    """
    try:
        api = _get_alpaca_api()
        side = signal["side"]
        symbol = signal["symbol"]
        qty = _portfolio.compute_position_size(signal)
        tp = signal.get("take_profit_price")
        sl = signal.get("stop_loss_price")

        logging.info(f"[Stock LIVE] {side.upper()} {qty:.4f} shares of {symbol}")

        order_params: Dict[str, Any] = dict(
            symbol=symbol,
            qty=round(qty, 4),
            side=side,
            type="market",
            time_in_force="day",
        )

        # Attach server-side TP/SL as a bracket order when both are available
        if tp and sl and tp > 0 and sl > 0:
            order_params["order_class"] = "bracket"
            order_params["take_profit"] = {"limit_price": round(tp, 2)}
            order_params["stop_loss"] = {"stop_price": round(sl, 2)}
            logging.info(f"[Stock LIVE] Bracket order: TP=${tp:.2f}, SL=${sl:.2f}")
        elif sl and sl > 0:
            order_params["order_class"] = "oto"
            order_params["stop_loss"] = {"stop_price": round(sl, 2)}
            logging.info(f"[Stock LIVE] OTO order with SL=${sl:.2f}")

        order = api.submit_order(**order_params)

        logging.info(f"[Stock LIVE] Order submitted: {order.id}")
        signal["broker_order_id"] = order.id
        _portfolio.record_position(signal, order_id=order.id)
        return True

    except Exception as e:
        logging.error(f"[Stock LIVE] Order failed: {e}", exc_info=True)
        return False


# ── Position monitoring (TP / SL / trailing) ─────────────────────

def check_open_positions(market_prices: Dict[str, float]) -> None:
    """
    Check all open positions against current prices and close any
    that have hit their TP, SL, or trailing-stop levels.
    """
    for pos in list(_portfolio.open_positions):
        sym = pos["symbol"]
        current_price = market_prices.get(sym)
        if current_price is None:
            continue

        side = pos["side"]
        tp = pos["take_profit_price"]
        sl = pos["stop_loss_price"]
        trail_pct = pos.get("trailing_stop_pct", 0)

        # Update trailing stop (only tightens, never loosens)
        if trail_pct > 0:
            if side == "buy":
                new_trail_sl = current_price * (1 - trail_pct)
                if new_trail_sl > sl:
                    pos["stop_loss_price"] = round(new_trail_sl, 4)
                    sl = pos["stop_loss_price"]
            else:
                new_trail_sl = current_price * (1 + trail_pct)
                if new_trail_sl < sl:
                    pos["stop_loss_price"] = round(new_trail_sl, 4)
                    sl = pos["stop_loss_price"]

        should_close = False
        result = "open"

        if side == "buy":
            if current_price >= tp:
                should_close, result = True, "tp_hit"
            elif current_price <= sl:
                should_close, result = True, "sl_hit"
        else:
            if current_price <= tp:
                should_close, result = True, "tp_hit"
            elif current_price >= sl:
                should_close, result = True, "sl_hit"

        if should_close:
            pnl = (current_price - pos["entry_price"]) if side == "buy" else (pos["entry_price"] - current_price)
            logging.info(
                f"[Monitor] Closing {sym} ({result}) @ {current_price:.2f}  "
                f"PnL=${pnl:.2f}"
            )
            _portfolio.close_position(pos, current_price, result)


def get_portfolio():
    """Expose the global portfolio manager for external callers."""
    return _portfolio
