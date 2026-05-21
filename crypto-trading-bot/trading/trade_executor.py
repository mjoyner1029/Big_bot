"""Trade executor — routes orders to the right venue.

Supports:
  • Paper trading  (simulated, logged locally)
  • Live crypto    (via ccxt → Coinbase / any ccxt-supported exchange)
  • Live stocks    (via Alpaca REST API — bracket orders with server-side TP/SL)

Server-side stop-loss sync:
  When placing live orders, TP and SL are attached as bracket / OCO orders
  directly on the broker so positions are protected even if the bot crashes.

High-Frequency Mode:
  Inspired by Claude's autonomous 48hr experiment (+1,322% return, 5,200 trades).
  Enables fast execution with minimal latency and comprehensive cost tracking.

PRODUCTION SAFETY:
  - Rate limiting to prevent API bans
  - Order validation before execution
  - Exponential backoff retry logic
"""
import logging
from typing import Dict, Any, Optional, Tuple

from config.config import CONFIG, is_crypto, get_mode_config
from trading.paper_trader import execute_paper_trade
from trading.portfolio_manager import PortfolioManager
from trading.rate_limiter import (
    get_coinbase_limiter,
    get_alpaca_limiter,
    get_default_retry,
)
from strategies.cost_tracker import get_cost_tracker
from strategies.velocity_optimizer import get_velocity_optimizer


_portfolio = PortfolioManager()


def execute_trade(signal: Dict[str, Any]) -> bool:
    """
    Execute a trade (paper or live) based on the signal dict.

    Returns True if the order was placed successfully (or simulated).
    """
    symbol = signal["symbol"]
    
    # Get trackers
    velocity_opt = get_velocity_optimizer()
    cost_tracker = get_cost_tracker()

    # ── Risk gate: check portfolio limits ─────────────────────────
    if not _portfolio.can_open_position(signal):
        logging.warning(f"[Executor] Position rejected by portfolio manager for {symbol}")
        velocity_opt.record_signal(executed=False)
        return False

    # Execute the trade
    success = False
    if CONFIG["use_paper_trading"]:
        success = _execute_paper(signal)
    elif is_crypto(symbol):
        success = _execute_crypto_live(signal)
    else:
        success = _execute_stock_live(signal)
    
    # Track velocity
    velocity_opt.record_signal(executed=success)
    
    # Track transaction costs
    if success:
        qty = _portfolio.compute_position_size(signal)
        trade_value = qty * signal["entry_price"]
        cost_tracker.record_trade_cost(symbol, trade_value)
    
    return success


# ── Paper trading ─────────────────────────────────────────────────

def _execute_paper(signal: Dict[str, Any]) -> bool:
    """Simulate the trade and record it."""
    execute_paper_trade(signal, _portfolio)
    return True


# ── Order Validation (PRODUCTION SAFETY) ─────────────────────────

def _validate_order(signal: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate order before execution to prevent fat-finger errors.
    
    Returns:
        (is_valid, reason)
    """
    if not CONFIG.get("order_validation_enabled", True):
        return True, "validation disabled"
    
    symbol = signal["symbol"]
    side = signal["side"]
    amount = _portfolio.compute_position_size(signal)
    entry_price = signal.get("entry_price", 0)
    
    if entry_price <= 0:
        return False, f"Invalid entry price: ${entry_price}"
    
    order_value = amount * entry_price
    
    # Check maximum single trade value
    max_single = CONFIG.get("max_single_trade_value", CONFIG["capital"] * 0.5)
    if order_value > max_single:
        return False, (
            f"Order value ${order_value:.2f} exceeds max ${max_single:.2f}. "
            f"Symbol: {symbol}, Amount: {amount:.6f}, Price: ${entry_price:.2f}"
        )
    
    # Check minimum trade value (avoid dust trades)
    min_trade = CONFIG.get("min_trade_value", 10)
    if order_value < min_trade:
        return False, f"Order value ${order_value:.2f} below minimum ${min_trade:.2f}"
    
    # Verify amount is reasonable
    if amount <= 0:
        return False, f"Invalid amount: {amount}"
    
    # For crypto, check absurd amounts
    if is_crypto(symbol):
        if amount > 1e6:  # More than 1 million units seems suspicious
            return False, f"Suspicious crypto amount: {amount:.2f} {symbol}"
    else:
        # For stocks, check absurd share counts
        if amount > 100000:  # More than 100k shares seems suspicious
            return False, f"Suspicious stock amount: {amount:.0f} shares of {symbol}"
    
    # Check max position value
    max_position = CONFIG.get("max_position_value", CONFIG["capital"])
    if order_value > max_position:
        return False, (
            f"Order value ${order_value:.2f} exceeds max position ${max_position:.2f}"
        )
    
    # Verify symbol format
    if is_crypto(symbol):
        if not ("/" in symbol or "-" in symbol):
            return False, f"Invalid crypto symbol format: {symbol}"
    
    # Log successful validation
    logging.info(
        f"[Order Validation] APPROVED: {side.upper()} {amount:.6f} {symbol} "
        f"@ ${entry_price:.2f} = ${order_value:.2f}"
    )
    
    return True, "validated"


# ── Paper trading ─────────────────────────────────────────────────

def _execute_paper(signal: Dict[str, Any]) -> bool:
    """Simulate the trade and record it."""
    execute_paper_trade(signal, _portfolio)
    return True


# ── Live crypto via ccxt ──────────────────────────────────────────

def _get_ccxt_exchange():
    """Lazy-init a ccxt exchange instance with mode-aware optimizations."""
    import ccxt
    exchange_id = CONFIG.get("exchange", "coinbase")
    exchange_cls = getattr(ccxt, exchange_id, None)
    if exchange_cls is None:
        raise ValueError(f"Unsupported ccxt exchange: {exchange_id}")
    
    # High-frequency mode: aggressive timeouts
    trading_mode = CONFIG.get("trading_mode", "balanced")
    timeout_ms = 5000 if trading_mode == "claude_hf" else 10000
    
    return exchange_cls({
        "apiKey": CONFIG["coinbase_api_key"],
        "secret": CONFIG["coinbase_api_secret"],
        "password": CONFIG.get("coinbase_passphrase", ""),
        "enableRateLimit": True,
        "timeout": timeout_ms,  # Faster for HF mode
    })


def _execute_crypto_live(signal: Dict[str, Any]) -> bool:
    """Place a real crypto order via ccxt with server-side stop-loss.

    After the main market order fills, places a stop-loss and take-profit
    limit order so the position is protected even if the bot goes offline.
    
    PRODUCTION SAFETY:
    - Order validation before execution
    - Rate limiting to prevent API bans
    - Retry logic with exponential backoff
    """
    # STEP 1: Validate order
    is_valid, reason = _validate_order(signal)
    if not is_valid:
        logging.error(f"[Crypto LIVE] ORDER REJECTED: {reason}")
        return False
    
    # STEP 2: Get rate limiter and retry handler
    rate_limiter = get_coinbase_limiter()
    retry_handler = get_default_retry()
    
    try:
        def _place_order():
            """Inner function for retry logic"""
            # Rate limit before API call
            if CONFIG.get("rate_limiting_enabled", True):
                wait_time = rate_limiter.wait_if_needed()
                if wait_time > 0:
                    logging.debug(f"[RateLimit] Waited {wait_time:.2f}s before order")
            
            exchange = _get_ccxt_exchange()
            side = signal["side"]
            symbol = signal["symbol"].replace("-", "/")  # BTC-USD → BTC/USD
            amount = _portfolio.compute_position_size(signal)
            
            logging.warning(
                f"[Crypto LIVE] EXECUTING: {side.upper()} {amount:.6f} {symbol} (market order)"
            )

            if side == "buy":
                order = exchange.create_market_buy_order(symbol, amount)
            else:
                order = exchange.create_market_sell_order(symbol, amount)
            
            return order
        
        # Execute with retry logic
        if CONFIG.get("retry_on_failure", True):
            order = retry_handler.execute(_place_order)
        else:
            order = _place_order()

        order_id = order.get("id", "N/A")
        fill_price = float(order.get("average", signal["entry_price"]))
        fill_value = float(order.get("cost", 0))
        logging.warning(
            f"[Crypto LIVE] ORDER FILLED: {order_id} @ ${fill_price:.2f} "
            f"(total: ${fill_value:.2f})"
        )

        # ── Server-side stop-loss & take-profit ───────────────────
        tp = signal.get("take_profit_price")
        sl = signal.get("stop_loss_price")
        close_side = "sell" if signal["side"] == "buy" else "buy"
        amount = _portfolio.compute_position_size(signal)
        symbol = signal["symbol"].replace("-", "/")

        if sl and sl > 0:
            try:
                # Rate limit before SL order
                if CONFIG.get("rate_limiting_enabled", True):
                    rate_limiter.wait_if_needed()
                
                exchange = _get_ccxt_exchange()
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
                # Rate limit before TP order
                if CONFIG.get("rate_limiting_enabled", True):
                    rate_limiter.wait_if_needed()
                
                exchange = _get_ccxt_exchange()
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
    
    PRODUCTION SAFETY:
    - Order validation before execution
    - Rate limiting to prevent API bans
    - Retry logic with exponential backoff
    """
    # STEP 1: Validate order
    is_valid, reason = _validate_order(signal)
    if not is_valid:
        logging.error(f"[Stock LIVE] ORDER REJECTED: {reason}")
        return False
    
    # STEP 2: Get rate limiter and retry handler
    rate_limiter = get_alpaca_limiter()
    retry_handler = get_default_retry()
    
    try:
        def _place_order():
            """Inner function for retry logic"""
            # Rate limit before API call
            if CONFIG.get("rate_limiting_enabled", True):
                wait_time = rate_limiter.wait_if_needed()
                if wait_time > 0:
                    logging.debug(f"[RateLimit] Waited {wait_time:.2f}s before order")
            
            api = _get_alpaca_api()
            side = signal["side"]
            symbol = signal["symbol"]
            qty = _portfolio.compute_position_size(signal)
            tp = signal.get("take_profit_price")
            sl = signal.get("stop_loss_price")

            logging.warning(
                f"[Stock LIVE] EXECUTING: {side.upper()} {qty:.4f} shares of {symbol}"
            )

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
            return order
        
        # Execute with retry logic
        if CONFIG.get("retry_on_failure", True):
            order = retry_handler.execute(_place_order)
        else:
            order = _place_order()

        logging.warning(f"[Stock LIVE] ORDER SUBMITTED: {order.id}")
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

    Supports partial exits (scaling out):
      • At TP1 (halfway to full TP): sell 50%, move SL to breakeven
      • At full TP: close remaining position
      • At SL: close entire position immediately
    """
    for pos in list(_portfolio.open_positions):
        sym = pos["symbol"]
        current_price = market_prices.get(sym)
        if current_price is None:
            continue

        side = pos["side"]
        tp = pos["take_profit_price"]
        sl = pos["stop_loss_price"]
        entry = pos["entry_price"]
        trail_pct = pos.get("trailing_stop_pct", 0)
        scale_out = CONFIG.get("enable_scale_out", True)

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

        # ── Partial exit (TP1): scale out at halfway to TP ───────
        if scale_out and not pos.get("_scaled_out", False):
            tp1_price = _compute_tp1(entry, tp, side)
            hit_tp1 = (
                (side == "buy" and current_price >= tp1_price) or
                (side == "sell" and current_price <= tp1_price)
            )
            if hit_tp1:
                _do_partial_exit(pos, current_price, tp1_price)
                continue  # don't also check full TP on same bar

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


def _compute_tp1(entry: float, tp: float, side: str) -> float:
    """Compute the partial-exit price (TP1) — halfway between entry and full TP."""
    scale_pct = CONFIG.get("scale_out_at_pct", 0.50)  # take profit on 50% at half-way
    if side == "buy":
        return entry + (tp - entry) * scale_pct
    else:
        return entry - (entry - tp) * scale_pct


def _do_partial_exit(pos: Dict[str, Any], current_price: float, tp1_price: float) -> None:
    """Execute a partial exit: sell configured fraction, move SL to breakeven."""
    sym = pos["symbol"]
    side = pos["side"]
    exit_fraction = CONFIG.get("scale_out_fraction", 0.50)  # sell 50% of position

    original_qty = pos["qty"]
    exit_qty = round(original_qty * exit_fraction, 6)
    remaining_qty = round(original_qty - exit_qty, 6)

    if remaining_qty <= 0 or exit_qty <= 0:
        return

    # Calculate P&L on the exited portion
    if side == "buy":
        partial_pnl = (current_price - pos["entry_price"]) * exit_qty
    else:
        partial_pnl = (pos["entry_price"] - current_price) * exit_qty

    # Update position: reduce qty, move SL to breakeven
    pos["qty"] = remaining_qty
    pos["stop_loss_price"] = round(pos["entry_price"], 4)  # breakeven SL
    pos["_scaled_out"] = True
    pos["_partial_pnl"] = round(partial_pnl, 2)

    # Return cash from partial exit
    _portfolio.cash += exit_qty * current_price
    _portfolio.save_state()

    logging.info(
        f"[ScaleOut] {sym}: sold {exit_fraction*100:.0f}% ({exit_qty:.6f}) @ "
        f"${current_price:.2f}  partial_PnL=${partial_pnl:.2f}  "
        f"remaining={remaining_qty:.6f}  SL moved to breakeven (${pos['entry_price']:.2f})"
    )


def flatten_positions(positions: list, current_prices: Dict[str, float]) -> int:
    """Force-close a list of positions (used for overnight risk management).

    Returns:
        Number of positions closed.
    """
    closed = 0
    for pos in positions:
        sym = pos["symbol"]
        price = current_prices.get(sym)
        if price is None:
            logging.warning(f"[Flatten] No price for {sym} — cannot flatten")
            continue

        logging.info(f"[Flatten] Force-closing {sym} for overnight risk management")
        _portfolio.close_position(pos, price, "overnight_flatten")
        closed += 1
    return closed


def get_portfolio():
    """Expose the global portfolio manager for external callers."""
    return _portfolio
