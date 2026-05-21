"""COMPLETE Multi-Asset Integration Example

This shows how to integrate ALL the new safety features into your main trading loop
for BOTH crypto AND stocks trading.

This is a COMPLETE reference implementation. Copy the patterns you need into main.py.
"""
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
from config.config import CONFIG, get_all_symbols, is_crypto
from data.fetcher import fetch_latest_market_data
from strategies.strategy_engine import generate_trade_signal
from trading.trade_executor import execute_trade
from trading.portfolio_manager import PortfolioManager

# NEW: Multi-asset safety features
from strategies.killswitch import (
    evaluate_multi_asset_kill_switch,
    evaluate_portfolio_drawdown,
    update_price_history,
)
from trading.position_health import (
    audit_positions,
    cleanup_closed_position,
)
from trading.profit_pile import (
    record_profit,
    get_pile_status,
    format_profit_summary,
)
from strategies.macro_snapshot import (
    get_macro_snapshot,
    should_suppress_trading,
    get_position_size_multiplier,
    update_macro_history,
)
from strategies.market_hours import (
    can_open_new_position,
    should_flatten_positions,
    get_market_hours_summary,
)

# ── Global state ──────────────────────────────────────────────────

_cycle_count = 0
_portfolio = PortfolioManager()
_last_macro_refresh = 0
_macro_snapshots: Dict[str, Dict[str, Any]] = {}


# ── Helper functions ──────────────────────────────────────────────

def _get_current_prices(symbols: List[str]) -> Dict[str, float]:
    """Get current prices for all symbols."""
    prices = {}
    for symbol in symbols:
        try:
            df = fetch_latest_market_data(ticker=symbol)
            if df is not None and not df.empty:
                prices[symbol] = float(df["Close"].iloc[-1])
        except Exception as e:
            logging.warning(f"Failed to get price for {symbol}: {e}")
    return prices


def _separate_symbols_by_asset_class(symbols: List[str]) -> Dict[str, List[str]]:
    """Separate symbols into crypto and stocks."""
    crypto = []
    stocks = []
    
    for symbol in symbols:
        if is_crypto(symbol):
            crypto.append(symbol)
        else:
            stocks.append(symbol)
    
    return {"crypto": crypto, "stocks": stocks}


# ── Main trading loop ─────────────────────────────────────────────

def multi_asset_trading_loop():
    """
    COMPLETE multi-asset trading loop with ALL safety features.
    
    This handles BOTH crypto AND stocks with appropriate checks for each.
    """
    global _cycle_count, _last_macro_refresh, _macro_snapshots
    
    logging.info("Starting MULTI-ASSET trading bot with comprehensive safety features")
    logging.info(f"Trading mode: {CONFIG.get('trading_mode', 'balanced')}")
    logging.info(f"Asset class: {CONFIG.get('asset_class', 'both')}")
    
    while True:
        _cycle_count += 1
        cycle_start = time.time()
        
        logging.info(f"\n{'='*60}\nCycle {_cycle_count}\n{'='*60}")
        
        try:
            # Get all symbols we're trading
            all_symbols = get_all_symbols()
            symbols_by_class = _separate_symbols_by_asset_class(all_symbols)
            
            logging.info(
                f"Trading {len(symbols_by_class['crypto'])} crypto + "
                f"{len(symbols_by_class['stocks'])} stock symbols"
            )
            
            # ══════════════════════════════════════════════════════
            # STEP 1: MARKET HOURS CHECK (Stocks Only)
            # ══════════════════════════════════════════════════════
            
            if symbols_by_class['stocks']:
                logging.info("\n" + get_market_hours_summary(all_symbols))
            
            # ══════════════════════════════════════════════════════
            # STEP 2: MULTI-ASSET KILL SWITCH CHECK
            # ══════════════════════════════════════════════════════
            
            logging.info("\n[Kill Switch] Evaluating multi-asset kill switch...")
            
            # Get current prices
            current_prices = _get_current_prices(
                [CONFIG.get("kill_market_symbol_crypto", "BTC-USD"),
                 CONFIG.get("kill_market_symbol_stock", "SPY")]
            )
            
            # Check both crypto AND stock markets
            kill_state = evaluate_multi_asset_kill_switch(
                crypto_symbols=[CONFIG.get("kill_market_symbol_crypto", "BTC-USD")],
                stock_symbols=[CONFIG.get("kill_market_symbol_stock", "SPY")],
                current_prices=current_prices,
            )
            
            # Also check portfolio-level drawdown
            portfolio_value = _portfolio.total_equity(current_prices)
            peak_value = CONFIG.get("peak_portfolio_value", portfolio_value)
            if portfolio_value > peak_value:
                peak_value = portfolio_value
                CONFIG["peak_portfolio_value"] = peak_value
            
            should_kill_drawdown, dd_reason = evaluate_portfolio_drawdown(
                portfolio_value=portfolio_value,
                peak_value=peak_value,
                max_drawdown_pct=CONFIG.get("kill_portfolio_drawdown_pct", 15.0),
            )
            
            if kill_state["tripped"] or should_kill_drawdown:
                reason = kill_state.get("reason") or dd_reason
                logging.error(
                    f"\n🚨🚨🚨 KILL SWITCH ACTIVE 🚨🚨🚨\n"
                    f"Reason: {reason}\n"
                    f"Cooldown until: {kill_state.get('cooldown_until')}\n"
                    f"Closing ALL positions immediately!\n"
                )
                
                # Close all positions
                for position in _portfolio.open_positions:
                    try:
                        symbol = position["symbol"]
                        current_price = current_prices.get(symbol)
                        
                        if not current_price:
                            # Try to fetch it
                            prices = _get_current_prices([symbol])
                            current_price = prices.get(symbol)
                        
                        if current_price:
                            _portfolio.close_position(
                                position,
                                exit_price=current_price,
                                result="KILL_SWITCH"
                            )
                            cleanup_closed_position(position.get("order_id", symbol))
                            logging.warning(f"Closed {symbol} on kill switch")
                    except Exception as e:
                        logging.error(f"Error closing position on kill switch: {e}")
                
                # Skip rest of cycle
                time.sleep(CONFIG.get("loop_interval_seconds", 300))
                continue
            
            logging.info("[Kill Switch] ✓ All clear - no kill conditions triggered")
            
            # ══════════════════════════════════════════════════════
            # STEP 3: MACRO SNAPSHOT & REGIME DETECTION
            # ══════════════════════════════════════════════════════
            
            # Refresh macro every N minutes
            refresh_interval = CONFIG.get("macro_refresh_interval_min", 15) * 60
            time_since_refresh = time.time() - _last_macro_refresh
            
            if time_since_refresh >= refresh_interval or not _macro_snapshots:
                logging.info("\n[Macro] Refreshing macro snapshots...")
                
                # Get macro for crypto market
                if symbols_by_class['crypto']:
                    crypto_indicator = CONFIG.get("kill_market_symbol_crypto", "BTC-USD")
                    crypto_price = current_prices.get(crypto_indicator)
                    
                    if crypto_price:
                        crypto_macro = get_macro_snapshot(
                            symbol=crypto_indicator,
                            current_price=crypto_price,
                            asset_type="crypto",
                        )
                        _macro_snapshots["crypto"] = crypto_macro
                        update_macro_history(crypto_indicator, crypto_price)
                        
                        logging.info(
                            f"  Crypto ({crypto_indicator}): {crypto_macro['regime'].upper()} "
                            f"(conf: {crypto_macro['confidence']:.0%})"
                        )
                
                # Get macro for stock market
                if symbols_by_class['stocks']:
                    stock_indicator = CONFIG.get("kill_market_symbol_stock", "SPY")
                    stock_price = current_prices.get(stock_indicator)
                    
                    if stock_price:
                        stock_macro = get_macro_snapshot(
                            symbol=stock_indicator,
                            current_price=stock_price,
                            asset_type="stock",
                        )
                        _macro_snapshots["stock"] = stock_macro
                        update_macro_history(stock_indicator, stock_price)
                        
                        logging.info(
                            f"  Stock ({stock_indicator}): {stock_macro['regime'].upper()} "
                            f"(conf: {stock_macro['confidence']:.0%})"
                        )
                
                _last_macro_refresh = time.time()
            
            # Check if we should suppress trading based on macro
            for asset_type, macro in _macro_snapshots.items():
                should_suppress, reason = should_suppress_trading(macro)
                if should_suppress:
                    logging.warning(
                        f"[Macro] {asset_type.upper()} trading suppressed: {reason}"
                    )
            
            # ══════════════════════════════════════════════════════
            # STEP 4: POSITION HEALTH MONITORING
            # ══════════════════════════════════════════════════════
            
            if _portfolio.open_positions:
                logging.info(f"\n[Position Health] Auditing {len(_portfolio.open_positions)} positions...")
                
                # Get current prices for all open positions
                position_symbols = [p["symbol"] for p in _portfolio.open_positions]
                position_prices = _get_current_prices(position_symbols)
                
                # Prepare positions for health check
                positions_for_check = []
                for pos in _portfolio.open_positions:
                    current_price = position_prices.get(pos["symbol"])
                    if current_price:
                        check_pos = {
                            "id": pos.get("order_id", pos["symbol"]),
                            "symbol": pos["symbol"],
                            "side": pos["side"],
                            "entry_price": pos["entry_price"],
                            "current_price": current_price,
                            "quantity": pos["qty"],
                            "opened_at": pos["opened_at"],
                            "fees_paid": pos.get("cost", 0) * CONFIG.get("exchange_fee_pct", 0.001),
                        }
                        positions_for_check.append(check_pos)
                
                # Audit positions
                positions_to_close = audit_positions(positions_for_check)
                
                # Close flagged positions
                for pos_to_close in positions_to_close:
                    try:
                        # Find the full position record
                        full_pos = next(
                            (p for p in _portfolio.open_positions 
                             if p["symbol"] == pos_to_close["symbol"]),
                            None
                        )
                        
                        if full_pos:
                            current_price = position_prices.get(full_pos["symbol"])
                            if current_price:
                                logging.warning(
                                    f"[Position Health] Closing {full_pos['symbol']}: "
                                    f"{pos_to_close['close_reason']}"
                                )
                                _portfolio.close_position(
                                    full_pos,
                                    exit_price=current_price,
                                    result="HEALTH_CHECK"
                                )
                                cleanup_closed_position(full_pos.get("order_id", full_pos["symbol"]))
                                
                    except Exception as e:
                        logging.error(f"Error closing unhealthy position: {e}")
                
                if not positions_to_close:
                    logging.info("[Position Health] ✓ All positions healthy")
            
            # ══════════════════════════════════════════════════════
            # STEP 5: FLATTEN POSITIONS BEFORE MARKET CLOSE (Stocks)
            # ══════════════════════════════════════════════════════
            
            stock_positions = [
                p for p in _portfolio.open_positions
                if not is_crypto(p["symbol"])
            ]
            
            if stock_positions:
                # Check each stock position for market close
                for pos in stock_positions:
                    should_flatten, reason = should_flatten_positions(pos["symbol"])
                    
                    if should_flatten:
                        logging.warning(f"[Market Hours] Flattening {pos['symbol']}: {reason}")
                        
                        prices = _get_current_prices([pos["symbol"]])
                        current_price = prices.get(pos["symbol"])
                        
                        if current_price:
                            _portfolio.close_position(
                                pos,
                                exit_price=current_price,
                                result="MARKET_CLOSE"
                            )
                            cleanup_closed_position(pos.get("order_id", pos["symbol"]))
            
            # ══════════════════════════════════════════════════════
            # STEP 6: GENERATE SIGNALS AND EXECUTE TRADES
            # ══════════════════════════════════════════════════════
            
            logging.info(f"\n[Trading] Generating signals for {len(all_symbols)} symbols...")
            
            for symbol in all_symbols:
                try:
                    asset_type = "crypto" if is_crypto(symbol) else "stock"
                    
                    # Check market hours (stocks only)
                    can_trade, hours_reason = can_open_new_position(symbol)
                    if not can_trade:
                        logging.debug(f"[{symbol}] Cannot trade: {hours_reason}")
                        continue
                    
                    # Check macro suppression for this asset class
                    macro = _macro_snapshots.get(asset_type)
                    if macro:
                        should_suppress, macro_reason = should_suppress_trading(macro)
                        if should_suppress:
                            logging.debug(f"[{symbol}] Macro suppressed: {macro_reason}")
                            continue
                    
                    # Fetch market data
                    market_data = fetch_latest_market_data(ticker=symbol)
                    if market_data is None or market_data.empty:
                        continue
                    
                    # Generate signal (uses your existing strategy engine)
                    # NOTE: generate_trade_signal expects (df, symbol) not (symbol, df)
                    signal = generate_trade_signal(market_data, symbol)
                    
                    # Validate signal is a dict (not a DataFrame or other type)
                    if not isinstance(signal, dict):
                        logging.warning(
                            f"[{symbol}] Signal is not a dict (got {type(signal).__name__}), skipping"
                        )
                        continue
                    
                    if signal.get("action") in ["buy", "sell"]:
                        # Apply macro-based position sizing
                        if macro:
                            size_mult = get_position_size_multiplier(macro)
                            signal["_size_mult"] = signal.get("_size_mult", 1.0) * size_mult
                            
                            logging.info(
                                f"[{symbol}] Macro size multiplier: {size_mult:.2f}x "
                                f"(regime: {macro['regime']})"
                            )
                        
                        # Execute trade
                        success = execute_trade(signal)
                        
                        if success:
                            logging.info(
                                f"[{symbol}] ✓ {signal['action'].upper()} executed "
                                f"(confidence: {signal.get('confidence', 0):.2%})"
                            )
                        
                except Exception as e:
                    logging.error(f"Error processing {symbol}: {e}")
            
            # ══════════════════════════════════════════════════════
            # STEP 7: PERIODIC STATUS UPDATES
            # ══════════════════════════════════════════════════════
            
            # Every 12 cycles (~1 hour), show profit pile status
            if _cycle_count % 12 == 0:
                logging.info("\n" + format_profit_summary())
                
                pile_status = get_pile_status()
                logging.info(
                    f"\nPortfolio: ${portfolio_value:.2f} | "
                    f"Profit Pile: ${pile_status['total_piled']:.2f} | "
                    f"Lifetime P&L: ${pile_status['net_lifetime']:.2f}"
                )
            
        except Exception as e:
            logging.error(f"Error in main loop: {e}", exc_info=True)
        
        # ══════════════════════════════════════════════════════════
        # CYCLE COMPLETE
        # ══════════════════════════════════════════════════════════
        
        cycle_duration = time.time() - cycle_start
        logging.info(f"\nCycle {_cycle_count} complete in {cycle_duration:.1f}s")
        
        # Sleep until next cycle
        sleep_time = max(0, CONFIG.get("loop_interval_seconds", 300) - cycle_duration)
        if sleep_time > 0:
            logging.info(f"Sleeping for {sleep_time:.0f}s until next cycle...")
            time.sleep(sleep_time)


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/multi_asset_bot.log"),
            logging.StreamHandler(),
        ],
    )
    
    logging.info("="*60)
    logging.info("MULTI-ASSET AUTONOMOUS TRADING BOT")
    logging.info("="*60)
    logging.info("Features:")
    logging.info("  ✓ Multi-asset kill switch (crypto + stocks)")
    logging.info("  ✓ Position health monitoring")
    logging.info("  ✓ Profit pile accounting")
    logging.info("  ✓ Macro regime detection")
    logging.info("  ✓ Market hours management")
    logging.info("  ✓ Atomic state persistence")
    logging.info("="*60)
    
    try:
        multi_asset_trading_loop()
    except KeyboardInterrupt:
        logging.info("\nShutdown requested by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logging.info("Bot stopped")
