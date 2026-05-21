"""
LIMITLESS - Autonomous AI Trading System

An autonomous AI trading bot that processes market data instantly, identifies
complex patterns, and executes trades at machine speed with zero human intervention.

Run modes:
  python main.py              -> continuous trading loop (paper or live)
  python main.py --once       -> single pass then exit
  python main.py --backtest   -> backtest all symbols and print metrics
  python main.py --train      -> train ML + RL models from latest data
  python main.py --pretrain   -> pre-train on maximum historical data

Production loop:
  1. WebSocket price feed provides real-time prices (with yfinance fallback).
  2. LLM watcher runs in the background, scoring every symbol every 15 min.
  3. Autonomous agent analyzes world events and learns from every trade.
  4. Each cycle: reconcile -> check TP/SL -> generate signals -> autonomous decision
     -> execute trades -> periodic rebalancing -> self-reflection.

Key Capabilities:
  - Instant information assimilation - processes all market data immediately
  - Perfect recall - learns from every trade outcome
  - Pattern recognition - spots tiny market clues invisible to humans
  - High-frequency execution - 500+ trades/day capacity
  - World events synthesis - analyzes geopolitical, economic, regulatory news
  - Self-evolution - continuously adapts strategies based on performance
  - Zero human intervention required

Benchmark:
  Claude Sonnet AI: +1,322% in 48hrs, 5,200+ trades, fully autonomous
"""
import argparse
import logging
import os
import sys
import time
from typing import Dict, Optional

import pandas as pd

from config.config import CONFIG, get_all_symbols, is_crypto, get_loop_interval
from data.fetcher import fetch_latest_market_data, fetch_multiple_symbols
from strategies.strategy_engine import generate_trade_signal
from trading.trade_executor import (
    execute_trade, check_open_positions, get_portfolio, flatten_positions,
)
from strategies.discipline import DisciplineGate
from strategies.session_manager import get_session_info

# Autonomous mode imports
from autonomous_mode import (
    autonomous_pre_cycle_checks,
    autonomous_enhance_signal,
    autonomous_post_cycle_tasks,
    get_autonomous_status,
)

# ── NEW: Multi-Asset Safety Features ─────────────────────────────
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

# ── Logging setup ─────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CONFIG.get("bot_log_path", "logs/bot.log")),
        logging.StreamHandler(),
    ],
)


# ── Subsystem lazy-init helpers ───────────────────────────────────

_feed_manager = None
_llm_watcher = None
_discipline: DisciplineGate = None  # type: ignore[assignment]
_cycle_count = 0

# Multi-asset safety state
_last_macro_refresh = 0
_macro_snapshots: Dict[str, Dict] = {}
_peak_portfolio_value = 0.0


def _start_subsystems() -> None:
    """Start WebSocket feeds and LLM watcher (best-effort, errors logged)."""
    global _feed_manager, _llm_watcher

    # ── WebSocket real-time price feed ────────────────────────────
    if CONFIG.get("use_websocket", True):
        try:
            from data.websocket_feed import get_feed_manager
            _feed_manager = get_feed_manager()
            symbols = get_all_symbols()
            _feed_manager.start(symbols)
            logging.info("[Main] WebSocket price feed started")
        except Exception as e:
            logging.warning(f"[Main] WebSocket feed failed to start (will use yfinance): {e}")
            _feed_manager = None

    # ── LLM continuous watcher ────────────────────────────────────
    if CONFIG.get("use_llm") and CONFIG.get("anthropic_api_key"):
        try:
            from models.llm_watcher import get_watcher
            _llm_watcher = get_watcher()
            _llm_watcher.start()
            logging.info("[Main] LLM watcher started")
        except Exception as e:
            logging.warning(f"[Main] LLM watcher failed to start: {e}")
            _llm_watcher = None


def _stop_subsystems() -> None:
    """Gracefully stop background subsystems."""
    if _feed_manager:
        try:
            _feed_manager.stop()
        except Exception:
            pass
    if _llm_watcher:
        try:
            _llm_watcher.stop()
        except Exception:
            pass


def _get_current_prices(symbols) -> Dict[str, float]:
    """Get current prices — prefer WebSocket, fall back to yfinance OHLCV."""
    prices: Dict[str, float] = {}

    # Try WebSocket first
    if _feed_manager:
        ws_prices = _feed_manager.get_latest_prices()
        for sym in symbols:
            if sym in ws_prices:
                prices[sym] = ws_prices[sym]

    # Fall back to yfinance for any missing symbols
    missing = [s for s in symbols if s not in prices]
    if missing:
        for sym in missing:
            df = fetch_latest_market_data(ticker=sym)
            if df is not None and not df.empty:
                prices[sym] = float(df["Close"].iloc[-1])

    return prices


# ── PRODUCTION SAFETY: Startup Configuration Logging (C7) ─────────

def log_startup_configuration() -> None:
    """
    [PRODUCTION SAFETY] Log critical system configuration at startup.
    Ensures we have a record of what parameters were active for each trading session.
    """
    import platform
    import hashlib
    
    logging.info("=" * 80)
    logging.info("STARTUP CONFIGURATION SNAPSHOT")
    logging.info("=" * 80)
    
    # System info
    logging.info(f"Platform: {platform.system()} {platform.release()}")
    logging.info(f"Python: {platform.python_version()}")
    
    # Trading mode
    trading_mode = CONFIG.get("trading_mode", "balanced")
    paper_trading = CONFIG.get("use_paper_trading", True)
    asset_class = CONFIG.get("asset_class", "crypto")
    logging.info(f"Trading Mode: {trading_mode.upper()}")
    logging.info(f"Paper Trading: {paper_trading}")
    logging.info(f"Asset Class: {asset_class}")
    
    # Capital and risk parameters
    initial_capital = CONFIG.get("initial_capital", 10000)
    logging.info(f"Initial Capital: ${initial_capital:,.2f}")
    logging.info(f"Position Size: {CONFIG.get('position_size_pct', 0.05)*100:.1f}%")
    logging.info(f"Max Daily Drawdown: {CONFIG.get('max_daily_drawdown_pct', 0.05)*100:.1f}%")
    logging.info(f"Stop Loss: {CONFIG.get('stop_loss_pct', 0.02)*100:.1f}%")
    logging.info(f"Take Profit: {CONFIG.get('take_profit_pct', 0.04)*100:.1f}%")
    
    # Production safety controls
    logging.info("─" * 80)
    logging.info("PRODUCTION SAFETY CONTROLS:")
    max_total_loss = CONFIG.get("max_total_loss_pct", 0.20)
    logging.info(f"  Max Total Loss (Circuit Breaker): {max_total_loss*100:.1f}%")
    logging.info(f"  Rapid Loss Threshold: {CONFIG.get('rapid_loss_threshold', 5)} losses")
    logging.info(f"  Rapid Loss Window: {CONFIG.get('rapid_loss_window_sec', 600)}s")
    max_trade = CONFIG.get("max_single_trade_value", 5000)
    min_trade = CONFIG.get("min_trade_value", 10)
    logging.info(f"  Max Single Trade: ${max_trade:,.2f}")
    logging.info(f"  Min Trade Value: ${min_trade:.2f}")
    logging.info(f"  Order Validation: {CONFIG.get('order_validation_enabled', True)}")
    logging.info(f"  Rate Limiting: {CONFIG.get('rate_limiting_enabled', True)}")
    
    # Emergency stop files
    emergency_file = CONFIG.get("emergency_stop_file", "EMERGENCY_STOP")
    pause_file = CONFIG.get("pause_trading_file", "PAUSE_TRADING")
    logging.info(f"  Emergency Stop File: {emergency_file}")
    logging.info(f"  Pause Trading File: {pause_file}")
    
    # API status (check if keys are set, but NEVER log the actual keys)
    logging.info("─" * 80)
    logging.info("API CONFIGURATION:")
    
    def check_api_key(key_name: str) -> str:
        """Check if API key exists without revealing it."""
        key_value = CONFIG.get(key_name, "")
        if not key_value or key_value == "your-key-here":
            return "❌ NOT SET"
        # Return truncated hash for verification
        key_hash = hashlib.md5(key_value.encode()).hexdigest()[:8]
        return f"✓ SET (hash:{key_hash})"
    
    if asset_class == "crypto":
        logging.info(f"  Coinbase API Key: {check_api_key('coinbase_api_key')}")
        logging.info(f"  Coinbase Secret: {check_api_key('coinbase_api_secret')}")
    else:
        logging.info(f"  Alpaca API Key: {check_api_key('alpaca_api_key')}")
        logging.info(f"  Alpaca Secret Key: {check_api_key('alpaca_secret_key')}")
    
    if CONFIG.get("use_llm", False):
        logging.info(f"  Anthropic API Key: {check_api_key('anthropic_api_key')}")
    
    # Symbol watchlist
    symbols = get_all_symbols()
    logging.info("─" * 80)
    logging.info(f"WATCHLIST: {len(symbols)} symbols")
    logging.info(f"  {', '.join(symbols[:10])}" + (f" ... (+{len(symbols)-10} more)" if len(symbols) > 10 else ""))
    
    # AI features
    logging.info("─" * 80)
    logging.info("AI FEATURES:")
    logging.info(f"  LLM Analysis: {CONFIG.get('use_llm', False)}")
    logging.info(f"  Autonomous Learning: {CONFIG.get('enable_autonomous_learning', True)}")
    logging.info(f"  World Events Analysis: {CONFIG.get('enable_world_events_analysis', True)}")
    logging.info(f"  WebSocket Feed: {'active' if _feed_manager else 'inactive (yfinance fallback)'}")
    logging.info(f"  LLM Watcher: {'active' if _llm_watcher else 'inactive'}")
    
    logging.info("=" * 80)
    logging.info("System initialization complete. Ready to trade.")
    logging.info("=" * 80)


# ── PRODUCTION SAFETY: Emergency Stop Mechanism (C8) ───────────────

def check_emergency_stop() -> None:
    """
    [PRODUCTION SAFETY] Check for emergency stop or pause files.
    
    EMERGENCY_STOP: Immediately halt all trading and exit.
    PAUSE_TRADING: Skip this cycle and continue monitoring.
    
    Usage:
      touch EMERGENCY_STOP  -> Bot will save state and exit
      touch PAUSE_TRADING   -> Bot will pause trading but keep running
      rm PAUSE_TRADING      -> Bot resumes trading
    
    Raises:
        SystemExit: If EMERGENCY_STOP file exists
    """
    emergency_file = CONFIG.get("emergency_stop_file", "EMERGENCY_STOP")
    pause_file = CONFIG.get("pause_trading_file", "PAUSE_TRADING")
    
    # Check for emergency stop
    if os.path.exists(emergency_file):
        logging.critical("=" * 80)
        logging.critical("⛔ EMERGENCY STOP FILE DETECTED")
        logging.critical("=" * 80)
        logging.critical(f"File: {emergency_file}")
        logging.critical("Halting all trading operations immediately.")
        logging.critical("Saving portfolio state...")
        
        try:
            portfolio = get_portfolio()
            portfolio.save_state()
            logging.critical("Portfolio state saved successfully.")
        except Exception as e:
            logging.critical(f"ERROR saving portfolio state: {e}")
        
        logging.critical("=" * 80)
        logging.critical("System shutdown complete. Remove EMERGENCY_STOP file to resume.")
        logging.critical("=" * 80)
        
        _stop_subsystems()
        raise SystemExit(1)
    
    # Check for trading pause
    if os.path.exists(pause_file):
        logging.warning("⏸️  PAUSE_TRADING file detected - skipping this cycle")
        logging.warning(f"File: {pause_file}")
        logging.warning("Remove file to resume trading.")
        # Return without raising - caller should skip trading logic
        return


# ── Core loop iteration ──────────────────────────────────────────

def run_one_cycle() -> None:
    """
    Execute one full cycle:
      0. Emergency stop check (PRODUCTION SAFETY)
      1. Discipline gate: check daily drawdown kill switch
      2. Reconcile positions with broker (live mode only)
      3. Fetch data for every symbol on the watchlist
      4. Flatten overnight stock positions if near close
      5. Check open positions against current prices (TP/SL/trailing + partial exits)
      6. Generate signals → discipline gate → execute trades
      7. Periodic rebalancing based on LLM watcher opinions
    """
    print("[DEBUG] Entering run_one_cycle()")
    global _cycle_count, _discipline
    print("[DEBUG] After global declaration")
    
    # ── STEP 0: EMERGENCY STOP CHECK (PRODUCTION SAFETY C8) ──────────
    # Check for EMERGENCY_STOP or PAUSE_TRADING files before any trading logic
    try:
        check_emergency_stop()
    except SystemExit:
        # Emergency stop triggered - re-raise to stop the bot
        raise
    
    # If PAUSE_TRADING file exists, check_emergency_stop() returns normally
    # but we should skip trading. Check again here:
    pause_file = CONFIG.get("pause_trading_file", "PAUSE_TRADING")
    if os.path.exists(pause_file):
        logging.warning(f"[Cycle {_cycle_count}] Trading paused - checking open positions only")
        # Still check open positions even when paused
        symbols = get_all_symbols()
        current_prices = _get_current_prices(symbols)
        check_open_positions(current_prices)
        return
    
    _cycle_count += 1
    print(f"[DEBUG] Cycle count: {_cycle_count}")

    symbols = get_all_symbols()
    print(f"[DEBUG] Got {len(symbols)} symbols")
    logging.info(f"── Cycle {_cycle_count} start ── {len(symbols)} symbols ──")
    
    # Separate symbols by asset class
    crypto_symbols = [s for s in symbols if is_crypto(s)]
    stock_symbols = [s for s in symbols if not is_crypto(s)]
    
    if stock_symbols:
        logging.info("\n" + get_market_hours_summary(symbols))

    portfolio = get_portfolio()

    # Initialise discipline gate (once)
    if _discipline is None:
        _discipline = DisciplineGate(portfolio)
    
    # ══════════════════════════════════════════════════════════════
    # NEW: MULTI-ASSET KILL SWITCH CHECK
    # ══════════════════════════════════════════════════════════════
    if CONFIG.get("kill_switch_enabled", True):
        # Get current prices for market indicators
        crypto_indicator = CONFIG.get("kill_market_symbol_crypto", "BTC-USD")
        stock_indicator = CONFIG.get("kill_market_symbol_stock", "SPY")
        
        indicator_prices = _get_current_prices([crypto_indicator, stock_indicator])
        
        # Update price history
        for sym, price in indicator_prices.items():
            update_price_history(sym, price)
        
        # Check multi-asset kill switch
        kill_state = evaluate_multi_asset_kill_switch(
            crypto_symbols=[crypto_indicator] if crypto_symbols else [],
            stock_symbols=[stock_indicator] if stock_symbols else [],
            current_prices=indicator_prices,
        )
        
        # Check portfolio-level drawdown
        global _peak_portfolio_value
        current_prices_all = _get_current_prices(symbols)
        portfolio_value = portfolio.total_equity(current_prices_all)
        
        if portfolio_value > _peak_portfolio_value:
            _peak_portfolio_value = portfolio_value
        
        should_kill_dd, dd_reason = evaluate_portfolio_drawdown(
            portfolio_value=portfolio_value,
            peak_value=_peak_portfolio_value,
            max_drawdown_pct=CONFIG.get("kill_portfolio_drawdown_pct", 15.0),
        )
        
        if kill_state["tripped"] or should_kill_dd:
            reason = kill_state.get("reason") or dd_reason
            logging.error(
                f"\n{'='*60}\n"
                f"🚨🚨🚨 KILL SWITCH ACTIVE 🚨🚨🚨\n"
                f"Reason: {reason}\n"
                f"Cooldown until: {kill_state.get('cooldown_until', 'N/A')}\n"
                f"Closing ALL positions immediately!\n"
                f"{'='*60}"
            )
            
            # Close all positions
            current_prices_all = _get_current_prices(symbols)
            for position in portfolio.open_positions:
                try:
                    sym = position["symbol"]
                    current_price = current_prices_all.get(sym)
                    
                    if current_price:
                        portfolio.close_position(
                            position,
                            exit_price=current_price,
                            result="KILL_SWITCH"
                        )
                        cleanup_closed_position(position.get("order_id", sym))
                        logging.warning(f"[Kill Switch] Closed {sym} at {current_price}")
                except Exception as e:
                    logging.error(f"[Kill Switch] Error closing position: {e}")
            
            # Skip rest of cycle
            return
        
        logging.info("[Kill Switch] ✓ All clear - no kill conditions triggered")
    
    # ══════════════════════════════════════════════════════════════
    # NEW: MACRO REGIME DETECTION & SNAPSHOT
    # ══════════════════════════════════════════════════════════════
    global _last_macro_refresh, _macro_snapshots
    
    if CONFIG.get("macro_monitoring_enabled", True):
        refresh_interval = CONFIG.get("macro_refresh_interval_min", 15) * 60
        time_since_refresh = time.time() - _last_macro_refresh
        
        if time_since_refresh >= refresh_interval or not _macro_snapshots:
            logging.info("\n[Macro] Refreshing macro snapshots...")
            
            # Get macro for crypto market
            if crypto_symbols:
                crypto_indicator = CONFIG.get("kill_market_symbol_crypto", "BTC-USD")
                crypto_prices = _get_current_prices([crypto_indicator])
                crypto_price = crypto_prices.get(crypto_indicator)
                
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
            if stock_symbols:
                stock_indicator = CONFIG.get("kill_market_symbol_stock", "SPY")
                stock_prices = _get_current_prices([stock_indicator])
                stock_price = stock_prices.get(stock_indicator)
                
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
                logging.warning(f"[Macro] {asset_type.upper()} trading suppressed: {reason}")
    
    # ── AUTONOMOUS PRE-CYCLE CHECKS ──────────────────────────────
    # Analyze world events, check for major market-moving events
    autonomous_checks = autonomous_pre_cycle_checks(_cycle_count)
    
    if autonomous_checks["should_pause"]:
        logging.warning(
            f"[Main] Trading paused by autonomous system: "
            f"{autonomous_checks['pause_reason']}"
        )
        # Still check open positions even when paused
        current_prices = _get_current_prices(symbols)
        check_open_positions(current_prices)
        return
    
    world_events = autonomous_checks.get("world_events")
    adaptive_params = autonomous_checks.get("adaptive_params", {})

    # 0. Position reconciliation (live mode only)
    if not CONFIG.get("use_paper_trading", True):
        try:
            from trading.reconciler import reconcile
            reconcile(portfolio)
        except Exception as e:
            logging.warning(f"[Main] Reconciliation error: {e}")

    # 1. Fetch market data (for TA indicators — always need full OHLCV)
    market_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_latest_market_data(ticker=sym)
        if df is not None and not df.empty:
            market_data[sym] = df
        else:
            logging.warning(f"Skipping {sym} — no data")

    if not market_data:
        logging.warning("No market data fetched for any symbol — skipping cycle")
        return

    # 2. Build a current-price map (prefer real-time WebSocket prices)
    current_prices = _get_current_prices(symbols)
    # Fill any still-missing from OHLCV
    for sym, df in market_data.items():
        if sym not in current_prices:
            current_prices[sym] = float(df["Close"].iloc[-1])

    # 3. NEW: Position health monitoring BEFORE standard TP/SL checks
    if CONFIG.get("position_health_monitor_enabled", True) and portfolio.open_positions:
        logging.info(f"\n[Position Health] Auditing {len(portfolio.open_positions)} positions...")
        
        # Prepare positions for health check
        positions_for_check = []
        for pos in portfolio.open_positions:
            current_price = current_prices.get(pos["symbol"])
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
                    (p for p in portfolio.open_positions 
                     if p["symbol"] == pos_to_close["symbol"]),
                    None
                )
                
                if full_pos:
                    current_price = current_prices.get(full_pos["symbol"])
                    if current_price:
                        logging.warning(
                            f"[Position Health] Closing {full_pos['symbol']}: "
                            f"{pos_to_close['close_reason']}"
                        )
                        
                        # Calculate P&L for profit pile
                        entry_cost = full_pos["entry_price"] * full_pos["qty"]
                        exit_value = current_price * full_pos["qty"]
                        
                        if full_pos["side"] == "long":
                            profit_usd = exit_value - entry_cost
                        else:
                            profit_usd = entry_cost - exit_value
                        
                        # Close position
                        portfolio.close_position(
                            full_pos,
                            exit_price=current_price,
                            result="HEALTH_CHECK"
                        )
                        cleanup_closed_position(full_pos.get("order_id", full_pos["symbol"]))
                        
                        # Record profit if positive
                        if profit_usd > 0:
                            record_profit(
                                profit_usd=profit_usd,
                                reinvest_pct=CONFIG.get("profit_reinvest_pct", 60),
                                metadata={
                                    "symbol": full_pos["symbol"],
                                    "reason": "position_health",
                                    "strategy": pos_to_close["close_reason"],
                                }
                            )
                        
            except Exception as e:
                logging.error(f"[Position Health] Error closing unhealthy position: {e}")
        
        if not positions_to_close:
            logging.info("[Position Health] ✓ All positions healthy")
    
    # 4. NEW: Flatten stock positions before market close
    stock_positions = [p for p in portfolio.open_positions if not is_crypto(p["symbol"])]
    
    if stock_positions and CONFIG.get("enforce_market_hours", True):
        for pos in stock_positions:
            should_flatten, reason = should_flatten_positions(pos["symbol"])
            
            if should_flatten:
                logging.warning(f"[Market Hours] Flattening {pos['symbol']}: {reason}")
                
                current_price = current_prices.get(pos["symbol"])
                if current_price:
                    # Calculate P&L for profit pile
                    entry_cost = pos["entry_price"] * pos["qty"]
                    exit_value = current_price * pos["qty"]
                    
                    if pos["side"] == "long":
                        profit_usd = exit_value - entry_cost
                    else:
                        profit_usd = entry_cost - exit_value
                    
                    portfolio.close_position(
                        pos,
                        exit_price=current_price,
                        result="MARKET_CLOSE"
                    )
                    cleanup_closed_position(pos.get("order_id", pos["symbol"]))
                    
                    # Record profit if positive
                    if profit_usd > 0:
                        record_profit(
                            profit_usd=profit_usd,
                            reinvest_pct=CONFIG.get("profit_reinvest_pct", 60),
                            metadata={
                                "symbol": pos["symbol"],
                                "reason": "market_close",
                            }
                        )
    
    # Legacy flatten overnight (for backwards compatibility)
    if CONFIG.get("flatten_overnight", True):
        to_flatten = _discipline.get_positions_to_flatten()
        if to_flatten:
            closed = flatten_positions(to_flatten, current_prices)
            logging.info(f"[Main] Flattened {closed} stock positions for overnight risk")

    # 5. Check open positions (TP/SL/trailing + partial exits)
    check_open_positions(current_prices)

    # Track closed trades for discipline counters
    _sync_discipline_results(portfolio)

    # 6. Generate signals for symbols without open positions
    held_symbols = {p["symbol"] for p in portfolio.open_positions}

    for sym, df in market_data.items():
        if sym in held_symbols:
            logging.info(f"[Main] {sym} — already holding, skip signal generation")
            continue
        
        asset_type = "crypto" if is_crypto(sym) else "stock"

        # NEW: Market hours check (stocks only)
        if asset_type == "stock" and CONFIG.get("enforce_market_hours", True):
            can_trade, hours_reason = can_open_new_position(sym)
            if not can_trade:
                logging.debug(f"[Main] {sym} — Cannot trade: {hours_reason}")
                continue
        
        # NEW: Check macro suppression for this asset class
        if _macro_snapshots:
            macro = _macro_snapshots.get(asset_type)
            if macro:
                should_suppress, macro_reason = should_suppress_trading(macro)
                if should_suppress:
                    logging.debug(f"[Main] {sym} — Macro suppressed: {macro_reason}")
                    continue

        # Legacy session check (for backwards compatibility)
        session_info = get_session_info(sym)
        if not session_info["trading_allowed"]:
            logging.info(f"[Main] {sym} — market {session_info['session']}, skipping")
            continue

        signal = generate_trade_signal(df, symbol=sym)

        # Incorporate LLM watcher opinion if available
        if signal and _llm_watcher:
            try:
                opinion = _llm_watcher.get_opinion(sym)
                if opinion:
                    llm_bias = opinion.get("bias", "neutral")
                    llm_score = opinion.get("score", 0)
                    # Reject signals that strongly contradict the LLM watcher
                    if signal["side"] == "buy" and llm_bias == "bearish" and llm_score < -0.5:
                        logging.info(f"[Main] LLM watcher bearish on {sym} (score={llm_score:.2f}) — suppressing buy")
                        continue
                    elif signal["side"] == "sell" and llm_bias == "bullish" and llm_score > 0.5:
                        logging.info(f"[Main] LLM watcher bullish on {sym} (score={llm_score:.2f}) — suppressing sell")
                        continue
            except Exception as e:
                logging.debug(f"[Main] LLM watcher opinion error for {sym}: {e}")

        if signal:
            # ── AUTONOMOUS DECISION ───────────────────────────────
            # Let the autonomous agent evaluate with world events & learning
            autonomous_decision = autonomous_enhance_signal(
                signal, df, world_events
            )
            
            if not autonomous_decision["should_execute"]:
                logging.info(
                    f"[Main] {sym} — REJECTED by autonomous agent: "
                    f"{autonomous_decision['reasoning']}"
                )
                continue
            
            # Update signal confidence based on autonomous decision
            signal["confidence"] = autonomous_decision["adjusted_confidence"]
            signal["autonomous_reasoning"] = autonomous_decision["reasoning"]
            
            # NEW: Apply macro-based position sizing
            if _macro_snapshots:
                macro = _macro_snapshots.get(asset_type)
                if macro:
                    size_mult = get_position_size_multiplier(macro)
                    signal["_size_mult"] = signal.get("_size_mult", 1.0) * size_mult
                    
                    logging.info(
                        f"[Main] {sym} — Macro size multiplier: {size_mult:.2f}x "
                        f"(regime: {macro['regime']})"
                    )
            
            # ── DISCIPLINE GATE — must pass all checks ────────────
            allowed, reason, adjustments = _discipline.check(
                signal, market_data=market_data, current_prices=current_prices,
            )
            if not allowed:
                logging.info(f"[Main] {sym} — BLOCKED by discipline: {reason}")
                continue

            # Apply discipline adjustments to the signal
            signal = _apply_adjustments(signal, adjustments)
            
            # Apply world events adaptive parameters
            if adaptive_params:
                size_mult = adaptive_params.get("position_size_mult", 1.0)
                if size_mult != 1.0:
                    signal["_size_mult"] = signal.get("_size_mult", 1.0) * size_mult
                    logging.info(
                        f"[Main] {sym} — world events size adjustment: ×{size_mult:.2f}"
                    )

            execute_trade(signal)
            _discipline.record_trade_opened()
        else:
            logging.info(f"[Main] {sym} — no signal")

    # 7. NEW: Record profits from closed positions
    # Check for any newly closed positions since last cycle
    for closed_trade in portfolio.closed_trades:
        trade_id = closed_trade.get("order_id", closed_trade.get("symbol"))
        pnl = closed_trade.get("pnl", 0)
        
        if pnl > 0 and not closed_trade.get("_profit_recorded"):
            record_profit(
                profit_usd=pnl,
                reinvest_pct=CONFIG.get("profit_reinvest_pct", 60),
                metadata={
                    "symbol": closed_trade.get("symbol"),
                    "strategy": closed_trade.get("result", "unknown"),
                }
            )
            # Mark as recorded so we don't double-count
            closed_trade["_profit_recorded"] = True

    # 8. Periodic rebalancing (every N cycles)
    rebalance_interval = CONFIG.get("rebalance_interval_cycles", 12)  # default every 12 cycles (~1hr at 5min)
    if rebalance_interval > 0 and _cycle_count % rebalance_interval == 0:
        _run_rebalance(portfolio, current_prices)

    # 9. Log portfolio & discipline summary
    summary = portfolio.summary(current_prices)
    disc_status = _discipline.get_status(current_prices)
    logging.info(
        f"── Cycle {_cycle_count} end ── equity=${summary['equity']:.2f}  "
        f"cash=${summary['cash']:.2f}  "
        f"open={summary['open_positions']}  "
        f"PnL=${summary['total_realised_pnl']:.2f}  "
        f"win_rate={summary['win_rate']*100:.1f}%  "
        f"daily_dd={disc_status['daily_drawdown_pct']:.2f}%  "
        f"losses_streak={disc_status['consecutive_losses']}  "
        f"trades_today={disc_status['daily_trades']}"
    )

    # SMS alert if daily drawdown exceeds 5%
    if disc_status['daily_drawdown_pct'] >= 5.0:
        try:
            from alerts.sms_notifier import send_alert
            send_alert(
                f"DAILY LOSS ALERT: Down {disc_status['daily_drawdown_pct']:.1f}% today. "
                f"Equity: ${disc_status['current_equity']:.2f}"
            )
        except Exception:
            pass
    
    # 10. NEW: Periodic status updates (profit pile, kill switch, etc.)
    # Every 12 cycles (~1 hour)
    if _cycle_count % 12 == 0:
        pile_status = get_pile_status()
        logging.info("\n" + format_profit_summary())
        logging.info(
            f"\n💰 Portfolio: ${summary['equity']:.2f} | "
            f"Profit Pile: ${pile_status['total_piled']:.2f} | "
            f"Lifetime P&L: ${pile_status['net_lifetime']:.2f}"
        )
    
    # 11. Autonomous post-cycle tasks
    # Self-reflection, cost tracking, velocity monitoring
    autonomous_post_cycle_tasks(_cycle_count, portfolio)


# ── Discipline helpers ────────────────────────────────────────────

# Track how many closed trades we've already reported to the discipline gate
_last_closed_count: int = 0


def _sync_discipline_results(portfolio) -> None:
    """Feed newly closed trades into the discipline gate for streak tracking."""
    global _last_closed_count, _discipline
    if _discipline is None:
        return

    closed = portfolio.closed_trades
    new_trades = closed[_last_closed_count:]
    for t in new_trades:
        _discipline.record_trade_result(t.get("pnl", 0))
    _last_closed_count = len(closed)


def _apply_adjustments(signal: Dict, adjustments: Dict[str, float]) -> Dict:
    """Apply discipline-layer adjustments (size/TP/SL multipliers) to a signal."""
    # Adjust TP/SL distances (multiply the distance from entry)
    entry = signal.get("entry_price", 0)
    side = signal.get("side", "buy")
    tp_mult = adjustments.get("tp_mult", 1.0)
    sl_mult = adjustments.get("sl_mult", 1.0)
    size_mult = adjustments.get("size_mult", 1.0)

    if tp_mult != 1.0 and "take_profit_price" in signal and entry > 0:
        tp = signal["take_profit_price"]
        tp_dist = abs(tp - entry) * tp_mult
        if side == "buy":
            signal["take_profit_price"] = round(entry + tp_dist, 4)
        else:
            signal["take_profit_price"] = round(entry - tp_dist, 4)

    if sl_mult != 1.0 and "stop_loss_price" in signal and entry > 0:
        sl = signal["stop_loss_price"]
        sl_dist = abs(sl - entry) * sl_mult
        if side == "buy":
            signal["stop_loss_price"] = round(entry - sl_dist, 4)
        else:
            signal["stop_loss_price"] = round(entry + sl_dist, 4)

    # Store size multiplier so portfolio manager can apply it
    if size_mult != 1.0:
        signal["_size_mult"] = size_mult

    # Apply confidence adjustment
    conf_adj = adjustments.get("confidence_adj", 0.0)
    if conf_adj != 0:
        signal["confidence"] = round(signal.get("confidence", 0.5) + conf_adj, 4)

    return signal


def _run_rebalance(portfolio, current_prices: Dict[str, float]) -> None:
    """Collect LLM watcher opinions and rebalance portfolio accordingly."""
    if not _llm_watcher:
        return

    try:
        # Build target allocations from LLM opinions
        target_allocs: Dict[str, float] = {}
        symbols = get_all_symbols()
        bullish_syms = []

        for sym in symbols:
            opinion = _llm_watcher.get_opinion(sym)
            if opinion and opinion.get("bias") == "bullish" and opinion.get("score", 0) > 0.3:
                bullish_syms.append((sym, opinion["score"]))

        if not bullish_syms:
            logging.info("[Main] No bullish symbols from LLM watcher — skipping rebalance")
            return

        # Equal-weight allocation among bullish symbols (capped at max_position_pct)
        max_pct = CONFIG.get("max_position_pct", 0.25)
        n = len(bullish_syms)
        per_sym = min(1.0 / n, max_pct)
        for sym, score in bullish_syms:
            target_allocs[sym] = per_sym

        orders = portfolio.rebalance(target_allocs, current_prices)
        for order in orders:
            signal = {
                "symbol": order["symbol"],
                "side": order["side"],
                "entry_price": order["current_price"],
                "asset_type": "crypto" if is_crypto(order["symbol"]) else "stock",
                "confidence": 0.6,
                "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            execute_trade(signal)

        if orders:
            logging.info(f"[Main] Rebalanced: executed {len(orders)} orders")

    except Exception as e:
        logging.warning(f"[Main] Rebalance error: {e}")


# ── Backtest mode ─────────────────────────────────────────────────

def run_backtest_mode(use_llm: bool = False) -> None:
    """Backtest the strategy on every symbol in the watchlist."""
    from backtest.backtester import run_backtest

    original_llm = CONFIG.get("use_llm", False)
    if not use_llm:
        CONFIG["use_llm"] = False

    # Flag so strategy engine skips live sentiment/news fetching per bar
    CONFIG["backtest_mode"] = True

    symbols = get_all_symbols()

    for sym in symbols:
        logging.info(f"[Backtest] Fetching data for {sym} …")
        df = fetch_latest_market_data(ticker=sym, period="1y", interval="1h")
        if df is None or len(df) < 100:
            logging.warning(f"[Backtest] Not enough data for {sym}, skipping")
            continue

        def strategy_fn(df_slice, symbol=sym):
            return generate_trade_signal(df_slice, symbol=symbol)

        result = run_backtest(strategy_fn, df, symbol=sym, use_llm=use_llm)
        m = result["metrics"]
        logging.info(
            f"[Backtest] {sym} DONE: "
            f"{m['total_trades']} trades | "
            f"Return={m['total_return_pct']:.2f}% | "
            f"Sharpe={m['sharpe_ratio']:.2f} | "
            f"MaxDD={m['max_drawdown_pct']:.2f}% | "
            f"WinRate={m['win_rate']*100:.1f}% | "
            f"Final=${m['final_equity']:.2f}"
        )

    CONFIG["use_llm"] = original_llm  # restore
    CONFIG["backtest_mode"] = False


# ── Train mode ────────────────────────────────────────────────────

def run_train_mode() -> None:
    """Train ML (XGBoost) and RL models from the latest data."""
    from models.price_predictor import train_model
    from models.reinforce_trainer import train_reinforcement_model

    symbols = get_all_symbols()
    for sym in symbols:
        logging.info(f"[Train] Fetching data for {sym} …")
        df = fetch_latest_market_data(ticker=sym, period="1y", interval="1h")
        if df is None or len(df) < 200:
            logging.warning(f"[Train] Not enough data for {sym}")
            continue

        logging.info(f"[Train] Training XGBoost for {sym} …")
        train_model(df, save=True, symbol=sym)

        logging.info(f"[Train] Training RL agent for {sym} …")
        train_reinforcement_model(df, episodes=50, save=True, symbol=sym)

    logging.info("[Train] All models trained")


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-asset trading bot")
    parser.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    parser.add_argument("--backtest", action="store_true", help="Run backtests on all symbols")
    parser.add_argument("--backtest-llm", action="store_true", help="Run backtests WITH LLM (cached)")
    parser.add_argument("--train", action="store_true", help="Train ML/RL models")
    parser.add_argument("--pretrain", action="store_true", help="Pre-train on max historical data")
    args = parser.parse_args()

    if args.backtest or args.backtest_llm:
        run_backtest_mode(use_llm=args.backtest_llm)
        return

    if args.train:
        run_train_mode()
        return

    if args.pretrain:
        # Delegate to the pretrain script
        from scripts.pretrain import main as pretrain_main
        pretrain_main()
        return

    # ── Start subsystems for live/paper loop ─────────────────────
    _start_subsystems()
    
    # ── PRODUCTION SAFETY: Log startup configuration (C7) ─────────
    log_startup_configuration()
    
    # ── LIMITLESS Startup Banner ─────────────────────────────
    print("\n" + "="*80)
    print("🧠  LIMITLESS — AUTONOMOUS AI TRADING SYSTEM")
    print("="*80)
    print("\"What if you could unlock 100% of your brain's trading potential?\"")
    print()
    print("Key Capabilities:")
    print("  - Instant information assimilation - processes all market data immediately")
    print("  - Perfect recall - learns from every trade outcome")
    print("  - Pattern recognition - spots invisible market clues")
    print("  - High-frequency execution - 500+ trades/day capacity")
    print("  - World events synthesis - analyzes news at machine speed")
    print("  - Self-evolution - continuous strategy adaptation")
    print()
    print("Benchmark: Claude Sonnet AI +1,322% in 48hrs | 5,200+ trades")
    print("="*80)
    print()
    if args.once:
        run_one_cycle()
        _stop_subsystems()
        return

    # Continuous loop
    interval = get_loop_interval()  # Mode-aware interval
    trading_mode = CONFIG.get("trading_mode", "balanced")
    
    logging.info(f"Starting continuous AUTONOMOUS trading loop")
    logging.info(f"Trading mode: {trading_mode} (interval={interval}s)")
    logging.info(f"Asset class: {CONFIG['asset_class']}")
    logging.info(f"Paper trading: {CONFIG['use_paper_trading']}")
    logging.info(f"LLM enabled: {CONFIG.get('use_llm', False)}")
    logging.info(f"Autonomous learning: {CONFIG.get('enable_autonomous_learning', True)}")
    logging.info(f"World events analysis: {CONFIG.get('enable_world_events_analysis', True)}")
    logging.info(f"WebSocket feed: {'active' if _feed_manager else 'inactive (yfinance fallback)'}")
    logging.info(f"LLM watcher: {'active' if _llm_watcher else 'inactive'}")
    logging.info(f"Symbols: {get_all_symbols()}")
    logging.info("─" * 80)
    logging.info("LIMITLESS MODE: Superhuman intelligence active")
    logging.info("   Zero learning curve. Instant pattern recognition. Perfect recall.")
    logging.info("   No human intervention required.")
    logging.info("─" * 80)

    while True:
        try:
            run_one_cycle()
        except SystemExit:
            # Emergency stop triggered (EMERGENCY_STOP file detected)
            logging.info("Emergency stop triggered - exiting gracefully")
            break
        except KeyboardInterrupt:
            logging.info("Interrupted — saving state and shutting down")
            portfolio = get_portfolio()
            portfolio.save_state()
            
            # Save autonomous learning
            if CONFIG.get("enable_autonomous_learning", True):
                from models.autonomous_agent import get_autonomous_agent
                agent = get_autonomous_agent()
                agent.save_journal()
                
                # Log final learning summary
                summary = agent.get_learning_summary()
                logging.info("─" * 80)
                logging.info("AUTONOMOUS LEARNING SUMMARY:")
                logging.info(f"  Total decisions: {summary['total_decisions']}")
                logging.info(f"  Trades executed: {summary['total_executed']}")
                logging.info(f"  Win rate: {summary['overall_win_rate']:.1%}")
                logging.info(f"  Symbols learned: {summary['symbols_learned']}")
                logging.info("─" * 80)
            
            _stop_subsystems()
            sys.exit(0)
        except Exception as e:
            logging.exception(f"Cycle error: {e}")
            try:
                from alerts.sms_notifier import send_alert
                send_alert(f"BOT CRASH: Cycle error - {str(e)[:120]}")
            except Exception:
                pass

        logging.info(f"Sleeping {interval}s until next cycle …")
        time.sleep(interval)


if __name__ == "__main__":
    main()
