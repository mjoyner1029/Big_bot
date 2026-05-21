"""Integration Guide: Solana Bot Features into Main Loop

This guide shows how to integrate the new safety features into your main trading loop.

KEY MODULES ADDED:
1. strategies/killswitch.py - Market crash protection
2. trading/position_health.py - Automatic position auditing
3. trading/profit_pile.py - Realized profit tracking
4. config/state_manager.py - Robust state persistence
5. strategies/macro_snapshot.py - Market regime detection

INTEGRATION STEPS:

═══════════════════════════════════════════════════════════════════
STEP 1: Update imports in main.py
═══════════════════════════════════════════════════════════════════

Add these imports at the top of main.py:

```python
from strategies.killswitch import (
    evaluate_kill_switch,
    should_pause_trading,
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
```

═══════════════════════════════════════════════════════════════════
STEP 2: Add kill switch check at START of each cycle
═══════════════════════════════════════════════════════════════════

BEFORE checking positions or generating signals, add:

```python
def main_loop():
    global _cycle_count
    
    while True:
        _cycle_count += 1
        logging.info(f"\\n{'='*60}\\nCycle {_cycle_count}\\n{'='*60}")
        
        # ──────────────────────────────────────────────────────────
        # KILL SWITCH CHECK (CRITICAL - DO THIS FIRST!)
        # ──────────────────────────────────────────────────────────
        try:
            # Get current market price
            market_symbol = CONFIG.get("kill_market_symbol", "BTC-USD")
            prices = _get_current_prices([market_symbol])
            current_price = prices.get(market_symbol)
            
            if current_price:
                # Evaluate kill switch
                kill_state = evaluate_kill_switch(
                    market_symbol=market_symbol,
                    current_price=current_price
                )
                
                # Update price history for future checks
                update_price_history(market_symbol, current_price)
                
                # If kill switch tripped, close all and skip cycle
                if kill_state["tripped"]:
                    logging.error(
                        f"🚨 KILL SWITCH ACTIVE: {kill_state['reason']}\\n"
                        f"Cooldown until: {kill_state['cooldown_until']}"
                    )
                    
                    # Close all open positions
                    portfolio = get_portfolio()
                    for position in portfolio.open_positions:
                        try:
                            current_price = prices.get(position["symbol"])
                            if current_price:
                                portfolio.close_position(
                                    position,
                                    exit_price=current_price,
                                    result="KILL_SWITCH"
                                )
                                cleanup_closed_position(position["symbol"])
                        except Exception as e:
                            logging.error(f"Error closing position on kill switch: {e}")
                    
                    # Skip rest of cycle
                    time.sleep(CONFIG.get("loop_interval_seconds", 300))
                    continue
                    
        except Exception as e:
            logging.error(f"Kill switch check failed: {e}")
        
        # Continue with normal cycle...
```

═══════════════════════════════════════════════════════════════════
STEP 3: Add macro snapshot and regime detection
═══════════════════════════════════════════════════════════════════

After kill switch check, add macro analysis:

```python
        # ──────────────────────────────────────────────────────────
        # MACRO SNAPSHOT & REGIME DETECTION
        # ──────────────────────────────────────────────────────────
        macro_snapshot = None
        try:
            market_symbol = CONFIG.get("kill_market_symbol", "BTC-USD")
            prices = _get_current_prices([market_symbol])
            current_price = prices.get(market_symbol)
            
            if current_price:
                macro_snapshot = get_macro_snapshot(
                    symbol=market_symbol,
                    current_price=current_price
                )
                
                # Check if we should suppress trading
                should_suppress, reason = should_suppress_trading(macro_snapshot)
                if should_suppress:
                    logging.warning(f"Trading suppressed: {reason}")
                    time.sleep(CONFIG.get("loop_interval_seconds", 300))
                    continue
                
                # Log regime info
                logging.info(
                    f"Market Regime: {macro_snapshot['regime'].upper()} "
                    f"(confidence: {macro_snapshot['confidence']:.0%})"
                )
                
        except Exception as e:
            logging.error(f"Macro snapshot failed: {e}")
```

═══════════════════════════════════════════════════════════════════
STEP 4: Add position health monitoring BEFORE TP/SL checks
═══════════════════════════════════════════════════════════════════

Replace or enhance your existing position check loop:

```python
        # ──────────────────────────────────────────────────────────
        # POSITION HEALTH MONITORING
        # ──────────────────────────────────────────────────────────
        try:
            portfolio = get_portfolio()
            if portfolio.open_positions:
                # Get current prices for all open positions
                position_symbols = [p["symbol"] for p in portfolio.open_positions]
                current_prices = _get_current_prices(position_symbols)
                
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
                                    f"Closing position: {pos_to_close['close_reason']}"
                                )
                                portfolio.close_position(
                                    full_pos,
                                    exit_price=current_price,
                                    result="HEALTH_CHECK"
                                )
                                cleanup_closed_position(full_pos["symbol"])
                                
                    except Exception as e:
                        logging.error(f"Error closing unhealthy position: {e}")
                        
        except Exception as e:
            logging.error(f"Position health monitoring failed: {e}")
```

═══════════════════════════════════════════════════════════════════
STEP 5: Apply macro-based position sizing to new signals
═══════════════════════════════════════════════════════════════════

When generating signals, apply regime-based sizing:

```python
        # ──────────────────────────────────────────────────────────
        # SIGNAL GENERATION (with macro adjustments)
        # ──────────────────────────────────────────────────────────
        for symbol in symbols:
            try:
                # ... (existing signal generation code) ...
                
                signal = generate_trade_signal(symbol, market_data)
                
                if signal["action"] == "buy" or signal["action"] == "sell":
                    # Apply macro-based position sizing
                    if macro_snapshot:
                        size_mult = get_position_size_multiplier(macro_snapshot)
                        signal["_size_mult"] = signal.get("_size_mult", 1.0) * size_mult
                        
                        logging.info(
                            f"Applied macro size multiplier: {size_mult:.2f}x "
                            f"(regime: {macro_snapshot['regime']})"
                        )
                    
                    # ... (continue with signal execution) ...
                    
            except Exception as e:
                logging.error(f"Signal generation failed for {symbol}: {e}")
```

═══════════════════════════════════════════════════════════════════
STEP 6: Record profits in profit pile when closing positions
═══════════════════════════════════════════════════════════════════

Modify the portfolio.close_position() method or add after closing:

```python
# In trading/portfolio_manager.py, in close_position() method:

def close_position(self, position: Dict[str, Any],
                   exit_price: float, result: str) -> None:
    \"\"\"Close a position, compute P&L, log it, and return cash.\"\"\"
    qty = position["qty"]
    entry = position["entry_price"]
    side = position["side"]

    if side == "buy":
        pnl = (exit_price - entry) * qty
    else:
        pnl = (entry - exit_price) * qty

    self.cash += qty * exit_price  # return proceeds

    trade_record = {
        **position,
        "exit_price": exit_price,
        "pnl": round(pnl, 2),
        "result": result,
        "closed_at": datetime.now(timezone.utc).isoformat(),
    }
    self.closed_trades.append(trade_record)

    if position in self.open_positions:
        self.open_positions.remove(position)

    self.save_state()

    # ──────────────────────────────────────────────────────────────
    # RECORD IN PROFIT PILE (NEW)
    # ──────────────────────────────────────────────────────────────
    try:
        from trading.profit_pile import record_profit
        
        split = record_profit(
            profit_usd=pnl,
            symbol=position["symbol"],
            trade_id=position.get("order_id"),
        )
        
        # Add reinvested amount back to cash
        if split["reinvested"] > 0:
            self.cash += split["reinvested"]
            logging.info(
                f"[ProfitPile] Reinvested ${split['reinvested']:.2f} "
                f"back to trading capital"
            )
            
    except Exception as e:
        logging.error(f"Profit pile recording failed: {e}")

    # ... (rest of existing code) ...
```

═══════════════════════════════════════════════════════════════════
STEP 7: Add profit pile status to periodic logging
═══════════════════════════════════════════════════════════════════

Add to your hourly/daily status logs:

```python
        # ──────────────────────────────────────────────────────────
        # PERIODIC STATUS UPDATE (every N cycles)
        # ──────────────────────────────────────────────────────────
        if _cycle_count % 12 == 0:  # Every ~1 hour
            try:
                from trading.profit_pile import format_profit_summary
                
                pile_summary = format_profit_summary()
                logging.info(f"\\n{pile_summary}")
                
            except Exception as e:
                logging.error(f"Profit pile summary failed: {e}")
```

═══════════════════════════════════════════════════════════════════
STEP 8: Use StateManager for bot state (optional enhancement)
═══════════════════════════════════════════════════════════════════

The portfolio manager already has atomic writes. For other state files:

```python
from config.state_manager import StateManager

# Replace manual JSON save/load with:
state_manager = StateManager("my_state.json")

# Save
state_manager.save({"key": "value"})

# Load
state = state_manager.load()

# List backups
backups = state_manager.list_backups()
```

═══════════════════════════════════════════════════════════════════
TESTING CHECKLIST
═══════════════════════════════════════════════════════════════════

Before going live:

□ Test kill switch in paper mode (manually adjust prices to trigger)
□ Test position health monitor (let positions go underwater)
□ Verify profit pile accounting (check state/profit_pile.json)
□ Check macro snapshot updates every 15 minutes
□ Verify state backups are created (state/backups/)
□ Test recovery from corrupted state file
□ Monitor logs for new health check messages
□ Verify positions close automatically when underwater
□ Check that kill switch cooldown works
□ Verify profit reinvestment appears in portfolio cash

═══════════════════════════════════════════════════════════════════
MONITORING & OPERATIONS
═══════════════════════════════════════════════════════════════════

New log patterns to watch for:

✓ "[KillSwitch] Kill switch tripped" - Market crash detected
✓ "[PosHealth] Position flagged for close" - Underwater position
✓ "[ProfitPile] Recorded profit" - Trade closed with profit
✓ "[Macro] Bull regime" - Market condition update
✓ "[StateManager] Recovered from backup" - State corruption recovered

New files to monitor:

- state/killswitch.json - Kill switch state
- state/position_health.json - Position health tracking
- state/profit_pile.json - Profit pile accounting
- state/backups/ - Automatic state backups

═══════════════════════════════════════════════════════════════════
EMERGENCY PROCEDURES
═══════════════════════════════════════════════════════════════════

Manual kill switch reset:
```python
from strategies.killswitch import reset_kill_switch
reset_kill_switch()
```

Reset position health strikes:
```python
from trading.position_health import reset_all_strikes
reset_all_strikes()
```

Restore from backup:
```python
from config.state_manager import get_portfolio_state_manager
manager = get_portfolio_state_manager()
manager.restore_from_backup("20260507-143022.json")
```

═══════════════════════════════════════════════════════════════════
CONFIGURATION TUNING
═══════════════════════════════════════════════════════════════════

Key parameters to adjust based on your risk tolerance:

Kill Switch (in config.py):
- kill_4h_drop_pct: 6.0 (more aggressive) to 10.0 (more tolerant)
- kill_24h_drop_pct: 10.0 (aggressive) to 15.0 (tolerant)

Position Health:
- loss_watchdog_strikes: 3 (aggressive) to 5 (patient)
- trailing_stop_pct: 50 (tight) to 70 (loose)

Profit Pile:
- profit_reinvest_pct: 60 (moderate growth) to 80 (aggressive compounding)

═══════════════════════════════════════════════════════════════════

For full documentation, see SOLANA_BOT_INTEGRATION.md
"""