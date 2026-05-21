"""Kill Switch — Market crash protection mechanism.

Automatically closes all positions when market drops beyond configured thresholds,
then enforces a cooldown period before allowing new trades.

Inspired by the Meteora LP bot's kill switch that prevented major losses during
SOL flash crashes.

Safety Rules:
  • Drop > 6% in 4 hours → close all positions, 24h cooldown
  • Drop > 10% in 24 hours → close all positions, 24h cooldown
  • During cooldown, no new positions allowed

Usage:
    from strategies.killswitch import evaluate_kill_switch, should_pause_trading
    
    # In main loop:
    kill_state = evaluate_kill_switch(prices, kill_state)
    if kill_state['tripped']:
        close_all_positions()
        continue  # Skip trading cycle
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json
import os

from config.config import CONFIG


# ── Price history tracking ────────────────────────────────────────

class PriceHistory:
    """Lightweight circular buffer for price history."""
    
    def __init__(self, hours: int = 24):
        self.max_age_hours = hours
        self.data: List[Dict[str, Any]] = []
    
    def add(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """Add a price point."""
        if timestamp is None:
            timestamp = datetime.now()
        
        entry = {
            "symbol": symbol,
            "price": price,
            "timestamp": timestamp.isoformat(),
        }
        self.data.append(entry)
        self._prune()
    
    def _prune(self) -> None:
        """Remove entries older than max_age_hours."""
        cutoff = datetime.now() - timedelta(hours=self.max_age_hours)
        self.data = [
            d for d in self.data
            if datetime.fromisoformat(d["timestamp"]) >= cutoff
        ]
    
    def get_price_at_offset(self, symbol: str, hours_ago: float) -> Optional[float]:
        """Get price approximately N hours ago."""
        target = datetime.now() - timedelta(hours=hours_ago)
        
        # Find closest entry for this symbol before target time
        candidates = [
            d for d in self.data
            if d["symbol"] == symbol
            and datetime.fromisoformat(d["timestamp"]) <= target
        ]
        
        if not candidates:
            return None
        
        # Return most recent one before target
        closest = max(candidates, key=lambda d: datetime.fromisoformat(d["timestamp"]))
        return closest["price"]
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get most recent price for symbol."""
        for d in reversed(self.data):
            if d["symbol"] == symbol:
                return d["price"]
        return None


# ── Kill switch state ─────────────────────────────────────────────

_kill_state: Dict[str, Any] = {
    "tripped": False,
    "tripped_at": None,
    "reason": "",
    "cooldown_until": None,
}

_price_history = PriceHistory(hours=25)  # Track 25h for 24h lookback + buffer

# State persistence
_state_file = os.path.join(CONFIG.get("state_dir", "state"), "killswitch.json")


def _load_state() -> None:
    """Load kill switch state from disk."""
    global _kill_state
    
    if not os.path.exists(_state_file):
        return
    
    try:
        with open(_state_file, "r") as f:
            loaded = json.load(f)
            _kill_state.update(loaded)
        logging.info(f"[KillSwitch] Loaded state: tripped={_kill_state['tripped']}")
    except Exception as e:
        logging.warning(f"[KillSwitch] Failed to load state: {e}")


def _save_state() -> None:
    """Save kill switch state to disk."""
    os.makedirs(os.path.dirname(_state_file), exist_ok=True)
    
    try:
        with open(_state_file, "w") as f:
            json.dump(_kill_state, f, indent=2)
    except Exception as e:
        logging.error(f"[KillSwitch] Failed to save state: {e}")


# Initialize on import
_load_state()


# ── Kill switch evaluation ────────────────────────────────────────

def update_price_history(symbol: str, price: float) -> None:
    """Add a price point to history. Call this from your main loop."""
    _price_history.add(symbol, price)


def evaluate_kill_switch(
    market_symbol: Optional[str] = None,
    current_price: Optional[float] = None,
    asset_type: str = "crypto",
) -> Dict[str, Any]:
    """Evaluate kill switch conditions and return current state.
    
    Args:
        market_symbol: Primary market indicator. If None, uses config default based on asset_type.
        current_price: Current price. If None, uses latest from history.
        asset_type: "crypto" or "stock" - determines which thresholds to use
    
    Returns:
        Dict with:
          - tripped: bool
          - reason: str
          - cooldown_until: str (ISO timestamp)
          - drop_4h_pct: float
          - drop_24h_pct: float
    """
    global _kill_state
    
    # If not enabled, always return safe state
    if not CONFIG.get("kill_switch_enabled", True):
        return {
            "tripped": False,
            "reason": "kill switch disabled",
            "cooldown_until": None,
            "drop_4h_pct": 0,
            "drop_24h_pct": 0,
        }
    
    # Auto-select market symbol based on asset type if not provided
    if market_symbol is None:
        if asset_type == "crypto":
            market_symbol = CONFIG.get("kill_market_symbol_crypto", "BTC-USD")
        else:
            market_symbol = CONFIG.get("kill_market_symbol_stock", "SPY")
    
    now = datetime.now()
    
    # Check if we're still in cooldown
    if _kill_state["tripped"] and _kill_state.get("cooldown_until"):
        cooldown_end = datetime.fromisoformat(_kill_state["cooldown_until"])
        if now < cooldown_end:
            remaining = (cooldown_end - now).total_seconds() / 3600
            logging.debug(
                f"[KillSwitch] Still in cooldown: {remaining:.1f}h remaining. "
                f"Reason: {_kill_state.get('reason', 'unknown')}"
            )
            return {
                "tripped": True,
                "reason": _kill_state["reason"],
                "cooldown_until": _kill_state["cooldown_until"],
                "drop_4h_pct": 0,
                "drop_24h_pct": 0,
            }
        else:
            # Cooldown expired
            logging.warning("[KillSwitch] Cooldown expired — resuming normal operation")
            _kill_state = {
                "tripped": False,
                "tripped_at": None,
                "reason": "",
                "cooldown_until": None,
            }
            _save_state()
    
    # Get current price
    if current_price is None:
        current_price = _price_history.get_latest_price(market_symbol)
    
    if current_price is None:
        logging.warning(f"[KillSwitch] No price data for {market_symbol}")
        return {
            "tripped": False,
            "reason": "no price data",
            "cooldown_until": None,
            "drop_4h_pct": 0,
            "drop_24h_pct": 0,
        }
    
    # Update price history
    update_price_history(market_symbol, current_price)
    
    # Get historical prices
    price_4h_ago = _price_history.get_price_at_offset(market_symbol, 4.0)
    price_24h_ago = _price_history.get_price_at_offset(market_symbol, 24.0)
    
    drop_4h_pct = 0.0
    drop_24h_pct = 0.0
    
    # Get asset-specific thresholds
    # Crypto is more volatile, so use higher thresholds
    # Stocks are less volatile, so use tighter thresholds
    if asset_type == "crypto":
        threshold_4h = CONFIG.get("kill_4h_drop_pct_crypto", 6.0)
        threshold_24h = CONFIG.get("kill_24h_drop_pct_crypto", 10.0)
    else:
        threshold_4h = CONFIG.get("kill_4h_drop_pct_stock", 4.0)
        threshold_24h = CONFIG.get("kill_24h_drop_pct_stock", 7.0)
    
    if price_4h_ago:
        drop_4h_pct = ((price_4h_ago - current_price) / price_4h_ago) * 100
    
    if price_24h_ago:
        drop_24h_pct = ((price_24h_ago - current_price) / price_24h_ago) * 100
    
    # Get thresholds from config
    threshold_4h = CONFIG.get("kill_4h_drop_pct", 6.0)
    threshold_24h = CONFIG.get("kill_24h_drop_pct", 10.0)
    cooldown_hours = CONFIG.get("kill_cooldown_hours", 24)
    
    # Check if we should trip the kill switch
    should_trip = False
    reason = ""
    
    if drop_4h_pct >= threshold_4h:
        should_trip = True
        reason = (
            f"{market_symbol} dropped {drop_4h_pct:.2f}% in 4h "
            f"(threshold {threshold_4h}%)"
        )
    elif drop_24h_pct >= threshold_24h:
        should_trip = True
        reason = (
            f"{market_symbol} dropped {drop_24h_pct:.2f}% in 24h "
            f"(threshold {threshold_24h}%)"
        )
    
    if should_trip:
        cooldown_until = (now + timedelta(hours=cooldown_hours)).isoformat()
        
        _kill_state = {
            "tripped": True,
            "tripped_at": now.isoformat(),
            "reason": reason,
            "cooldown_until": cooldown_until,
        }
        _save_state()
        
        logging.error(
            f"[KillSwitch] 🚨 KILL SWITCH TRIPPED 🚨\n"
            f"  Reason: {reason}\n"
            f"  Current price: ${current_price:.2f}\n"
            f"  4h drop: {drop_4h_pct:.2f}%\n"
            f"  24h drop: {drop_24h_pct:.2f}%\n"
            f"  Cooldown until: {cooldown_until}"
        )
        try:
            from alerts.sms_notifier import send_alert
            send_alert(f"KILL SWITCH TRIPPED: {reason}. Cooldown: {cooldown_hours}h.")
        except Exception:
            pass
    
    return {
        "tripped": _kill_state["tripped"],
        "reason": _kill_state.get("reason", ""),
        "cooldown_until": _kill_state.get("cooldown_until"),
        "drop_4h_pct": drop_4h_pct,
        "drop_24h_pct": drop_24h_pct,
    }


def should_pause_trading() -> tuple[bool, str]:
    """Check if trading should be paused due to kill switch.
    
    Returns:
        (should_pause, reason)
    """
    if not _kill_state["tripped"]:
        return False, ""
    
    # Check if still in cooldown
    if _kill_state.get("cooldown_until"):
        cooldown_end = datetime.fromisoformat(_kill_state["cooldown_until"])
        if datetime.now() < cooldown_end:
            return True, _kill_state.get("reason", "Kill switch active")
    
    return False, ""


def reset_kill_switch() -> None:
    """Manually reset the kill switch (use with caution!)."""
    global _kill_state
    
    logging.warning("[KillSwitch] Manual reset triggered")
    _kill_state = {
        "tripped": False,
        "tripped_at": None,
        "reason": "",
        "cooldown_until": None,
    }
    _save_state()


def get_kill_switch_status() -> Dict[str, Any]:
    """Get current kill switch state (for monitoring/dashboard)."""
    return _kill_state.copy()


# ── Multi-asset kill switch (advanced) ────────────────────────────

def evaluate_portfolio_drawdown(
    portfolio_value: float,
    peak_value: float,
    max_drawdown_pct: float = 15.0,
) -> tuple[bool, str]:
    """Evaluate if portfolio drawdown exceeds threshold.
    
    This is an alternative/complementary kill switch based on total
    portfolio performance rather than market price.
    
    Args:
        portfolio_value: Current total portfolio value
        peak_value: Historical peak portfolio value
        max_drawdown_pct: Maximum allowed drawdown %
    
    Returns:
        (should_kill, reason)
    """
    if peak_value <= 0:
        return False, ""
    
    drawdown_pct = ((peak_value - portfolio_value) / peak_value) * 100
    
    if drawdown_pct >= max_drawdown_pct:
        reason = (
            f"Portfolio drawdown {drawdown_pct:.1f}% exceeds threshold "
            f"{max_drawdown_pct}% (peak=${peak_value:.2f}, "
            f"current=${portfolio_value:.2f})"
        )
        return True, reason
    
    return False, ""


def evaluate_multi_asset_kill_switch(
    crypto_symbols: list = None,
    stock_symbols: list = None,
    current_prices: dict = None,
) -> Dict[str, Any]:
    """Evaluate kill switch across multiple asset classes.
    
    Monitors both crypto AND stock markets. Triggers if EITHER crashes.
    
    Args:
        crypto_symbols: List of crypto symbols to monitor (e.g., ["BTC-USD", "ETH-USD"])
        stock_symbols: List of stock symbols to monitor (e.g., ["SPY", "QQQ"])
        current_prices: Dict of {symbol: price}
    
    Returns:
        Dict with kill switch state
    """
    if crypto_symbols is None:
        crypto_symbols = [CONFIG.get("kill_market_symbol_crypto", "BTC-USD")]
    
    if stock_symbols is None:
        stock_symbols = [CONFIG.get("kill_market_symbol_stock", "SPY")]
    
    current_prices = current_prices or {}
    
    results = {
        "tripped": False,
        "reason": "",
        "cooldown_until": None,
        "crypto_states": {},
        "stock_states": {},
    }
    
    # Check crypto markets
    for symbol in crypto_symbols:
        price = current_prices.get(symbol)
        state = evaluate_kill_switch(symbol, price, asset_type="crypto")
        results["crypto_states"][symbol] = state
        
        if state["tripped"]:
            results["tripped"] = True
            results["reason"] = f"Crypto: {state['reason']}"
            results["cooldown_until"] = state["cooldown_until"]
            return results
    
    # Check stock markets
    for symbol in stock_symbols:
        price = current_prices.get(symbol)
        state = evaluate_kill_switch(symbol, price, asset_type="stock")
        results["stock_states"][symbol] = state
        
        if state["tripped"]:
            results["tripped"] = True
            results["reason"] = f"Stock: {state['reason']}"
            results["cooldown_until"] = state["cooldown_until"]
            return results
    
    return results

