"""Macro Snapshot — Market regime detection and analysis.

Periodically analyzes market conditions (multi-timeframe price changes, RSI,
volume) to classify the current regime and adjust trading strategy accordingly.

Inspired by the Meteora LP bot's macro monitoring that prevented aggressive
positioning during downtrends.

Regimes:
  • Bull: Strong uptrend, high confidence for longs
  • Bear: Strong downtrend, reduce risk or short bias
  • Ranging: Sideways, mean reversion strategies
  • Volatile: High volatility, reduce position sizes
  • Crisis: Extreme conditions, defensive mode

Usage:
    from strategies.macro_snapshot import get_macro_snapshot, get_regime
    
    # In main loop:
    macro = get_macro_snapshot("BTC-USD")
    regime = macro["regime"]
    
    if regime == "crisis":
        # Skip trading or close positions
        pass
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

from config.config import CONFIG


# ── Price history for multi-timeframe analysis ───────────────────

class MacroHistory:
    """Track price history for macro analysis."""
    
    def __init__(self, max_hours: int = 168):  # 7 days
        self.max_hours = max_hours
        self.data: List[Dict[str, Any]] = []
    
    def add(self, symbol: str, price: float, volume: float = 0) -> None:
        """Add a price/volume snapshot."""
        self.data.append({
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "timestamp": datetime.now().isoformat(),
        })
        self._prune()
    
    def _prune(self) -> None:
        """Remove old entries."""
        cutoff = datetime.now() - timedelta(hours=self.max_hours)
        self.data = [
            d for d in self.data
            if datetime.fromisoformat(d["timestamp"]) >= cutoff
        ]
    
    def get_price_at_offset(self, symbol: str, hours_ago: float) -> Optional[float]:
        """Get price approximately N hours ago."""
        target = datetime.now() - timedelta(hours=hours_ago)
        
        candidates = [
            d for d in self.data
            if d["symbol"] == symbol
            and datetime.fromisoformat(d["timestamp"]) <= target
        ]
        
        if not candidates:
            return None
        
        closest = max(
            candidates,
            key=lambda d: datetime.fromisoformat(d["timestamp"])
        )
        return closest["price"]
    
    def get_prices_since(self, symbol: str, hours_ago: float) -> List[float]:
        """Get all prices since N hours ago."""
        cutoff = datetime.now() - timedelta(hours=hours_ago)
        
        prices = [
            d["price"] for d in self.data
            if d["symbol"] == symbol
            and datetime.fromisoformat(d["timestamp"]) >= cutoff
        ]
        
        return prices


# Global history tracker
_macro_history = MacroHistory()


# ── RSI calculation ───────────────────────────────────────────────

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate Wilder's RSI.
    
    Args:
        prices: List of prices (chronological order)
        period: RSI period (default 14)
    
    Returns:
        RSI value [0, 100]
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral if not enough data
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    # Wilder's smoothing
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


# ── Macro snapshot ────────────────────────────────────────────────

def update_macro_history(symbol: str, price: float, volume: float = 0) -> None:
    """Update price history. Call this from main loop."""
    _macro_history.add(symbol, price, volume)


def get_macro_snapshot(
    symbol: str,
    current_price: Optional[float] = None,
    current_volume: Optional[float] = None,
    asset_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Get macro market snapshot for regime detection.
    
    Args:
        symbol: Trading symbol (e.g., "BTC-USD", "AAPL")
        current_price: Current price (if None, uses latest from history)
        current_volume: Current volume
        asset_type: "crypto" or "stock" (auto-detected if None)
    
    Returns:
        Dict with:
            - price_1h_pct: % change in 1 hour
            - price_4h_pct: % change in 4 hours
            - price_24h_pct: % change in 24 hours
            - price_7d_pct: % change in 7 days
            - rsi_14: RSI(14)
            - regime: str (bull/bear/ranging/volatile/crisis)
            - confidence: float [0, 1]
            - recommendation: str
            - asset_type: str
    """
    if not CONFIG.get("macro_monitoring_enabled", True):
        return {
            "enabled": False,
            "regime": "unknown",
            "confidence": 0.5,
            "recommendation": "macro monitoring disabled",
        }
    
    # Auto-detect asset type
    if asset_type is None:
        from config.config import is_crypto
        asset_type = "crypto" if is_crypto(symbol) else "stock"
    
    # Update history with current price
    if current_price is not None:
        update_macro_history(symbol, current_price, current_volume or 0)
    
    # Get multi-timeframe price changes
    price_1h_ago = _macro_history.get_price_at_offset(symbol, 1.0)
    price_4h_ago = _macro_history.get_price_at_offset(symbol, 4.0)
    price_24h_ago = _macro_history.get_price_at_offset(symbol, 24.0)
    price_7d_ago = _macro_history.get_price_at_offset(symbol, 168.0)
    
    # Get recent prices for RSI
    recent_prices = _macro_history.get_prices_since(symbol, 24.0)
    
    # Calculate % changes
    current = current_price or recent_prices[-1] if recent_prices else 0
    
    price_1h_pct = ((current / price_1h_ago) - 1) * 100 if price_1h_ago else 0
    price_4h_pct = ((current / price_4h_ago) - 1) * 100 if price_4h_ago else 0
    price_24h_pct = ((current / price_24h_ago) - 1) * 100 if price_24h_ago else 0
    price_7d_pct = ((current / price_7d_ago) - 1) * 100 if price_7d_ago else 0
    
    # Calculate RSI
    rsi_14 = calculate_rsi(recent_prices, period=14) if len(recent_prices) >= 14 else 50
    
    # Classify regime with asset-specific thresholds
    regime, confidence, recommendation = _classify_regime(
        price_1h_pct, price_4h_pct, price_24h_pct, price_7d_pct, rsi_14, asset_type
    )
    
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "price": current,
        "price_1h_pct": round(price_1h_pct, 2),
        "price_4h_pct": round(price_4h_pct, 2),
        "price_24h_pct": round(price_24h_pct, 2),
        "price_7d_pct": round(price_7d_pct, 2),
        "rsi_14": round(rsi_14, 1),
        "regime": regime,
        "confidence": round(confidence, 2),
        "recommendation": recommendation,
        "data_points": len(recent_prices),
    }
    
    logging.info(
        f"[Macro] {symbol}: {regime.upper()} regime (conf: {confidence:.2f}) | "
        f"1h: {price_1h_pct:+.1f}%, 24h: {price_24h_pct:+.1f}%, "
        f"7d: {price_7d_pct:+.1f}%, RSI: {rsi_14:.1f}"
    )
    
    return snapshot


def _classify_regime(
    pct_1h: float,
    pct_4h: float,
    pct_24h: float,
    pct_7d: float,
    rsi: float,
    asset_type: str = "crypto",
) -> tuple[str, float, str]:
    """Classify market regime from price changes and RSI.
    
    Args:
        pct_1h, pct_4h, pct_24h, pct_7d: Price change percentages
        rsi: RSI value
        asset_type: "crypto" or "stock" - affects thresholds
    
    Returns:
        (regime, confidence, recommendation)
    """
    # Get thresholds from config (asset-specific)
    # Crypto is more volatile, needs higher thresholds
    if asset_type == "crypto":
        crisis_threshold = CONFIG.get("regime_crisis_threshold_crypto", 10)
        volatile_threshold = CONFIG.get("regime_volatile_threshold_crypto", 5)
        bull_threshold = CONFIG.get("regime_bull_threshold_crypto", 2)
        bear_threshold = CONFIG.get("regime_bear_threshold_crypto", -2)
    else:
        # Stocks are less volatile
        crisis_threshold = CONFIG.get("regime_crisis_threshold_stock", 7)
        volatile_threshold = CONFIG.get("regime_volatile_threshold_stock", 3)
        bull_threshold = CONFIG.get("regime_bull_threshold_stock", 1.5)
        bear_threshold = CONFIG.get("regime_bear_threshold_stock", -1.5)
    
    # Crisis regime: Severe drop
    if pct_4h <= -crisis_threshold or pct_24h <= -crisis_threshold:
        return "crisis", 0.9, "CLOSE ALL POSITIONS - Market crash detected"
    
    # Volatile regime: Large swings regardless of direction
    volatility = abs(pct_1h) + abs(pct_4h)
    if volatility > volatile_threshold * 2:
        return "volatile", 0.8, "Reduce position sizes - High volatility"
    
    # Bull regime: Consistent uptrend
    bull_signals = sum([
        pct_1h > 0,
        pct_4h > bull_threshold,
        pct_24h > bull_threshold * 2,
        pct_7d > 0,
        rsi > 50,
    ])
    
    if bull_signals >= 4:
        confidence = min(0.9, bull_signals / 5 + 0.2)
        return "bull", confidence, "Favorable for long positions"
    
    # Bear regime: Consistent downtrend
    bear_signals = sum([
        pct_1h < 0,
        pct_4h < bear_threshold,
        pct_24h < bear_threshold * 2,
        pct_7d < 0,
        rsi < 50,
    ])
    
    if bear_signals >= 4:
        confidence = min(0.9, bear_signals / 5 + 0.2)
        return "bear", confidence, "Reduce longs or consider shorts"
    
    # Ranging regime: Low movement, no clear trend
    if abs(pct_24h) < 1 and abs(pct_7d) < 3:
        return "ranging", 0.6, "Range-bound - Mean reversion strategies"
    
    # Default: Uncertain
    return "uncertain", 0.4, "Mixed signals - Trade with caution"


def get_regime(symbol: str) -> str:
    """Get current market regime (quick helper).
    
    Returns:
        Regime string: "bull", "bear", "ranging", "volatile", "crisis", "uncertain"
    """
    snapshot = get_macro_snapshot(symbol)
    return snapshot.get("regime", "uncertain")


def should_suppress_trading(snapshot: Dict[str, Any]) -> tuple[bool, str]:
    """Check if trading should be suppressed based on macro conditions.
    
    Args:
        snapshot: Macro snapshot from get_macro_snapshot()
    
    Returns:
        (should_suppress, reason)
    """
    regime = snapshot.get("regime", "unknown")
    confidence = snapshot.get("confidence", 0)
    
    # Crisis mode: suppress all trading
    if regime == "crisis":
        return True, "Crisis regime - trading suspended"
    
    # Extreme volatility: suppress
    if regime == "volatile" and confidence > 0.7:
        return True, "Extreme volatility - trading suspended"
    
    # Strong bear + high confidence: suppress aggressive longs
    if regime == "bear" and confidence > 0.7:
        suppress_aggressive = CONFIG.get("suppress_longs_in_bear", True)
        if suppress_aggressive:
            return True, "Strong bear market - long positions suppressed"
    
    # RSI extreme + downtrend: suppress
    rsi = snapshot.get("rsi_14", 50)
    price_7d = snapshot.get("price_7d_pct", 0)
    
    rsi_floor = CONFIG.get("regime_rsi_floor", 40)
    if price_7d < 0 and rsi < rsi_floor:
        return True, f"Downtrend + RSI {rsi:.0f} < {rsi_floor} - trading suppressed"
    
    return False, ""


def get_position_size_multiplier(snapshot: Dict[str, Any]) -> float:
    """Get position size multiplier based on macro conditions.
    
    Args:
        snapshot: Macro snapshot from get_macro_snapshot()
    
    Returns:
        Multiplier [0.5, 1.5] to adjust position sizes
    """
    regime = snapshot.get("regime", "uncertain")
    confidence = snapshot.get("confidence", 0.5)
    
    # Crisis: no new positions
    if regime == "crisis":
        return 0.0
    
    # Volatile: reduce size
    if regime == "volatile":
        return 0.5
    
    # Strong bull: increase size slightly
    if regime == "bull" and confidence > 0.7:
        return 1.3
    
    # Ranging or uncertain: normal size
    if regime in ["ranging", "uncertain"]:
        return 1.0
    
    # Bear: reduce size
    if regime == "bear":
        return 0.7
    
    return 1.0


def get_macro_summary(symbol: str) -> str:
    """Get formatted macro summary for logging/dashboard."""
    snapshot = get_macro_snapshot(symbol)
    
    return f"""
╔════════════════════════════════════════════════╗
║          MACRO SNAPSHOT - {symbol:<17}║
╠════════════════════════════════════════════════╣
║ Regime:             {snapshot['regime'].upper():<21}  ║
║ Confidence:         {snapshot['confidence']:<21.0%}  ║
║ ────────────────────────────────────────────── ║
║ Price Changes:                                 ║
║   1 hour:           {snapshot['price_1h_pct']:>5.1f}%                    ║
║   4 hours:          {snapshot['price_4h_pct']:>5.1f}%                    ║
║   24 hours:         {snapshot['price_24h_pct']:>5.1f}%                    ║
║   7 days:           {snapshot['price_7d_pct']:>5.1f}%                    ║
║ ────────────────────────────────────────────── ║
║ RSI (14):           {snapshot['rsi_14']:>5.1f}                     ║
║ ────────────────────────────────────────────── ║
║ Recommendation:                                ║
║ {snapshot['recommendation']:<46} ║
╚════════════════════════════════════════════════╝
    """.strip()
