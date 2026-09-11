"""Canonical signal object shared across every strategy and engine.

Every strategy must return a Signal (or a plain dict matching this shape).
This is the single source of truth for what a trade signal looks like.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"
    CALL = "CALL"   # options: buy a call (bullish)
    PUT = "PUT"     # options: buy a put (bearish / hedge)


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    STOCK = "stock"
    ETF = "etf"
    OPTION = "option"
    UNKNOWN = "unknown"


@dataclass
class Signal:
    """Immutable trade signal produced by any strategy.

    Fields
    ------
    symbol        : Ticker string (e.g. "AAPL", "BTC-USD")
    signal        : BUY | SELL | NO_TRADE
    confidence    : 0-100 score produced by master scoring system
    entry         : Suggested entry price (None for NO_TRADE)
    stop_loss     : Hard stop-loss price (None for NO_TRADE)
    targets       : Ordered list of profit targets
    strategy_name : Name of the strategy that produced this signal
    reason        : Human-readable explanation
    asset_class   : Asset class of the symbol
    metadata      : Arbitrary extra data (box levels, indicators, etc.)
    """

    symbol: str
    signal: SignalType = SignalType.NO_TRADE
    confidence: float = 0.0
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    targets: List[float] = field(default_factory=list)
    strategy_name: str = "unknown"
    reason: str = ""
    asset_class: AssetClass = AssetClass.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Convenience helpers ───────────────────────────────────────

    @property
    def is_actionable(self) -> bool:
        """True when the signal is BUY, SELL, CALL, or PUT (not NO_TRADE)."""
        return self.signal in (SignalType.BUY, SignalType.SELL, SignalType.CALL, SignalType.PUT)

    @property
    def risk_reward(self) -> Optional[float]:
        """R:R ratio using first target.  None when data is absent."""
        if (
            self.entry is not None
            and self.stop_loss is not None
            and self.targets
        ):
            risk = abs(self.entry - self.stop_loss)
            reward = abs(self.targets[0] - self.entry)
            if risk > 0:
                return round(reward / risk, 2)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        d = asdict(self)
        d["signal"] = self.signal.value
        d["asset_class"] = self.asset_class.value
        return d

    @classmethod
    def no_trade(
        cls,
        symbol: str,
        strategy_name: str = "unknown",
        reason: str = "No trade conditions met",
        **metadata: Any,
    ) -> "Signal":
        """Factory: create a NO_TRADE signal with minimal boilerplate."""
        return cls(
            symbol=symbol,
            signal=SignalType.NO_TRADE,
            confidence=0.0,
            strategy_name=strategy_name,
            reason=reason,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        rr = self.risk_reward
        rr_str = f"  R:R={rr}" if rr else ""
        return (
            f"Signal({self.symbol} {self.signal.value} "
            f"conf={self.confidence:.0f}{rr_str}  [{self.strategy_name}])"
        )
