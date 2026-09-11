"""Abstract base class for every strategy in the platform.

How to add a new strategy
--------------------------
1. Create ``strategies/my_strategy.py``
2. Subclass ``StrategyBase``
3. Implement ``generate_signal()``
4. Decorate with ``@StrategyRegistry.register("my_strategy")``
   (or call ``StrategyRegistry.register("my_strategy")(MyStrategy)`` at
   module bottom)
5. Drop the file into strategies/ — the registry auto-discovers on import.

That's it.  No other file needs to change.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from core.signal_flipper import Signal, AssetClass


class StrategyBase(ABC):
    """Interface every strategy must implement.

    Parameters passed to the constructor are strategy-specific and should
    all carry sensible defaults so the strategy is usable out of the box.
    """

    #: Override in subclasses; used for logging and registry keys.
    name: str = "base"

    #: Asset classes this strategy supports.  Override to restrict.
    supported_asset_classes: tuple = (
        AssetClass.STOCK,
        AssetClass.ETF,
        AssetClass.CRYPTO,
        AssetClass.OPTION,
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self.logger = logging.getLogger(f"strategy.{self.name}")

    # ── Required interface ────────────────────────────────────────

    @abstractmethod
    def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Signal:
        """Analyse ``data`` and return a Signal for ``symbol``.

        Parameters
        ----------
        symbol : str
            Ticker / instrument identifier.
        data : dict
            Arbitrary market data dict.  Each strategy documents what keys
            it expects (e.g. ``current_candle``, ``previous_day_ohlc``,
            ``atr``, ``df``, ...).

        Returns
        -------
        Signal
            Always return a Signal — never raise or return None.
            Use ``Signal.no_trade()`` when conditions are not met.
        """

    # ── Optional hooks ────────────────────────────────────────────

    def on_trade_result(self, symbol: str, _profit_pct: float) -> None:
        """Called after a position closes so strategies can self-adapt.

        Override to implement feedback learning.
        """

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Pre-flight check on incoming data.  Return False to skip."""
        return True

    # ── Helpers ───────────────────────────────────────────────────

    def _cfg(self, key: str, default: Any) -> Any:
        """Convenience accessor: checks injected config first."""
        return self._config.get(key, default)

    def _no_trade(self, symbol: str, reason: str, **meta: Any) -> Signal:
        return Signal.no_trade(
            symbol=symbol, strategy_name=self.name, reason=reason, **meta
        )

    def __repr__(self) -> str:
        return f"<Strategy:{self.name}>"
