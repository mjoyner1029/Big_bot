"""core package — platform architecture layer."""

from core.signal_flipper import Signal, SignalType, AssetClass
from core.strategy_base import StrategyBase
from core.strategy_registry import StrategyRegistry

# New safety modules (Aug 1, 2026)
from core.safety_manager import SafetyManager
from core.transaction_costs import TransactionCostModel
from core.health_monitor import HealthMonitor  
from core.position_manager import PositionManager

__all__ = [
    "Signal",
    "SignalType",
    "AssetClass",
    "StrategyBase",
    "StrategyRegistry",
    # New safety modules
    "SafetyManager",
    "TransactionCostModel",
    "HealthMonitor",
    "PositionManager",
]
