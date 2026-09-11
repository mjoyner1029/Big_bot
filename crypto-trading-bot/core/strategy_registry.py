"""Strategy Registry — plugin system for the trading platform.

Usage
-----
Register at definition time::

    from core.strategy_registry import StrategyRegistry
    from core.strategy_base import StrategyBase

    @StrategyRegistry.register("my_strategy")
    class MyStrategy(StrategyBase):
        name = "my_strategy"
        ...

Or register after the fact::

    StrategyRegistry.register("my_strategy")(MyStrategy)

Retrieve::

    strategy = StrategyRegistry.get("my_strategy")()
    strategies = StrategyRegistry.all_instances()  # one of each, default config
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any, Dict, Iterator, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from core.strategy_base import StrategyBase

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Class-level registry.  No instantiation needed."""

    _registry: Dict[str, Type["StrategyBase"]] = {}

    # ── Registration ──────────────────────────────────────────────

    @classmethod
    def register(cls, name: str):
        """Decorator / callable that registers a strategy class."""

        def decorator(strategy_cls: Type["StrategyBase"]) -> Type["StrategyBase"]:
            if name in cls._registry:
                logger.warning(
                    "[Registry] Overwriting existing strategy '%s' with %s",
                    name,
                    strategy_cls.__name__,
                )
            cls._registry[name] = strategy_cls
            logger.debug("[Registry] Registered strategy '%s' → %s", name, strategy_cls.__name__)
            return strategy_cls

        return decorator

    # ── Lookup ────────────────────────────────────────────────────

    @classmethod
    def get(cls, name: str) -> Optional[Type["StrategyBase"]]:
        """Return the class registered under *name*, or None."""
        return cls._registry.get(name)

    @classmethod
    def get_or_raise(cls, name: str) -> Type["StrategyBase"]:
        """Return the class or raise KeyError."""
        if name not in cls._registry:
            raise KeyError(
                f"Strategy '{name}' not found. "
                f"Available: {list(cls._registry)}"
            )
        return cls._registry[name]

    @classmethod
    def list_names(cls) -> List[str]:
        """Sorted list of registered strategy names."""
        return sorted(cls._registry)

    @classmethod
    def all_instances(
        cls, config: Optional[Dict[str, Any]] = None
    ) -> List["StrategyBase"]:
        """Instantiate every registered strategy with *config*."""
        return [klass(config=config) for klass in cls._registry.values()]

    # ── Auto-discovery ────────────────────────────────────────────

    @classmethod
    def autodiscover(cls, package_name: str = "strategies") -> None:
        """Import every module in *package_name* so decorators fire.

        Call once at startup::

            StrategyRegistry.autodiscover()
        """
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError:
            logger.warning("[Registry] Package '%s' not found for autodiscover", package_name)
            return

        pkg_path = getattr(package, "__path__", [])
        for _, module_name, _ in pkgutil.walk_packages(
            pkg_path, prefix=f"{package_name}."
        ):
            try:
                importlib.import_module(module_name)
                logger.debug("[Registry] Autodiscovered module: %s", module_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Registry] Could not import '%s': %s", module_name, exc)

    # ── Iterator ─────────────────────────────────────────────────

    @classmethod
    def items(cls) -> Iterator[tuple]:
        """Iterate (name, class) pairs."""
        return iter(cls._registry.items())

    def __repr__(cls) -> str:
        return f"<StrategyRegistry strategies={cls.list_names()}>"
