"""Return unit conventions.

CONVENTION: all EV/return quantities are stored internally as FRACTIONAL
returns (+1% = 0.01, -2.5% = -0.025). Convert to percent / bps / dollars ONLY
at UI, reporting, or execution boundaries — never inside EV math.

Fields following this convention: mean_net_return, median_net_return,
expected_net_return, return_std, standard_error(_return),
confidence_lower/upper(_return|_bound), expected_win/loss(_return).
Anything in other units must say so in its name (…_usd, …_bps, …_pct).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def to_fractional_return(value: float, unit: str) -> float:
    """Convert value in ('fraction'|'percent'|'bps') to fractional return."""
    unit = unit.lower()
    if unit in ("fraction", "fractional", "frac"):
        return value
    if unit in ("percent", "pct", "%"):
        return value / 100.0
    if unit == "bps":
        return value / 10_000.0
    raise ValueError(f"Unknown return unit '{unit}'")


def to_percent(fractional: float) -> float:
    return fractional * 100.0


def to_bps(fractional: float) -> float:
    return fractional * 10_000.0


def to_dollars(fractional: float, notional_usd: float) -> float:
    return fractional * notional_usd


# Per-trade sanity ceilings by asset class (fractional return, |value|).
# Deliberately loose — these catch UNIT mistakes (e.g. dollars or percent
# leaking into fractional fields), not genuine tail moves.
_SANITY_CEILING = {
    "stock": 0.5,     # ±50% per trade
    "etf": 0.3,
    "crypto": 1.0,    # ±100% per trade
    "unknown": 1.0,
}


def validate_return_units(
    value: Optional[float],
    field_name: str,
    asset_class: str = "unknown",
    holding_hours: Optional[float] = None,
    context: str = "",
) -> bool:
    """Sanity-check a fractional-return value; log RETURN_UNIT_ANOMALY when it
    looks like a unit mistake. Returns False for anomalous values."""
    if value is None:
        return True
    ceiling = _SANITY_CEILING.get(asset_class, _SANITY_CEILING["unknown"])
    # Short holds can't plausibly produce huge fractional returns
    if holding_hours is not None and holding_hours <= 24:
        ceiling = min(ceiling, 0.25)
    if abs(value) > ceiling:
        logger.warning(
            f"RETURN_UNIT_ANOMALY: {field_name}={value:+.4f} "
            f"({value * 100:+.1f}%) exceeds {ceiling:.0%} sanity ceiling for "
            f"{asset_class}"
            + (f" holding={holding_hours}h" if holding_hours else "")
            + (f" [{context}]" if context else "")
        )
        return False
    return True
