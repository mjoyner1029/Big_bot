"""
Canonical trading mode definitions.

Two separate concepts are defined here:

ExecutionMode
─────────────
WHERE orders go — the deployment environment.

    BACKTEST   Historical replay, no live data, no broker
    PAPER      Simulated orders against a paper broker account
    SHADOW     Live market data, real decision pipeline, NO order submission
    LIVE       Real capital, real broker — requires explicit dual authorization

StrategyProfile
───────────────
HOW the strategy behaves — risk/frequency settings.

    CONSERVATIVE   1% risk/trade, 10 min loop, ≤3 positions
    BALANCED       2% risk/trade,  5 min loop, ≤5 positions
    AGGRESSIVE     3% risk/trade,  2 min loop, ≤8 positions

Usage
─────
    from core.trading_mode import ExecutionMode, resolve_execution_mode

    mode = resolve_execution_mode()   # reads EXECUTION_MODE env var
    if mode == ExecutionMode.LIVE:
        ...
"""
from __future__ import annotations

import os
from enum import Enum


class ExecutionMode(str, Enum):
    """Canonical execution modes. Unknown values are rejected."""
    BACKTEST = "BACKTEST"
    PAPER    = "PAPER"
    SHADOW   = "SHADOW"
    LIVE     = "LIVE"


# Aliases accepted in the environment (normalized to canonical)
_ALIASES: dict[str, ExecutionMode] = {
    "backtest": ExecutionMode.BACKTEST,
    "paper":    ExecutionMode.PAPER,
    "shadow":   ExecutionMode.SHADOW,
    "live":     ExecutionMode.LIVE,
}

PAPER_BROKER_DOMAINS = ("paper-api.alpaca.markets",)
LIVE_BROKER_DOMAINS  = ("api.alpaca.markets",)


def resolve_execution_mode(raw: str | None = None) -> ExecutionMode:
    """
    Parse and normalize an execution mode string.

    Accepts any case variation of BACKTEST / PAPER / SHADOW / LIVE.
    Raises ValueError for unknown values (including obsolete strategy
    profiles such as 'claude_hf', 'balanced', 'conservative').

    Args:
        raw: mode string; if None, reads EXECUTION_MODE env var
             (falling back to TRADING_MODE for backward compat).

    Returns:
        ExecutionMode enum value.

    Raises:
        ValueError: if the value is unrecognized.
    """
    if raw is None:
        raw = (
            os.getenv("EXECUTION_MODE")
            or os.getenv("TRADING_MODE")
            or ""
        )

    normalized = raw.strip().lower()

    if normalized in _ALIASES:
        return _ALIASES[normalized]

    # Reject obsolete / confusing values explicitly
    _OBSOLETE = {"claude_hf", "balanced", "conservative", "aggressive"}
    if normalized in _OBSOLETE:
        raise ValueError(
            f"'{raw}' is a strategy profile, not an execution mode. "
            f"Set EXECUTION_MODE to one of: BACKTEST, PAPER, SHADOW, LIVE"
        )

    raise ValueError(
        f"Unknown execution mode: '{raw}'. "
        f"Allowed values: BACKTEST, PAPER, SHADOW, LIVE"
    )


def is_live_authorized() -> bool:
    """
    Return True ONLY if BOTH conditions are met:
        EXECUTION_MODE=LIVE (or TRADING_MODE=LIVE)
        ENABLE_LIVE_TRADING=true

    Presence of API credentials NEVER activates live trading alone.
    """
    try:
        mode = resolve_execution_mode()
    except ValueError:
        return False

    if mode != ExecutionMode.LIVE:
        return False

    return os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"


def verify_broker_matches_mode(base_url: str, mode: ExecutionMode) -> tuple[bool, str]:
    """
    Verify that the Alpaca base URL matches the requested execution mode.

    Returns (ok, reason).
    """
    url = (base_url or "").lower().rstrip("/")

    is_paper_url = any(d in url for d in PAPER_BROKER_DOMAINS)
    is_live_url  = "api.alpaca.markets" in url and "paper" not in url

    if mode == ExecutionMode.PAPER:
        if is_live_url:
            return False, (
                f"EXECUTION_MODE=PAPER but ALPACA_BASE_URL looks like a LIVE endpoint: {base_url}"
            )
        if not is_paper_url:
            return False, (
                f"EXECUTION_MODE=PAPER but ALPACA_BASE_URL is not the paper endpoint. "
                f"Expected something like https://paper-api.alpaca.markets, got: {base_url}"
            )

    if mode == ExecutionMode.LIVE:
        if is_paper_url:
            return False, (
                f"EXECUTION_MODE=LIVE but ALPACA_BASE_URL looks like a PAPER endpoint: {base_url}"
            )

    if mode in (ExecutionMode.SHADOW, ExecutionMode.BACKTEST):
        # No broker order submission — URL mismatch is a warning, not a failure
        if is_live_url:
            return False, (
                f"EXECUTION_MODE={mode.value} — no orders will be submitted, "
                f"but ALPACA_BASE_URL points to live endpoint: {base_url}. "
                f"Recommend switching to paper URL for safety."
            )

    return True, "broker URL matches execution mode"
