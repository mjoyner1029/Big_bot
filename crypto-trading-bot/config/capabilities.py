"""Capability matrix — configuration validation WITHOUT import-time exits.

Each capability reports AVAILABLE / UNAVAILABLE / MISCONFIGURED with a reason.
Modes validate only what they actually need:

    RESEARCH  — no broker or LLM keys required
    BACKTEST  — no broker or LLM keys required
    PAPER     — no broker keys required (simulated fills)
    LIVE      — Alpaca credentials + explicit live flags required
    LLM       — Anthropic key required only when Claude features are used
    STOCKS    — Alpaca market-data credentials
    CRYPTO    — public data, no credentials required
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

AVAILABLE = "AVAILABLE"
UNAVAILABLE = "UNAVAILABLE"
MISCONFIGURED = "MISCONFIGURED"


@dataclass
class CapabilityStatus:
    name: str
    status: str
    reason: str = ""
    required_env: List[str] = field(default_factory=list)


def _has(var: str) -> bool:
    return bool(os.getenv(var, "").strip())


def check_capability(name: str) -> CapabilityStatus:
    name = name.upper()
    if name in ("RESEARCH", "BACKTEST", "CRYPTO"):
        return CapabilityStatus(name, AVAILABLE, "no credentials required")

    if name == "PAPER":
        return CapabilityStatus(name, AVAILABLE, "simulated fills — no broker keys required")

    if name == "LIVE":
        required = ["ALPACA_API_KEY", "ALPACA_API_SECRET"]
        missing = [v for v in required if not _has(v)]
        if missing:
            return CapabilityStatus(name, UNAVAILABLE,
                                    f"missing {missing}", required)
        mode = os.getenv("TRADING_MODE", "PAPER").upper()
        enable = os.getenv("ENABLE_LIVE_TRADING", "false").lower()
        if mode != "LIVE" or enable != "true":
            return CapabilityStatus(
                name, MISCONFIGURED,
                "credentials present but TRADING_MODE=LIVE and "
                "ENABLE_LIVE_TRADING=true are also required", required)
        return CapabilityStatus(name, AVAILABLE, "live credentials and flags set", required)

    if name == "LLM":
        if _has("ANTHROPIC_API_KEY"):
            return CapabilityStatus(name, AVAILABLE, "", ["ANTHROPIC_API_KEY"])
        return CapabilityStatus(name, UNAVAILABLE,
                                "ANTHROPIC_API_KEY not set — rule-based fallbacks active",
                                ["ANTHROPIC_API_KEY"])

    if name == "STOCKS":
        required = ["ALPACA_API_KEY", "ALPACA_API_SECRET"]
        missing = [v for v in required if not _has(v)]
        if missing:
            return CapabilityStatus(name, UNAVAILABLE, f"missing {missing}", required)
        return CapabilityStatus(name, AVAILABLE, "", required)

    return CapabilityStatus(name, UNAVAILABLE, f"unknown capability '{name}'")


def capability_matrix() -> Dict[str, CapabilityStatus]:
    return {n: check_capability(n) for n in
            ("RESEARCH", "BACKTEST", "PAPER", "LIVE", "LLM", "STOCKS", "CRYPTO")}


def check_startup_requirements(mode: str) -> List[str]:
    """Errors preventing startup in ``mode``. Empty list = OK.

    Raise/exit belongs to the caller's startup path — never at import.
    """
    mode = mode.upper()
    errors: List[str] = []
    needed = {
        "RESEARCH": [], "BACKTEST": [], "PAPER": [],
        "LIVE": ["LIVE"],
    }.get(mode)
    if needed is None:
        return [f"unknown mode '{mode}'"]
    for cap in needed:
        status = check_capability(cap)
        if status.status != AVAILABLE:
            errors.append(f"{cap}: {status.status} — {status.reason}")
    return errors
