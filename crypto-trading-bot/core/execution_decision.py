"""ExecutionDecision: carries the optimizer's choice all the way to the broker.

    OpportunityEnricher → ExecutionDecision → execute_trade() → Broker

Nothing is silently downgraded: unsupported methods are re-costed against the
best supported method and the trade is re-evaluated (spec §2-13). Every
decision and its realized cost components are persisted separately —
spread ≠ slippage ≠ impact ≠ fees.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc)

EXECUTION_MODEL_VERSION = "2.0.0"

# Resolution outcomes (spec §6)
USE_AS_IS = "USE_AS_IS"
USE_SUPPORTED_ALTERNATIVE = "USE_SUPPORTED_ALTERNATIVE"
NO_TRADE_EXECUTION_COST = "NO_TRADE_EXECUTION_COST"
NO_TRADE_SIGNAL_EXPIRED = "NO_TRADE_SIGNAL_EXPIRED"
NO_TRADE_EV_GONE = "NO_TRADE_EV_GONE"

# method → broker order_type mapping
_METHOD_TO_ORDER_TYPE = {
    "market": "MARKET",
    "limit_mid": "LIMIT",
    "limit_passive": "LIMIT",
    "marketable_limit": "LIMIT",
    "delayed_1m": "MARKET",
    "delayed_5m": "MARKET",
}

# expected minutes-to-fill priors per method (for half-life-aware fallback)
_METHOD_FILL_MINUTES = {
    "market": 0.1, "marketable_limit": 0.5, "limit_mid": 3.0,
    "limit_passive": 12.0, "delayed_1m": 1.2, "delayed_5m": 5.5,
}


@dataclass
class BrokerCapabilities:
    supports_limit_orders: bool = True
    supports_post_only: bool = False
    supports_auction: bool = False
    supports_venue_selection: bool = False
    supports_time_in_force: bool = False

    @classmethod
    def detect(cls, broker: Any) -> "BrokerCapabilities":
        """Explicit attributes win; otherwise conservative defaults per class."""
        def cap(name: str, default: bool) -> bool:
            return bool(getattr(broker, name, default))
        return cls(
            supports_limit_orders=cap("supports_limit_orders", True),
            supports_post_only=cap("supports_post_only", False),
            supports_auction=cap("supports_auction", False),
            supports_venue_selection=cap("supports_venue_selection", False),
            supports_time_in_force=cap("supports_time_in_force", False),
        )

    def supports_method(self, method: str) -> bool:
        if method in ("market", "delayed_1m", "delayed_5m"):
            return True
        if method in ("limit_mid", "limit_passive", "marketable_limit"):
            return self.supports_limit_orders
        if method == "post_only":
            return self.supports_post_only
        if method == "auction":
            return self.supports_auction
        return False


@dataclass
class ExecutionDecision:
    candidate_id: str
    alpha_id: str
    symbol: str
    side: str                                 # BUY | SELL
    quantity: float
    method: str                               # optimizer method name
    order_type: str                           # broker order type
    limit_price: Optional[float] = None
    venue: Optional[str] = None
    time_in_force: str = "GTC"
    urgency: str = "NORMAL"
    signal_time: Optional[str] = None
    signal_expiration_time: Optional[str] = None
    max_execution_delay_seconds: float = 300.0
    expected_fill_probability: float = 1.0
    # cost components — NEVER lumped together as "slippage" (spec §11)
    expected_spread_bps: float = 0.0
    expected_slippage_bps: float = 0.0
    expected_impact_bps: float = 0.0
    expected_fee_bps: float = 0.0
    expected_borrow_bps: float = 0.0
    expected_funding_bps: float = 0.0
    expected_entry_cost_bps: float = 0.0
    expected_round_trip_cost_bps: float = 0.0
    gross_ev: Optional[float] = None
    net_ev: Optional[float] = None
    fallback_policy: str = "recost_then_decide"
    resolution: str = USE_AS_IS
    execution_model_version: str = EXECUTION_MODEL_VERSION
    detail: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if not self.signal_expiration_time:
            return False
        now = now or _utcnow()
        try:
            exp = datetime.fromisoformat(
                self.signal_expiration_time.replace("Z", "+00:00"))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return now >= exp
        except ValueError:
            return False


def deterministic_limit_price(*, side: str, bid: Optional[float],
                              ask: Optional[float], mid: float,
                              spread_pct: float, urgency: str,
                              method: str) -> Optional[float]:
    """Deterministic limit-price logic from current quote only (spec §8).
    Marketable limits carry price protection — never unbounded."""
    if method == "market":
        return None
    half_spread = mid * max(spread_pct, 0.0) / 2
    bid = bid if bid is not None else mid - half_spread
    ask = ask if ask is not None else mid + half_spread
    buy = side.upper() == "BUY"
    if method == "limit_passive":
        return bid if buy else ask                     # rest at the touch
    if method == "limit_mid":
        return mid
    if method == "marketable_limit":
        # cross the spread but cap chase at 2 half-spreads past the touch
        protection = half_spread * 2
        return (ask + protection) if buy else (bid - protection)
    return mid


def build_execution_decision(candidate: Any, *, allocation_usd: float,
                             price: float,
                             bid: Optional[float] = None,
                             ask: Optional[float] = None,
                             bar_minutes: float = 1440.0,
                             signal_time: Optional[str] = None) -> ExecutionDecision:
    """Build the order instruction from an ENRICHED candidate — nothing the
    optimizer chose is dropped."""
    method = candidate.recommended_order_type or "market"
    side = "BUY" if candidate.direction == "long" else "SELL"
    spread = (candidate.features or {}).get("spread_pct") or 0.001
    mid = price
    urgency = candidate.urgency or "NORMAL"

    # signal expiration from half-life: stale fills are worthless (spec §62)
    expiration = None
    max_delay = 300.0
    if candidate.half_life_bars:
        half_life_minutes = candidate.half_life_bars * bar_minutes
        max_delay = min(max(half_life_minutes * 60 / 4, 30.0), 3600.0)
        base = signal_time or candidate.signal_time
        try:
            t0 = datetime.fromisoformat(str(base).replace("Z", "+00:00"))
            if t0.tzinfo is None:
                t0 = t0.replace(tzinfo=timezone.utc)
            expiration = (t0 + timedelta(minutes=half_life_minutes)).isoformat()
        except (ValueError, TypeError):
            expiration = None

    components = (candidate.features or {}).get("expected_cost_components", {})
    entry_cost = components.get("entry_cost_bps",
                                (candidate.expected_total_cost_bps or 0.0) / 2)
    return ExecutionDecision(
        candidate_id=candidate.candidate_id,
        alpha_id=candidate.alpha_id,
        symbol=candidate.symbol,
        side=side,
        quantity=(allocation_usd / price) if price > 0 else 0.0,
        method=method,
        order_type=_METHOD_TO_ORDER_TYPE.get(method, "MARKET"),
        limit_price=deterministic_limit_price(
            side=side, bid=bid, ask=ask, mid=mid, spread_pct=float(spread),
            urgency=urgency, method=method),
        venue=candidate.recommended_venue,
        urgency=urgency,
        signal_time=signal_time or candidate.signal_time,
        signal_expiration_time=expiration,
        max_execution_delay_seconds=max_delay,
        expected_fill_probability=(candidate.features or {}).get(
            "expected_fill_probability", 1.0),
        expected_spread_bps=components.get("spread_bps", 0.0),
        expected_slippage_bps=components.get("slippage_bps",
                                             candidate.expected_slippage_bps or 0.0),
        expected_impact_bps=components.get("impact_bps", 0.0),
        expected_fee_bps=components.get("fee_bps", 0.0),
        expected_borrow_bps=components.get("borrow_bps", 0.0),
        expected_funding_bps=components.get("funding_bps", 0.0),
        expected_entry_cost_bps=entry_cost,
        expected_round_trip_cost_bps=candidate.expected_total_cost_bps or 0.0,
        gross_ev=candidate.raw_expected_net_return,
        net_ev=candidate.expected_net_return,
    )


def resolve_for_broker(decision: ExecutionDecision,
                       capabilities: BrokerCapabilities,
                       *, now: Optional[datetime] = None) -> ExecutionDecision:
    """Capability-aware resolution (spec §5-7). Unsupported methods are
    NEVER silently replaced with MARKET: the alternative is re-costed and the
    trade re-decided; passive fills slower than the signal's half-life are
    rejected as execution options."""
    if decision.is_expired(now):
        decision.resolution = NO_TRADE_SIGNAL_EXPIRED
        return decision

    method = decision.method
    half_life_minutes = None
    if decision.signal_expiration_time and decision.signal_time:
        try:
            t0 = datetime.fromisoformat(
                str(decision.signal_time).replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(
                decision.signal_expiration_time.replace("Z", "+00:00"))
            half_life_minutes = (t1 - t0).total_seconds() / 60
        except ValueError:
            pass

    def fill_too_slow(m: str) -> bool:
        return (half_life_minutes is not None
                and _METHOD_FILL_MINUTES.get(m, 1.0) > half_life_minutes / 2)

    supported = capabilities.supports_method(method) and not fill_too_slow(method)
    if supported:
        decision.resolution = USE_AS_IS
        return decision

    # Re-cost with the best supported, fast-enough alternative
    alternatives = [m for m in ("limit_mid", "marketable_limit", "market")
                    if capabilities.supports_method(m) and not fill_too_slow(m)]
    if not alternatives:
        decision.resolution = NO_TRADE_SIGNAL_EXPIRED
        decision.detail["why"] = "no supported method fills within signal half-life"
        return decision
    alt = alternatives[0]
    # market-family methods cost the full spread instead of the passive share
    cost_multiplier = {"market": 1.0, "marketable_limit": 0.9, "limit_mid": 0.55}
    old_slip = decision.expected_slippage_bps
    baseline = old_slip / cost_multiplier.get(decision.method, 0.55) \
        if decision.method != "market" else old_slip
    new_slip = baseline * cost_multiplier[alt]
    delta_bps = (new_slip - old_slip) * 2          # entry + exit
    new_net = (decision.net_ev or 0.0) - delta_bps / 10_000
    if new_net <= 0:
        decision.resolution = NO_TRADE_EXECUTION_COST
        decision.detail["why"] = (f"{method} unsupported; best supported {alt} "
                                  f"leaves net EV {new_net:.5f} <= 0")
        return decision
    decision.method = alt
    decision.order_type = _METHOD_TO_ORDER_TYPE.get(alt, "MARKET")
    decision.expected_slippage_bps = new_slip
    decision.net_ev = new_net
    decision.resolution = USE_SUPPORTED_ALTERNATIVE
    decision.detail["fallback_from"] = method
    return decision


def pre_submission_recheck(decision: ExecutionDecision, *,
                           current_spread_pct: float,
                           reference_spread_pct: float) -> ExecutionDecision:
    """Final EV recheck against the LIVE quote right before submission
    (spec §63): if conditions deteriorated and net EV is gone, cancel."""
    widening_bps = max(current_spread_pct - reference_spread_pct, 0.0) * 10_000
    if widening_bps <= 0:
        return decision
    new_net = (decision.net_ev or 0.0) - widening_bps / 10_000   # pay extra spread
    decision.detail["pre_submission_spread_widening_bps"] = widening_bps
    if new_net <= 0:
        decision.resolution = NO_TRADE_EV_GONE
        decision.detail["why"] = (f"spread widened {widening_bps:.1f} bps; "
                                  f"net EV {new_net:.5f} <= 0")
    else:
        decision.net_ev = new_net
    return decision


def remaining_ev_sufficient(decision: ExecutionDecision, *,
                            filled_qty: float, requested_qty: float,
                            current_spread_pct: float,
                            reference_spread_pct: float) -> bool:
    """Partial-fill policy (spec §10): chase the remainder only if its EV
    still clears costs under CURRENT conditions."""
    if requested_qty <= 0 or filled_qty >= requested_qty:
        return False
    probe = ExecutionDecision(**{**asdict(decision)})
    probe = pre_submission_recheck(probe, current_spread_pct=current_spread_pct,
                                   reference_spread_pct=reference_spread_pct)
    return probe.resolution not in (NO_TRADE_EV_GONE,) and (probe.net_ev or 0) > 0


def realized_execution_attribution(
    decision: ExecutionDecision, *, fill_price: float,
    mid_at_decision: float, fees_usd: float, notional_usd: float,
    borrow_funding_usd: float = 0.0) -> Dict[str, float]:
    """Separate REALIZED cost components (spec §12). Adverse price vs the
    decision mid is split: expected-spread share = spread cost, remainder =
    slippage+impact (impact isolated via the model estimate)."""
    if notional_usd <= 0 or mid_at_decision <= 0:
        return {}
    sign = 1.0 if decision.side == "BUY" else -1.0
    adverse_bps = sign * (fill_price - mid_at_decision) / mid_at_decision * 10_000
    spread_cost = min(max(adverse_bps, 0.0), decision.expected_spread_bps)
    residual = max(adverse_bps - spread_cost, 0.0)
    impact_est = min(residual, decision.expected_impact_bps)
    slippage = residual - impact_est
    fee_bps = fees_usd / notional_usd * 10_000
    carry_bps = borrow_funding_usd / notional_usd * 10_000
    return {
        "realized_spread_bps": spread_cost,
        "realized_slippage_bps": slippage,
        "realized_impact_bps": impact_est,
        "realized_fee_bps": fee_bps,
        "realized_borrow_funding_bps": carry_bps,
        "realized_total_cost_bps": max(adverse_bps, 0.0) + fee_bps + carry_bps,
        "predicted_entry_cost_bps": decision.expected_entry_cost_bps,
    }


class ExecutionDecisionStore:
    """Persists every ExecutionDecision and its realized attribution."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite") -> None:
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    candidate_id TEXT, alpha_id TEXT, symbol TEXT, side TEXT,
                    method TEXT, order_type TEXT, limit_price REAL,
                    urgency TEXT, resolution TEXT,
                    quantity REAL, expected_fill_probability REAL,
                    expected_spread_bps REAL, expected_slippage_bps REAL,
                    expected_impact_bps REAL, expected_fee_bps REAL,
                    expected_borrow_bps REAL, expected_round_trip_cost_bps REAL,
                    net_ev REAL, execution_model_version TEXT,
                    realized_json TEXT, detail_json TEXT
                )""")

    def record(self, decision: ExecutionDecision,
               realized: Optional[Dict[str, float]] = None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO execution_decisions (recorded_at, candidate_id, "
                "alpha_id, symbol, side, method, order_type, limit_price, "
                "urgency, resolution, quantity, expected_fill_probability, "
                "expected_spread_bps, expected_slippage_bps, expected_impact_bps, "
                "expected_fee_bps, expected_borrow_bps, "
                "expected_round_trip_cost_bps, net_ev, execution_model_version, "
                "realized_json, detail_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (_utcnow().isoformat(), decision.candidate_id, decision.alpha_id,
                 decision.symbol, decision.side, decision.method,
                 decision.order_type, decision.limit_price, decision.urgency,
                 decision.resolution, decision.quantity,
                 decision.expected_fill_probability,
                 decision.expected_spread_bps, decision.expected_slippage_bps,
                 decision.expected_impact_bps, decision.expected_fee_bps,
                 decision.expected_borrow_bps,
                 decision.expected_round_trip_cost_bps, decision.net_ev,
                 decision.execution_model_version,
                 json.dumps(realized) if realized else None,
                 json.dumps(decision.detail, default=str)))
            return int(cur.lastrowid)

    def attach_realized(self, decision_id: int,
                        realized: Dict[str, float]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE execution_decisions SET realized_json=? WHERE id=?",
                         (json.dumps(realized), decision_id))
