"""Execution alpha: reduce trading friction without touching signal validity.

    SlippageModel        : expected + uncertainty + tail slippage
    MarketImpactModel    : square-root impact, capacity aware
    ExecutionOptimizer   : method/venue selection by NET expected outcome
    BorrowChecker        : equity short feasibility (borrow/HTB/SSR)
    ExecutionLearner     : predicted-vs-actual calibration, damped updates

The optimizer never changes whether an Alpha is statistically valid; it only
changes HOW a validated opportunity is executed (spec §25-32, §52).
"""
from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


# ── Slippage (spec §28) ───────────────────────────────────────────────────────


@dataclass
class SlippageEstimate:
    expected_bps: float
    uncertainty_bps: float
    tail_bps: float               # ~p95 adverse case
    spread_bps: float = 0.0       # half-spread component (distinct from slippage)
    size_bps: float = 0.0         # size/liquidity slippage component


class SlippageModel:
    """Structural estimate: half-spread + size/liquidity + volatility terms.
    Calibration multiplier is learned from realized fills (ExecutionLearner)."""

    def __init__(self, calibration_multiplier: float = 1.0) -> None:
        self.calibration_multiplier = calibration_multiplier

    def estimate(self, *, spread_pct: float, order_notional: float,
                 adv_usd: Optional[float], volatility_daily: float = 0.02,
                 order_type: str = "market", urgency: str = "normal"
                 ) -> SlippageEstimate:
        half_spread_bps = max(spread_pct, 0.0) * 10_000 / 2
        participation = (order_notional / adv_usd) if adv_usd else 0.02
        size_bps = 10_000 * 0.1 * volatility_daily * math.sqrt(max(participation, 0.0))
        base = half_spread_bps + size_bps
        mult = 1.0
        if order_type == "limit":
            mult = 0.35          # maker: no spread crossing, but adverse-selection residual
        if urgency == "high":
            mult *= 1.5
        mult *= self.calibration_multiplier
        base *= mult
        return SlippageEstimate(
            expected_bps=base,
            uncertainty_bps=base * 0.6,
            tail_bps=base * 3.0,
            spread_bps=half_spread_bps * mult,
            size_bps=size_bps * mult,
        )


# ── Market impact (spec §29) ──────────────────────────────────────────────────


def market_impact_bps(*, order_notional: float, adv_usd: Optional[float],
                      volatility_daily: float = 0.02,
                      impact_coeff: float = 0.6) -> float:
    """Square-root impact: coeff * sigma * sqrt(Q/ADV), in bps. Unknown
    liquidity → conservative (assume thin book)."""
    if order_notional <= 0:
        return 0.0
    participation = (order_notional / adv_usd) if adv_usd and adv_usd > 0 else 0.05
    return 10_000 * impact_coeff * volatility_daily * math.sqrt(min(participation, 1.0))


# ── Borrow constraints for equity shorts (spec §52) ───────────────────────────


@dataclass
class BorrowInfo:
    available: bool = True
    borrow_rate_annual: float = 0.003
    hard_to_borrow: bool = False
    short_sale_restricted: bool = False
    source: str = "unknown"


class BorrowChecker:
    """Feasibility of equity shorts. Without a borrow-data provider we are
    conservative for stocks and permissive for crypto perps."""

    def __init__(self, borrow_provider=None) -> None:
        self._provider = borrow_provider

    def check(self, symbol: str, asset_class: str,
              direction: str) -> Dict[str, Any]:
        if direction != "short":
            return {"executable": True, "reason": None}
        if asset_class == "crypto":
            return {"executable": True, "reason": None,
                    "note": "perp short — funding applies, no borrow needed"}
        info = self._provider.borrow_info(symbol) if self._provider else None
        if info is None:
            return {"executable": False, "reason": "BORROW_UNAVAILABLE",
                    "note": "no borrow data provider — equity shorts fail closed"}
        if not info.available or info.short_sale_restricted:
            return {"executable": False, "reason": "BORROW_UNAVAILABLE"}
        return {"executable": True, "reason": None,
                "borrow_rate_annual": info.borrow_rate_annual,
                "hard_to_borrow": info.hard_to_borrow}


# ── Venue selection (spec §30) ────────────────────────────────────────────────


@dataclass
class VenueQuote:
    venue: str
    fee_bps: float
    spread_pct: float
    depth_usd: Optional[float] = None
    funding_rate_8h: float = 0.0
    healthy: bool = True


def select_venue(quotes: Sequence[VenueQuote], *, order_notional: float,
                 volatility_daily: float = 0.02,
                 holding_hours: float = 24.0,
                 direction: str = "long") -> Optional[Dict[str, Any]]:
    """Pick the venue with the best NET expected cost. Only venues actually
    provided are compared — never fabricate venues."""
    best = None
    slip = SlippageModel()
    for q in quotes:
        if not q.healthy:
            continue
        s = slip.estimate(spread_pct=q.spread_pct, order_notional=order_notional,
                          adv_usd=q.depth_usd * 20 if q.depth_usd else None,
                          volatility_daily=volatility_daily)
        funding_bps = q.funding_rate_8h * 10_000 * (holding_hours / 8.0)
        if direction == "short":
            funding_bps = -funding_bps       # shorts receive positive funding
        total = q.fee_bps + s.expected_bps + max(funding_bps, -50.0)
        cand = {"venue": q.venue, "total_cost_bps": total, "fee_bps": q.fee_bps,
                "slippage_bps": s.expected_bps, "funding_bps": funding_bps}
        if best is None or total < best["total_cost_bps"]:
            best = cand
    return best


# ── Execution method optimization + net EV (spec §26-27, §31) ─────────────────

ENTRY_METHODS = ("market", "limit_passive", "limit_mid", "delayed_1m", "delayed_5m")
EXIT_METHODS = ("market", "limit_passive", "time_sliced", "partial")


@dataclass
class ExecutionPlan:
    method: str
    expected_cost_bps: float
    fill_probability: float
    detail: Dict[str, Any] = field(default_factory=dict)


class ExecutionOptimizer:
    """Chooses execution method by expected NET cost including the cost of
    non-fills (missed edge). Uses learned method-cost table when available."""

    # priors: (cost multiplier vs market order, fill probability)
    _METHOD_PRIORS = {
        "market": (1.0, 1.0),
        "limit_mid": (0.55, 0.85),
        "limit_passive": (0.30, 0.65),
        "delayed_1m": (0.90, 1.0),
        "delayed_5m": (0.85, 1.0),
        "time_sliced": (0.60, 0.97),
        "partial": (0.70, 1.0),
    }

    def __init__(self, slippage_model: Optional[SlippageModel] = None,
                 method_costs: Optional[Dict[str, Dict[str, float]]] = None,
                 fee_bps: float = 2.0) -> None:
        self.slippage = slippage_model or SlippageModel()
        self.method_costs = method_costs or {}   # learned: method -> {cost_bps, fill_prob}
        self.fee_bps = fee_bps

    def cost_of_method(self, method: str, *, spread_pct: float,
                       order_notional: float, adv_usd: Optional[float],
                       volatility_daily: float = 0.02) -> ExecutionPlan:
        learned = self.method_costs.get(method)
        base = self.slippage.estimate(
            spread_pct=spread_pct, order_notional=order_notional,
            adv_usd=adv_usd, volatility_daily=volatility_daily)
        mult, fill_p = self._METHOD_PRIORS.get(method, (1.0, 1.0))
        cost = learned["cost_bps"] if learned else base.expected_bps * mult
        fill = learned.get("fill_prob", fill_p) if learned else fill_p
        impact = market_impact_bps(order_notional=order_notional, adv_usd=adv_usd,
                                   volatility_daily=volatility_daily)
        plan = ExecutionPlan(method=method,
                             expected_cost_bps=cost + impact + self.fee_bps,
                             fill_probability=fill)
        # separated components — total execution cost is NOT "slippage"
        plan.detail["components"] = {
            "spread_bps": base.spread_bps * mult,
            "slippage_bps": base.size_bps * mult if not learned else
            max(cost - base.spread_bps * mult, 0.0),
            "impact_bps": impact,
            "fee_bps": self.fee_bps,
        }
        return plan

    def best_entry(self, *, gross_alpha_bps: float, spread_pct: float,
                   order_notional: float, adv_usd: Optional[float],
                   volatility_daily: float = 0.02,
                   urgency: str = "normal",
                   methods: Sequence[str] = ENTRY_METHODS) -> ExecutionPlan:
        """Maximize fill_prob*(alpha − cost): a passive order that misses a
        fast signal is worse than paying the spread on a slow one."""
        best: Optional[ExecutionPlan] = None
        best_net = -float("inf")
        for m in methods:
            if urgency == "high" and m.startswith(("delayed", "limit_passive")):
                continue
            plan = self.cost_of_method(m, spread_pct=spread_pct,
                                       order_notional=order_notional,
                                       adv_usd=adv_usd,
                                       volatility_daily=volatility_daily)
            net = plan.fill_probability * (gross_alpha_bps - plan.expected_cost_bps)
            if net > best_net:
                best_net, best = net, plan
        assert best is not None
        best.detail["expected_net_bps"] = best_net
        return best

    def net_execution_ev(self, *, gross_alpha_ev: float, spread_pct: float,
                         order_notional: float, adv_usd: Optional[float],
                         volatility_daily: float = 0.02,
                         funding_borrow_bps: float = 0.0) -> Dict[str, Any]:
        """gross − commission − spread/slippage − impact − funding/borrow,
        as FRACTIONAL return. Round trip = 2 executions."""
        plan = self.best_entry(gross_alpha_bps=gross_alpha_ev * 10_000,
                               spread_pct=spread_pct,
                               order_notional=order_notional, adv_usd=adv_usd,
                               volatility_daily=volatility_daily)
        round_trip_bps = plan.expected_cost_bps * 2 + funding_borrow_bps
        net = gross_alpha_ev - round_trip_bps / 10_000
        return {
            "gross_alpha_ev": gross_alpha_ev,
            "execution_cost_bps": round_trip_bps,
            "net_execution_ev": net,
            "entry_method": plan.method,
            "fill_probability": plan.fill_probability,
            "viable": net > 0,
        }


# ── Execution learning (spec §32) ─────────────────────────────────────────────


class ExecutionLearner:
    """Compares predicted vs realized execution and updates the calibration
    multiplier with heavy damping — never overreacts to single fills."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite",
                 min_samples: int = 20, learning_rate: float = 0.1) -> None:
        self.db_path = db_path
        self.min_samples = min_samples
        self.learning_rate = learning_rate
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    symbol TEXT, method TEXT,
                    predicted_slippage_bps REAL, actual_slippage_bps REAL,
                    predicted_spread_pct REAL, actual_spread_pct REAL,
                    predicted_fill_prob REAL, filled INTEGER
                )""")

    def record(self, *, symbol: str, method: str,
               predicted_slippage_bps: float, actual_slippage_bps: float,
               predicted_spread_pct: float = 0.0, actual_spread_pct: float = 0.0,
               predicted_fill_prob: float = 1.0, filled: bool = True) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO execution_calibration (recorded_at, symbol, method, "
                "predicted_slippage_bps, actual_slippage_bps, predicted_spread_pct, "
                "actual_spread_pct, predicted_fill_prob, filled) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (_utcnow(), symbol, method, predicted_slippage_bps,
                 actual_slippage_bps, predicted_spread_pct, actual_spread_pct,
                 predicted_fill_prob, int(filled)))

    def calibration_multiplier(self) -> float:
        """Damped ratio of realized to predicted slippage over the sample.
        Returns 1.0 until enough evidence accumulates."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT predicted_slippage_bps, actual_slippage_bps FROM "
                "execution_calibration ORDER BY id DESC LIMIT 500").fetchall()
        usable = [(p, a) for p, a in rows if p and p > 0]
        if len(usable) < self.min_samples:
            return 1.0
        ratio = sum(a / p for p, a in usable) / len(usable)
        # damped toward 1.0 — periodic recalibration, not per-fill twitching
        return 1.0 + self.learning_rate * (min(max(ratio, 0.25), 4.0) - 1.0)

    def method_cost_table(self) -> Dict[str, Dict[str, float]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT method, AVG(actual_slippage_bps), AVG(filled), COUNT(*) "
                "FROM execution_calibration GROUP BY method").fetchall()
        return {m: {"cost_bps": c or 0.0, "fill_prob": f or 1.0}
                for m, c, f, n in rows if n >= self.min_samples}
