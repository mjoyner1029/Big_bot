"""
RealisticExecutionSimulator — simulate realistic order execution.

PHASE 8

Simulates all execution imperfections:
    - Bid/ask spread
    - Slippage (market impact)
    - Exchange fees
    - Latency (entry at slightly different price)
    - Partial fills
    - Missed fills (order expires, rate limited, broker outage)
    - Order rejection
    - Worse entry/exit prices

Do NOT assume fills at candle close. Use realistic mid-price offsets.

All parameters are configurable and tracked per fill for attribution.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


@dataclass
class ExecutionParams:
    """Parameters for a single execution simulation."""
    # Spread (each side) in basis points
    spread_bps:           float = 3.0
    # Market slippage (impact) in basis points
    slippage_bps:         float = 5.0
    # Exchange fee as fraction of notional
    fee_pct:              float = 0.001   # 0.1%
    # Probability of a missed fill (order expires / not matched)
    missed_fill_prob:     float = 0.01
    # Probability of a partial fill
    partial_fill_prob:    float = 0.03
    # Fraction filled on partial fill
    partial_fill_frac:    float = 0.70
    # Probability of a worse entry price (latency / queue position)
    worse_entry_prob:     float = 0.04
    # Scale for worse entry (fraction of price)
    worse_entry_scale:    float = 0.0015
    # Probability of rejection
    rejection_prob:       float = 0.002
    # RNG seed
    seed:                 int = 42


@dataclass
class SimulatedFill:
    """Result of simulating an order execution."""
    filled:           bool
    fill_price:       float       # actual fill price (after spread/slippage/latency)
    fill_qty:         float       # quantity filled
    requested_qty:    float
    fee:              float
    slippage_cost:    float
    spread_cost:      float
    total_cost:       float
    latency_ms:       float
    rejection_reason: Optional[str]
    partial:          bool
    theoretical_price: float      # mid-price at order submission

    @property
    def effective_price(self) -> float:
        return self.fill_price

    @property
    def cost_bps(self) -> float:
        if self.theoretical_price <= 0:
            return 0.0
        return abs(self.fill_price - self.theoretical_price) / self.theoretical_price * 10_000

    def summary(self) -> str:
        if not self.filled:
            return f"MISSED FILL — {self.rejection_reason}"
        s = f"Fill: {self.fill_qty:.4f} @ {self.fill_price:.4f}"
        if self.partial:
            s += f" [PARTIAL {self.fill_qty/self.requested_qty:.0%}]"
        s += f" | Cost: {self.cost_bps:.1f}bps ({self.total_cost:.4f})"
        return s


@dataclass
class ExecutionReport:
    """Summary of a batch of simulated fills."""
    n_orders:          int
    n_filled:          int
    n_partial:         int
    n_missed:          int
    n_rejected:        int
    total_fee:         float
    total_slippage:    float
    total_spread_cost: float
    total_cost:        float
    avg_cost_bps:      float
    fill_rate:         float
    fills:             List[SimulatedFill] = field(default_factory=list)

    def summary(self) -> str:
        return "\n".join([
            "EXECUTION QUALITY REPORT",
            "─" * 40,
            f"  Orders:          {self.n_orders:>8}",
            f"  Filled:          {self.n_filled:>8}",
            f"  Partial fills:   {self.n_partial:>8}",
            f"  Missed fills:    {self.n_missed:>8}",
            f"  Rejected:        {self.n_rejected:>8}",
            f"  Fill rate:       {self.fill_rate:>8.1%}",
            f"  Avg cost (bps):  {self.avg_cost_bps:>8.2f}",
            f"  Total fee:       ${self.total_fee:>10,.4f}",
            f"  Total slippage:  ${self.total_slippage:>10,.4f}",
            f"  Total spread:    ${self.total_spread_cost:>10,.4f}",
            f"  Total cost:      ${self.total_cost:>10,.4f}",
        ])


class RealisticExecutionSimulator:
    """
    Simulates realistic order execution for backtesting and paper trading.

    Usage:
        sim    = RealisticExecutionSimulator(params=ExecutionParams(spread_bps=5))
        fill   = sim.simulate_entry(mid_price=50_000.0, qty=0.1, direction='LONG')
        report = sim.batch_simulate(prices, qtys, directions)
    """

    def __init__(self, params: ExecutionParams = None, seed: int = None):
        self.params = params or ExecutionParams()
        self._rng   = random.Random(seed or self.params.seed)

    # ── Public API ─────────────────────────────────────────────────────────

    def simulate_entry(
        self,
        mid_price:  float,
        qty:        float,
        direction:  str = 'LONG',
    ) -> SimulatedFill:
        """Simulate entering a position."""
        return self._simulate_order(mid_price, qty, direction, is_entry=True)

    def simulate_exit(
        self,
        mid_price:  float,
        qty:        float,
        direction:  str = 'LONG',
    ) -> SimulatedFill:
        """Simulate exiting a position."""
        return self._simulate_order(mid_price, qty, direction, is_entry=False)

    def simulate_round_trip(
        self,
        entry_mid:   float,
        exit_mid:    float,
        qty:         float,
        direction:   str = 'LONG',
    ) -> Tuple[SimulatedFill, SimulatedFill, float]:
        """
        Simulate a complete round-trip trade.
        Returns (entry_fill, exit_fill, net_pnl_after_costs).
        """
        entry_fill = self.simulate_entry(entry_mid, qty, direction)
        exit_fill  = self.simulate_exit(exit_mid, qty, direction)

        if not entry_fill.filled or not exit_fill.filled:
            return entry_fill, exit_fill, 0.0

        if direction == 'LONG':
            gross = (exit_fill.fill_price - entry_fill.fill_price) * min(entry_fill.fill_qty, exit_fill.fill_qty)
        else:
            gross = (entry_fill.fill_price - exit_fill.fill_price) * min(entry_fill.fill_qty, exit_fill.fill_qty)

        total_costs = entry_fill.total_cost + exit_fill.total_cost
        net_pnl     = gross - total_costs
        return entry_fill, exit_fill, net_pnl

    def batch_simulate(
        self,
        orders: List[Dict],   # each: {'mid_price', 'qty', 'direction', 'side'}
    ) -> ExecutionReport:
        """Simulate a batch of orders and produce an aggregate report."""
        fills = []
        for order in orders:
            side      = order.get('side', 'entry')
            mid_price = float(order.get('mid_price', 0))
            qty       = float(order.get('qty', 1.0))
            direction = order.get('direction', 'LONG')
            if side == 'entry':
                fill = self.simulate_entry(mid_price, qty, direction)
            else:
                fill = self.simulate_exit(mid_price, qty, direction)
            fills.append(fill)

        n_filled   = sum(1 for f in fills if f.filled)
        n_partial  = sum(1 for f in fills if f.filled and f.partial)
        n_missed   = sum(1 for f in fills if not f.filled and f.rejection_reason == 'missed_fill')
        n_rejected = sum(1 for f in fills if not f.filled and f.rejection_reason != 'missed_fill')

        filled_fills = [f for f in fills if f.filled]
        total_fee      = sum(f.fee for f in filled_fills)
        total_slip     = sum(f.slippage_cost for f in filled_fills)
        total_spread   = sum(f.spread_cost for f in filled_fills)
        total_cost     = sum(f.total_cost for f in filled_fills)
        avg_cost_bps   = sum(f.cost_bps for f in filled_fills) / max(len(filled_fills), 1)
        fill_rate      = n_filled / max(len(fills), 1)

        return ExecutionReport(
            n_orders=len(fills),
            n_filled=n_filled,
            n_partial=n_partial,
            n_missed=n_missed,
            n_rejected=n_rejected,
            total_fee=total_fee,
            total_slippage=total_slip,
            total_spread_cost=total_spread,
            total_cost=total_cost,
            avg_cost_bps=avg_cost_bps,
            fill_rate=fill_rate,
            fills=fills,
        )

    def theoretical_to_net(
        self,
        theoretical_pnl: float,
        notional:        float,
        n_legs:          int = 2,   # entry + exit
    ) -> float:
        """
        Convert a theoretical (spread/cost-free) PnL to realistic net PnL.

        Used to retrofit backtests that assumed mid-price execution.
        """
        bps_per_leg  = self.params.spread_bps / 2 + self.params.slippage_bps
        total_bps    = bps_per_leg * n_legs
        fee_cost     = notional * self.params.fee_pct * n_legs
        spread_cost  = notional * total_bps / 10_000
        return theoretical_pnl - fee_cost - spread_cost

    # ── Private ────────────────────────────────────────────────────────────

    def _simulate_order(
        self,
        mid_price:  float,
        qty:        float,
        direction:  str,
        is_entry:   bool,
    ) -> SimulatedFill:
        p = self.params
        r = self._rng

        # Rejection
        if r.random() < p.rejection_prob:
            return SimulatedFill(
                filled=False, fill_price=0, fill_qty=0, requested_qty=qty,
                fee=0, slippage_cost=0, spread_cost=0, total_cost=0,
                latency_ms=0, rejection_reason='order_rejected', partial=False,
                theoretical_price=mid_price,
            )

        # Missed fill
        if r.random() < p.missed_fill_prob:
            return SimulatedFill(
                filled=False, fill_price=0, fill_qty=0, requested_qty=qty,
                fee=0, slippage_cost=0, spread_cost=0, total_cost=0,
                latency_ms=0, rejection_reason='missed_fill', partial=False,
                theoretical_price=mid_price,
            )

        # Partial fill
        partial = r.random() < p.partial_fill_prob
        fill_qty = qty * p.partial_fill_frac if partial else qty

        # Spread cost (paying half the spread)
        # LONG entry: buy at ask = mid + spread/2
        # LONG exit:  sell at bid = mid - spread/2
        half_spread = mid_price * p.spread_bps / 2 / 10_000
        if (direction == 'LONG' and is_entry) or (direction == 'SHORT' and not is_entry):
            spread_adjusted = mid_price + half_spread
        else:
            spread_adjusted = mid_price - half_spread

        # Slippage (market impact, always against you)
        slippage_px  = mid_price * p.slippage_bps / 10_000
        if (direction == 'LONG' and is_entry) or (direction == 'SHORT' and not is_entry):
            slippage_adjusted = spread_adjusted + slippage_px
        else:
            slippage_adjusted = spread_adjusted - slippage_px

        # Latency / worse fill price
        worse_px = 0.0
        if r.random() < p.worse_entry_prob:
            worse_px = mid_price * p.worse_entry_scale * r.random()
            if (direction == 'LONG' and is_entry) or (direction == 'SHORT' and not is_entry):
                slippage_adjusted += worse_px
            else:
                slippage_adjusted -= worse_px

        fill_price   = slippage_adjusted
        fee          = abs(fill_qty * fill_price) * p.fee_pct
        slippage_cost = abs(slippage_px + worse_px) * fill_qty
        spread_cost  = abs(half_spread) * fill_qty
        total_cost   = fee + slippage_cost + spread_cost
        latency_ms   = r.uniform(5, 200)

        return SimulatedFill(
            filled=True,
            fill_price=fill_price,
            fill_qty=fill_qty,
            requested_qty=qty,
            fee=fee,
            slippage_cost=slippage_cost,
            spread_cost=spread_cost,
            total_cost=total_cost,
            latency_ms=latency_ms,
            rejection_reason=None,
            partial=partial,
            theoretical_price=mid_price,
        )
