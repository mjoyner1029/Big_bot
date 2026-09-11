"""
PerformanceAttributionEngine — decompose PnL by every meaningful dimension.

PHASE 15

Separate SIGNAL PnL from EXECUTION PnL:
    Strategy theoretical PnL: +$1,000
    Execution costs:           -$240
    Actual PnL:                 +$760

Attribution dimensions:
    asset            symbol (BTC/USD, ETH/USD, ...)
    strategy         strategy name
    direction        LONG / SHORT
    regime           bull / bear / ranging / volatile
    time_of_day      0-4h / 4-8h / 8-12h / 12-16h / 16-20h / 20-24h
    day_of_week      Mon / Tue / Wed / Thu / Fri / Sat / Sun
    holding_period   <1h / 1-4h / 4-24h / 1-3d / 3d+
    confidence       50-60% / 60-70% / 70-80% / 80%+

Each dimension produces a breakdown showing:
    count, total_pnl_net, avg_pnl, win_rate, expectancy, sharpe
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timezone
from typing import Dict, List, Optional

from validation.engine import Trade, ValidationEngine, ValidationResult


@dataclass
class AttributionBucket:
    """Performance metrics for a single attribution bucket."""
    name:       str
    count:      int
    total_pnl:  float
    avg_pnl:    float
    win_rate:   float
    expectancy: float
    sharpe:     float
    fees:       float
    slippage:   float

    def summary_row(self) -> str:
        return (f"  {self.name:<25} n={self.count:>5}  "
                f"PnL=${self.total_pnl:>10,.2f}  "
                f"WR={self.win_rate:.1%}  "
                f"E=${self.expectancy:>7.3f}  "
                f"Sharpe={self.sharpe:>6.3f}")


@dataclass
class AttributionReport:
    """Full attribution report across all dimensions."""
    label:           str
    trade_count:     int
    total_pnl_net:   float
    total_pnl_gross: float
    total_fees:      float
    total_slippage:  float
    cost_drag_pct:   float    # (total_pnl_gross - total_pnl_net) / |total_pnl_gross|

    by_asset:         Dict[str, AttributionBucket] = field(default_factory=dict)
    by_strategy:      Dict[str, AttributionBucket] = field(default_factory=dict)
    by_direction:     Dict[str, AttributionBucket] = field(default_factory=dict)
    by_regime:        Dict[str, AttributionBucket] = field(default_factory=dict)
    by_time_of_day:   Dict[str, AttributionBucket] = field(default_factory=dict)
    by_day_of_week:   Dict[str, AttributionBucket] = field(default_factory=dict)
    by_holding:       Dict[str, AttributionBucket] = field(default_factory=dict)
    by_confidence:    Dict[str, AttributionBucket] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            f"  PERFORMANCE ATTRIBUTION: {self.label}",
            "=" * 70,
            f"  Trades:    {self.trade_count}",
            f"  Net PnL:   ${self.total_pnl_net:,.2f}",
            f"  Gross PnL: ${self.total_pnl_gross:,.2f}",
            f"  Fees:      ${self.total_fees:,.2f}",
            f"  Slippage:  ${self.total_slippage:,.2f}",
            f"  Cost drag: {self.cost_drag_pct:.1%}",
        ]

        for dim_name, dim_dict in [
            ("BY ASSET",       self.by_asset),
            ("BY STRATEGY",    self.by_strategy),
            ("BY DIRECTION",   self.by_direction),
            ("BY REGIME",      self.by_regime),
            ("BY TIME OF DAY", self.by_time_of_day),
            ("BY HOLDING",     self.by_holding),
        ]:
            if dim_dict:
                lines.append(f"\n  {dim_name}:")
                for bucket in sorted(dim_dict.values(), key=lambda b: -b.total_pnl):
                    lines.append(bucket.summary_row())

        lines.append("=" * 70)
        return "\n".join(lines)


class PerformanceAttributionEngine:
    """
    Decomposes PnL across all meaningful dimensions.

    Usage:
        eng    = PerformanceAttributionEngine()
        report = eng.attribute(trades, label="MyStrategy")
        print(report.summary())
    """

    def attribute(
        self,
        trades: List[Trade],
        label:  str = "Strategy",
    ) -> AttributionReport:
        """Run full attribution on a trade list."""
        if not trades:
            return AttributionReport(
                label=label, trade_count=0, total_pnl_net=0, total_pnl_gross=0,
                total_fees=0, total_slippage=0, cost_drag_pct=0,
            )

        total_pnl_net   = sum(t.pnl_net for t in trades)
        total_pnl_gross = sum(t.pnl_gross for t in trades)
        total_fees      = sum(t.fees for t in trades)
        total_slippage  = sum(t.slippage for t in trades)

        if total_pnl_gross != 0:
            cost_drag = (total_pnl_gross - total_pnl_net) / abs(total_pnl_gross)
        else:
            cost_drag = 0.0

        report = AttributionReport(
            label=label,
            trade_count=len(trades),
            total_pnl_net=total_pnl_net,
            total_pnl_gross=total_pnl_gross,
            total_fees=total_fees,
            total_slippage=total_slippage,
            cost_drag_pct=cost_drag,
        )

        report.by_asset       = self._bucket_by(trades, self._key_asset)
        report.by_strategy    = self._bucket_by(trades, self._key_strategy)
        report.by_direction   = self._bucket_by(trades, self._key_direction)
        report.by_regime      = self._bucket_by(trades, self._key_regime)
        report.by_time_of_day = self._bucket_by(trades, self._key_time_of_day)
        report.by_day_of_week = self._bucket_by(trades, self._key_day_of_week)
        report.by_holding     = self._bucket_by(trades, self._key_holding)

        return report

    # ── Key functions ──────────────────────────────────────────────────────

    @staticmethod
    def _key_asset(t: Trade) -> str:
        return t.symbol or "UNKNOWN"

    @staticmethod
    def _key_strategy(t: Trade) -> str:
        return t.strategy or "UNKNOWN"

    @staticmethod
    def _key_direction(t: Trade) -> str:
        return t.direction or "UNKNOWN"

    @staticmethod
    def _key_regime(t: Trade) -> str:
        return t.regime or "UNKNOWN"

    @staticmethod
    def _key_time_of_day(t: Trade) -> str:
        if t.entry_time is None:
            return "UNKNOWN"
        dt = t.entry_time
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        hour = dt.hour
        if hour < 4:
            return "00-04h"
        elif hour < 8:
            return "04-08h"
        elif hour < 12:
            return "08-12h"
        elif hour < 16:
            return "12-16h"
        elif hour < 20:
            return "16-20h"
        else:
            return "20-24h"

    @staticmethod
    def _key_day_of_week(t: Trade) -> str:
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        if t.entry_time is None:
            return "UNKNOWN"
        return days[t.entry_time.weekday()]

    @staticmethod
    def _key_holding(t: Trade) -> str:
        hours = t.holding_hours
        if hours < 1:
            return "<1h"
        elif hours < 4:
            return "1-4h"
        elif hours < 24:
            return "4-24h"
        elif hours < 72:
            return "1-3d"
        else:
            return "3d+"

    # ── Bucketing ──────────────────────────────────────────────────────────

    @staticmethod
    def _bucket_by(trades: List[Trade], key_fn) -> Dict[str, AttributionBucket]:
        groups: Dict[str, List[Trade]] = {}
        for t in trades:
            k = key_fn(t)
            groups.setdefault(k, []).append(t)

        result = {}
        for name, group in groups.items():
            pnls     = [t.pnl_net for t in group]
            wins     = [p for p in pnls if p > 0]
            total    = sum(pnls)
            avg      = total / len(pnls) if pnls else 0.0
            wr       = len(wins) / len(pnls) if pnls else 0.0
            exp      = avg

            # Sharpe for the bucket
            if len(pnls) > 1:
                mean = avg
                std  = math.sqrt(sum((p - mean) ** 2 for p in pnls) / len(pnls))
                sharpe = mean / std * math.sqrt(252) if std > 0 else 0.0
            else:
                sharpe = 0.0

            result[name] = AttributionBucket(
                name=name,
                count=len(group),
                total_pnl=total,
                avg_pnl=avg,
                win_rate=wr,
                expectancy=exp,
                sharpe=sharpe,
                fees=sum(t.fees for t in group),
                slippage=sum(t.slippage for t in group),
            )

        return result
