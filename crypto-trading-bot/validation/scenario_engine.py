"""
ScenarioEngine — stress-test strategies against historical and synthetic scenarios.

PHASE 7

Historical scenarios:
    COVID_CRASH_2020      Feb–Mar 2020, -50% equity in 30d
    CRYPTO_MANIA_2021     Nov 2020 – Nov 2021, +500% BTC
    CRYPTO_COLLAPSE_2022  Nov 2021 – Nov 2022, -75% BTC
    EQUITY_BEAR_2022      Jan–Oct 2022, -25% SPX
    FTX_COLLAPSE_2022     Nov 2022, -30% crypto in 1 week
    RATE_SHOCK_2022       Fed hikes 75bps four times
    BANKING_CRISIS_2023   SVB collapse, March 2023

Synthetic scenarios:
    VOL_SPIKE_2X          Double volatility
    VOL_SPIKE_5X          5x volatility  
    LIQUIDITY_COLLAPSE    Spread 10x, fill rate 50%
    TRENDING_MARKET       All trends — mean reversion fails
    RANGING_MARKET        All ranging — trend following fails
    BROKER_OUTAGE         50% missed fills
    DELAYED_EXECUTION     5x latency
    EXTREME_SPREAD        Spread 20x
    ZERO_EDGE             Market is random walk — no signal

Each scenario modifies trade-level PnL via a scaling model.
The ValidationEngine evaluates performance within each scenario.
"""
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from validation.engine import Trade, ValidationEngine, ValidationResult

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    # Historical
    COVID_CRASH_2020     = "COVID_CRASH_2020"
    CRYPTO_MANIA_2021    = "CRYPTO_MANIA_2021"
    CRYPTO_COLLAPSE_2022 = "CRYPTO_COLLAPSE_2022"
    EQUITY_BEAR_2022     = "EQUITY_BEAR_2022"
    FTX_COLLAPSE_2022    = "FTX_COLLAPSE_2022"
    RATE_SHOCK_2022      = "RATE_SHOCK_2022"
    BANKING_CRISIS_2023  = "BANKING_CRISIS_2023"
    # Synthetic
    VOL_SPIKE_2X         = "VOL_SPIKE_2X"
    VOL_SPIKE_5X         = "VOL_SPIKE_5X"
    LIQUIDITY_COLLAPSE   = "LIQUIDITY_COLLAPSE"
    TRENDING_MARKET      = "TRENDING_MARKET"
    RANGING_MARKET       = "RANGING_MARKET"
    BROKER_OUTAGE_50PCT  = "BROKER_OUTAGE_50PCT"
    DELAYED_EXECUTION    = "DELAYED_EXECUTION"
    EXTREME_SPREAD       = "EXTREME_SPREAD"
    ZERO_EDGE            = "ZERO_EDGE"


@dataclass
class ScenarioSpec:
    """Specification for a scenario — how it modifies trades."""
    name:              str
    scenario_type:     ScenarioType
    description:       str
    pnl_scale:         float = 1.0    # multiply gross PnL by this factor
    vol_mult:          float = 1.0    # volatility multiplier (affects PnL spread)
    fill_rate:         float = 1.0    # fraction of trades that fill
    slippage_mult:     float = 1.0    # multiply slippage by this factor
    spread_mult:       float = 1.0    # multiply spread cost by this factor
    mean_revert_bias:  float = 0.0    # extra loss for mean-reversion in trending mkt
    trend_bias:        float = 0.0    # extra loss for trend-following in ranging mkt
    seed:              int = 42


@dataclass
class ScenarioResult:
    """Result of running a scenario against a set of trades."""
    scenario_name:        str
    scenario_type:        ScenarioType
    n_original_trades:    int
    n_scenario_trades:    int
    original_result:      ValidationResult
    scenario_result:      ValidationResult
    pnl_change_pct:       float     # (scenario_pnl - original_pnl) / |original_pnl|
    sharpe_change:        float
    survived:             bool      # True if scenario doesn't trigger ruin
    notes:                str = ''

    def summary(self) -> str:
        status = "✓ SURVIVED" if self.survived else "✗ RUIN/EXTREME LOSS"
        return "\n".join([
            f"Scenario: {self.scenario_name} [{status}]",
            f"  Trades:  {self.n_original_trades} → {self.n_scenario_trades}",
            f"  PnL:     ${self.original_result.total_pnl_net:,.2f} → "
            f"${self.scenario_result.total_pnl_net:,.2f} "
            f"({self.pnl_change_pct:+.1f}%)",
            f"  Sharpe:  {self.original_result.sharpe:.3f} → "
            f"{self.scenario_result.sharpe:.3f}",
        ])


@dataclass
class ScenarioReport:
    label:       str
    results:     List[ScenarioResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            f"  SCENARIO STRESS TEST: {self.label}",
            "=" * 70,
        ]
        survivors = sum(1 for r in self.results if r.survived)
        lines.append(f"  Scenarios: {len(self.results)}   Survived: {survivors}/{len(self.results)}")
        lines.append("─" * 70)
        for r in self.results:
            s = "✓" if r.survived else "✗"
            lines.append(
                f"  {s} {r.scenario_name:<30} "
                f"PnL: {r.pnl_change_pct:>+7.1f}%  "
                f"Sharpe: {r.scenario_result.sharpe:>+7.3f}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)


# ── Built-in scenario specs ───────────────────────────────────────────────────

BUILTIN_SCENARIOS = [
    ScenarioSpec(
        name="COVID Crash 2020",
        scenario_type=ScenarioType.COVID_CRASH_2020,
        description="Rapid -50% equity drawdown in 4 weeks",
        pnl_scale=0.50, vol_mult=3.5, fill_rate=0.90, slippage_mult=4.0, spread_mult=5.0,
    ),
    ScenarioSpec(
        name="Crypto Mania 2021",
        scenario_type=ScenarioType.CRYPTO_MANIA_2021,
        description="Extreme uptrend — mean reversion fails, trend wins",
        pnl_scale=1.40, vol_mult=2.0, mean_revert_bias=-0.30,
    ),
    ScenarioSpec(
        name="Crypto Collapse 2022",
        scenario_type=ScenarioType.CRYPTO_COLLAPSE_2022,
        description="-75% BTC over 12 months — long strategies destroyed",
        pnl_scale=0.30, vol_mult=2.5, slippage_mult=2.0, fill_rate=0.85,
    ),
    ScenarioSpec(
        name="Equity Bear 2022",
        scenario_type=ScenarioType.EQUITY_BEAR_2022,
        description="-25% SPX, rising rates, correlation spikes",
        pnl_scale=0.60, vol_mult=1.8, spread_mult=1.5,
    ),
    ScenarioSpec(
        name="FTX Collapse Nov 2022",
        scenario_type=ScenarioType.FTX_COLLAPSE_2022,
        description="-30% crypto in 1 week, exchange halt risk",
        pnl_scale=0.40, vol_mult=5.0, fill_rate=0.70, slippage_mult=8.0, spread_mult=10.0,
    ),
    ScenarioSpec(
        name="Rate Shock 2022",
        scenario_type=ScenarioType.RATE_SHOCK_2022,
        description="75bps Fed hikes: risk-off, crypto -50%",
        pnl_scale=0.45, vol_mult=2.0, spread_mult=2.0,
    ),
    ScenarioSpec(
        name="Banking Crisis 2023",
        scenario_type=ScenarioType.BANKING_CRISIS_2023,
        description="SVB collapse: flight to safety, crypto volatile",
        pnl_scale=0.70, vol_mult=2.5, fill_rate=0.92,
    ),
    ScenarioSpec(
        name="2x Volatility",
        scenario_type=ScenarioType.VOL_SPIKE_2X,
        description="Synthetic: market volatility doubles",
        vol_mult=2.0,
    ),
    ScenarioSpec(
        name="5x Volatility",
        scenario_type=ScenarioType.VOL_SPIKE_5X,
        description="Synthetic: extreme 5x volatility spike",
        vol_mult=5.0,
    ),
    ScenarioSpec(
        name="Liquidity Collapse",
        scenario_type=ScenarioType.LIQUIDITY_COLLAPSE,
        description="Spread 10x, fill rate 50%",
        fill_rate=0.50, spread_mult=10.0, slippage_mult=5.0,
    ),
    ScenarioSpec(
        name="Strong Trend Market",
        scenario_type=ScenarioType.TRENDING_MARKET,
        description="All assets in strong trend — mean reversion loses",
        mean_revert_bias=-0.40,
    ),
    ScenarioSpec(
        name="Ranging Market",
        scenario_type=ScenarioType.RANGING_MARKET,
        description="All assets in tight ranges — trend following loses",
        trend_bias=-0.30,
    ),
    ScenarioSpec(
        name="50% Broker Outage",
        scenario_type=ScenarioType.BROKER_OUTAGE_50PCT,
        description="Half of all orders missed due to broker outage",
        fill_rate=0.50,
    ),
    ScenarioSpec(
        name="Extreme Spread",
        scenario_type=ScenarioType.EXTREME_SPREAD,
        description="20x spread — nearly all edge is consumed by costs",
        spread_mult=20.0, slippage_mult=3.0,
    ),
    ScenarioSpec(
        name="Zero Edge (Random Walk)",
        scenario_type=ScenarioType.ZERO_EDGE,
        description="Market is a random walk — no signal exists",
        pnl_scale=0.0,   # all signal removed, only costs remain
    ),
]


class ScenarioEngine:
    """
    Stress-test a strategy against historical and synthetic scenarios.

    Usage:
        engine = ScenarioEngine(capital=10_000)
        report = engine.run(trades, label="MyStrategy")
        print(report.summary())
    """

    # Ruin threshold: scenario PnL < -50% of capital
    RUIN_THRESHOLD_PCT = -0.50

    def __init__(
        self,
        capital: float = 10_000.0,
        scenarios: List[ScenarioSpec] = None,
    ):
        self.capital  = capital
        self.scenarios = scenarios or BUILTIN_SCENARIOS
        self._ve      = ValidationEngine(capital=capital)

    def run(
        self,
        trades: List[Trade],
        label:  str = "Strategy",
        scenarios: List[ScenarioSpec] = None,
    ) -> ScenarioReport:
        """Run all scenarios against the given trade list."""
        specs   = scenarios or self.scenarios
        baseline = self._ve.evaluate(trades, label=f"{label} Baseline")
        results  = []

        for spec in specs:
            try:
                scenario_trades = self._apply_scenario(trades, spec)
                scenario_result = self._ve.evaluate(scenario_trades, label=f"{label} {spec.name}")
                survived = self._check_survival(scenario_result)
                pnl_change = 0.0
                if baseline.total_pnl_net != 0:
                    pnl_change = (
                        (scenario_result.total_pnl_net - baseline.total_pnl_net)
                        / abs(baseline.total_pnl_net) * 100
                    )
                results.append(ScenarioResult(
                    scenario_name=spec.name,
                    scenario_type=spec.scenario_type,
                    n_original_trades=len(trades),
                    n_scenario_trades=len(scenario_trades),
                    original_result=baseline,
                    scenario_result=scenario_result,
                    pnl_change_pct=pnl_change,
                    sharpe_change=scenario_result.sharpe - baseline.sharpe,
                    survived=survived,
                ))
            except Exception as e:
                logger.warning(f"ScenarioEngine: error in '{spec.name}': {e}")

        return ScenarioReport(label=label, results=results)

    def run_single(
        self,
        trades: List[Trade],
        scenario_type: ScenarioType,
        label: str = "Strategy",
    ) -> Optional[ScenarioResult]:
        """Run a single named scenario."""
        spec = next((s for s in BUILTIN_SCENARIOS if s.scenario_type == scenario_type), None)
        if not spec:
            raise ValueError(f"Unknown scenario: {scenario_type}")
        report = self.run(trades, label=label, scenarios=[spec])
        return report.results[0] if report.results else None

    # ── Private ───────────────────────────────────────────────────────────────

    def _apply_scenario(self, trades: List[Trade], spec: ScenarioSpec) -> List[Trade]:
        """Transform trades according to scenario specification."""
        rng     = random.Random(spec.seed)
        result  = []

        for t in trades:
            # Missed fill
            if rng.random() > spec.fill_rate:
                continue

            # Copy trade
            t2 = copy.copy(t)

            # Scale gross PnL
            pnl_gross = t2.pnl_gross * spec.pnl_scale

            # Apply volatility multiplier (increases wins AND losses)
            if spec.vol_mult != 1.0:
                # Amplify deviations from zero
                pnl_gross = pnl_gross * spec.vol_mult * rng.uniform(0.5, 1.5)

            # Mean-reversion bias (in trending markets)
            if spec.mean_revert_bias != 0 and t2.strategy and 'mean' in t2.strategy.lower():
                pnl_gross += abs(pnl_gross) * spec.mean_revert_bias

            # Trend bias (in ranging markets)
            if spec.trend_bias != 0 and t2.strategy and 'trend' in t2.strategy.lower():
                pnl_gross += abs(pnl_gross) * spec.trend_bias

            # Adjust costs
            slippage = t2.slippage * spec.slippage_mult
            spread_cost = t2.size * 0.0001 * (spec.spread_mult - 1)  # extra spread
            fees = t2.fees
            total_costs = fees + slippage + spread_cost

            t2 = Trade(
                entry_time=t2.entry_time,
                exit_time=t2.exit_time,
                pnl_gross=pnl_gross,
                pnl_net=pnl_gross - total_costs,
                size=t2.size,
                fees=fees,
                slippage=slippage,
                symbol=t2.symbol,
                strategy=t2.strategy,
                direction=t2.direction,
                regime=t2.regime,
            )
            result.append(t2)

        return result

    def _check_survival(self, result: ValidationResult) -> bool:
        """Returns False if the scenario causes ruin (>50% drawdown of capital)."""
        max_dd_pct = result.max_drawdown / max(self.capital, 1)
        return max_dd_pct < abs(self.RUIN_THRESHOLD_PCT)
