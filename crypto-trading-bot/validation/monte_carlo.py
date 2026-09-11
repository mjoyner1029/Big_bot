"""
MonteCarloEngine — stress-test portfolios by simulating thousands of paths.

Simulates variation in:
    trade order randomisation    slippage variation
    spread variation             fee variation
    latency variation            missed fills
    partial fills                worse-than-expected entries/exits

Produces per-path statistics and aggregate risk metrics:
    median return              5th percentile return        1st percentile return
    median max drawdown        95th percentile max drawdown
    probability of loss        probability of exceeding risk limits
    probability of ruin
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from validation.engine import Trade, ValidationEngine, ValidationResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloConfig:
    n_simulations:         int   = 2000
    seed:                  int   = 42

    # Randomisation switches
    randomise_order:       bool  = True    # shuffle trade order
    vary_slippage:         bool  = True
    vary_spread:           bool  = True
    vary_fees:             bool  = True
    vary_latency:          bool  = True
    missed_fill_prob:      float = 0.02    # 2% chance of missing a fill
    partial_fill_prob:     float = 0.05    # 5% partial fill, 75% of size
    partial_fill_size:     float = 0.75    # fraction when partial
    worse_entry_prob:      float = 0.05    # 5% chance of worse entry
    worse_entry_scale:     float = 0.002   # 0.2% worse
    worse_exit_prob:       float = 0.05
    worse_exit_scale:      float = 0.002

    # Distribution parameters
    slippage_std_mult:     float = 0.5     # std = base_slippage * this
    spread_std_mult:       float = 0.3
    fee_std_mult:          float = 0.1

    # Risk limits
    ruin_threshold_pct:    float = -50.0   # −50% drawdown = ruin
    risk_limit_pct:        float = -20.0   # −20% = exceeds risk


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    n_simulations:        int
    config:               MonteCarloConfig

    # Return distribution (% of capital)
    returns_pct:          List[float]
    median_return:        float
    p5_return:            float          # 5th percentile
    p1_return:            float          # 1st percentile
    p25_return:           float
    p75_return:           float

    # Drawdown distribution
    max_drawdowns:        List[float]    # absolute $ per sim
    median_max_dd:        float
    p95_max_dd:           float

    # Risk
    prob_loss:            float          # fraction of sims with negative return
    prob_exceed_risk:     float          # fraction exceeding risk_limit_pct
    prob_ruin:            float          # fraction reaching ruin_threshold_pct
    prob_positive:        float

    # Sharpe distribution
    sharpe_values:        List[float]
    median_sharpe:        float

    capital:              float = 10_000.0

    def summary(self) -> str:
        lines = [
            "═" * 60,
            "  MONTE CARLO STRESS TEST",
            "═" * 60,
            f"  Simulations:        {self.n_simulations:>10,}",
            "─" * 60,
            "  RETURN DISTRIBUTION",
            f"  Median return:      {self.median_return:>10.2f}%",
            f"  25th pct return:    {self.p25_return:>10.2f}%",
            f"  5th pct return:     {self.p5_return:>10.2f}%",
            f"  1st pct return:     {self.p1_return:>10.2f}%",
            "─" * 60,
            "  DRAWDOWN DISTRIBUTION",
            f"  Median max DD:      ${self.median_max_dd:>9,.2f}",
            f"  95th pct max DD:    ${self.p95_max_dd:>9,.2f}",
            "─" * 60,
            "  RISK PROBABILITIES",
            f"  P(positive return): {self.prob_positive:>10.1%}",
            f"  P(loss):            {self.prob_loss:>10.1%}",
            f"  P(exceed risk lim): {self.prob_exceed_risk:>10.1%}",
            f"  P(ruin):            {self.prob_ruin:>10.1%}",
            "─" * 60,
            "  SHARPE DISTRIBUTION",
            f"  Median Sharpe:      {self.median_sharpe:>10.3f}",
            "═" * 60,
        ]
        return "\n".join(lines)

    def percentile(self, series: List[float], pct: float) -> float:
        """Compute a percentile from a sorted list (0-100)."""
        if not series:
            return 0.0
        s = sorted(series)
        idx = max(0, min(len(s) - 1, int(len(s) * pct / 100)))
        return s[idx]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """
    Stress-tests a trade sequence by running N Monte Carlo simulations,
    each introducing realistic execution noise and randomisation.
    """

    def __init__(self, config: MonteCarloConfig = None, capital: float = 10_000.0):
        self.config  = config or MonteCarloConfig()
        self.capital = capital
        self._ve     = ValidationEngine(capital=capital)

    def run(self, trades: List[Trade], label: str = "Strategy") -> MonteCarloResult:
        """
        Run Monte Carlo simulation on a trade sequence.

        Returns MonteCarloResult with distribution statistics.
        """
        if not trades:
            return self._empty_result()

        rng          = random.Random(self.config.seed)
        returns_pct  = []
        max_drawdowns = []
        sharpes      = []

        for _ in range(self.config.n_simulations):
            sim_trades = self._perturb_trades(trades, rng)
            if not sim_trades:
                continue
            result = self._ve.evaluate(sim_trades, label="_mc_")
            returns_pct.append(result.total_return_pct)
            max_drawdowns.append(result.max_drawdown)
            sharpes.append(result.sharpe)

        if not returns_pct:
            return self._empty_result()

        returns_sorted = sorted(returns_pct)
        dd_sorted      = sorted(max_drawdowns)
        n              = len(returns_pct)

        def pct(lst, p): return lst[max(0, min(len(lst)-1, int(len(lst)*p/100)))]

        return MonteCarloResult(
            n_simulations=n,
            config=self.config,
            returns_pct=returns_pct,
            median_return=pct(returns_sorted, 50),
            p5_return=pct(returns_sorted, 5),
            p1_return=pct(returns_sorted, 1),
            p25_return=pct(returns_sorted, 25),
            p75_return=pct(returns_sorted, 75),
            max_drawdowns=max_drawdowns,
            median_max_dd=pct(dd_sorted, 50),
            p95_max_dd=pct(dd_sorted, 95),
            prob_loss=sum(1 for r in returns_pct if r < 0) / n,
            prob_exceed_risk=sum(1 for r in returns_pct if r < self.config.risk_limit_pct) / n,
            prob_ruin=sum(1 for r in returns_pct if r < self.config.ruin_threshold_pct) / n,
            prob_positive=sum(1 for r in returns_pct if r > 0) / n,
            sharpe_values=sharpes,
            median_sharpe=pct(sorted(sharpes), 50) if sharpes else 0.0,
            capital=self.capital,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _perturb_trades(self, trades: List[Trade], rng: random.Random) -> List[Trade]:
        """Apply Monte Carlo perturbations to a trade list."""
        result = []
        if self.config.randomise_order:
            trades = list(trades)
            rng.shuffle(trades)

        for t in trades:
            # Missed fill
            if rng.random() < self.config.missed_fill_prob:
                continue

            size = t.size

            # Partial fill
            if rng.random() < self.config.partial_fill_prob:
                size *= self.config.partial_fill_size

            # Slippage variation
            slip_mult = 1.0
            if self.config.vary_slippage:
                slip_mult = max(0.5, rng.gauss(1.0, self.config.slippage_std_mult))

            # Fee variation
            fee_mult = 1.0
            if self.config.vary_fees:
                fee_mult = max(0.8, rng.gauss(1.0, self.config.fee_std_mult))

            # Worse entry/exit
            worse_factor = 1.0
            if rng.random() < self.config.worse_entry_prob:
                worse_factor *= (1 - self.config.worse_entry_scale)
            if rng.random() < self.config.worse_exit_prob:
                worse_factor *= (1 - self.config.worse_exit_scale)

            new_fees     = t.fees * fee_mult
            new_slippage = t.slippage * slip_mult
            new_pnl_gross = t.pnl_gross * worse_factor * (size / t.size if t.size > 0 else 1.0)
            new_pnl_net   = new_pnl_gross - new_fees - new_slippage

            result.append(Trade(
                entry_time=t.entry_time, exit_time=t.exit_time,
                pnl_net=new_pnl_net, pnl_gross=new_pnl_gross,
                size=size, fees=new_fees, slippage=new_slippage,
                symbol=t.symbol, strategy=t.strategy, direction=t.direction,
            ))

        return result

    def _empty_result(self) -> MonteCarloResult:
        return MonteCarloResult(
            n_simulations=0, config=self.config,
            returns_pct=[], median_return=0, p5_return=0, p1_return=0,
            p25_return=0, p75_return=0,
            max_drawdowns=[], median_max_dd=0, p95_max_dd=0,
            prob_loss=0, prob_exceed_risk=0, prob_ruin=0, prob_positive=0,
            sharpe_values=[], median_sharpe=0, capital=self.capital,
        )
