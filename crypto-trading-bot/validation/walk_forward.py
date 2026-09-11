"""
Walk-Forward Validator + OOS Splitter + Random Baseline Tester

PHASES 3, 4, 5

── WALK-FORWARD VALIDATION (Phase 4) ─────────────────────────────────────────

Strict time-series cross-validation. Data is NEVER randomly shuffled.

Structure:
    [TRAIN] → [VALIDATE] → roll forward → [TRAIN] → [VALIDATE] → ...

Protections:
    • No look-ahead bias (train uses only data before validation start)
    • Purging:  remove trades that overlap train/validation boundary
    • Embargo:  skip N bars after purge to prevent leakage from fast features
    • Labels are never overlapping

── OOS SPLITTER (Phase 5) ────────────────────────────────────────────────────

Maintains a final dataset that cannot be seen during development.

Labelling:
    TRAIN        → used for fitting
    VALIDATION   → used for in-sample tuning / feature selection
    OUT_OF_SAMPLE → touched only after candidate passes validation
    LIVE          → forward test data (accumulated in real time)

── RANDOM BASELINE (Phase 3) ────────────────────────────────────────────────

Generates thousands of random trade sequences with equivalent:
    • trade frequency
    • holding duration
    • position sizing
    • market exposure
    • asset universe

Compares real strategy against distribution:
    • percentile ranking
    • probability random beats real
    • distribution of random Sharpe ratios
    • distribution of random expectancy
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from validation.engine import Trade, ValidationEngine, ValidationResult


# ---------------------------------------------------------------------------
# OOS Splitter (Phase 5)
# ---------------------------------------------------------------------------

class DataSplit(str, Enum):
    TRAIN          = "TRAIN"
    VALIDATION     = "VALIDATION"
    OUT_OF_SAMPLE  = "OUT_OF_SAMPLE"
    LIVE           = "LIVE"


@dataclass
class SplitConfig:
    """Configuration for train/val/OOS/live splits."""
    train_pct:     float = 0.60    # 60% training
    val_pct:       float = 0.20    # 20% validation
    oos_pct:       float = 0.20    # 20% OOS (never touched during development)
    # Note: LIVE is everything after the dataset end date


@dataclass
class DataWindow:
    split:      DataSplit
    start:      datetime
    end:        datetime
    trades:     List[Trade] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.trades)


class OOSSplitter:
    """
    Splits a trade sequence into TRAIN/VALIDATION/OOS/LIVE windows.

    CRITICAL: OOS data must never be accessed until after all development
    decisions (feature selection, parameter tuning) are complete.

    Call oos_result.trades only from ValidationEngine after freezing development.
    """

    def __init__(self, config: SplitConfig = None):
        self.config = config or SplitConfig()
        self._oos_accessed = False

    def split(self, trades: List[Trade]) -> Dict[DataSplit, DataWindow]:
        """
        Split trades chronologically into TRAIN/VAL/OOS windows.

        Returns dict of DataSplit → DataWindow.
        """
        trades = sorted(trades, key=lambda t: t.entry_time)
        n      = len(trades)
        if n == 0:
            return {}

        train_end = int(n * self.config.train_pct)
        val_end   = int(n * (self.config.train_pct + self.config.val_pct))

        train_trades = trades[:train_end]
        val_trades   = trades[train_end:val_end]
        oos_trades   = trades[val_end:]

        windows = {}

        if train_trades:
            windows[DataSplit.TRAIN] = DataWindow(
                split=DataSplit.TRAIN,
                start=train_trades[0].entry_time,
                end=train_trades[-1].exit_time,
                trades=train_trades,
            )
        if val_trades:
            windows[DataSplit.VALIDATION] = DataWindow(
                split=DataSplit.VALIDATION,
                start=val_trades[0].entry_time,
                end=val_trades[-1].exit_time,
                trades=val_trades,
            )
        if oos_trades:
            # Store but DO NOT expose until explicitly requested
            self._oos_window = DataWindow(
                split=DataSplit.OUT_OF_SAMPLE,
                start=oos_trades[0].entry_time,
                end=oos_trades[-1].exit_time,
                trades=oos_trades,
            )
            # Return a window without trades visible until unlocked
            windows[DataSplit.OUT_OF_SAMPLE] = DataWindow(
                split=DataSplit.OUT_OF_SAMPLE,
                start=oos_trades[0].entry_time,
                end=oos_trades[-1].exit_time,
                trades=[],   # hidden until accessed
            )

        return windows

    def unlock_oos(self, justification: str) -> Optional[DataWindow]:
        """
        Access the OOS data. Requires explicit justification.
        Records that OOS was accessed — cannot be called again.
        """
        if self._oos_accessed:
            raise RuntimeError(
                "OOS data has already been accessed. "
                "Create a new OOSSplitter instance for a fresh hold-out."
            )
        if not justification or len(justification) < 10:
            raise ValueError("Must provide justification (min 10 chars) to unlock OOS")
        self._oos_accessed = True
        return getattr(self, '_oos_window', None)

    def label(self, trade: Trade, windows: Dict[DataSplit, DataWindow]) -> DataSplit:
        """Return which split a given trade belongs to."""
        for split, window in windows.items():
            if window.start <= trade.entry_time <= window.end:
                return split
        return DataSplit.LIVE


# ---------------------------------------------------------------------------
# Walk-Forward Validator (Phase 4)
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    train_periods:     int = 90     # training window in days
    val_periods:       int = 30     # validation window in days
    step_periods:      int = 30     # roll forward by N days
    purge_periods:     int = 1      # remove N days at train/val boundary
    embargo_periods:   int = 1      # skip N additional days after purge
    min_train_trades:  int = 20     # minimum trades in training window
    min_val_trades:    int = 5      # minimum trades in validation window


@dataclass
class WalkForwardFold:
    fold_number:      int
    train_window:     DataWindow
    val_window:       DataWindow
    train_result:     ValidationResult
    val_result:       ValidationResult
    train_start:      datetime
    train_end:        datetime
    val_start:        datetime
    val_end:          datetime
    purge_count:      int = 0
    embargo_count:    int = 0


@dataclass
class WalkForwardResult:
    folds:              List[WalkForwardFold]
    config:             WalkForwardConfig
    combined_val_result: Optional[ValidationResult] = None

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def avg_val_sharpe(self) -> float:
        sharpes = [f.val_result.sharpe for f in self.folds if f.val_result.sufficient_trades]
        return statistics.mean(sharpes) if sharpes else 0.0

    @property
    def avg_val_expectancy(self) -> float:
        exps = [f.val_result.expectancy for f in self.folds if f.val_result.sufficient_trades]
        return statistics.mean(exps) if exps else 0.0

    @property
    def pct_folds_positive(self) -> float:
        pos = sum(1 for f in self.folds if f.val_result.expectancy > 0)
        return pos / len(self.folds) if self.folds else 0.0

    def summary(self) -> str:
        lines = [
            "═" * 65,
            "  WALK-FORWARD VALIDATION RESULTS",
            "═" * 65,
            f"  Folds:              {self.n_folds}",
            f"  Avg val Sharpe:     {self.avg_val_sharpe:.3f}",
            f"  Avg val expectancy: ${self.avg_val_expectancy:.4f}",
            f"  Positive folds:     {self.pct_folds_positive:.1%}",
            "",
            f"  {'Fold':<5} {'TrainN':>7} {'ValN':>7} {'ValSharpe':>10} {'ValExp':>10} {'Pass?':>6}",
            "  " + "─" * 45,
        ]
        for f in self.folds:
            passed = f.val_result.expectancy > 0 and f.val_result.sufficient_trades
            lines.append(
                f"  {f.fold_number:<5} "
                f"{f.train_window.n:>7} "
                f"{f.val_window.n:>7} "
                f"{f.val_result.sharpe:>10.3f} "
                f"${f.val_result.expectancy:>9.4f} "
                f"{'✓' if passed else '✗':>6}"
            )
        lines.append("═" * 65)
        return "\n".join(lines)


class WalkForwardValidator:
    """
    Performs strict time-series walk-forward validation.

    Prevents look-ahead bias by ensuring training data never contains
    any information about the validation period.
    """

    def __init__(
        self,
        config: WalkForwardConfig = None,
        capital: float = 10_000.0,
        risk_free_rate: float = 0.05,
    ):
        self.config = config or WalkForwardConfig()
        self._ve    = ValidationEngine(capital=capital, risk_free_rate=risk_free_rate)

    def validate(self, trades: List[Trade], label: str = "Strategy") -> WalkForwardResult:
        """
        Run walk-forward validation on a trade sequence.

        Trades must be sorted chronologically.
        """
        trades = sorted(trades, key=lambda t: t.entry_time)
        if not trades:
            return WalkForwardResult(folds=[], config=self.config)

        start = trades[0].entry_time
        end   = trades[-1].exit_time

        folds      = []
        fold_num   = 0
        cursor     = start

        train_td  = timedelta(days=self.config.train_periods)
        val_td    = timedelta(days=self.config.val_periods)
        step_td   = timedelta(days=self.config.step_periods)
        purge_td  = timedelta(days=self.config.purge_periods)
        embargo_td = timedelta(days=self.config.embargo_periods)

        while True:
            train_start = cursor
            train_end   = cursor + train_td
            val_start   = train_end + purge_td + embargo_td
            val_end     = val_start + val_td

            if val_end > end:
                break

            # Assign trades to windows with purge/embargo
            train_trades = [
                t for t in trades
                if train_start <= t.entry_time < train_end - purge_td
            ]
            embargo_start = train_end + purge_td
            val_trades = [
                t for t in trades
                if embargo_start + embargo_td <= t.entry_time < val_end
            ]

            purge_count   = sum(1 for t in trades
                                if train_end - purge_td <= t.entry_time < train_end)
            embargo_count = sum(1 for t in trades
                                if embargo_start <= t.entry_time < embargo_start + embargo_td)

            train_window = DataWindow(
                split=DataSplit.TRAIN, start=train_start, end=train_end, trades=train_trades
            )
            val_window = DataWindow(
                split=DataSplit.VALIDATION, start=val_start, end=val_end, trades=val_trades
            )

            if (len(train_trades) >= self.config.min_train_trades and
                    len(val_trades) >= self.config.min_val_trades):
                train_res = self._ve.evaluate(train_trades, label=f"{label} Train F{fold_num}")
                val_res   = self._ve.evaluate(val_trades,   label=f"{label} Val F{fold_num}")

                folds.append(WalkForwardFold(
                    fold_number=fold_num,
                    train_window=train_window,
                    val_window=val_window,
                    train_result=train_res,
                    val_result=val_res,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    purge_count=purge_count,
                    embargo_count=embargo_count,
                ))
                fold_num += 1

            cursor += step_td

        # Combined out-of-sample validation result across all folds
        all_val_trades = []
        for f in folds:
            all_val_trades.extend(f.val_window.trades)
        combined = self._ve.evaluate(all_val_trades, label=f"{label} WalkForward Combined") if all_val_trades else None

        return WalkForwardResult(
            folds=folds,
            config=self.config,
            combined_val_result=combined,
        )


# ---------------------------------------------------------------------------
# Random Baseline (Phase 3)
# ---------------------------------------------------------------------------

@dataclass
class RandomBaselineResult:
    """Results from random baseline comparison."""
    real_sharpe:        float
    real_expectancy:    float
    n_random_runs:      int
    random_sharpes:     List[float]
    random_expectancies: List[float]

    # Distribution stats
    sharpe_mean:        float = 0.0
    sharpe_std:         float = 0.0
    sharpe_percentile:  float = 0.0   # real strategy's percentile in random dist
    expectancy_percentile: float = 0.0
    prob_random_beats:  float = 0.0   # fraction of random strategies beating real
    verdict:            str = ''

    def summary(self) -> str:
        return "\n".join([
            "═" * 60,
            "  RANDOM BASELINE TEST",
            "═" * 60,
            f"  Random runs:       {self.n_random_runs:>10}",
            f"  Real Sharpe:       {self.real_sharpe:>10.3f}",
            f"  Random Sharpe μ:   {self.sharpe_mean:>10.3f}",
            f"  Random Sharpe σ:   {self.sharpe_std:>10.3f}",
            f"  Sharpe percentile: {self.sharpe_percentile:>10.1f}%",
            f"  P(random beats):   {self.prob_random_beats:>10.1%}",
            f"  Real expectancy:   ${self.real_expectancy:>9.4f}",
            f"  Exp percentile:    {self.expectancy_percentile:>10.1f}%",
            "─" * 60,
            f"  Verdict: {self.verdict}",
            "═" * 60,
        ])


class RandomBaselineTester:
    """
    Tests whether strategy performance could plausibly be random luck.

    Generates N random trade sequences with equivalent:
        - trade count
        - holding duration distribution
        - position sizing
        - market exposure
        - asset universe

    Then compares the real strategy's Sharpe/expectancy against the
    distribution of random strategies.
    """

    def __init__(
        self,
        n_runs: int = 1000,
        capital: float = 10_000.0,
        risk_free_rate: float = 0.05,
        seed: int = 42,
    ):
        self.n_runs          = n_runs
        self.capital         = capital
        self.risk_free_rate  = risk_free_rate
        self.seed            = seed
        self._ve             = ValidationEngine(capital=capital, risk_free_rate=risk_free_rate)

    def test(
        self,
        real_result: ValidationResult,
        template_trades: List[Trade],
        market_daily_returns: List[float] = None,
    ) -> RandomBaselineResult:
        """
        Generate N random strategies and compare against real result.

        Args:
            real_result:          ValidationResult from real strategy
            template_trades:      Real trades (used to extract timing/sizing stats)
            market_daily_returns: Optional daily market returns for realistic simulation
        """
        rng = random.Random(self.seed)

        if not template_trades:
            return RandomBaselineResult(
                real_sharpe=real_result.sharpe,
                real_expectancy=real_result.expectancy,
                n_random_runs=0,
                random_sharpes=[],
                random_expectancies=[],
                verdict="Insufficient template trades for random baseline",
            )

        # Extract stats from real trades
        avg_hold = real_result.avg_hold_hours
        avg_size = sum(t.size for t in template_trades) / len(template_trades)
        avg_fee  = sum(t.fees for t in template_trades) / len(template_trades)
        n_trades = real_result.trade_count

        # Build a daily return pool from market returns or synthetic
        if market_daily_returns and len(market_daily_returns) >= 20:
            returns_pool = market_daily_returns
        else:
            # Synthetic: zero-mean, realistic crypto volatility
            rng2 = random.Random(self.seed + 1)
            returns_pool = [rng2.gauss(0.0002, 0.025) for _ in range(1000)]

        random_sharpes      = []
        random_expectancies = []

        eval_start = template_trades[0].entry_time
        eval_end   = template_trades[-1].exit_time

        for _ in range(self.n_runs):
            random_trades = self._generate_random_trades(
                rng, n_trades, avg_hold, avg_size, avg_fee,
                returns_pool, eval_start, eval_end,
            )
            if not random_trades:
                continue
            res = self._ve.evaluate(random_trades, label="_random_")
            random_sharpes.append(res.sharpe)
            random_expectancies.append(res.expectancy)

        if not random_sharpes:
            return RandomBaselineResult(
                real_sharpe=real_result.sharpe,
                real_expectancy=real_result.expectancy,
                n_random_runs=0,
                random_sharpes=[],
                random_expectancies=[],
                verdict="Failed to generate random baselines",
            )

        sharpe_mean = statistics.mean(random_sharpes)
        sharpe_std  = statistics.stdev(random_sharpes) if len(random_sharpes) > 1 else 0.0
        n_worse_sharpe = sum(1 for s in random_sharpes if s <= real_result.sharpe)
        n_worse_exp    = sum(1 for e in random_expectancies if e <= real_result.expectancy)
        sharpe_pct   = n_worse_sharpe / len(random_sharpes) * 100
        exp_pct      = n_worse_exp    / len(random_expectancies) * 100
        prob_beats   = 1 - sharpe_pct / 100

        if sharpe_pct >= 95:
            verdict = f"SIGNIFICANT EDGE — real strategy beats {sharpe_pct:.0f}% of random strategies"
        elif sharpe_pct >= 80:
            verdict = f"POSSIBLE EDGE — real strategy beats {sharpe_pct:.0f}% of random strategies (not significant)"
        else:
            verdict = f"NO CLEAR EDGE — real strategy only beats {sharpe_pct:.0f}% of random strategies (LIKELY LUCK)"

        return RandomBaselineResult(
            real_sharpe=real_result.sharpe,
            real_expectancy=real_result.expectancy,
            n_random_runs=self.n_runs,
            random_sharpes=random_sharpes,
            random_expectancies=random_expectancies,
            sharpe_mean=sharpe_mean,
            sharpe_std=sharpe_std,
            sharpe_percentile=sharpe_pct,
            expectancy_percentile=exp_pct,
            prob_random_beats=prob_beats,
            verdict=verdict,
        )

    def _generate_random_trades(
        self,
        rng:          random.Random,
        n_trades:     int,
        avg_hold_h:   float,
        avg_size:     float,
        avg_fee:      float,
        returns_pool: List[float],
        start:        datetime,
        end:          datetime,
    ) -> List[Trade]:
        """Generate a random sequence of N trades using market return pool."""
        duration = (end - start).total_seconds()
        trades   = []

        for _ in range(n_trades):
            # Random entry time
            offset     = rng.uniform(0, duration * 0.9)
            entry_time = start + timedelta(seconds=offset)
            hold_h     = max(rng.expovariate(1.0 / max(avg_hold_h, 1.0)), 0.5)
            exit_time  = entry_time + timedelta(hours=hold_h)
            if exit_time > end:
                exit_time = end

            # Sample return from pool
            hold_bars  = max(int(hold_h / 24), 1)
            pnl_pct    = sum(rng.choice(returns_pool) for _ in range(hold_bars))
            size       = avg_size * rng.uniform(0.5, 1.5)
            pnl_gross  = pnl_pct * size
            fees       = avg_fee * rng.uniform(0.5, 1.5)
            slippage   = fees * 0.3
            pnl_net    = pnl_gross - fees - slippage

            trades.append(Trade(
                entry_time=entry_time, exit_time=exit_time,
                pnl_net=pnl_net, pnl_gross=pnl_gross,
                size=size, fees=fees, slippage=slippage,
            ))

        return sorted(trades, key=lambda t: t.entry_time)
