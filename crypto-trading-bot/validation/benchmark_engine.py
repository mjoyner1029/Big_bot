"""
BenchmarkEngine — compare strategy performance against standard benchmarks.

Supported benchmarks
--------------------
CASH         Zero-return baseline
BUY_HOLD_BTC Buy-and-hold BTC for the evaluation period
SPY          Buy-and-hold SPY ETF
QQQ          Buy-and-hold QQQ ETF
EQUAL_WEIGHT Equal-weight buy-and-hold of the traded universe
RANDOM       Random-entry baseline (average of N random runs)
PREVIOUS     Previous production strategy/model version

Computed comparators
--------------------
excess_return       Strategy net return − benchmark return
alpha               Excess return after beta adjustment
relative_sharpe     Strategy Sharpe / Benchmark Sharpe
relative_drawdown   Strategy max DD / Benchmark max DD
information_ratio   Excess return / Tracking error
benchmark_corr      Correlation of strategy returns with benchmark
beta                Regression slope against benchmark

A strategy making money does NOT automatically demonstrate edge —
it must beat appropriate risk-adjusted benchmarks.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from validation.engine import Trade, ValidationEngine, ValidationResult, _parse_dt


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSeries:
    """Daily or per-trade equity curve for a benchmark."""
    name:       str
    timestamps: List[datetime]
    prices:     List[float]          # price series, same length as timestamps

    @property
    def total_return_pct(self) -> float:
        if len(self.prices) < 2 or self.prices[0] == 0:
            return 0.0
        return (self.prices[-1] / self.prices[0] - 1.0) * 100.0

    @property
    def sharpe(self) -> float:
        returns = [(self.prices[i] / self.prices[i-1] - 1.0)
                   for i in range(1, len(self.prices))]
        if len(returns) < 2:
            return 0.0
        avg = statistics.mean(returns)
        std = statistics.stdev(returns)
        return (avg / std * math.sqrt(252)) if std > 0 else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        peak, max_dd = 0.0, 0.0
        for p in self.prices:
            if p > peak:
                peak = p
            dd = (peak - p) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd * 100.0


@dataclass
class BenchmarkComparison:
    """Comparison of a strategy against a single benchmark."""
    strategy_label:     str
    benchmark_name:     str
    strategy_return:    float    # % net
    benchmark_return:   float    # %
    excess_return:      float    # strategy − benchmark
    strategy_sharpe:    float
    benchmark_sharpe:   float
    relative_sharpe:    float    # strategy / benchmark (>1 = strategy better)
    strategy_max_dd:    float
    benchmark_max_dd:   float    # % 
    relative_drawdown:  float    # strategy / benchmark (<1 = strategy better)
    information_ratio:  float
    benchmark_corr:     float    # correlation of period returns
    beta:               float
    alpha:              float    # Jensen's alpha (annual)
    beats_benchmark:    bool
    verdict:            str = ''


@dataclass
class BenchmarkReport:
    label:       str
    comparisons: List[BenchmarkComparison] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["═" * 70, f"  BENCHMARK COMPARISON: {self.label}", "═" * 70]
        header = f"  {'Benchmark':<18} {'Strat%':>7} {'BM%':>7} {'Excess':>7} {'rSharpe':>8} {'Beat?':>6}"
        lines.append(header)
        lines.append("  " + "─" * 60)
        for c in self.comparisons:
            lines.append(
                f"  {c.benchmark_name:<18} "
                f"{c.strategy_return:>7.2f} "
                f"{c.benchmark_return:>7.2f} "
                f"{c.excess_return:>7.2f} "
                f"{c.relative_sharpe:>8.3f} "
                f"{'✓' if c.beats_benchmark else '✗':>6}"
            )
        lines.append("═" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BenchmarkEngine:
    """
    Compares a ValidationResult against one or more benchmark series.

    If live market data is unavailable (common in paper-trading environments),
    use synthetic benchmarks based on provided return assumptions.
    """

    # Default annual return assumptions when live data unavailable
    _DEFAULT_ASSUMPTIONS = {
        'CASH':         0.045,    # 4.5% risk-free
        'BUY_HOLD_BTC': 0.60,     # 60% BTC historical annual (conservative)
        'SPY':          0.10,
        'QQQ':          0.12,
        'EQUAL_WEIGHT': 0.15,     # assume moderate crypto portfolio
        'RANDOM':       0.00,     # zero-alpha assumption
    }

    def __init__(
        self,
        data_fetcher=None,
        risk_free_rate: float = 0.05,
    ):
        self.data_fetcher   = data_fetcher
        self.risk_free_rate = risk_free_rate
        self._ve            = ValidationEngine(risk_free_rate=risk_free_rate)

    # ── Public API ────────────────────────────────────────────────────────────

    def compare(
        self,
        result: ValidationResult,
        benchmarks: List[str] = None,
        benchmark_series: Dict[str, BenchmarkSeries] = None,
    ) -> BenchmarkReport:
        """
        Compare a ValidationResult against benchmarks.

        Args:
            result:           Strategy validation result
            benchmarks:       List of benchmark names (use _DEFAULT_ASSUMPTIONS)
            benchmark_series: Optional pre-computed price series per benchmark
        """
        if benchmarks is None:
            benchmarks = ['CASH', 'BUY_HOLD_BTC', 'SPY', 'EQUAL_WEIGHT', 'RANDOM']

        bm_series = benchmark_series or {}
        comparisons = []

        for bm_name in benchmarks:
            series = bm_series.get(bm_name)
            bm_ret = self._benchmark_return(bm_name, result.eval_days, series)
            bm_sharpe  = self._benchmark_sharpe(bm_name, series)
            bm_max_dd  = self._benchmark_max_dd(bm_name, series)

            # Strategy returns and risk
            strat_ret    = result.total_return_pct
            strat_sharpe = result.sharpe
            strat_max_dd = result.max_drawdown

            excess    = strat_ret - bm_ret
            rel_sh    = (strat_sharpe / bm_sharpe) if abs(bm_sharpe) > 0.01 else (1.0 if strat_sharpe > 0 else 0.0)
            rel_dd    = (strat_max_dd / bm_max_dd) if bm_max_dd > 0 else 0.0

            ir = self._information_ratio(result, bm_name, bm_ret, series)
            corr, beta = self._corr_beta(result, bm_name, series)
            alpha = self._alpha(result, bm_name, beta)

            beats = (excess > 0 and strat_sharpe > bm_sharpe)
            verdict = (
                "BEATS BENCHMARK (excess return + higher Sharpe)"
                if beats else
                "DOES NOT BEAT BENCHMARK"
            )

            comparisons.append(BenchmarkComparison(
                strategy_label=result.label,
                benchmark_name=bm_name,
                strategy_return=strat_ret,
                benchmark_return=bm_ret,
                excess_return=excess,
                strategy_sharpe=strat_sharpe,
                benchmark_sharpe=bm_sharpe,
                relative_sharpe=rel_sh,
                strategy_max_dd=strat_max_dd,
                benchmark_max_dd=bm_max_dd,
                relative_drawdown=rel_dd,
                information_ratio=ir,
                benchmark_corr=corr,
                beta=beta,
                alpha=alpha,
                beats_benchmark=beats,
                verdict=verdict,
            ))

        return BenchmarkReport(label=result.label, comparisons=comparisons)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _benchmark_return(
        self,
        name:     str,
        days:     float,
        series:   Optional[BenchmarkSeries],
    ) -> float:
        """Return total % return for a benchmark over the given period."""
        if series:
            return series.total_return_pct
        annual = self._DEFAULT_ASSUMPTIONS.get(name, 0.0)
        years  = days / 365.0
        return ((1 + annual) ** years - 1) * 100.0

    def _benchmark_sharpe(self, name: str, series: Optional[BenchmarkSeries]) -> float:
        if series:
            return series.sharpe
        annual = self._DEFAULT_ASSUMPTIONS.get(name, 0.0)
        # Approximate Sharpe from return assumption (very rough)
        vols = {'CASH': 0.0, 'BUY_HOLD_BTC': 0.8, 'SPY': 0.15,
                'QQQ': 0.20, 'EQUAL_WEIGHT': 0.50, 'RANDOM': 0.30}
        vol = vols.get(name, 0.2)
        return ((annual - self.risk_free_rate) / vol) if vol > 0 else 0.0

    def _benchmark_max_dd(self, name: str, series: Optional[BenchmarkSeries]) -> float:
        if series:
            return series.max_drawdown_pct
        defaults = {'CASH': 0.0, 'BUY_HOLD_BTC': 80.0, 'SPY': 34.0,
                    'QQQ': 35.0, 'EQUAL_WEIGHT': 60.0, 'RANDOM': 40.0}
        return defaults.get(name, 30.0)

    def _information_ratio(
        self,
        result:  ValidationResult,
        name:    str,
        bm_ret:  float,
        series:  Optional[BenchmarkSeries],
    ) -> float:
        """Excess return / tracking error (approximated)."""
        excess = result.total_return_pct - bm_ret
        te = result.ann_volatility * 0.1 or 1.0   # rough tracking error proxy
        return excess / te if te > 0 else 0.0

    def _corr_beta(
        self,
        result: ValidationResult,
        name:   str,
        series: Optional[BenchmarkSeries],
    ) -> Tuple[float, float]:
        """Return (correlation, beta) — approximated without per-trade benchmark data."""
        if series and len(series.prices) >= 2:
            # Would compute actual correlation with trade-time prices
            # For now return approximate
            pass
        corr_map = {
            'CASH':         0.0,
            'BUY_HOLD_BTC': 0.65,
            'SPY':          0.20,
            'QQQ':          0.25,
            'EQUAL_WEIGHT': 0.70,
            'RANDOM':       0.0,
        }
        corr = corr_map.get(name, 0.3)
        beta = corr * (result.ann_volatility / 100.0) / 0.15 if result.ann_volatility > 0 else 0.0
        return corr, beta

    def _alpha(self, result: ValidationResult, name: str, beta: float) -> float:
        """Jensen's alpha (annual %)."""
        bm_annual = self._DEFAULT_ASSUMPTIONS.get(name, 0.0) * 100
        strat_annual = result.ann_return_pct
        rf_annual    = self.risk_free_rate * 100
        return strat_annual - rf_annual - beta * (bm_annual - rf_annual)
