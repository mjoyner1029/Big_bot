"""Statistical validation toolbox for financial time-series research.

Provides the quantitative machinery required before any edge can be trusted:

- Walk-forward and purged time-series splits (with embargo)
- Benjamini-Hochberg false discovery rate control
- Deflated Sharpe ratio (Bailey & Lopez de Prado)
- Bootstrap confidence intervals and permutation tests
- Effective sample size for autocorrelated returns
- Parameter robustness scoring (plateau vs cliff)
- Event / profit concentration scoring

All functions are pure and deterministic given a seed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

# ── Time-series splitting ─────────────────────────────────────────────────────


def walk_forward_splits(
    n: int,
    n_folds: int = 5,
    min_train: int = 20,
) -> List[Tuple[range, range]]:
    """Expanding-window walk-forward splits over ``n`` time-ordered samples.

    Returns list of (train_indices, test_indices). Train always precedes test.
    """
    if n < min_train + n_folds:
        return []
    test_size = max(1, (n - min_train) // n_folds)
    splits: List[Tuple[range, range]] = []
    for i in range(n_folds):
        train_end = min_train + i * test_size
        test_end = min(train_end + test_size, n)
        if train_end >= test_end:
            break
        splits.append((range(0, train_end), range(train_end, test_end)))
    return splits


def purged_time_series_splits(
    n: int,
    n_folds: int = 5,
    embargo: int = 0,
    label_horizon: int = 0,
) -> List[Tuple[List[int], List[int]]]:
    """Purged K-fold for time series (Lopez de Prado style).

    Each fold's test block is contiguous. Training samples whose label window
    (``label_horizon`` bars ahead) overlaps the test block are PURGED, and an
    additional ``embargo`` samples after the test block are dropped from
    training to prevent leakage through serial correlation.
    """
    if n < n_folds * 2:
        return []
    fold_size = n // n_folds
    splits: List[Tuple[List[int], List[int]]] = []
    for k in range(n_folds):
        test_start = k * fold_size
        test_end = n if k == n_folds - 1 else (k + 1) * fold_size
        test_idx = list(range(test_start, test_end))
        train_idx = []
        for i in range(n):
            if test_start <= i < test_end:
                continue
            # Purge: label window of train sample i must not reach into test block
            if i < test_start and i + label_horizon >= test_start:
                continue
            # Embargo: drop samples immediately after the test block
            if test_end <= i < test_end + embargo:
                continue
            train_idx.append(i)
        splits.append((train_idx, test_idx))
    return splits


# ── Multiple hypothesis correction ────────────────────────────────────────────


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Benjamini-Hochberg FDR control. Returns pass/fail per hypothesis."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    passed = [False] * m
    max_k = -1
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= (rank / m) * alpha:
            max_k = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= max_k:
            passed[idx] = True
    return passed


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Probability that the observed Sharpe exceeds the expected max Sharpe
    from ``n_trials`` unskilled tries (Bailey & Lopez de Prado 2014).

    Returns a probability in [0, 1]; > 0.95 is a common acceptance level.
    """
    if n_obs < 2 or n_trials < 1:
        return 0.0
    e = 0.5772156649  # Euler-Mascheroni
    n_trials = max(n_trials, 1)
    if n_trials == 1:
        sr0 = 0.0
    else:
        z1 = _norm_ppf(1 - 1.0 / n_trials)
        z2 = _norm_ppf(1 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(max(_sharpe_variance_proxy(n_trials), 1e-12)) * ((1 - e) * z1 + e * z2)
    denom = math.sqrt(
        max(1 - skew * observed_sharpe + ((kurt - 1) / 4.0) * observed_sharpe**2, 1e-12)
        / max(n_obs - 1, 1)
    )
    return _norm_cdf((observed_sharpe - sr0) / denom)


def probability_of_backtest_overfitting(
    returns_matrix: Sequence[Sequence[float]],
    n_partitions: int = 8,
    min_configurations: int = 4,
    min_observations: int = 40,
    max_combinations: int = 200,
) -> Dict[str, object]:
    """Probability of Backtest Overfitting via Combinatorially Symmetric
    Cross-Validation (CSCV, Bailey et al. 2015).

    Input: matrix of TIME-ALIGNED returns, one row per competing strategy /
    parameter configuration (columns = chronological observations). NOT a
    single return stream — PBO measures SELECTION overfitting across variants.

    Method:
      1. Split the time axis into S chronological blocks (S even).
      2. For every symmetric combination of S/2 train blocks vs S/2 test
         blocks: rank configurations by in-sample Sharpe, take the IS winner,
         find its OOS relative rank ω = rank/(N+1), logit λ = ln(ω/(1−ω)).
      3. PBO = fraction of combinations where the IS winner performs at or
         below the OOS median (λ ≤ 0).

    Returns dict with pbo, median_oos_rank, rank_logits, n_configurations,
    n_partitions, n_cscv_splits — or {"status": "UNAVAILABLE", ...} when the
    inputs cannot support an honest estimate.
    """
    from itertools import combinations

    n_cfg = len(returns_matrix)
    if n_cfg < min_configurations:
        return {"status": "UNAVAILABLE",
                "reason": "PBO_UNAVAILABLE_INSUFFICIENT_VARIANTS",
                "n_configurations": n_cfg,
                "min_configurations": min_configurations}
    t = min(len(r) for r in returns_matrix)
    if t < min_observations:
        return {"status": "UNAVAILABLE",
                "reason": "PBO_UNAVAILABLE_INSUFFICIENT_OBSERVATIONS",
                "n_observations": t, "min_observations": min_observations}
    if n_partitions % 2 or n_partitions < 2:
        raise ValueError("n_partitions must be even and >= 2")
    matrix = [list(r[:t]) for r in returns_matrix]

    block_size = t // n_partitions
    if block_size < 2:
        n_partitions = max(2, (t // 2) * 2 // max(t // 4, 1))
        block_size = t // n_partitions
    blocks = [list(range(i * block_size,
                         (i + 1) * block_size if i < n_partitions - 1 else t))
              for i in range(n_partitions)]

    combos = list(combinations(range(n_partitions), n_partitions // 2))
    if len(combos) > max_combinations:
        step = len(combos) / max_combinations
        combos = [combos[int(i * step)] for i in range(max_combinations)]

    def perf(cfg_idx: int, idxs: List[int]) -> float:
        vals = [matrix[cfg_idx][i] for i in idxs]
        return sharpe_ratio(vals)

    logits: List[float] = []
    oos_ranks: List[float] = []
    for train_blocks in combos:
        train_idx = [i for b in train_blocks for i in blocks[b]]
        test_idx = [i for b in range(n_partitions) if b not in train_blocks
                    for i in blocks[b]]
        is_scores = [perf(c, train_idx) for c in range(n_cfg)]
        winner = max(range(n_cfg), key=lambda c: is_scores[c])
        oos_scores = [perf(c, test_idx) for c in range(n_cfg)]
        # OOS relative rank of the IS winner (1 = worst, N = best)
        rank = 1 + sum(1 for c in range(n_cfg)
                       if c != winner and oos_scores[c] < oos_scores[winner])
        omega = rank / (n_cfg + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1 - omega)))
        oos_ranks.append(omega)

    pbo = sum(1 for l in logits if l <= 0) / len(logits)
    oos_ranks.sort()
    return {
        "status": "OK",
        "pbo": pbo,
        "median_oos_rank": oos_ranks[len(oos_ranks) // 2],
        "rank_logits": logits[:50],
        "n_configurations": n_cfg,
        "n_observations": t,
        "n_partitions": n_partitions,
        "n_cscv_splits": len(combos),
    }


def _sharpe_variance_proxy(n_trials: int) -> float:
    # Variance of Sharpe estimates across trials; assume unit variance proxy
    return 1.0 / max(n_trials, 1)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    """Acklam's inverse normal CDF approximation."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ── Resampling ────────────────────────────────────────────────────────────────


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI of the mean. Returns (mean, lower, upper)."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    mean = sum(values) / n
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_i = int(((1 - ci) / 2) * n_boot)
    hi_i = int((1 - (1 - ci) / 2) * n_boot) - 1
    return mean, means[lo_i], means[hi_i]


def permutation_test_mean(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 2000,
    seed: int = 42,
) -> float:
    """Two-sided permutation test p-value for difference in means of a vs b."""
    if not a or not b:
        return 1.0
    rng = random.Random(seed)
    observed = abs(sum(a) / len(a) - sum(b) / len(b))
    combined = list(a) + list(b)
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        pa = combined[:n_a]
        pb = combined[n_a:]
        if abs(sum(pa) / len(pa) - sum(pb) / len(pb)) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def sign_test_p_value(values: Sequence[float]) -> float:
    """One-sided binomial sign test p-value that mean > 0 (H0: p(win)=0.5)."""
    n = len(values)
    if n == 0:
        return 1.0
    wins = sum(1 for v in values if v > 0)
    # P(X >= wins) under Binomial(n, 0.5)
    p = 0.0
    for k in range(wins, n + 1):
        p += math.comb(n, k) * (0.5 ** n)
    return min(max(p, 0.0), 1.0)


# ── Sample-size & uncertainty ─────────────────────────────────────────────────


def effective_sample_size(values: Sequence[float], max_lag: int = 20) -> float:
    """ESS adjusted for autocorrelation: n / (1 + 2*sum(rho_k))."""
    n = len(values)
    if n < 3:
        return float(n)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    if var <= 0:
        return float(n)
    rho_sum = 0.0
    for lag in range(1, min(max_lag, n - 1) + 1):
        cov = sum((values[i] - mean) * (values[i + lag] - mean) for i in range(n - lag)) / n
        rho = cov / var
        if rho <= 0:
            break
        rho_sum += rho
    return max(1.0, n / (1 + 2 * rho_sum))


def uncertainty_summary(values: Sequence[float], seed: int = 42) -> Dict[str, float]:
    """Mean, median, stderr, CI, P(net>0), expected shortfall, ESS."""
    n = len(values)
    if n == 0:
        return {
            "mean": 0.0, "median": 0.0, "std_error": 0.0,
            "ci_lower": 0.0, "ci_upper": 0.0,
            "p_net_positive": 0.0, "expected_shortfall_5pct": 0.0,
            "effective_sample_size": 0.0, "n": 0,
        }
    ordered = sorted(values)
    mean, lo, hi = bootstrap_ci(values, seed=seed)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    var = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    ess = effective_sample_size(values)
    stderr = math.sqrt(var / max(ess, 1.0))
    tail_n = max(1, int(n * 0.05))
    es = sum(ordered[:tail_n]) / tail_n
    return {
        "mean": mean,
        "median": median,
        "std_error": stderr,
        "ci_lower": lo,
        "ci_upper": hi,
        "p_net_positive": sum(1 for v in values if v > 0) / n,
        "expected_shortfall_5pct": es,
        "effective_sample_size": ess,
        "n": n,
    }


# ── Robustness / concentration ────────────────────────────────────────────────


def parameter_robustness_score(param_to_metric: Dict[float, float]) -> float:
    """Score in [0,1] rewarding broad plateaus over narrow peaks.

    Input: mapping of parameter value -> performance metric (e.g. expectancy)
    for the chosen value and its neighbors. Score is the fraction of the best
    metric retained on average across neighbors, floored at 0.
    """
    if len(param_to_metric) < 3:
        return 0.0
    metrics = list(param_to_metric.values())
    best = max(metrics)
    if best <= 0:
        return 0.0
    retained = [max(m, 0.0) / best for m in metrics]
    return sum(retained) / len(retained)


def profit_concentration(pnls: Sequence[float]) -> Dict[str, float]:
    """Fraction of total profit contributed by the biggest winners."""
    total = sum(p for p in pnls if p > 0)
    result = {
        "top_1_trade_pct": 0.0, "top_5_trades_pct": 0.0, "top_10_trades_pct": 0.0,
        "top_1pct_pct": 0.0, "top_5pct_pct": 0.0, "concentration_score": 0.0,
    }
    if total <= 0 or not pnls:
        return result
    winners = sorted((p for p in pnls if p > 0), reverse=True)
    n = len(pnls)

    def frac(k: int) -> float:
        return sum(winners[:k]) / total

    result["top_1_trade_pct"] = frac(1)
    result["top_5_trades_pct"] = frac(5)
    result["top_10_trades_pct"] = frac(10)
    result["top_1pct_pct"] = frac(max(1, int(n * 0.01)))
    result["top_5pct_pct"] = frac(max(1, int(n * 0.05)))
    # High score = concentrated (bad). Use top-5-trades share as headline.
    result["concentration_score"] = result["top_5_trades_pct"]
    return result


def sharpe_ratio(pnls: Sequence[float]) -> float:
    n = len(pnls)
    if n < 2:
        return 0.0
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / n
    std = math.sqrt(var)
    return mean / std if std > 0 else 0.0


def sortino_ratio(pnls: Sequence[float]) -> float:
    n = len(pnls)
    if n < 2:
        return 0.0
    mean = sum(pnls) / n
    downside = [p for p in pnls if p < 0]
    if not downside:
        return float("inf") if mean > 0 else 0.0
    dvar = sum(p ** 2 for p in downside) / n
    dstd = math.sqrt(dvar)
    return mean / dstd if dstd > 0 else 0.0


def max_drawdown(pnls: Sequence[float]) -> float:
    """Max drawdown of the cumulative P&L curve (absolute units)."""
    peak = 0.0
    cum = 0.0
    mdd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return mdd


def profit_factor(pnls: Sequence[float]) -> float:
    gains = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


# ── Hypothesis family registry (for FDR bookkeeping) ─────────────────────────


@dataclass
class TestedHypothesis:
    hypothesis_id: str
    family: str
    p_value: float
    metric: float = 0.0
    metadata: Dict = field(default_factory=dict)


class HypothesisFamilyTracker:
    """Track every tested hypothesis so FDR can be applied per family."""

    def __init__(self) -> None:
        self._tests: List[TestedHypothesis] = []

    def record(self, hypothesis_id: str, family: str, p_value: float,
               metric: float = 0.0, **metadata) -> None:
        self._tests.append(TestedHypothesis(hypothesis_id, family, p_value, metric, metadata))

    def n_trials(self, family: Optional[str] = None) -> int:
        if family is None:
            return len(self._tests)
        return sum(1 for t in self._tests if t.family == family)

    def fdr_pass(self, hypothesis_id: str, alpha: float = 0.05) -> bool:
        """Does this hypothesis survive BH-FDR within its family?"""
        target = next((t for t in self._tests if t.hypothesis_id == hypothesis_id), None)
        if target is None:
            return False
        family_tests = [t for t in self._tests if t.family == target.family]
        passed = benjamini_hochberg([t.p_value for t in family_tests], alpha)
        for t, ok in zip(family_tests, passed):
            if t.hypothesis_id == hypothesis_id:
                return ok
        return False
