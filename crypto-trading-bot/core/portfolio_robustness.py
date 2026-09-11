"""Portfolio robustness metrics (spec §52-59).

The goal is NOT one universal strategy — it is a portfolio of independent
alphas covering DIFFERENT regimes, sized for capacity and constrained by
risk of ruin and geometric growth, never raw arithmetic return.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── Regime coverage matrix (spec §53) ─────────────────────────────────────────


def regime_coverage_matrix(
    alpha_returns: Dict[str, Sequence[Tuple[str, float]]],
    min_regime_sample: int = 8,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """alpha_id -> [(regime_label, return)] → measured (never hard-coded)
    per-regime expectancy/sample. Signs come from real data."""
    matrix: Dict[str, Dict[str, Dict[str, float]]] = {}
    for alpha_id, pairs in alpha_returns.items():
        by_regime: Dict[str, List[float]] = {}
        for regime, ret in pairs:
            by_regime.setdefault(regime, []).append(ret)
        matrix[alpha_id] = {}
        for regime, rets in by_regime.items():
            if len(rets) < min_regime_sample:
                continue
            matrix[alpha_id][regime] = {
                "n": len(rets),
                "expectancy": sum(rets) / len(rets),
                "win_rate": sum(1 for r in rets if r > 0) / len(rets),
            }
    return matrix


def regime_coverage_score(matrix: Dict[str, Dict[str, Dict[str, float]]],
                          regimes: Optional[Sequence[str]] = None) -> float:
    """Fraction of observed regimes covered by ≥1 alpha with positive
    measured expectancy. Diversification across ENVIRONMENTS, not tickers."""
    all_regimes = set(regimes or [])
    if not all_regimes:
        for per_alpha in matrix.values():
            all_regimes.update(per_alpha)
    if not all_regimes:
        return 0.0
    covered = sum(
        1 for regime in all_regimes
        if any(stats.get(regime, {}).get("expectancy", 0) > 0
               for stats in matrix.values()))
    return covered / len(all_regimes)


# ── Downside / stress correlation (spec §55) ──────────────────────────────────


def downside_correlation(a: Sequence[float], b: Sequence[float],
                         quantile: float = 0.3) -> Optional[float]:
    """Correlation measured on each series' own worst-``quantile`` days
    (symmetrized) — ten strategies that crash together are not diversified.
    Conditioning on each side separately avoids the selection artifact of
    conditioning on the joint sum."""
    from core.strategy_correlation import _pearson
    n = min(len(a), len(b))
    if n < 10:
        return None
    k = max(5, int(n * quantile))

    def _tail_corr(cond: Sequence[float]) -> Optional[float]:
        idx = sorted(range(n), key=lambda i: cond[i])[:k]
        return _pearson([a[i] for i in idx], [b[i] for i in idx])
    corrs = [c for c in (_tail_corr(a), _tail_corr(b)) if c is not None]
    return sum(corrs) / len(corrs) if corrs else None


# ── Alpha capacity (spec §56) ─────────────────────────────────────────────────


def estimate_alpha_capacity(
    *, adv_usd: Optional[float], expected_net_return: float,
    signal_frequency_per_day: float, holding_days: float,
    max_participation: float = 0.01, impact_coeff: float = 0.001,
) -> Dict[str, Any]:
    """Max deployable notional before impact erodes the edge:
    size where sqrt-impact(participation) ≈ half the expected edge, capped at
    max ADV participation. Unknown liquidity → conservative tiny capacity."""
    if not adv_usd or adv_usd <= 0:
        return {"estimated_alpha_capacity_usd": 1_000.0,
                "constraint": "unknown_liquidity"}
    participation_cap = adv_usd * max_participation
    if expected_net_return <= 0:
        return {"estimated_alpha_capacity_usd": 0.0, "constraint": "no_edge"}
    # impact_pct = impact_coeff * sqrt(participation) = expected/2 → solve
    target_impact = expected_net_return / 2
    impact_participation = (target_impact / impact_coeff) ** 2
    impact_cap = adv_usd * min(impact_participation, 1.0)
    capacity = min(participation_cap, impact_cap)
    return {
        "estimated_alpha_capacity_usd": capacity,
        "constraint": ("participation" if participation_cap < impact_cap
                       else "market_impact"),
        "daily_capacity_usd": capacity * signal_frequency_per_day,
        "capital_turnover_days": holding_days,
    }


def return_on_capital_time(expected_net_return: float, capital_required: float,
                           holding_days: float) -> Optional[float]:
    """Capital-efficiency metric (spec §57): edge per dollar-day tied up."""
    if capital_required <= 0 or holding_days <= 0:
        return None
    return expected_net_return / holding_days


# ── Geometric growth + risk of ruin (spec §58-59) ─────────────────────────────


def expected_log_growth(returns: Sequence[float], fraction: float = 1.0) -> float:
    """Mean log growth at the given capital fraction (bounded Kelly territory).
    Any single ruinous outcome (-100% at fraction) → -inf growth."""
    total = 0.0
    n = 0
    for r in returns:
        g = 1.0 + fraction * r
        if g <= 0:
            return float("-inf")
        total += math.log(g)
        n += 1
    return total / n if n else 0.0


def risk_of_ruin(
    returns: Sequence[float], *, fraction: float = 1.0,
    ruin_drawdown: float = 0.5, horizon_trades: int = 250,
    n_sims: int = 1000, block_size: int = 5, seed: int = 42,
) -> Dict[str, float]:
    """Block-bootstrap probability of hitting the ruin drawdown within the
    horizon. Portfolios above the configured limit are hard-rejected."""
    rets = list(returns)
    if len(rets) < 20:
        return {"risk_of_ruin": 1.0, "note_insufficient_sample": 1.0}
    rng = random.Random(seed)
    n = len(rets)
    block = max(2, min(block_size, n // 4))
    ruined = 0
    worst_dds = []
    for _ in range(n_sims):
        equity = peak = 1.0
        dd_max = 0.0
        steps = 0
        is_ruined = False
        while steps < horizon_trades:
            start = rng.randrange(0, n - block + 1)
            for r in rets[start:start + block]:
                equity *= max(1.0 + fraction * r, 0.0)
                peak = max(peak, equity)
                dd = 1.0 - equity / peak if peak > 0 else 1.0
                dd_max = max(dd_max, dd)
                steps += 1
                if dd >= ruin_drawdown or equity <= 0:
                    is_ruined = True
                    break
            if is_ruined:
                break
        if is_ruined:
            ruined += 1
        worst_dds.append(dd_max)
    worst_dds.sort()
    return {
        "risk_of_ruin": ruined / n_sims,
        "p95_max_drawdown": worst_dds[int(0.95 * len(worst_dds)) - 1],
        "median_max_drawdown": worst_dds[len(worst_dds) // 2],
        "fraction": fraction,
        "horizon_trades": horizon_trades,
    }


def portfolio_regime_preference(
    candidate_sets: Dict[str, Dict[str, Any]],
) -> str:
    """Prefer the candidate PORTFOLIO with better regime coverage and lower
    downside correlation over a cluster of correlated alphas — even when the
    correlated cluster's headline return is higher (spec §54, test Q).

    candidate_sets: name -> {coverage: float, avg_downside_corr: float,
                             expected_return: float}
    """
    def score(s: Dict[str, Any]) -> float:
        return (s.get("coverage", 0.0)
                - max(s.get("avg_downside_corr", 0.0), 0.0)
                + 0.1 * s.get("expected_return", 0.0))
    return max(candidate_sets, key=lambda k: score(candidate_sets[k]))


# ── Research gap direction (spec §72-73) ──────────────────────────────────


class ResearchGapDirector:
    """Finds what the alpha PORTFOLIO is missing (regimes, directions,
    families) and produces research priorities. A gap directs research — it
    never forces an alpha into existence; cash remains acceptable."""

    CORE_REGIMES = ("BULL", "BEAR", "SIDEWAYS", "HIGH_VOL", "LOW_VOL")

    def analyze(
        self,
        coverage_matrix: Dict[str, Dict[str, Dict[str, float]]],
        *,
        alpha_families: Optional[Dict[str, str]] = None,
        alpha_directions: Optional[Dict[str, str]] = None,
        regimes: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        regimes = list(regimes or self.CORE_REGIMES)
        gaps: List[str] = []
        uncovered = [
            r for r in regimes
            if not any(stats.get(r, {}).get("expectancy", 0) > 0
                       for stats in coverage_matrix.values())]
        for r in uncovered:
            gaps.append(f"REGIME_COVERAGE_GAP:{r}")

        directions = set((alpha_directions or {}).values())
        if directions and "short" not in directions:
            gaps.append("DIRECTION_GAP:short")
        families = set((alpha_families or {}).values())
        if families and "MARKET_NEUTRAL" not in families:
            gaps.append("FAMILY_GAP:MARKET_NEUTRAL")

        priorities: List[Dict[str, str]] = []
        for g in gaps:
            if g.startswith("REGIME_COVERAGE_GAP:BEAR") or \
                    g.startswith("REGIME_COVERAGE_GAP:HIGH_VOL"):
                priorities.append({
                    "gap": g,
                    "research_directive": "prioritize market-neutral, short-side, "
                                          "relative-value and volatility families",
                })
            elif g.startswith("DIRECTION_GAP"):
                priorities.append({
                    "gap": g,
                    "research_directive": "prioritize short-side discovery: negative "
                                          "overnight anomalies, relative weakness, "
                                          "post-event negative drift",
                })
            else:
                priorities.append({"gap": g,
                                   "research_directive": "broaden family search"})
        return {
            "gaps": gaps,
            "uncovered_regimes": uncovered,
            "priorities": priorities,
            "note": "gaps direct research budget; they never force trades — "
                    "cash is acceptable when nothing validates",
        }
