"""Portfolio Opportunity Allocator.

Ranks and sizes the surviving OpportunityCandidates TOGETHER (opportunity-
first, portfolio-aware) under hard constraints. Every candidate competes with
CASH: nothing is allocated unless its conservative (lower-bound) EV clears the
configured hurdle. Conflicting signals on one asset are resolved
deterministically, never by summing votes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.alpha_signal_engine import OpportunityCandidate, ReasonCode
from core.strategy_correlation import family_of

logger = logging.getLogger(__name__)


@dataclass
class AllocationDecision:
    candidate: OpportunityCandidate
    accepted: bool
    allocation_usd: float = 0.0
    reason_codes: List[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class AllocatorConfig:
    capital: float = 10_000.0
    min_conservative_ev: float = 0.0     # lower-bound EV must exceed this (cash hurdle)
    max_total_exposure_frac: float = 0.6
    max_asset_exposure_frac: float = 0.15
    max_family_exposure_frac: float = 0.40
    max_correlated_cluster_frac: float = 0.45
    correlation_threshold: float = 0.7   # candidates above this are one cluster
    base_position_frac: float = 0.05
    conflict_quality_margin: float = 1.5  # winner must be this much better, else abstain
    max_risk_of_ruin: float = 0.05       # hard gate (spec: risk-of-ruin limit)
    ruin_drawdown: float = 0.5
    min_usable_capacity_usd: float = 100.0
    min_ev_improvement: float = 0.001    # rebalancing hysteresis (spec §67)


class PortfolioAllocator:
    """Constrained scoring allocator (deliberately not a fragile optimizer)."""

    def __init__(self, config: Optional[AllocatorConfig] = None,
                 alpha_correlations: Optional[Dict[Tuple[str, str], float]] = None,
                 downside_correlations: Optional[Dict[Tuple[str, str], float]] = None):
        self.config = config or AllocatorConfig()
        self.alpha_correlations = alpha_correlations or {}
        # Strategies uncorrelated normally but crashing together cluster too
        self.downside_correlations = downside_correlations or {}

    def allocate(
        self,
        candidates: Sequence[OpportunityCandidate],
        open_positions: Optional[List[Dict[str, Any]]] = None,
        family_of_alpha: Optional[Dict[str, str]] = None,
        alpha_return_samples: Optional[Dict[str, List[float]]] = None,
        alpha_return_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[AllocationDecision]:
        cfg = self.config
        open_positions = open_positions or []
        family_of_alpha = family_of_alpha or {}
        decisions: List[AllocationDecision] = []

        # 1. HARD GATES first (spec: gates before any heuristic score)
        viable: List[OpportunityCandidate] = []
        open_symbol_dirs = {(p.get("symbol"), p.get("direction", "long"))
                            for p in open_positions}
        for c in candidates:
            lcb = c.ev_lower_bound if c.ev_lower_bound is not None else c.conservative_ev
            ev = c.expected_net_return
            if ReasonCode.BORROW_UNAVAILABLE in c.reason_codes:
                decisions.append(AllocationDecision(
                    c, False, reason_codes=[ReasonCode.BORROW_UNAVAILABLE],
                    detail="equity short without borrow — not executable"))
                continue
            if ReasonCode.EDGE_DECAY in c.reason_codes:
                decisions.append(AllocationDecision(
                    c, False, reason_codes=[ReasonCode.EDGE_DECAY],
                    detail="edge survival PAUSED — fresh evidence required"))
                continue
            if ev is None or lcb is None:
                decisions.append(AllocationDecision(
                    c, False, reason_codes=[ReasonCode.EXPECTED_EV_TOO_LOW],
                    detail="no EV estimate — cash wins by default"))
                continue
            if ev <= cfg.min_conservative_ev:
                decisions.append(AllocationDecision(
                    c, False, reason_codes=[ReasonCode.EXPECTED_EV_TOO_LOW],
                    detail=f"EV={ev:.5f} <= hurdle {cfg.min_conservative_ev}"))
                continue
            if lcb <= cfg.min_conservative_ev:
                decisions.append(AllocationDecision(
                    c, False, reason_codes=[ReasonCode.LOWER_BOUND_NEGATIVE],
                    detail=f"lower bound {lcb:.5f} <= hurdle — uncertainty too high"))
                continue
            if (c.practical_capacity_usd is not None
                    and c.practical_capacity_usd < cfg.min_usable_capacity_usd):
                decisions.append(AllocationDecision(
                    c, False, reason_codes=[ReasonCode.INSUFFICIENT_CAPACITY],
                    detail=f"capacity ${c.practical_capacity_usd:,.0f} unusable"))
                continue
            # Rebalancing hysteresis: adding to an existing same-direction
            # exposure requires a meaningfully better edge, not churn
            if (c.symbol, c.direction) in open_symbol_dirs and \
                    lcb < cfg.min_conservative_ev + cfg.min_ev_improvement:
                decisions.append(AllocationDecision(
                    c, False, reason_codes=[ReasonCode.EXPECTED_EV_TOO_LOW],
                    detail="hysteresis: insufficient improvement over open exposure"))
                continue
            viable.append(c)

        # 2. Signal collision handling per symbol
        viable, conflict_decisions = self._resolve_conflicts(viable)
        decisions.extend(conflict_decisions)

        # 3. Rank by conservative NET-execution EV × quality multipliers
        def score(c: OpportunityCandidate) -> float:
            s = (c.ev_lower_bound or 0.0)
            s *= (c.edge_health_score if c.edge_health_score is not None else 0.7)
            s *= c.regime_fit
            s *= (c.execution_feasibility / 100.0 if c.execution_feasibility else 0.7)
            s *= c.correlation_penalty
            s *= c.survival_multiplier
            s *= (1.0 - 0.5 * c.crowding_score)
            if c.capital_time_efficiency:
                s *= (1.0 + min(c.capital_time_efficiency * 100.0, 0.25))
            # Economic scale: bounded log1p transform of expected dollar alpha
            # (spec: reflects scalable dollars without dominating safety)
            if c.expected_dollar_alpha and c.expected_dollar_alpha > 0:
                import math as _m
                s *= 1.0 + min(0.25, _m.log1p(c.expected_dollar_alpha) / 40.0)
            return s

        for c in viable:
            c.final_opportunity_score = score(c)
        ranked = sorted(viable, key=lambda c: c.final_opportunity_score, reverse=True)

        # 4. Greedy constrained allocation
        total_alloc = sum(float(p.get("size") or 0.0) for p in open_positions)
        asset_alloc: Dict[str, float] = {}
        family_alloc: Dict[str, float] = {}
        for p in open_positions:
            asset_alloc[p.get("symbol", "?")] = asset_alloc.get(p.get("symbol", "?"), 0.0) \
                + float(p.get("size") or 0.0)
            fam = family_of(p.get("strategy") or "unknown")
            family_alloc[fam] = family_alloc.get(fam, 0.0) + float(p.get("size") or 0.0)
        cluster_alloc: Dict[int, float] = {}
        accepted_alphas: List[str] = []

        clusters = self._cluster_by_correlation(ranked)

        for c in ranked:
            size = cfg.capital * cfg.base_position_frac
            size *= c.survival_multiplier          # decaying edges get less
            # Capacity is a HARD limit — high EV never overrides liquidity
            if c.practical_capacity_usd is not None:
                size = min(size, c.practical_capacity_usd)
            # Correlated-cluster sizing: reduce, don't treat as independent
            cluster_id = clusters[c.candidate_id]
            n_in_cluster_accepted = sum(
                1 for a in accepted_alphas
                if clusters.get(a) == cluster_id
            )
            if n_in_cluster_accepted:
                size *= 0.5 ** n_in_cluster_accepted
                c.correlation_penalty = min(c.correlation_penalty,
                                            0.5 ** n_in_cluster_accepted)

            codes: List[str] = []
            if total_alloc + size > cfg.capital * cfg.max_total_exposure_frac:
                codes.append(ReasonCode.PORTFOLIO_CORRELATION_LIMIT)
                decisions.append(AllocationDecision(
                    c, False, reason_codes=codes, detail="total exposure limit"))
                continue
            if asset_alloc.get(c.symbol, 0.0) + size > cfg.capital * cfg.max_asset_exposure_frac:
                decisions.append(AllocationDecision(
                    c, False, reason_codes=[ReasonCode.PORTFOLIO_CORRELATION_LIMIT],
                    detail=f"asset exposure limit for {c.symbol}"))
                continue
            fam = family_of_alpha.get(c.alpha_id) or family_of(c.alpha_id.split(":")[0])
            if family_alloc.get(fam, 0.0) + size > cfg.capital * cfg.max_family_exposure_frac:
                decisions.append(AllocationDecision(
                    c, False, reason_codes=[ReasonCode.FAMILY_RISK_LIMIT],
                    detail=f"family {fam} exposure limit"))
                continue
            if cluster_alloc.get(cluster_id, 0.0) + size > cfg.capital * cfg.max_correlated_cluster_frac:
                decisions.append(AllocationDecision(
                    c, False, reason_codes=[ReasonCode.PORTFOLIO_CORRELATION_LIMIT],
                    detail="correlated cluster limit"))
                continue

            total_alloc += size
            asset_alloc[c.symbol] = asset_alloc.get(c.symbol, 0.0) + size
            family_alloc[fam] = family_alloc.get(fam, 0.0) + size
            cluster_alloc[cluster_id] = cluster_alloc.get(cluster_id, 0.0) + size
            accepted_alphas.append(c.candidate_id)
            decisions.append(AllocationDecision(c, True, allocation_usd=size))

        # 5. Hard risk-of-ruin gate on the accepted set (geometric-growth view)
        if alpha_return_matrix:
            decisions = self._enforce_ruin_limit_synchronized(
                decisions, alpha_return_matrix)
        elif alpha_return_samples:
            decisions = self._enforce_ruin_limit(decisions, alpha_return_samples)

        n_accept = sum(1 for d in decisions if d.accepted)
        if n_accept == 0:
            logger.info(
                "PortfolioAllocator: NO TRADE — no opportunity beats cash "
                f"({len(candidates)} candidates evaluated)"
            )
        return decisions

    # ── Conflict resolution (spec §47) ────────────────────────────────────────
    def _enforce_ruin_limit_synchronized(
        self, decisions: List[AllocationDecision],
        alpha_return_matrix: Dict[str, Dict[str, float]],
    ) -> List[AllocationDecision]:
        """Ruin gate on the SYNCHRONIZED portfolio path: alphas that lose on
        the same timestamps compound — concatenation would hide that."""
        from core.portfolio_paths import (
            build_return_matrix,
            portfolio_return_series,
            simulate_portfolio_paths,
        )
        cfg = self.config
        while True:
            accepted = [d for d in decisions if d.accepted]
            if not accepted:
                return decisions
            active = {d.candidate.alpha_id: alpha_return_matrix[d.candidate.alpha_id]
                      for d in accepted
                      if d.candidate.alpha_id in alpha_return_matrix}
            if not active:
                return decisions
            _, matrix = build_return_matrix(active)
            weights = {d.candidate.alpha_id: d.allocation_usd / cfg.capital
                       for d in accepted if cfg.capital > 0}
            series = portfolio_return_series(matrix, weights=weights)
            if len(series) < 20:
                return decisions
            sim = simulate_portfolio_paths(
                series, ruin_drawdowns=(cfg.ruin_drawdown,))
            ruin = sim["risk_of_ruin"][str(cfg.ruin_drawdown)]
            if ruin <= cfg.max_risk_of_ruin and \
                    sim.get("expected_log_growth", 0.0) > float("-inf"):
                return decisions

            def downside(d: AllocationDecision) -> float:
                col = alpha_return_matrix.get(d.candidate.alpha_id, {})
                return min(col.values()) if col else 0.0

            worst = min(accepted, key=downside)
            worst.accepted = False
            worst.allocation_usd = 0.0
            worst.reason_codes.append(ReasonCode.RISK_OF_RUIN_LIMIT)
            worst.detail = (f"synchronized portfolio ruin {ruin:.2%} > "
                            f"limit {cfg.max_risk_of_ruin:.2%}")
    def _enforce_ruin_limit(
        self, decisions: List[AllocationDecision],
        alpha_return_samples: Dict[str, List[float]],
    ) -> List[AllocationDecision]:
        """Hard gate: portfolio risk of ruin must stay below the limit AND
        expected log growth must be positive (geometric objective — spec §70).
        Removes the wildest accepted candidates (worst per-trade downside)
        until compliant — never accepts a ruinous portfolio for its upside."""
        from core.portfolio_robustness import expected_log_growth, risk_of_ruin

        cfg = self.config

        def portfolio_sample(active: List[AllocationDecision]) -> List[float]:
            pooled: List[float] = []
            for d in active:
                rets = alpha_return_samples.get(d.candidate.alpha_id) or []
                w = d.allocation_usd / cfg.capital if cfg.capital > 0 else 0.0
                pooled.extend(w * r for r in rets)
            return pooled

        while True:
            accepted = [d for d in decisions if d.accepted]
            sample = portfolio_sample(accepted)
            if not accepted or len(sample) < 20:
                return decisions
            ruin = risk_of_ruin(sample, ruin_drawdown=cfg.ruin_drawdown)
            log_growth = expected_log_growth(sample)
            if ruin["risk_of_ruin"] <= cfg.max_risk_of_ruin and log_growth > float("-inf"):
                return decisions

            def downside(d: AllocationDecision) -> float:
                rets = alpha_return_samples.get(d.candidate.alpha_id) or [0.0]
                return min(rets)

            worst = min(accepted, key=downside)
            worst.accepted = False
            worst.allocation_usd = 0.0
            worst.reason_codes.append(ReasonCode.RISK_OF_RUIN_LIMIT)
            worst.detail = (f"portfolio risk of ruin {ruin['risk_of_ruin']:.2%} > "
                            f"limit {cfg.max_risk_of_ruin:.2%}")


    def _resolve_conflicts(
        self, candidates: List[OpportunityCandidate]
    ) -> Tuple[List[OpportunityCandidate], List[AllocationDecision]]:
        """Same asset, opposing directions: pick a clear winner by conservative
        EV, or abstain entirely when evidence conflicts too strongly."""
        by_symbol: Dict[str, List[OpportunityCandidate]] = {}
        for c in candidates:
            by_symbol.setdefault(c.symbol, []).append(c)

        keep: List[OpportunityCandidate] = []
        rejected: List[AllocationDecision] = []
        for symbol, group in by_symbol.items():
            directions = {c.direction for c in group}
            if len(directions) <= 1:
                keep.extend(group)   # compatible independent signals coexist
                continue
            longs = [c for c in group if c.direction == "long"]
            shorts = [c for c in group if c.direction == "short"]
            long_ev = max((c.ev_lower_bound or 0.0) for c in longs) if longs else 0.0
            short_ev = max((c.ev_lower_bound or 0.0) for c in shorts) if shorts else 0.0
            margin = self.config.conflict_quality_margin
            if long_ev > short_ev * margin and long_ev > 0:
                winner_side, losers = longs, shorts
            elif short_ev > long_ev * margin and short_ev > 0:
                winner_side, losers = shorts, longs
            else:
                # Conflicting evidence too strong — abstain on this asset
                for c in group:
                    rejected.append(AllocationDecision(
                        c, False, reason_codes=[ReasonCode.SIGNAL_CONFLICT],
                        detail=f"conflicting long/short evidence on {symbol} — abstain"))
                continue
            keep.extend(winner_side)
            for c in losers:
                rejected.append(AllocationDecision(
                    c, False, reason_codes=[ReasonCode.SIGNAL_CONFLICT],
                    detail=f"lost conflict resolution on {symbol}"))
        return keep, rejected

    # ── Correlation clustering ────────────────────────────────────────────────

    def _cluster_by_correlation(
        self, candidates: List[OpportunityCandidate]
    ) -> Dict[str, int]:
        """Union-find clusters over the alpha-correlation matrix; candidates on
        the same symbol+direction are trivially clustered too."""
        parent: Dict[str, str] = {c.candidate_id: c.candidate_id for c in candidates}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            parent[find(a)] = find(b)

        for i, a in enumerate(candidates):
            for b in candidates[i + 1:]:
                corr = self.alpha_correlations.get((a.alpha_id, b.alpha_id)) \
                    or self.alpha_correlations.get((b.alpha_id, a.alpha_id))
                dcorr = self.downside_correlations.get((a.alpha_id, b.alpha_id)) \
                    or self.downside_correlations.get((b.alpha_id, a.alpha_id))
                eff = max(c for c in (corr, dcorr) if c is not None) \
                    if (corr is not None or dcorr is not None) else None
                same_exposure = a.symbol == b.symbol and a.direction == b.direction
                if same_exposure or (eff is not None and eff >= self.config.correlation_threshold):
                    union(a.candidate_id, b.candidate_id)

        roots: Dict[str, int] = {}
        out: Dict[str, int] = {}
        for c in candidates:
            r = find(c.candidate_id)
            if r not in roots:
                roots[r] = len(roots)
            out[c.candidate_id] = roots[r]
        return out


def alpha_return_correlations(
    db_path: str = "data/trade_memory.sqlite",
    lookback_days: int = 90,
) -> Dict[Tuple[str, str], float]:
    """Rolling correlation between attributed ALPHA return streams (spec §44)."""
    import sqlite3
    from collections import defaultdict
    from core.strategy_correlation import _pearson

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT ta.alpha_id, substr(tm.exit_time,1,10), SUM(tm.net_pnl) "
                "FROM trade_attribution ta JOIN trade_memory tm ON tm.id=ta.trade_memory_id "
                "WHERE tm.exit_time IS NOT NULL GROUP BY ta.alpha_id, substr(tm.exit_time,1,10)"
            ).fetchall()
    except sqlite3.OperationalError:
        return {}
    streams: Dict[str, Dict[str, float]] = defaultdict(dict)
    for alpha_id, day, pnl in rows:
        streams[alpha_id][day] = float(pnl or 0.0)
    names = sorted(streams)
    out: Dict[Tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            days = sorted(set(streams[a]) & set(streams[b]))[-lookback_days:]
            if len(days) < 10:
                continue
            corr = _pearson([streams[a][d] for d in days], [streams[b][d] for d in days])
            if corr is not None:
                out[(a, b)] = corr
                out[(b, a)] = corr
    return out
