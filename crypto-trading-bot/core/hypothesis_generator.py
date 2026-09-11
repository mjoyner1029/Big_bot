"""Autonomous hypothesis generation with aggressive anti-overfitting controls.

Converts anomalies (outliers, change points, relative-performance extremes,
return decompositions, external-intelligence events) into MACHINE-READABLE
hypotheses for the existing validation stack. Claude may propose
interpretations, but only structured DSL conditions are ever tested.

Anti-overfitting (spec §46-51):
  - hierarchical staged search with hard budgets
  - campaign-wide search-breadth accounting → breadth-adjusted FDR alpha
  - complexity penalty (MDL-flavored: conditions + parameters + precision)
  - feature ablation: every feature must earn incremental value
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.discovery_detectors import DiscoveredHypothesis
from core.universal_discovery import (
    ChangePoint,
    CrossSectionalRow,
    Outlier,
    ReturnDecomposition,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneratorBudget:
    max_hypotheses_total: int = 2000
    max_hypotheses_per_family: int = 300
    max_external_feature_hypotheses: int = 500
    max_features_per_hypothesis: int = 3
    max_interaction_depth: int = 2
    # Research-gap direction: per-family boosts/cuts (spec: gaps modify
    # research BUDGETS, never validation thresholds)
    per_family_overrides: Dict[str, int] = field(default_factory=dict)

    def family_cap(self, family: str) -> int:
        return self.per_family_overrides.get(family, self.max_hypotheses_per_family)


class AutonomousHypothesisGenerator:
    """Anomaly → structured hypothesis. Emits condition-DSL specs compatible
    with the campaign; NOTHING here is a trade signal (AnomalyScore is not
    OpportunityScore)."""

    def __init__(self, budget: Optional[GeneratorBudget] = None) -> None:
        self.budget = budget or GeneratorBudget()
        self._emitted = 0
        self._per_family: Dict[str, int] = {}

    def _allow(self, family: str) -> bool:
        if self._emitted >= self.budget.max_hypotheses_total:
            return False
        if self._per_family.get(family, 0) >= self.budget.family_cap(family):
            return False
        self._emitted += 1
        self._per_family[family] = self._per_family.get(family, 0) + 1
        return True

    def from_return_decomposition(
        self, decomp: ReturnDecomposition) -> List[Dict[str, Any]]:
        """EXTREME_OVERNIGHT_CONTRIBUTION → overnight-holding hypothesis, etc.
        This is what turns a Micron-style observation into a testable spec."""
        out = []
        for flag in decomp.extreme_flags:
            if not self._allow("TEMPORAL"):
                break
            if flag == "EXTREME_OVERNIGHT_CONTRIBUTION":
                out.append({
                    "family": "TEMPORAL", "subfamily": "overnight",
                    "symbol": decomp.symbol, "direction": "long",
                    "entry_conditions": [],   # unconditional close→open
                    "holding_bars": 1,
                    "rationale": f"{decomp.overnight_contribution:.0%} of "
                                 f"long-run log return earned overnight",
                    "anomaly_score": decomp.overnight_contribution,
                })
            elif flag.startswith("EXTREME_") and flag.endswith("_CONTRIBUTION") \
                    and flag not in ("EXTREME_INTRADAY_CONTRIBUTION",):
                day = flag.replace("EXTREME_", "").replace("_CONTRIBUTION", "")
                if day in ("MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
                           "FRIDAY", "SATURDAY", "SUNDAY"):
                    out.append({
                        "family": "TEMPORAL", "subfamily": "day_of_week",
                        "symbol": decomp.symbol, "direction": "long",
                        "entry_conditions": [{"feature": "day_of_week",
                                              "op": "==", "value": day}],
                        "holding_bars": 1,
                        "rationale": f"return concentrated on {day}",
                        "anomaly_score": decomp.weekday_contributions.get(day, 0),
                    })
        return out

    # Predeclared threshold grid (spec: never derive thresholds from one
    # observed asset's exact value — test the percentile FAMILY instead)
    RETURN_PERCENTILE_GRID = (0.80, 0.90, 0.95, 0.99)

    def from_cross_sectional(self, rows: Sequence[CrossSectionalRow]
                             ) -> List[Dict[str, Any]]:
        out = []
        # percentile-family thresholds computed across the WHOLE universe
        all_20d = sorted(r.returns[20] for r in rows if 20 in r.returns)

        def pct_threshold(q: float) -> float:
            if not all_20d:
                return 0.0
            i = min(int(q * len(all_20d)), len(all_20d) - 1)
            return round(all_20d[i], 4)

        for row in rows:
            for flag in row.anomaly_flags:
                if not self._allow("MOMENTUM"):
                    return out
                if flag.startswith("EXTREME_RELATIVE_OUTPERFORMANCE"):
                    # test the percentile family, not the observed value
                    for q in self.RETURN_PERCENTILE_GRID:
                        if not self._allow("MOMENTUM"):
                            return out
                        out.append({
                            "family": "MOMENTUM", "subfamily": "relative_strength",
                            "symbol": row.symbol, "direction": "long",
                            "entry_conditions": [
                                {"feature": "returns_20d", "op": ">",
                                 "value": pct_threshold(q)}],
                            "holding_bars": 20,
                            "rationale": f"{flag}: universe {q:.0%} percentile family",
                            "anomaly_score": row.return_percentiles.get(20, 0),
                        })
                elif flag.startswith("EXTREME_UNDERPERFORMANCE"):
                    # short-side symmetry (spec §51)
                    for q in self.RETURN_PERCENTILE_GRID:
                        if not self._allow("MOMENTUM"):
                            return out
                        out.append({
                            "family": "MOMENTUM", "subfamily": "relative_weakness",
                            "symbol": row.symbol, "direction": "short",
                            "entry_conditions": [
                                {"feature": "returns_20d", "op": "<",
                                 "value": pct_threshold(1 - q)}],
                            "holding_bars": 20,
                            "rationale": f"{flag}: universe {1 - q:.0%} percentile family",
                            "anomaly_score": 1 - row.return_percentiles.get(20, 1),
                        })
                elif flag == "MOMENTUM_ACCELERATION":
                    out.append({
                        "family": "MOMENTUM", "subfamily": "acceleration",
                        "symbol": row.symbol, "direction": "long",
                        "entry_conditions": [
                            {"feature": "returns_5d", "op": ">", "value": 0.0}],
                        "holding_bars": 5,
                        "rationale": "momentum acceleration percentile extreme",
                        "anomaly_score": row.momentum_acceleration_score or 0,
                    })
        return out

    def from_outliers_and_changepoints(
        self, outliers: Sequence[Outlier],
        change_points: Sequence[ChangePoint]) -> List[Dict[str, Any]]:
        out = []
        cp_symbols = {c.symbol for c in change_points}
        for o in outliers:
            if o.symbol not in cp_symbols:
                continue   # stage-2 interaction: outlier AND structural change
            if not self._allow("EVENT"):
                break
            out.append({
                "family": "EVENT", "subfamily": f"outlier_{o.feature}"[:40],
                "symbol": o.symbol, "direction": "long" if o.kind == "high" else "short",
                "entry_conditions": [
                    {"feature": "relative_volume_20d", "op": ">", "value": 1.5}],
                "holding_bars": 5,
                "rationale": f"{o.feature} robust-z={o.robust_z:.1f} with "
                             "coincident change point",
                "anomaly_score": abs(o.robust_z) / 10,
            })
        return out

    def from_external_events(self, feature_rows: Dict[str, Dict[str, Any]]
                             ) -> List[Dict[str, Any]]:
        """External-intelligence features (government/congress/wallets) →
        hypotheses, capped by the external budget. Features only — the
        conditions reference validated FeatureRegistry context names."""
        out = []
        emitted = 0
        for symbol, feats in feature_rows.items():
            if emitted >= self.budget.max_external_feature_hypotheses:
                break
            if feats.get("new_federal_award") and \
                    (feats.get("award_amount_vs_market_cap") or 0) > 0.01:
                if self._allow("EVENT"):
                    emitted += 1
                    out.append({
                        "family": "EVENT", "subfamily": "federal_award",
                        "symbol": symbol, "direction": "long",
                        "entry_conditions": [
                            {"feature": "market_regime", "op": "!=", "value": "PANIC"}],
                        "holding_bars": 60,
                        "rationale": "material federal award vs market cap "
                                     f"({feats['award_amount_vs_market_cap']:.1%})",
                        "anomaly_score": min(
                            feats["award_amount_vs_market_cap"] * 10, 1.0),
                    })
            if feats.get("clustered_buying"):
                if self._allow("EVENT"):
                    emitted += 1
                    out.append({
                        "family": "EVENT", "subfamily": "congress_cluster",
                        "symbol": symbol, "direction": "long",
                        "entry_conditions": [],
                        "holding_bars": 60,
                        "rationale": f"{feats.get('unique_members_buying')} members "
                                     "disclosed buys within 30d (post-disclosure)",
                        "anomaly_score": min(
                            feats.get("unique_members_buying", 0) / 10, 1.0),
                    })
            swp = feats.get("skill_weighted_positioning")
            if swp is not None and abs(swp) > 0.6:
                if self._allow("STRUCTURAL_CRYPTO"):
                    emitted += 1
                    out.append({
                        "family": "STRUCTURAL_CRYPTO", "subfamily": "smart_wallets",
                        "symbol": symbol,
                        "direction": "long" if swp > 0 else "short",
                        "entry_conditions": [],
                        "holding_bars": 1,
                        "rationale": f"skill-weighted positioning {swp:+.2f}",
                        "anomaly_score": abs(swp),
                    })
        return out


# ── Complexity penalty / MDL (spec §50-51) ────────────────────────────────────


def hypothesis_complexity(spec: Dict[str, Any]) -> float:
    """Description-length proxy: conditions + numeric parameters + precision.
    Rare conjunctions and precise thresholds cost more."""
    conditions = spec.get("entry_conditions") or []
    cost = 1.0 + len(conditions)
    for c in conditions:
        v = c.get("value")
        if isinstance(v, float):
            decimals = len(str(v).split(".")[-1]) if "." in str(v) else 0
            cost += 0.25 * min(decimals, 4)     # precise thresholds cost more
        if c.get("op") == "BETWEEN":
            cost += 0.5
    return cost


def complexity_adjusted_score(oos_sharpe: float, spec: Dict[str, Any],
                              penalty_per_unit: float = 0.05) -> float:
    """Prefer the simpler strategy when performance is similar: 2 conditions
    at Sharpe 1.5 beats 11 conditions at 1.55."""
    return oos_sharpe - penalty_per_unit * hypothesis_complexity(spec)


# ── Feature ablation (spec §49) ───────────────────────────────────────────────


def feature_ablation(
    conditions: List[Dict[str, Any]],
    evaluate_fn: Callable[[List[Dict[str, Any]]], float],
    min_incremental: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate full strategy vs each leave-one-out variant. Features whose
    removal does NOT hurt performance contribute nothing and are flagged."""
    full = evaluate_fn(conditions)
    contributions = {}
    non_contributing = []
    for i, cond in enumerate(conditions):
        reduced = conditions[:i] + conditions[i + 1:]
        without = evaluate_fn(reduced)
        incremental = full - without
        key = f"{cond.get('feature')}{cond.get('op')}{cond.get('value')}"
        contributions[key] = incremental
        if incremental <= min_incremental:
            non_contributing.append(key)
    return {
        "full_performance": full,
        "incremental_alpha_contribution": contributions,
        "non_contributing_features": non_contributing,
        "recommended_conditions": [
            c for c in conditions
            if f"{c.get('feature')}{c.get('op')}{c.get('value')}"
            not in non_contributing],
    }


# ── Search-breadth-adjusted significance (spec §47) ───────────────────────────


def breadth_adjusted_alpha(base_alpha: float, total_hypotheses: int,
                           reference_breadth: int = 100) -> float:
    """More search breadth → stricter FDR alpha. Testing a million
    relationships must not enjoy the same threshold as testing a hundred:
        alpha_eff = base / (1 + log10(breadth / reference))
    """
    if total_hypotheses <= reference_breadth:
        return base_alpha
    factor = 1.0 + math.log10(total_hypotheses / reference_breadth)
    return base_alpha / factor


# Spec §19 name: AlphaComplexityScore
alpha_complexity_score = hypothesis_complexity
