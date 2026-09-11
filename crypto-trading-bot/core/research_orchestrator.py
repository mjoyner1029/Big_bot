"""AutonomousResearchOrchestrator — ONE research loop wiring every discovery
component into the existing campaign/validation funnel (spec Priority Zero).

    UPDATE MARKET DATA → UPDATE EXTERNAL INTELLIGENCE → POINT-IN-TIME FEATURES
    → CROSS-SECTIONAL SCAN → RETURN DECOMPOSITION → OUTLIERS → CHANGE POINTS
    → KNOWN DETECTOR FAMILIES → STRUCTURED HYPOTHESES → SEARCH LEDGER
    → CHEAP PRELIMINARY TESTS → FULL VALIDATION → ALPHA LIBRARY → PAPER

No new validation framework is created: generated hypotheses are wrapped as a
DiscoveryDetector so they flow through ResearchCampaignRunner's existing FDR /
dedup / holdout machinery. External-source failures never stop unrelated
research (fail-soft per source, health recorded).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core.discovery_detectors import (
    DiscoveredHypothesis,
    DiscoveryDetector,
    SearchLimits,
    default_detectors,
)
from core.hypothesis_generator import (
    AutonomousHypothesisGenerator,
    GeneratorBudget,
    breadth_adjusted_alpha,
)
from core.market_neutral import BasketRelativeValueDetector, PairsRelativeValueDetector
from core.research_campaign import CampaignConfig, ResearchCampaignRunner
from core.search_ledger import SearchLedger, research_roi_scores
from core.universal_discovery import (
    BenchmarkResolver,
    CrossSectionalOpportunityScanner,
    MarketChangePointDetector,
    MarketOutlierDetector,
    ReturnDecompositionEngine,
)

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


# ── Generated-hypothesis adapter ──────────────────────────────────────────────


class GeneratedHypothesisDetector(DiscoveryDetector):
    """Adapts AutonomousHypothesisGenerator output into the standard detector
    interface: each structured spec is backtested on discovery data to build a
    real conditioned return sample, then emitted through the SAME `_emit`
    evidence gates (min sample, sign test, family budget) as any detector."""

    family = "UNIVERSAL"

    def __init__(self, specs: Sequence[Dict[str, Any]],
                 limits: Optional[SearchLimits] = None, tracker=None) -> None:
        super().__init__(limits, tracker)
        self.specs = list(specs)

    def scan(self, data: Dict[str, pd.DataFrame]) -> List[DiscoveredHypothesis]:
        out: List[DiscoveredHypothesis] = []
        for spec in self.specs:
            df = data.get(spec.get("symbol", ""))
            if df is None or len(df) < 60:
                continue
            rets = self._conditioned_returns(spec, df)
            if rets is None or rets.empty:
                continue
            h = self._emit(
                symbol=spec["symbol"],
                subfamily=f"{spec['family']}:{spec['subfamily']}"[:60].lower(),
                direction=spec.get("direction", "long"),
                conditions=spec.get("entry_conditions", []),
                returns=rets,
                holding_bars=int(spec.get("holding_bars", 5)),
                description=spec.get("rationale", ""),
                anomaly_score=spec.get("anomaly_score"),
                origin="universal_discovery")
            if h:
                out.append(h)
        return out

    def _conditioned_returns(self, spec: Dict[str, Any],
                             df: pd.DataFrame) -> Optional[pd.Series]:
        """Vectorized historical evaluation of the structured conditions.
        Unsupported/context conditions → skip the spec (never guess)."""
        work = df.copy()
        work.columns = [str(c).lower() for c in work.columns]
        if "close" not in work.columns:
            return None
        close = pd.to_numeric(work["close"], errors="coerce")
        hold = int(spec.get("holding_bars", 5))

        sub = str(spec.get("subfamily", ""))
        if sub == "overnight" and "open" in work.columns:
            o = pd.to_numeric(work["open"], errors="coerce")
            rets = (o / close.shift(1) - 1).dropna()
            return rets.iloc[:-1] if len(rets) > 1 else None

        mask = pd.Series(True, index=work.index)
        for cond in spec.get("entry_conditions", []):
            feat, op, val = cond.get("feature"), cond.get("op"), cond.get("value")
            series = self._feature_series(feat, work, close)
            if series is None:
                if feat in ("market_regime", "asset_class", "sector"):
                    continue        # context passthrough — no historical veto
                return None         # can't evaluate honestly → don't fabricate
            if op == ">":
                mask &= series > float(val)
            elif op == "<":
                mask &= series < float(val)
            elif op == "==":
                mask &= series.astype(str).str.upper() == str(val).upper()
            elif op == "!=":
                mask &= series.astype(str).str.upper() != str(val).upper()
            else:
                return None

        fwd = close.pct_change(hold).shift(-hold)
        rets = fwd[mask & fwd.notna()]
        return rets if len(rets) else None

    @staticmethod
    def _feature_series(feature: str, work: pd.DataFrame,
                        close: pd.Series) -> Optional[pd.Series]:
        if feature and feature.startswith("returns_") and feature.endswith("d"):
            try:
                bars = int(feature[len("returns_"):-1])
            except ValueError:
                return None
            return close.pct_change(bars)
        if feature == "relative_volume_20d" and "volume" in work.columns:
            v = pd.to_numeric(work["volume"], errors="coerce")
            base = v.rolling(20).mean()
            return v / base.replace(0, np.nan)
        if feature == "day_of_week" and isinstance(work.index, pd.DatetimeIndex):
            return pd.Series(work.index.strftime("%A").str.upper(), index=work.index)
        return None


# ── Orchestrator ──────────────────────────────────────────────────────────────


@dataclass
class OrchestratorConfig:
    campaign: CampaignConfig = field(default_factory=CampaignConfig)
    generator_budget: GeneratorBudget = field(default_factory=GeneratorBudget)
    include_market_neutral: bool = True
    include_default_detectors: bool = True
    external_since_days: int = 90
    fdr_base_alpha: float = 0.05
    # Research scale (spec §35): campaign batching handles large universes
    daily_incremental_max_instruments: int = 150
    weekly_full_max_instruments: int = 500
    monthly_deep_max_instruments: int = 2500
    gap_family_budget_boost: float = 2.0
    # Exploration vs exploitation of research compute (spec §54-55)
    research_exploitation_fraction: float = 0.8
    research_exploration_fraction: float = 0.2


class AutonomousResearchOrchestrator:
    """Coordinates the full research loop. Extends — never replaces — the
    existing ResearchCampaignRunner."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite",
                 config: Optional[OrchestratorConfig] = None,
                 source_registry=None,
                 alpha_library=None,
                 sector_of: Optional[Callable[[str], Optional[str]]] = None,
                 peer_basket_fn: Optional[Callable[[str], List[str]]] = None) -> None:
        self.db_path = db_path
        self.config = config or OrchestratorConfig()
        self.ledger = SearchLedger(db_path)
        self.registry = source_registry
        self.alpha_library = alpha_library
        self.scanner = CrossSectionalOpportunityScanner()
        self.decomposer = ReturnDecompositionEngine()
        self.outliers = MarketOutlierDetector()
        self.change_points = MarketChangePointDetector()
        if sector_of is None or peer_basket_fn is None:
            resolver = BenchmarkResolver()
            sector_of = sector_of or (
                lambda s: getattr(resolver.universe.get(s), "sector", None))
            peer_basket_fn = peer_basket_fn or resolver.peer_basket
        self._sector_of = sector_of
        self._peer_basket_fn = peer_basket_fn
        # component invocation counters — wiring is observable and testable
        self.invocations: Dict[str, int] = {}

    def _mark(self, component: str) -> None:
        self.invocations[component] = self.invocations.get(component, 0) + 1

    # ── Research-gap direction (spec: gaps steer BUDGETS, not thresholds) ─────

    def apply_research_gaps(
        self,
        alpha_regime_returns: Dict[str, List[tuple]],
        alpha_families: Optional[Dict[str, str]] = None,
        alpha_directions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Analyze portfolio gaps and boost hypothesis budgets for missing
        families/directions. Validation thresholds are untouched."""
        from core.portfolio_robustness import (
            ResearchGapDirector,
            regime_coverage_matrix,
        )
        self._mark("research_gap_direction")
        matrix = regime_coverage_matrix(alpha_regime_returns)
        analysis = ResearchGapDirector().analyze(
            matrix, alpha_families=alpha_families,
            alpha_directions=alpha_directions)
        budget = self.config.generator_budget
        boost = self.config.gap_family_budget_boost
        base = budget.max_hypotheses_per_family
        for gap in analysis["gaps"]:
            if gap.startswith(("REGIME_COVERAGE_GAP:BEAR",
                               "REGIME_COVERAGE_GAP:HIGH_VOL",
                               "FAMILY_GAP:MARKET_NEUTRAL")):
                budget.per_family_overrides["MARKET_NEUTRAL"] = int(base * boost)
                budget.per_family_overrides["STRUCTURAL_CRYPTO"] = int(base * boost)
            if gap.startswith("DIRECTION_GAP:short"):
                budget.per_family_overrides["MOMENTUM"] = int(base * boost)
        analysis["family_budget_overrides"] = dict(budget.per_family_overrides)
        self.last_gap_analysis = analysis
        self._persist_research_budgets(budget.per_family_overrides)
        return analysis

    def _persist_research_budgets(self, overrides: Dict[str, int]) -> None:
        """Gap-directed budgets survive to the NEXT campaign (spec §52)."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS research_budgets (family TEXT "
                "PRIMARY KEY, budget INTEGER, updated_at TEXT)")
            for family, cap in overrides.items():
                conn.execute(
                    "INSERT INTO research_budgets (family, budget, updated_at) "
                    "VALUES (?,?,datetime('now')) ON CONFLICT(family) DO UPDATE "
                    "SET budget=excluded.budget, updated_at=excluded.updated_at",
                    (family, int(cap)))

    def _load_research_budgets(self) -> Dict[str, int]:
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT family, budget FROM research_budgets").fetchall()
            return {r[0]: int(r[1]) for r in rows}
        except sqlite3.OperationalError:
            return {}

    # ── Step 1-2: data + external intelligence refresh ────────────────────────

    def refresh_external(self, since_iso: Optional[str] = None) -> Dict[str, int]:
        """Deterministic source refresh; per-source failures are isolated and
        recorded in source_health by the registry (fail-soft)."""
        self._mark("external_refresh")
        if self.registry is None:
            return {}
        since = since_iso or (
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0).isoformat())
        try:
            return self.registry.refresh_all(since)
        except Exception as e:                              # never stop research
            logger.warning(f"external refresh failed soft: {e}")
            return {}

    def external_feature_rows(self, symbols: Sequence[str],
                              as_of_iso: Optional[str] = None
                              ) -> Dict[str, Dict[str, Any]]:
        """Point-in-time features per symbol from the raw event store."""
        self._mark("external_features")
        if self.registry is None or getattr(self.registry, "event_store", None) is None:
            return {}
        from data.external_sources import (
            WalletIntelligenceEngine,
            congressional_features,
        )
        as_of = as_of_iso or _utcnow()
        store = self.registry.event_store
        rows: Dict[str, Dict[str, Any]] = {}
        wallet_engine = WalletIntelligenceEngine(store)
        for symbol in symbols:
            feats: Dict[str, Any] = {}
            try:
                feats.update(congressional_features(store, symbol, as_of))
            except Exception as e:
                logger.debug(f"congressional features failed for {symbol}: {e}")
            try:
                if symbol.endswith("-USD"):
                    feats.update(wallet_engine.positioning_features(symbol, as_of))
            except Exception as e:
                logger.debug(f"wallet features failed for {symbol}: {e}")
            if feats:
                rows[symbol] = feats
        return rows

    # ── Step 3-8: universal discovery ─────────────────────────────────────────

    def discover(self, data: Dict[str, pd.DataFrame],
                 external_rows: Optional[Dict[str, Dict[str, Any]]] = None
                 ) -> Dict[str, Any]:
        """Run every discovery engine and generate structured hypotheses."""
        self._mark("cross_sectional_scan")
        rows = self.scanner.scan(data)

        self._mark("return_decomposition")
        decomps = []
        for symbol, df in data.items():
            d = self.decomposer.decompose(symbol, df)
            if d and d.extreme_flags:
                decomps.append(d)

        self._mark("change_point_detection")
        cps = []
        for symbol, df in data.items():
            cps.extend(self.change_points.detect(symbol, df))

        self._mark("outlier_detection")
        feature_matrix = {
            r.symbol: {
                "returns_20d": r.returns.get(20),
                "risk_adjusted_20d": r.risk_adjusted_20d,
                "volume_acceleration": r.volume_acceleration,
                "volatility_change": r.volatility_change,
            } for r in rows}
        outliers = self.outliers.detect(feature_matrix)

        self._mark("hypothesis_generation")
        gen = AutonomousHypothesisGenerator(self.config.generator_budget)
        specs: List[Dict[str, Any]] = []
        specs += gen.from_cross_sectional(rows)
        for d in decomps:
            specs += gen.from_return_decomposition(d)
        specs += gen.from_outliers_and_changepoints(outliers, cps)
        if external_rows:
            specs += gen.from_external_events(external_rows)

        return {
            "cross_sectional_rows": rows,
            "anomaly_flags": {r.symbol: r.anomaly_flags for r in rows
                              if r.anomaly_flags},
            "decompositions": decomps,
            "change_points": cps,
            "outliers": outliers,
            "generated_specs": specs,
        }

    # ── Steps 9-14: ledger → campaign → validation → library → paper ─────────

    def run_campaign(self, data: Dict[str, pd.DataFrame],
                     campaign_id: Optional[str] = None,
                     since_iso: Optional[str] = None) -> Dict[str, Any]:
        started = time.time()
        # Persisted gap-directed budgets steer THIS campaign (spec §52);
        # exploration floor keeps under-explored families alive (spec §55)
        persisted = self._load_research_budgets()
        if persisted:
            base = self.config.generator_budget.max_hypotheses_per_family
            floor = int(base * self.config.research_exploration_fraction)
            self.config.generator_budget.per_family_overrides.update(
                {f: max(b, floor) for f, b in persisted.items()})
        refresh_counts = self.refresh_external(since_iso)
        external_rows = self.external_feature_rows(list(data), as_of_iso=None)
        discovery = self.discover(data, external_rows)
        specs = discovery["generated_specs"]

        # Search-breadth registration BEFORE validation so FDR knows the truth
        by_source: Dict[str, int] = {}
        for s in specs:
            by_source[s["family"]] = by_source.get(s["family"], 0) + 1
        for source, n in by_source.items():
            self.ledger.record_search(source, n, campaign_id=campaign_id,
                                      detector="AutonomousHypothesisGenerator")
        total_breadth = max(self.ledger.total_breadth(), len(specs), 1)
        effective_alpha = breadth_adjusted_alpha(
            self.config.fdr_base_alpha, total_breadth)

        # Campaign config inherits the breadth-adjusted FDR alpha
        cfg = self.config.campaign
        cfg.fdr_alpha = min(cfg.fdr_alpha, effective_alpha)

        def detectors_factory(limits=None, tracker=None):
            dets: List[DiscoveryDetector] = []
            if self.config.include_default_detectors:
                dets.extend(default_detectors(limits=limits, tracker=tracker))
            dets.append(GeneratedHypothesisDetector(specs, limits=limits,
                                                    tracker=tracker))
            if self.config.include_market_neutral:
                dets.append(PairsRelativeValueDetector(
                    limits=limits, tracker=tracker, sector_of=self._sector_of))
                dets.append(BasketRelativeValueDetector(
                    limits=limits, tracker=tracker,
                    peer_basket_fn=self._peer_basket_fn))
            return dets

        self._mark("campaign_validation")
        runner = ResearchCampaignRunner(
            self.db_path, config=cfg,
            detectors_factory=detectors_factory,
            alpha_library=self.alpha_library)
        report = runner.run(data, campaign_id=campaign_id)

        # Ledger: detector-side breadth + outcomes
        detector_tests = report.get("hypotheses_tested", 0)
        if detector_tests:
            self.ledger.record_search("PRICE", detector_tests,
                                      campaign_id=report.get("campaign_id"),
                                      detector="campaign_detectors")
        promoted = report.get("new_alphas", []) or []
        self.ledger.record_outcome(
            "UNIVERSAL", "validated", len(promoted),
            compute_seconds=time.time() - started)

        return self._universal_report(report, discovery, refresh_counts,
                                      external_rows, effective_alpha,
                                      total_breadth, started)

    def _universal_report(self, campaign_report, discovery, refresh_counts,
                          external_rows, effective_alpha, total_breadth,
                          started) -> Dict[str, Any]:
        rej = campaign_report.get("rejections", {}) or {}
        return {
            "universal_research_report": True,
            "generated_at": _utcnow(),
            "elapsed_seconds": round(time.time() - started, 1),
            "campaign": campaign_report,
            "universe_count": campaign_report.get("universe_scanned"),
            "external_events_refreshed": refresh_counts,
            "external_feature_symbols": len(external_rows or {}),
            "anomalies_by_family": {
                "cross_sectional": len(discovery["anomaly_flags"]),
                "return_decomposition": len(discovery["decompositions"]),
                "change_points": len(discovery["change_points"]),
                "outliers": len(discovery["outliers"]),
            },
            "hypotheses_generated_universal": len(discovery["generated_specs"]),
            "search_breadth_total": total_breadth,
            "breadth_adjusted_fdr_alpha": effective_alpha,
            "search_ledger": self.ledger.report(),
            "rejections": rej,
            "new_paper_alphas": campaign_report.get("new_alphas", []),
            "component_invocations": dict(self.invocations),
            "research_roi": research_roi_scores(self.ledger),
        }
