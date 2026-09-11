"""Autonomous research campaign.

    UNIVERSE → FEATURES → DETECTORS → HYPOTHESES → CHEAP FILTER → FDR
    → AlphaValidationPipeline (OOS, walk-forward, purge/embargo, parameter
      robustness, concentration, regime, cost stress, Monte Carlo, deflated
      Sharpe, extreme-return sanity, final holdout)
    → ALPHA LIBRARY → PAPER CAMPAIGN

Promotion happens ONLY through AlphaValidationPipeline PASS — the runner has
no path that creates a PAPER alpha from partial validation. Campaigns are
batched (campaign-global FDR/dedup/graveyard context) and checkpoint/resumable.
Zero promotions is a valid outcome.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from core.alpha_library import AlphaLibrary, AlphaState
from core.alpha_validation import (
    AlphaValidationPipeline,
    AlphaValidationResult,
    ValidationArtifactStore,
    ValidationConfig,
    family_signature,
    parameter_signature,
)
from core.data_quality import DataQualityMonitor
from core.holdout_manager import HoldoutManager
from core.discovery_detectors import (
    DiscoveredHypothesis,
    SearchLimits,
    default_detectors,
)
from core.validation_stats import (
    HypothesisFamilyTracker,
    TestedHypothesis,
    benjamini_hochberg,
    profit_concentration,
)

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


class RejectReason:
    LOW_EFFECTIVE_SAMPLE = "LOW_EFFECTIVE_SAMPLE"
    LOW_EFFECT_SIZE = "LOW_EFFECT_SIZE"
    FAILED_FDR = "FAILED_FDR"
    FAILED_FULL_VALIDATION = "FAILED_FULL_VALIDATION"
    OUTLIER_DEPENDENT = "OUTLIER_DEPENDENT"
    DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
    GRAVEYARD_DUPLICATE = "GRAVEYARD_DUPLICATE"
    ALREADY_PROMOTED = "ALREADY_PROMOTED"
    NEAR_DUPLICATE = "NEAR_DUPLICATE"
    SURVIVORSHIP_BIAS_RISK = "SURVIVORSHIP_BIAS_RISK"
    VALIDATION_BUDGET_EXCEEDED = "VALIDATION_BUDGET_EXCEEDED"


@dataclass
class CampaignConfig:
    # Data splits: discovery / independent OOS / final untouched holdout
    discovery_fraction: float = 0.6
    oos_fraction: float = 0.2                 # remainder is holdout
    min_holdout_bars: int = 60                # below this → HOLDOUT_UNAVAILABLE
    data_timeframe: str = "1d"
    # Cheap preliminary filter
    min_gross_effect: float = 0.0005
    max_concentration_top5: float = 0.85
    # FDR
    fdr_alpha: float = 0.05
    # Compute budgets
    instrument_batch_size: int = 100
    max_hypotheses_per_campaign: int = 20_000
    max_validation_candidates: int = 200      # deep validation budget
    limits: SearchLimits = field(
        default_factory=lambda: SearchLimits(max_hypotheses_per_family=2000))
    oos_limits: SearchLimits = field(
        default_factory=lambda: SearchLimits(minimum_effective_sample=8,
                                             max_hypotheses_per_family=2000))
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    seed: int = 42
    code_version: str = "campaign_v2"
    feature_version: str = "1.0.0"


_CREATE_GRAVEYARD = """
CREATE TABLE IF NOT EXISTS campaign_hypotheses (
    signature           TEXT NOT NULL,
    family_signature    TEXT,
    parameter_signature TEXT,
    campaign_id         TEXT NOT NULL,
    outcome             TEXT NOT NULL,      -- PROMOTED | REJECTED
    failure_stage       TEXT,
    reason              TEXT,
    mean_return         REAL,
    p_value             REAL,
    tested_at           TEXT NOT NULL
)
"""

_CREATE_CHECKPOINTS = """
CREATE TABLE IF NOT EXISTS campaign_checkpoints (
    campaign_id      TEXT NOT NULL,
    batch_index      INTEGER NOT NULL,
    hypotheses_json  TEXT,
    tracker_json     TEXT,
    completed_at     TEXT NOT NULL,
    PRIMARY KEY (campaign_id, batch_index)
)
"""


def _signature(h: DiscoveredHypothesis) -> str:
    """Backward-compatible identity — the exact parameter version."""
    return parameter_signature(h)


def _hyp_to_json(h: DiscoveredHypothesis) -> Dict:
    return {
        "family": h.family, "subfamily": h.subfamily, "symbol": h.symbol,
        "direction": h.direction, "entry_conditions": h.entry_conditions,
        "holding_bars": h.holding_bars, "sample_size": h.sample_size,
        "effective_sample": h.effective_sample, "mean_return": h.mean_return,
        "p_value": h.p_value, "description": h.description,
        "hypothesis_id": h.hypothesis_id, "metadata": h.metadata,
    }


def _hyp_from_json(d: Dict) -> DiscoveredHypothesis:
    return DiscoveredHypothesis(**d)


class ResearchCampaignRunner:
    """Runs one full discovery→validation→alpha-library campaign."""

    def __init__(
        self,
        db_path: str = "data/trade_memory.sqlite",
        config: Optional[CampaignConfig] = None,
        detectors_factory: Optional[Callable] = None,
        alpha_library: Optional[AlphaLibrary] = None,
    ) -> None:
        self.db_path = db_path
        self.config = config or CampaignConfig()
        self.detectors_factory = detectors_factory or default_detectors
        self.alpha_library = alpha_library or AlphaLibrary(db_path)
        self.data_quality = DataQualityMonitor()
        # Real HoldoutManager is ENFORCED — never None in campaign operation
        self.holdout_manager = HoldoutManager(db_path)
        self.pipeline = AlphaValidationPipeline(
            self.config.validation, db_path,
            holdout_manager=self.holdout_manager)
        self.artifacts = ValidationArtifactStore(db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_GRAVEYARD)
            conn.execute(_CREATE_CHECKPOINTS)
            # Additive migration for pre-existing campaign_hypotheses tables
            existing = {row[1] for row in
                        conn.execute("PRAGMA table_info(campaign_hypotheses)")}
            for col in ("family_signature", "parameter_signature", "failure_stage"):
                if col not in existing:
                    conn.execute(
                        f"ALTER TABLE campaign_hypotheses ADD COLUMN {col} TEXT")
            conn.commit()

    # ── Campaign ──────────────────────────────────────────────────────────────

    def run(self, data: Dict[str, pd.DataFrame],
            universe_has_membership_data: bool = False,
            campaign_id: Optional[str] = None,
            resume: bool = False) -> Dict[str, Any]:
        cfg = self.config
        campaign_id = campaign_id or str(uuid.uuid4())[:8]
        report: Dict[str, Any] = {
            "campaign_id": campaign_id,
            "started_at": _utcnow(),
            "universe_scanned": sorted(data.keys()),
            "survivorship_bias_risk": not universe_has_membership_data,
            "rejections": {},
            "reproducibility": {
                "seed": cfg.seed, "code_version": cfg.code_version,
                "feature_version": cfg.feature_version,
                "validation_config": cfg.validation.stage_policy,
            },
        }
        if not universe_has_membership_data:
            logger.warning("ResearchCampaign: SURVIVORSHIP_BIAS_RISK — universe "
                           "has no historical membership data")

        def reject(reason: str, n: int = 1):
            report["rejections"][reason] = report["rejections"].get(reason, 0) + n

        # 1. Data quality gate
        clean: Dict[str, pd.DataFrame] = {}
        for symbol, df in sorted(data.items()):
            work = df.copy()
            work.columns = [str(c).lower() for c in work.columns]
            work = work.loc[:, ~work.columns.duplicated()]
            result = self.data_quality.check(work, symbol=symbol,
                                             timeframe=cfg.data_timeframe)
            if result.passed and len(work) >= 100:
                clean[symbol] = work
            else:
                reject(RejectReason.DATA_QUALITY_FAILURE)
        report["instruments_clean"] = len(clean)
        if not clean:
            report["finished_at"] = _utcnow()
            return report

        # 2. Three-way chronological split per instrument
        discovery, oos, holdout = {}, {}, {}
        holdout_periods: Dict[str, Tuple[str, str]] = {}
        holdout_available = True
        for s, df in clean.items():
            n = len(df)
            d_end = int(n * cfg.discovery_fraction)
            o_end = int(n * (cfg.discovery_fraction + cfg.oos_fraction))
            discovery[s] = df.iloc[:d_end]
            oos[s] = df.iloc[d_end:o_end]
            if n - o_end < cfg.min_holdout_bars:
                holdout_available = False
            holdout[s] = df.iloc[o_end:]
            if len(holdout[s]):
                holdout_periods[s] = (str(holdout[s].index[0]),
                                      str(holdout[s].index[-1]))
        report["holdout_available"] = holdout_available

        # 3. Detectors in BATCHES with campaign-GLOBAL tracker/dedup context
        symbols = sorted(clean.keys())
        batches = [symbols[i:i + cfg.instrument_batch_size]
                   for i in range(0, len(symbols), cfg.instrument_batch_size)]
        tracker = HypothesisFamilyTracker()
        hypotheses: List[DiscoveredHypothesis] = []
        completed = self._load_checkpoint(campaign_id, tracker, hypotheses) \
            if resume else set()

        detectors_run, detectors_skipped = [], []
        for batch_idx, batch in enumerate(batches):
            if batch_idx in completed:
                continue
            batch_data = {s: discovery[s] for s in batch}
            detectors = self.detectors_factory(limits=cfg.limits, tracker=tracker)
            for det in detectors:
                name = det.__class__.__name__
                if det.requires_data:
                    if f"{name} (needs {det.requires_data})" not in detectors_skipped:
                        detectors_skipped.append(f"{name} (needs {det.requires_data})")
                    continue
                det.tracker = tracker   # GLOBAL context across batches
                try:
                    hypotheses.extend(det.scan(batch_data))
                    if name not in detectors_run:
                        detectors_run.append(name)
                except Exception as e:
                    logger.error(f"{name} failed on batch {batch_idx}: {e}")
            self._save_checkpoint(campaign_id, batch_idx, hypotheses, tracker)

        hypotheses = hypotheses[: cfg.max_hypotheses_per_campaign]
        report["batches"] = len(batches)
        report["detectors_run"] = detectors_run
        report["detectors_skipped_missing_data"] = detectors_skipped
        report["hypotheses_tested"] = tracker.n_trials()
        report["hypotheses_emitted"] = len(hypotheses)

        # 4. Campaign-global graveyard + dedup (exact parameter identity)
        seen: Dict[str, DiscoveredHypothesis] = {}
        graveyard = self._graveyard_signatures()
        promoted = self._promoted_signatures()
        for h in hypotheses:
            sig = parameter_signature(h)
            if sig in promoted:
                reject(RejectReason.ALREADY_PROMOTED)
                continue
            if sig in graveyard:
                reject(RejectReason.GRAVEYARD_DUPLICATE)
                continue
            prev = seen.get(sig)
            if prev is not None:
                reject(RejectReason.NEAR_DUPLICATE)
                if h.p_value < prev.p_value:
                    seen[sig] = h
                continue
            seen[sig] = h
        survivors = list(seen.values())
        report["after_dedup"] = len(survivors)

        # 5. Cheap preliminary filter
        prelim = []
        for h in survivors:
            if h.mean_return < cfg.min_gross_effect:
                reject(RejectReason.LOW_EFFECT_SIZE)
                self._bury(campaign_id, h, "preliminary", RejectReason.LOW_EFFECT_SIZE)
                continue
            conc = profit_concentration(h.metadata.get("returns_sample") or [])
            if conc["concentration_score"] > cfg.max_concentration_top5:
                reject(RejectReason.OUTLIER_DEPENDENT)
                self._bury(campaign_id, h, "preliminary", RejectReason.OUTLIER_DEPENDENT)
                continue
            prelim.append(h)
        report["preliminary_survivors"] = len(prelim)

        # 6. Campaign-global FDR within hypothesis families
        fdr_survivors = self._apply_fdr(campaign_id, prelim, tracker, reject)
        report["fdr_survivors"] = len(fdr_survivors)

        # 7. Deep validation queue — prioritized by evidence, budget-capped
        fdr_survivors.sort(
            key=lambda h: h.mean_return * h.effective_sample, reverse=True)
        if len(fdr_survivors) > cfg.max_validation_candidates:
            for h in fdr_survivors[cfg.max_validation_candidates:]:
                reject(RejectReason.VALIDATION_BUDGET_EXCEEDED)
            fdr_survivors = fdr_survivors[: cfg.max_validation_candidates]

        # 8. Full validation pipeline (the ONLY road to promotion)
        oos_index = self._scan_segment(oos, cfg.oos_limits)
        holdout_index = self._scan_segment(holdout, cfg.oos_limits) \
            if holdout_available else {}
        siblings_index = self._group_siblings(prelim)
        regime_index = {s: self._regime_labels(df) for s, df in clean.items()}

        stage_survivor_counts: Dict[str, int] = {}
        validated: List[Tuple[DiscoveredHypothesis, AlphaValidationResult]] = []
        for h in fdr_survivors:
            fam_sig = family_signature(h)
            param_sig = parameter_signature(h)
            result = self.pipeline.validate(
                h,
                oos_match=oos_index.get(param_sig),
                holdout_match=holdout_index.get(param_sig),
                holdout_available=holdout_available,
                sibling_variants=[s for s in siblings_index.get(fam_sig, [])
                                  if parameter_signature(s) != param_sig],
                family_trials=max(tracker.n_trials(f"{h.family}:{h.subfamily}"), 1),
                campaign_id=campaign_id,
                regime_series=regime_index.get(h.symbol),
                holdout_period=holdout_periods.get(h.symbol),
            )
            for stage in result.stages:
                if stage.status == "PASS":
                    stage_survivor_counts[stage.stage_name] = \
                        stage_survivor_counts.get(stage.stage_name, 0) + 1
            if result.passed:
                validated.append((h, result))
            else:
                failed_stage = next(
                    (s.stage_name for s in result.stages if s.status == "FAIL"),
                    "unknown")
                reject(RejectReason.FAILED_FULL_VALIDATION)
                self._bury(campaign_id, h, failed_stage,
                           ";".join(result.reason_codes)[:200])
                self.artifacts.save(result)
        report["stage_pass_counts"] = stage_survivor_counts
        report["full_validation_survivors"] = len(validated)

        # 9. Alpha creation via validated results ONLY (auto-PAPER, never live)
        new_alphas = []
        for h, result in validated:
            alpha_id = self.promote_validated(campaign_id, h, result)
            new_alphas.append(alpha_id)
        report["new_alphas"] = new_alphas
        report["new_paper_campaigns"] = new_alphas

        report["family_conversion"] = self._family_conversion(hypotheses, validated)
        report["top_candidates"] = [
            self._describe(h, r) for h, r in
            sorted(validated, key=lambda x: x[0].mean_return, reverse=True)[:10]
        ]
        report["finished_at"] = _utcnow()
        self._clear_checkpoints(campaign_id)
        logger.info(self.format_report(report))
        return report

    # ── Promotion (spec §25, §56): ONLY via a passing validation result ───────

    def promote_validated(self, campaign_id: str, h: DiscoveredHypothesis,
                          result: AlphaValidationResult) -> str:
        if not result.passed:
            raise PermissionError(
                "Alpha promotion requires a PASSING AlphaValidationResult — "
                f"decision={result.decision}, reasons={result.reason_codes}"
            )
        import hashlib
        sig_hash = hashlib.md5(result.parameter_signature.encode()).hexdigest()[:6]
        alpha_id = (f"{h.family.lower()}_{h.subfamily}_{h.symbol}_"
                    f"{h.direction}_{sig_hash}")
        cost_stage = next((s for s in result.stages
                           if s.stage_name == "cost_stress"), None)
        self.alpha_library.register(
            alpha_id, name=h.description or alpha_id,
            strategy_family=h.family, subfamily=h.subfamily,
            direction=h.direction,
            universe=[h.symbol],
            entry_conditions=json.dumps(h.entry_conditions),
            time_stop_bars=h.holding_bars,
            alpha_version="1",
            sample_size=h.sample_size,
            effective_sample_size=h.effective_sample,
            gross_expectancy=h.mean_return,
            net_expectancy=result.conservative_prior.get("mean_net_return"),
            p_value=h.p_value,
            adjusted_significance=1,
            valid_regimes=result.valid_regimes,
            cost_stress_results=(cost_stage.metrics if cost_stage else {}),
            oos_metrics=result.conservative_prior,
            event_concentration=next(
                (s.metrics.get("concentration_score") for s in result.stages
                 if s.stage_name == "profit_concentration"), None),
        )
        self.alpha_library.transition(alpha_id, AlphaState.VALIDATING,
                                      reason=f"campaign {campaign_id}")
        self.alpha_library.transition(alpha_id, AlphaState.PAPER,
                                      reason=f"campaign {campaign_id}: "
                                             "full validation PASS → paper")
        self.artifacts.save(result, alpha_id=alpha_id)
        self._record(campaign_id, h, "PROMOTED", "", "")
        return alpha_id

    # ── FDR (campaign-global) ─────────────────────────────────────────────────

    def _apply_fdr(self, campaign_id, prelim, tracker, reject) -> List:
        cfg = self.config
        out = []
        by_family: Dict[str, List[DiscoveredHypothesis]] = {}
        for h in prelim:
            by_family.setdefault(f"{h.family}:{h.subfamily}", []).append(h)
        for fam, hyps in by_family.items():
            fam_tests = [t for t in tracker._tests if t.family == fam]
            passed_flags = benjamini_hochberg([t.p_value for t in fam_tests],
                                              alpha=cfg.fdr_alpha)
            passed_ids = {t.hypothesis_id for t, ok in zip(fam_tests, passed_flags) if ok}
            for h in hyps:
                if h.hypothesis_id in passed_ids:
                    out.append(h)
                else:
                    reject(RejectReason.FAILED_FDR)
                    self._bury(campaign_id, h, "fdr", RejectReason.FAILED_FDR)
        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _scan_segment(self, segment_data: Dict[str, pd.DataFrame],
                      limits: SearchLimits) -> Dict[str, DiscoveredHypothesis]:
        tracker = HypothesisFamilyTracker()
        detectors = self.detectors_factory(limits=limits, tracker=tracker)
        out: Dict[str, DiscoveredHypothesis] = {}
        for det in detectors:
            if det.requires_data:
                continue
            det.tracker = tracker
            try:
                for h in det.scan(segment_data):
                    out[parameter_signature(h)] = h
            except Exception as e:
                logger.debug(f"segment scan {det.__class__.__name__}: {e}")
        return out

    @staticmethod
    def _group_siblings(hyps: List[DiscoveredHypothesis]
                        ) -> Dict[str, List[DiscoveredHypothesis]]:
        out: Dict[str, List[DiscoveredHypothesis]] = {}
        for h in hyps:
            out.setdefault(family_signature(h), []).append(h)
        return out

    @staticmethod
    def _regime_labels(df: pd.DataFrame) -> Dict[str, str]:
        """Volatility-regime labels per bar timestamp (HIGH_VOL / LOW_VOL)."""
        try:
            close = pd.to_numeric(df["close"], errors="coerce")
            vol = close.pct_change().rolling(20).std()
            median = vol.rolling(120, min_periods=40).median()
            labels = {}
            for ts, v, m in zip(df.index, vol, median):
                if pd.isna(v) or pd.isna(m):
                    continue
                labels[str(ts)] = "HIGH_VOL" if v > m else "LOW_VOL"
            return labels
        except Exception:
            return {}

    # ── Checkpointing (spec §36) ──────────────────────────────────────────────

    def _save_checkpoint(self, campaign_id: str, batch_index: int,
                         hypotheses: List, tracker) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO campaign_checkpoints "
                "(campaign_id, batch_index, hypotheses_json, tracker_json, completed_at) "
                "VALUES (?,?,?,?,?)",
                (campaign_id, batch_index,
                 json.dumps([_hyp_to_json(h) for h in hypotheses], default=str),
                 json.dumps([{"hypothesis_id": t.hypothesis_id, "family": t.family,
                              "p_value": t.p_value, "metric": t.metric}
                             for t in tracker._tests], default=str),
                 _utcnow()),
            )
            conn.commit()

    def _load_checkpoint(self, campaign_id: str, tracker,
                         hypotheses: List) -> set:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT batch_index, hypotheses_json, tracker_json "
                "FROM campaign_checkpoints WHERE campaign_id=? "
                "ORDER BY batch_index",
                (campaign_id,),
            ).fetchall()
        if not rows:
            return set()
        last = rows[-1]
        for d in json.loads(last[1] or "[]"):
            hypotheses.append(_hyp_from_json(d))
        for t in json.loads(last[2] or "[]"):
            tracker._tests.append(TestedHypothesis(
                hypothesis_id=t["hypothesis_id"], family=t["family"],
                p_value=t["p_value"], metric=t.get("metric", 0.0)))
        completed = {r[0] for r in rows}
        logger.info(f"ResearchCampaign: resumed {campaign_id} — "
                    f"{len(completed)} batches, {len(hypotheses)} hypotheses restored")
        return completed

    def _clear_checkpoints(self, campaign_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM campaign_checkpoints WHERE campaign_id=?",
                         (campaign_id,))
            conn.commit()

    # ── Graveyard ─────────────────────────────────────────────────────────────

    def _graveyard_signatures(self) -> set:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT COALESCE(parameter_signature, signature) "
                "FROM campaign_hypotheses WHERE outcome='REJECTED'"
            ).fetchall()
        return {r[0] for r in rows}

    def _promoted_signatures(self) -> set:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT COALESCE(parameter_signature, signature) "
                "FROM campaign_hypotheses WHERE outcome='PROMOTED'"
            ).fetchall()
        return {r[0] for r in rows}

    def _bury(self, campaign_id: str, h: DiscoveredHypothesis,
              failure_stage: str, reason: str) -> None:
        self._record(campaign_id, h, "REJECTED", failure_stage, reason)

    def _record(self, campaign_id: str, h: DiscoveredHypothesis,
                outcome: str, failure_stage: str, reason: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO campaign_hypotheses "
                "(signature, family_signature, parameter_signature, campaign_id, "
                " outcome, failure_stage, reason, mean_return, p_value, tested_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (parameter_signature(h), family_signature(h),
                 parameter_signature(h), campaign_id, outcome, failure_stage,
                 reason, h.mean_return, h.p_value, _utcnow()),
            )
            conn.commit()

    # ── Reporting ─────────────────────────────────────────────────────────────

    @staticmethod
    def _family_conversion(hypotheses: List[DiscoveredHypothesis],
                           validated: List) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for h in hypotheses:
            fam = out.setdefault(h.family, {"hypotheses": 0, "final_alphas": 0})
            fam["hypotheses"] += 1
        for h, _ in validated:
            out.setdefault(h.family, {"hypotheses": 0, "final_alphas": 0})
            out[h.family]["final_alphas"] += 1
        return out

    @staticmethod
    def _describe(h: DiscoveredHypothesis, result: AlphaValidationResult) -> Dict:
        stages = {s.stage_name: s for s in result.stages}
        oos = stages.get("oos")
        cost = stages.get("cost_stress")
        return {
            "signature": result.parameter_signature,
            "family_signature": result.family_signature,
            "description": h.description,
            "family": h.family, "subfamily": h.subfamily,
            "symbol": h.symbol, "direction": h.direction,
            "conditions": h.entry_conditions,
            "holding_bars": h.holding_bars,
            "discovery_samples": h.sample_size,
            "oos_samples": (oos.metrics.get("oos_sample") if oos else None),
            "gross_expectancy_return": h.mean_return,
            "net_expectancy_return_1x": (cost.metrics.get("net_1x") if cost else None),
            "net_expectancy_return_2x": (cost.metrics.get("net_2x") if cost else None),
            "net_expectancy_return_3x": (cost.metrics.get("net_3x") if cost else None),
            "validation_score": result.validation_score,
            "valid_regimes": result.valid_regimes,
            "p_value": h.p_value,
        }

    @staticmethod
    def format_report(report: Dict[str, Any]) -> str:
        lines = [
            "═" * 60,
            f"RESEARCH CAMPAIGN {report['campaign_id']}",
            f"Universe scanned: {len(report['universe_scanned'])} instruments "
            f"({report.get('instruments_clean', 0)} passed data quality, "
            f"{report.get('batches', 1)} batches)",
            f"Holdout available: {report.get('holdout_available')}",
            f"Hypotheses tested: {report.get('hypotheses_tested', 0)}",
            f"Hypotheses emitted: {report.get('hypotheses_emitted', 0)}",
            f"After dedup/graveyard: {report.get('after_dedup', 0)}",
            f"Preliminary survivors: {report.get('preliminary_survivors', 0)}",
            f"FDR survivors: {report.get('fdr_survivors', 0)}",
            f"Stage pass counts: {report.get('stage_pass_counts', {})}",
            f"Full validation survivors: {report.get('full_validation_survivors', 0)}",
            f"New alphas (auto-PAPER, never live): {report.get('new_alphas', [])}",
            f"Family conversion: {report.get('family_conversion', {})}",
            f"Detectors run: {report.get('detectors_run', [])}",
            f"Skipped (missing data): {report.get('detectors_skipped_missing_data', [])}",
            f"Rejections: {report.get('rejections', {})}",
        ]
        if report.get("survivorship_bias_risk"):
            lines.append("⚠ SURVIVORSHIP_BIAS_RISK: no historical membership data")
        for c in report.get("top_candidates", []):
            lines.append(
                f"  → {c['signature']}: gross={c['gross_expectancy_return']:+.4%} "
                f"net1x={c['net_expectancy_return_1x'] or 0:+.4%} "
                f"score={c['validation_score']:.0f} p={c['p_value']:.4f}"
            )
        lines.append("═" * 60)
        return "\n".join(lines)
