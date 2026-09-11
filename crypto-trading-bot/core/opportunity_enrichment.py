"""OpportunityEnricher — runs every profitability module on each candidate
BEFORE allocation, so the allocator ranks true net economics (spec §2-18).

    gross EV → meta-alpha regime adjustment → edge survival → crowding →
    half-life/urgency → execution method + costs → borrow → net_execution_EV →
    capacity → expected dollar alpha → capital-time efficiency

Hard rules preserved: enrichment never validates alphas, never rescues
negative EV, never exceeds hard caps, and meta-alpha starts in SHADOW.
"""
from __future__ import annotations

import logging
import sqlite3
import statistics as st
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.alpha_signal_engine import OpportunityCandidate, ReasonCode
from core.execution_optimizer import (
    BorrowChecker,
    ExecutionOptimizer,
    SlippageModel,
)
from core.meta_alpha import CrowdingMonitor, EdgeSurvivalModel, MetaAlphaModel
from core.opportunity_economics import opportunity_economics

logger = logging.getLogger(__name__)

ENRICHMENT_VERSION = "1.0.0"


def classify_urgency(half_life_bars: Optional[float],
                     bar_minutes: float = 1440.0) -> str:
    """Half-life → order urgency (spec §9). Bars default to daily."""
    if half_life_bars is None:
        return "NORMAL"
    half_life_minutes = half_life_bars * bar_minutes
    if half_life_minutes <= 10:
        return "IMMEDIATE"
    if half_life_minutes <= 240:
        return "FAST"
    if half_life_minutes <= 20 * 1440:
        return "NORMAL"
    return "PATIENT"


def methods_for_urgency(urgency: str) -> Tuple[str, ...]:
    """Feasible entry methods per urgency: a passive order that fills after
    the signal decays is not an execution option at all (spec §76)."""
    if urgency == "IMMEDIATE":
        return ("market",)
    if urgency == "FAST":
        return ("market", "limit_mid")
    if urgency == "PATIENT":
        return ("limit_passive", "limit_mid", "delayed_5m", "market")
    return ("market", "limit_mid", "limit_passive")


@dataclass
class EnrichmentConfig:
    meta_alpha_mode: str = "shadow"        # shadow | advisory | active
    advisory_fit_bounds: Tuple[float, float] = (0.5, 1.2)
    survival_pause_below: float = 0.25
    crowding_capital_penalty: float = 0.5  # max fractional haircut at score=1
    default_holding_days: float = 5.0
    borrow_rate_fallback_annual: float = 0.03


class OpportunityEnricher:
    """One enrichment pass per trading cycle. All model outputs land on the
    candidate; hard-gate failures append reason codes the allocator enforces."""

    def __init__(self,
                 config: Optional[EnrichmentConfig] = None,
                 meta_alpha: Optional[MetaAlphaModel] = None,
                 survival_model: Optional[EdgeSurvivalModel] = None,
                 crowding_monitor: Optional[CrowdingMonitor] = None,
                 execution_optimizer: Optional[ExecutionOptimizer] = None,
                 borrow_checker: Optional[BorrowChecker] = None,
                 returns_fetcher: Optional[Callable[[str], Dict[str, Any]]] = None,
                 half_life_provider: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
                 db_path: str = "data/trade_memory.sqlite") -> None:
        self.config = config or EnrichmentConfig()
        self.meta = meta_alpha or MetaAlphaModel(mode=self.config.meta_alpha_mode)
        self.meta.mode = self.config.meta_alpha_mode
        self.survival = survival_model or EdgeSurvivalModel()
        self.crowding = crowding_monitor or CrowdingMonitor()
        self.executor = execution_optimizer or ExecutionOptimizer()
        self.borrow = borrow_checker or BorrowChecker()
        self._returns_fetcher = returns_fetcher
        self._half_life_provider = half_life_provider
        self.db_path = db_path
        self.return_samples: Dict[str, List[float]] = {}
        self.re_research_queue: List[Dict[str, Any]] = []

    # ── Attributed-return history ─────────────────────────────────────────────

    def _alpha_history(self, alpha_id: str) -> Dict[str, Any]:
        """{recent: [...], older: [...], by_regime: [(regime, ret)]} from the
        attribution store (or the injected fetcher in tests). Uses the
        CANONICAL normalizer (net_pnl / size_dollars); schema failures are
        loud and answered conservatively — never treated as empty history."""
        if self._returns_fetcher:
            return self._returns_fetcher(alpha_id) or {}
        from core.trade_history import TradeMemorySchemaError, fetch_attributed_returns
        try:
            trades = fetch_attributed_returns(self.db_path, alpha_id=alpha_id)
        except TradeMemorySchemaError:
            # Broken history ≠ zero risk: mark the alpha unsizable this cycle
            logger.critical(
                f"enrichment: schema failure loading history for {alpha_id} — "
                "conservative response: survival multiplier 0")
            return {"schema_error": True}
        rets = [t.return_fraction for t in trades if t.return_fraction is not None]
        regimes = [t.market_regime for t in trades
                   if t.return_fraction is not None]
        half = len(rets) // 2
        return {"recent": rets[half:], "older": rets[:half],
                "by_regime": list(zip(regimes, rets))}

    # ── Enrichment ────────────────────────────────────────────────────────────

    def enrich_all(self, candidates: Sequence[OpportunityCandidate],
                   regime: str = "unknown",
                   capital: float = 10_000.0,
                   base_position_frac: float = 0.05) -> None:
        for c in candidates:
            try:
                self.enrich(c, regime=regime, capital=capital,
                            base_position_frac=base_position_frac)
            except Exception as e:                     # enrichment never crashes trading
                logger.warning(f"enrichment failed for {c.candidate_id}: {e}")

    def enrich(self, c: OpportunityCandidate, *, regime: str,
               capital: float, base_position_frac: float) -> None:
        cfg = self.config
        c.model_versions = {
            "enrichment": ENRICHMENT_VERSION,
            "meta_alpha_mode": cfg.meta_alpha_mode,
            "execution_optimizer": "1.0.0",
            "edge_survival": "1.0.0",
        }
        gross_ev = c.expected_net_return
        c.raw_expected_net_return = gross_ev
        if gross_ev is None:
            return
        history = self._alpha_history(c.alpha_id)
        if history.get("schema_error"):
            # conservative: no history → no sizing confidence, never "no risk"
            c.survival_multiplier = 0.0
            c.reason_codes.append(ReasonCode.EDGE_DECAY)
            return
        recent = history.get("recent") or []
        older = history.get("older") or []
        for reg, r in history.get("by_regime") or []:
            self.meta.observe(c.alpha_id, reg, r)
        if recent or older:
            self.return_samples[c.alpha_id] = older + recent

        # 1. Meta-alpha regime conditioning (shadow logs only; advisory/active
        #    modulate regime_fit within bounds; never a validation override)
        pred = self.meta.predict(c.alpha_id, regime)
        c.meta_alpha_ev = pred.get("expected_alpha_ev")
        c.meta_alpha_confidence = pred.get("confidence")
        if cfg.meta_alpha_mode == "active":
            c.regime_fit = min(c.regime_fit * pred["regime_fit"], 1.5)
        elif cfg.meta_alpha_mode == "advisory":
            lo, hi = cfg.advisory_fit_bounds
            c.regime_fit = min(c.regime_fit * max(lo, min(hi, pred["regime_fit"])),
                               1.5)
        else:
            self.meta.shadow_compare([c.alpha_id], regime)

        # 2. Edge survival — sizing multiplier + PAUSE hard gate
        if recent or older:
            assessment = self.survival.assess(
                c.alpha_id, recent_returns=recent, older_returns=older)
            c.edge_survival_probability = assessment.survival_probability
            c.survival_multiplier = assessment.capital_multiplier
            if assessment.recommended_state == "PAUSED":
                c.reason_codes.append(ReasonCode.EDGE_DECAY)

        # 3. Crowding — usable-EV haircut + re-research flag
        if len(older) >= 10 and len(recent) >= 10:
            crowd = self.crowding.assess(
                c.alpha_id, post_signal_returns_early=older,
                post_signal_returns_recent=recent)
            c.crowding_score = min(len(crowd["flags"]) / 3.0, 1.0) \
                if crowd["crowded"] else 0.0
            if "EDGE_TIMING_SHIFT" in crowd["flags"]:
                self.re_research_queue.append(
                    {"alpha_id": c.alpha_id, "reason": "EDGE_TIMING_SHIFT"})

        # 4. Half-life → urgency. Empirical decay curve wins when learned and
        #    stable; the planned holding period is only a MARKED fallback.
        empirical = None
        if self._half_life_provider:
            try:
                empirical = self._half_life_provider(c.alpha_id)
            except Exception:
                empirical = None
        if empirical and empirical.get("half_life") is not None \
                and empirical.get("stable"):
            c.half_life_bars = float(empirical["half_life"])
            c.features["half_life_source"] = "EMPIRICAL"
            c.features["half_life_ci"] = (empirical.get("ci_low"),
                                          empirical.get("ci_high"))
        else:
            hold_bars = (c.exit_plan or {}).get("time_stop_bars") \
                or (c.features or {}).get("holding_bars")
            c.half_life_bars = float(hold_bars) if hold_bars else None
            c.features["half_life_source"] = "HALF_LIFE_FALLBACK"
        c.urgency = classify_urgency(c.half_life_bars)

        # 5. Execution method + true round-trip cost
        notional = capital * base_position_frac
        adv = (c.features or {}).get("dollar_volume_24h")
        spread = (c.features or {}).get("spread_pct") \
            or (c.features or {}).get("bar_range_pct") or 0.001
        vol = (c.features or {}).get("atr_pct") or 0.02
        gross_after_crowding = gross_ev * (1 - cfg.crowding_capital_penalty
                                           * c.crowding_score)
        plan = self.executor.best_entry(
            gross_alpha_bps=gross_after_crowding * 10_000,
            spread_pct=float(spread), order_notional=notional, adv_usd=adv,
            volatility_daily=float(vol),
            methods=methods_for_urgency(c.urgency))
        c.recommended_order_type = plan.method
        c.expected_slippage_bps = plan.expected_cost_bps
        round_trip_bps = plan.expected_cost_bps * 2
        components = dict(plan.detail.get("components", {}))
        components["entry_cost_bps"] = plan.expected_cost_bps
        c.features["expected_fill_probability"] = plan.fill_probability

        # 6. Borrow feasibility + carry for shorts (spec §29, §85)
        borrow_bps = 0.0
        if c.direction == "short":
            check = self.borrow.check(c.symbol, c.asset_class, "short")
            if not check["executable"]:
                c.reason_codes.append(ReasonCode.BORROW_UNAVAILABLE)
            else:
                holding_days = c.half_life_bars or cfg.default_holding_days
                rate = check.get("borrow_rate_annual",
                                 cfg.borrow_rate_fallback_annual)
                borrow_bps = rate * 10_000 * holding_days / 365.0
        components["borrow_bps"] = borrow_bps
        c.features["expected_cost_components"] = components
        c.expected_total_cost_bps = round_trip_bps + borrow_bps

        # 7. NET execution EV — every downstream ranking uses this
        net = gross_after_crowding - c.expected_total_cost_bps / 10_000
        cost_delta = gross_ev - net
        c.expected_net_return = net
        if c.ev_lower_bound is not None:
            c.ev_lower_bound -= cost_delta
        if c.conservative_ev is not None:
            c.conservative_ev -= cost_delta

        # 8. Capacity, dollar alpha, capital-time efficiency
        holding_days = c.half_life_bars or cfg.default_holding_days
        econ = opportunity_economics(
            expected_net_return=max(net, 0.0), adv_usd=adv,
            holding_days=holding_days, signals_per_year=252.0 / holding_days)
        c.practical_capacity_usd = econ["estimated_alpha_capacity_usd"]
        c.capacity_utilization = (notional / c.practical_capacity_usd
                                  if c.practical_capacity_usd else None)
        deployable = min(notional, c.practical_capacity_usd or notional)
        c.expected_dollar_alpha = max(net, 0.0) * deployable
        c.capital_time_efficiency = econ["capital_time_efficiency"]

    # ── Cycle reports (spec §47-48) ───────────────────────────────────────────

    @staticmethod
    def allocation_report(decisions: Sequence[Any]) -> List[Dict[str, Any]]:
        out = []
        for d in decisions:
            c = d.candidate
            out.append({
                "candidate": c.candidate_id, "alpha": c.alpha_id,
                "symbol": c.symbol, "direction": c.direction,
                "raw_ev": c.raw_expected_net_return,
                "net_execution_ev": c.expected_net_return,
                "ev_lower_bound": c.ev_lower_bound,
                "meta_alpha_ev": c.meta_alpha_ev,
                "edge_survival": c.edge_survival_probability,
                "crowding": c.crowding_score,
                "half_life_bars": c.half_life_bars,
                "urgency": c.urgency,
                "order_type": c.recommended_order_type,
                "total_cost_bps": c.expected_total_cost_bps,
                "capacity_usd": c.practical_capacity_usd,
                "expected_dollar_alpha": c.expected_dollar_alpha,
                "capital_time_efficiency": c.capital_time_efficiency,
                "accepted": d.accepted,
                "allocation_usd": d.allocation_usd,
                "reasons": list(d.reason_codes) or list(c.reason_codes),
            })
        return out


# ── Meta-alpha promotion gate (spec §4, §90) ──────────────────────────────────


def meta_alpha_promotion_ready(
    resolved: Sequence[Tuple[float, float]], *,
    min_samples: int = 30, min_sign_agreement: float = 0.55,
) -> Dict[str, Any]:
    """resolved: [(predicted_conditional_ev, realized_return)]. Shadow →
    advisory/active requires calibration on enough forward outcomes."""
    n = len(resolved)
    if n < min_samples:
        return {"ready": False, "resolved": n, "why": "insufficient_samples"}
    agree = sum(1 for p, r in resolved
                if (p or 0) * r > 0 or (p == 0 and abs(r) < 1e-9)) / n
    pred_mean = st.mean(p for p, _ in resolved)
    real_mean = st.mean(r for _, r in resolved)
    calibrated = abs(pred_mean - real_mean) <= max(abs(real_mean), 0.002)
    ready = agree >= min_sign_agreement and calibrated
    return {"ready": ready, "resolved": n, "sign_agreement": agree,
            "predicted_mean": pred_mean, "realized_mean": real_mean,
            "calibrated": calibrated}
