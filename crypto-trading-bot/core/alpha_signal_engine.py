"""AlphaSignalEngine — the Alpha Library actively drives trade generation.

Each cycle:
    eligible alphas  ×  their universe  ×  current market state
    → standardized OpportunityCandidate objects (opportunity-first, not
      symbol-first: one symbol can carry several independent opportunities).

Lifecycle enforcement:
    LIVE_ELIGIBLE / LIVE_LIMITED / LIVE_SCALED  → live candidates
    PAPER                                       → paper candidates only
    everything else                             → never generates candidates

Every rejection carries a structured reason code.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from core.alpha_conditions import evaluate_conditions, parse_conditions_json
from core.alpha_library import (
    AlphaLibrary,
    AlphaState,
    LIVE_ELIGIBLE_STATES,
    SIGNAL_ELIGIBLE_STATES,
)
from core.feature_registry import FeatureRegistry
from core.ohlcv import normalize_ohlcv

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


# Structured rejection / observability codes (spec §56)
class ReasonCode:
    ALPHA_NOT_LIVE_ELIGIBLE = "ALPHA_NOT_LIVE_ELIGIBLE"
    ALPHA_NOT_SIGNAL_ELIGIBLE = "ALPHA_NOT_SIGNAL_ELIGIBLE"
    REGIME_MISMATCH = "REGIME_MISMATCH"
    EDGE_HEALTH_TOO_LOW = "EDGE_HEALTH_TOO_LOW"
    DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
    CONDITIONS_NOT_MET = "CONDITIONS_NOT_MET"
    EXPECTED_EV_TOO_LOW = "EXPECTED_EV_TOO_LOW"
    LOWER_BOUND_NEGATIVE = "LOWER_BOUND_NEGATIVE"
    EXECUTION_COST_TOO_HIGH = "EXECUTION_COST_TOO_HIGH"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    META_MODEL_REJECT = "META_MODEL_REJECT"
    PORTFOLIO_CORRELATION_LIMIT = "PORTFOLIO_CORRELATION_LIMIT"
    FAMILY_RISK_LIMIT = "FAMILY_RISK_LIMIT"
    SIGNAL_CONFLICT = "SIGNAL_CONFLICT"
    EVENT_RISK = "EVENT_RISK"
    NO_QUOTE_DATA = "NO_QUOTE_DATA"
    STALE_DATA = "STALE_DATA"
    NO_DATA = "NO_DATA"
    RISK_OF_RUIN_LIMIT = "RISK_OF_RUIN_LIMIT"
    BORROW_UNAVAILABLE = "BORROW_UNAVAILABLE"
    INSUFFICIENT_CAPACITY = "INSUFFICIENT_CAPACITY"
    POOR_REGIME_FIT = "POOR_REGIME_FIT"
    EDGE_DECAY = "EDGE_DECAY"


@dataclass
class OpportunityCandidate:
    """Standardized opportunity — the unit the whole pipeline ranks."""
    candidate_id: str
    alpha_id: str
    alpha_version: str
    symbol: str
    asset_class: str
    direction: str                      # 'long' | 'short'
    signal_time: str
    execution_mode: str                 # 'live' | 'paper'
    entry_plan: Dict[str, Any] = field(default_factory=dict)
    exit_plan: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    # Economics (filled by EV model / ranker)
    expected_net_return: Optional[float] = None
    probability_positive: Optional[float] = None
    expected_upside: Optional[float] = None
    expected_downside: Optional[float] = None
    expected_shortfall: Optional[float] = None
    conservative_ev: Optional[float] = None
    ev_lower_bound: Optional[float] = None
    # Quality scores
    edge_health_score: Optional[float] = None
    regime_fit: float = 1.0
    liquidity_score: Optional[float] = None
    execution_feasibility: Optional[float] = None
    meta_model_score: Optional[float] = None
    correlation_penalty: float = 1.0
    portfolio_fit: Optional[float] = None
    final_opportunity_score: Optional[float] = None
    # Enrichment (spec: enriched opportunity — filled by OpportunityEnricher)
    raw_expected_net_return: Optional[float] = None
    meta_alpha_ev: Optional[float] = None
    meta_alpha_confidence: Optional[float] = None
    edge_survival_probability: Optional[float] = None
    survival_multiplier: float = 1.0
    crowding_score: float = 0.0
    half_life_bars: Optional[float] = None
    urgency: Optional[str] = None            # IMMEDIATE|FAST|NORMAL|PATIENT
    recommended_order_type: Optional[str] = None
    recommended_venue: Optional[str] = None
    expected_slippage_bps: Optional[float] = None
    expected_total_cost_bps: Optional[float] = None
    practical_capacity_usd: Optional[float] = None
    capacity_utilization: Optional[float] = None
    expected_dollar_alpha: Optional[float] = None
    capital_time_efficiency: Optional[float] = None
    model_versions: Dict[str, str] = field(default_factory=dict)
    reason_codes: List[str] = field(default_factory=list)
    condition_details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Rejection:
    alpha_id: str
    symbol: str
    reason_code: str
    detail: str = ""


class AlphaSignalEngine:
    """Matches eligible alphas against current market state."""

    def __init__(
        self,
        alpha_library: AlphaLibrary,
        feature_registry: Optional[FeatureRegistry] = None,
        edge_health_floor: float = 0.3,
        data_quality_monitor=None,
        universe_resolver=None,
    ) -> None:
        self.library = alpha_library
        self.features = feature_registry or FeatureRegistry()
        self.edge_health_floor = edge_health_floor
        self.data_quality_monitor = data_quality_monitor
        if universe_resolver is None:
            from core.instruments import UniverseResolver
            universe_resolver = UniverseResolver()
        self.universe_resolver = universe_resolver
        self.last_rejections: List[Rejection] = []

    def required_symbols(self, scanner_symbols: Optional[List[str]] = None) -> List[str]:
        """Union of all eligible alphas' resolved universes.

        This is what makes alpha universes DRIVE scanning: an alpha targeting
        `sector:semiconductor` pulls the whole semiconductor universe into the
        cycle rather than being limited to a generic top-N list.
        """
        symbols: List[str] = []
        seen = set()
        for alpha in self.library.eligible_alphas():
            tokens = alpha.get("universe") or ["*"]
            for s in self.universe_resolver.resolve(tokens, scanner_symbols):
                if s not in seen:
                    seen.add(s)
                    symbols.append(s)
        return symbols

    def generate_candidates(
        self,
        market_data: Dict[str, pd.DataFrame],
        regime: str = "unknown",
        context: Optional[Dict[str, Dict[str, Any]]] = None,
        live_mode: bool = False,
    ) -> List[OpportunityCandidate]:
        """Evaluate every eligible alpha against every applicable symbol.

        market_data: symbol -> OHLCV df (current data, decision bar last)
        context:     symbol -> extra context features (regime, sector, ...)
        live_mode:   True when the caller intends live execution — PAPER
                     alphas then still produce paper-only candidates.
        """
        self.last_rejections = []
        context = context or {}
        candidates: List[OpportunityCandidate] = []

        alphas = self.library.eligible_alphas()
        if not alphas:
            logger.info("Alpha pipeline NO TRADE — no signal-eligible alphas in library")
            return []

        for alpha in alphas:
            state = AlphaState(alpha["lifecycle_state"])
            if state not in SIGNAL_ELIGIBLE_STATES:
                self._reject(alpha["alpha_id"], "*", ReasonCode.ALPHA_NOT_SIGNAL_ELIGIBLE,
                             state.value)
                continue

            execution_mode = "live" if state in LIVE_ELIGIBLE_STATES else "paper"
            if live_mode and execution_mode != "live":
                execution_mode = "paper"  # PAPER alphas never emit live candidates

            # Regime compatibility
            valid_regimes = alpha.get("valid_regimes") or []
            invalid_regimes = alpha.get("invalid_regimes") or []
            if valid_regimes and regime not in valid_regimes:
                self._reject(alpha["alpha_id"], "*", ReasonCode.REGIME_MISMATCH,
                             f"regime={regime} not in {valid_regimes}")
                continue
            if invalid_regimes and regime in invalid_regimes:
                self._reject(alpha["alpha_id"], "*", ReasonCode.REGIME_MISMATCH,
                             f"regime={regime} in invalid list")
                continue

            # Edge health
            health = alpha.get("edge_health_score")
            if health is not None and health < self.edge_health_floor:
                self._reject(alpha["alpha_id"], "*", ReasonCode.EDGE_HEALTH_TOO_LOW,
                             f"health={health:.2f} < {self.edge_health_floor}")
                continue

            universe = self.universe_resolver.resolve(
                alpha.get("universe") or ["*"],
                scanner_symbols=list(market_data.keys()),
            )
            for symbol in universe:
                df = market_data.get(symbol)
                if df is None or df.empty:
                    self._reject(alpha["alpha_id"], symbol, ReasonCode.NO_DATA)
                    continue
                cand = self._evaluate_symbol(alpha, symbol, df,
                                             context.get(symbol, {}),
                                             regime, execution_mode)
                if cand is not None:
                    candidates.append(cand)

        logger.info(
            f"AlphaSignalEngine: {len(candidates)} candidates from "
            f"{len(alphas)} eligible alphas ({len(self.last_rejections)} rejections)"
        )
        return candidates

    # ── Per-symbol evaluation ─────────────────────────────────────────────────

    def _evaluate_symbol(self, alpha, symbol, df, symbol_context, regime,
                         execution_mode) -> Optional[OpportunityCandidate]:
        alpha_id = alpha["alpha_id"]

        try:
            df = normalize_ohlcv(df)
        except ValueError as exc:
            self._reject(alpha_id, symbol, ReasonCode.DATA_QUALITY_FAILURE, str(exc))
            return None
        # Validate against the actual feed timeframe, not a hardcoded 15m default.
        if self.data_quality_monitor is not None:
            timeframe = df.attrs.get("timeframe", "15m")
            result = self.data_quality_monitor.check(df, symbol=symbol, timeframe=timeframe)
            if not result.passed:
                self._reject(alpha_id, symbol, ReasonCode.DATA_QUALITY_FAILURE,
                             "; ".join(result.issues[:3]))
                return None

        try:
            entry_conditions = parse_conditions_json(alpha.get("entry_conditions"))
        except Exception as e:
            self._reject(alpha_id, symbol, ReasonCode.CONDITIONS_NOT_MET,
                         f"invalid stored conditions: {e}")
            return None

        symbol_context = dict(symbol_context)
        symbol_context.setdefault("market_regime", regime)
        needed = list({c.feature for c in entry_conditions}) + [
            "price", "atr_pct", "dollar_volume_24h", "bar_range_pct"]
        try:
            feats = self.features.compute(symbol, df, list(set(needed)), symbol_context)
        except ValueError as e:
            self._reject(alpha_id, symbol, ReasonCode.CONDITIONS_NOT_MET, str(e))
            return None

        # Liquidity floor from the execution contract
        min_liq = alpha.get("min_liquidity_usd")
        dollar_vol = feats.get("dollar_volume_24h")
        if min_liq and (dollar_vol is None or dollar_vol < min_liq):
            self._reject(alpha_id, symbol, ReasonCode.LOW_LIQUIDITY,
                         f"dollar_vol={dollar_vol} < {min_liq}")
            return None

        if entry_conditions:
            ok, details = evaluate_conditions(entry_conditions, feats)
            if not ok:
                self._reject(alpha_id, symbol, ReasonCode.CONDITIONS_NOT_MET,
                             "; ".join(d for d in details if d.startswith("FAIL"))[:200])
                return None
        else:
            details = ["no entry conditions declared — alpha matches unconditionally"]

        price = feats.get("price")
        direction = (alpha.get("direction") or "long").lower()
        entry_plan = {
            "type": "market",
            "reference_price": price,
            "entry_window": alpha.get("entry_window"),
        }
        exit_plan = self._build_exit_plan(alpha, price, feats)

        return OpportunityCandidate(
            candidate_id=str(uuid.uuid4()),
            alpha_id=alpha_id,
            alpha_version=str(alpha.get("alpha_version") or "1"),
            symbol=symbol,
            asset_class=alpha.get("asset_class") or alpha.get("market") or "unknown",
            direction=direction,
            signal_time=_utcnow(),
            execution_mode=execution_mode,
            entry_plan=entry_plan,
            exit_plan=exit_plan,
            features=feats,
            edge_health_score=alpha.get("edge_health_score"),
            regime_fit=1.0,
            liquidity_score=self._liquidity_score(dollar_vol),
            condition_details=details,
        )

    @staticmethod
    def _build_exit_plan(alpha, price, feats) -> Dict[str, Any]:
        """Strategy-specific exits from the execution contract; ATR fallback."""
        import json as _json

        def _load(key):
            raw = alpha.get(key)
            if isinstance(raw, str):
                try:
                    return _json.loads(raw)
                except _json.JSONDecodeError:
                    return None
            return raw

        stop_policy = _load("stop_policy")
        profit_policy = _load("profit_policy")
        plan: Dict[str, Any] = {
            "stop_policy": stop_policy,
            "profit_policy": profit_policy,
            "time_stop_bars": alpha.get("time_stop_bars"),
            "exit_window": alpha.get("exit_window"),
            "exit_conditions": alpha.get("exit_conditions"),
            "source": "alpha_contract",
        }
        atr_pct = feats.get("atr_pct") or 0.02
        if price:
            if stop_policy and stop_policy.get("type") == "pct":
                plan["stop_price"] = price * (1 - stop_policy["value"])
            elif stop_policy and stop_policy.get("type") == "atr":
                plan["stop_price"] = price * (1 - atr_pct * stop_policy.get("value", 2.0))
            else:
                plan["stop_price"] = price * (1 - atr_pct * 2.0)
                plan["source"] = "atr_fallback"
            if profit_policy and profit_policy.get("type") == "pct":
                plan["target_price"] = price * (1 + profit_policy["value"])
            elif profit_policy and profit_policy.get("type") == "atr":
                plan["target_price"] = price * (1 + atr_pct * profit_policy.get("value", 3.0))
            else:
                plan["target_price"] = price * (1 + atr_pct * 3.0)
        return plan

    @staticmethod
    def _liquidity_score(dollar_vol: Optional[float]) -> Optional[float]:
        if dollar_vol is None or dollar_vol <= 0:
            return None
        import math
        return max(0.0, min(1.0, (math.log10(dollar_vol) - 4) / 5))  # 10k→0, 1B→1

    def _reject(self, alpha_id: str, symbol: str, code: str, detail: str = "") -> None:
        self.last_rejections.append(Rejection(alpha_id, symbol, code, detail))
        logger.debug(f"AlphaSignalEngine: [{code}] {alpha_id} {symbol} {detail}")

    def rejection_summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.last_rejections:
            out[r.reason_code] = out.get(r.reason_code, 0) + 1
        return out
