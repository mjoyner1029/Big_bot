"""Research data-source interfaces (spec §57-69).

Adapter interfaces for high-value data families that have NO configured
provider yet: options, short interest / borrow, earnings revisions, SEC
filing changes, insider transactions, institutional (13F), macro.

Every interface follows the ExternalDataSource contract: without a real
provider it reports UNAVAILABLE and returns nothing — data is NEVER
fabricated. All text from filings is untrusted and sanitized. Availability
timestamps always use filing/publication dates, never portfolio dates.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from data.external_intelligence import (
    AVAILABLE,
    UNAVAILABLE,
    ExternalDataSource,
    ExternalEvent,
    sanitize_external_text,
)

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


# ── Canonical event identity (spec §85-86) ────────────────────────────────────


def canonical_event_id(*, event_type: str, primary_entity: str,
                       event_time: str, magnitude: Optional[float] = None) -> str:
    """Same underlying real-world event reported by multiple sources maps to
    ONE canonical id: type + entity + day (+ magnitude bucket)."""
    day = str(event_time)[:10]
    bucket = ""
    if magnitude is not None and magnitude > 0:
        import math
        bucket = f":{int(math.log10(max(magnitude, 1.0)))}"
    payload = f"{event_type}:{primary_entity.strip().upper()}:{day}{bucket}"
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def deduplicate_events(events: Sequence[ExternalEvent]) -> List[ExternalEvent]:
    """Keep the EARLIEST-available record per canonical event; attach the
    others as corroborating sources."""
    by_canonical: Dict[str, List[ExternalEvent]] = {}
    for e in events:
        cid = e.payload.get("canonical_event_id") or canonical_event_id(
            event_type=e.event_type,
            primary_entity=(e.symbols[0] if e.symbols else
                            str(e.payload.get("recipient_name", ""))[:40]),
            event_time=e.event_time,
            magnitude=e.payload.get("award_amount") or e.payload.get("size"))
        e.payload["canonical_event_id"] = cid
        by_canonical.setdefault(cid, []).append(e)
    out = []
    for cid, group in by_canonical.items():
        group.sort(key=lambda e: e.available_from)
        primary = group[0]
        if len(group) > 1:
            primary.payload["corroborating_sources"] = [
                g.source_name for g in group[1:]]
        out.append(primary)
    return out


# ── Base for provider-less interfaces ─────────────────────────────────────────


class _AdapterSource(ExternalDataSource):
    """Interface-complete source that is UNAVAILABLE until a provider/fetch
    function is injected. Never fabricates data."""

    def __init__(self, fetch_fn: Optional[Callable[[str], List[Dict]]] = None) -> None:
        self._fetch_fn = fetch_fn

    def availability(self) -> str:
        return AVAILABLE if self._fetch_fn else UNAVAILABLE

    def fetch_since(self, since_iso: str) -> List[ExternalEvent]:
        if not self._fetch_fn:
            return []
        return [e for e in (self.normalize(r) for r in self._fetch_fn(since_iso)) if e]

    def normalize(self, r: Dict) -> Optional[ExternalEvent]:  # pragma: no cover
        raise NotImplementedError


# ── Options intelligence (spec §57-59) ────────────────────────────────────────


class OptionsIntelligenceSource(_AdapterSource):
    """IV / skew / term structure / unusual activity as research FEATURES.
    No options execution is implied or required."""

    source_name = "options_intelligence"
    source_type = "market"
    collection_method = "official_api"
    typical_publication_lag_days = 0.0

    FEATURES = ("implied_volatility", "iv_percentile", "skew_25d",
                "term_structure_slope", "put_call_volume_ratio",
                "expected_move_pct", "unusual_volume_ratio", "oi_change_1d")

    def normalize(self, r: Dict) -> Optional[ExternalEvent]:
        try:
            obs = str(r["observation_time"])
            return ExternalEvent(
                source_name=self.source_name, event_type="options_snapshot",
                event_time=obs, publication_time=obs, first_seen_time=obs,
                symbols=[str(r["symbol"]).upper()],
                payload={k: r.get(k) for k in self.FEATURES} | {
                    "symbol": str(r["symbol"]).upper()})
        except (KeyError, TypeError):
            return None


def volatility_risk_premium(implied_vol: float, realized_vol: float
                            ) -> Optional[float]:
    """IV − RV (annualized). Research feature only (spec §59)."""
    if implied_vol is None or realized_vol is None or realized_vol < 0:
        return None
    return implied_vol - realized_vol


# ── Short interest / borrow (spec §60) ────────────────────────────────────────


class ShortInterestSource(_AdapterSource):
    source_name = "short_interest"
    source_type = "market"
    collection_method = "official_api"
    typical_publication_lag_days = 10.0     # exchange SI data is delayed

    def normalize(self, r: Dict) -> Optional[ExternalEvent]:
        try:
            pub = str(r.get("publication_date") or r["report_date"])
            return ExternalEvent(
                source_name=self.source_name, event_type="short_interest",
                event_time=str(r["report_date"]), publication_time=pub,
                first_seen_time=r.get("first_seen", pub),
                symbols=[str(r["symbol"]).upper()],
                payload={
                    "short_interest_shares": r.get("short_interest_shares"),
                    "days_to_cover": r.get("days_to_cover"),
                    "borrow_fee_annual": r.get("borrow_fee_annual"),
                    "utilization": r.get("utilization"),
                    "shares_available": r.get("shares_available"),
                })
        except (KeyError, TypeError):
            return None


# ── Earnings / revenue revisions (spec §61-62) ────────────────────────────────


class EarningsRevisionSource(_AdapterSource):
    source_name = "earnings_revisions"
    source_type = "market"
    collection_method = "official_api"
    typical_publication_lag_days = 1.0

    def normalize(self, r: Dict) -> Optional[ExternalEvent]:
        try:
            pub = str(r["revision_date"])
            return ExternalEvent(
                source_name=self.source_name, event_type="estimate_revision",
                event_time=pub, publication_time=pub,
                first_seen_time=r.get("first_seen", pub),
                symbols=[str(r["symbol"]).upper()],
                payload={
                    "metric": r.get("metric", "eps"),
                    "direction": r.get("direction"),            # up | down
                    "magnitude_pct": r.get("magnitude_pct"),
                    "analyst_count": r.get("analyst_count"),
                    "period": r.get("period"),
                })
        except (KeyError, TypeError):
            return None


def revision_acceleration_score(revisions: Sequence[Dict[str, Any]],
                                window_days: int = 30) -> Dict[str, Any]:
    """Count/magnitude/concentration of upward revisions in the window
    (already filtered point-in-time by the caller)."""
    ups = [r for r in revisions if r.get("direction") == "up"]
    downs = [r for r in revisions if r.get("direction") == "down"]
    up_mag = sum(abs(r.get("magnitude_pct") or 0) for r in ups)
    down_mag = sum(abs(r.get("magnitude_pct") or 0) for r in downs)
    n = len(ups) + len(downs)
    return {
        "up_revisions": len(ups),
        "down_revisions": len(downs),
        "net_revision_count": len(ups) - len(downs),
        "net_revision_magnitude": up_mag - down_mag,
        "revision_breadth": (len(ups) - len(downs)) / n if n else 0.0,
        "revision_acceleration_score": (len(ups) - len(downs)) * (1 + up_mag - down_mag)
        if n else 0.0,
    }


# ── SEC filing change detection (spec §63-64) ─────────────────────────────────


class SECFilingChangeSource(_AdapterSource):
    """Diff between a filing and the previous comparable filing. Filing text
    is UNTRUSTED — sanitized, summarized, never followed as instructions."""

    source_name = "sec_filing_changes"
    source_type = "government"
    collection_method = "official_api"       # EDGAR full-text/API
    typical_publication_lag_days = 0.0       # availability = EDGAR acceptance time

    SECTIONS = ("risk_factors", "revenue_discussion", "margins", "capex",
                "inventory", "debt", "liquidity", "customer_concentration",
                "buybacks", "guidance_language")

    def normalize(self, r: Dict) -> Optional[ExternalEvent]:
        try:
            pub = str(r["filing_date"])
            changes = {k: sanitize_external_text(str(r.get(f"{k}_change") or ""), 500)
                       for k in self.SECTIONS if r.get(f"{k}_change")}
            return ExternalEvent(
                source_name=self.source_name, event_type="filing_change",
                event_time=pub, publication_time=pub,
                first_seen_time=r.get("first_seen", pub),
                symbols=[str(r["symbol"]).upper()],
                payload={"form_type": r.get("form_type"),
                         "sections_changed": sorted(changes),
                         "changes": changes,
                         "change_count": len(changes)})
        except (KeyError, TypeError):
            return None


# ── Insider transactions (spec §65-66) ────────────────────────────────────────


class InsiderTransactionSource(_AdapterSource):
    source_name = "insider_transactions"
    source_type = "government"
    collection_method = "official_api"       # Form 4 via EDGAR
    typical_publication_lag_days = 2.0       # Form 4 due within 2 business days

    TRANSACTION_TYPES = ("open_market_purchase", "sale", "option_exercise",
                         "award", "automatic_plan")

    def normalize(self, r: Dict) -> Optional[ExternalEvent]:
        try:
            pub = str(r["filing_date"])                 # availability = filing
            return ExternalEvent(
                source_name=self.source_name, event_type="insider_transaction",
                event_time=str(r["transaction_date"]), publication_time=pub,
                first_seen_time=r.get("first_seen", pub),
                symbols=[str(r["symbol"]).upper()],
                payload={
                    "insider_name": sanitize_external_text(
                        str(r.get("insider_name", "")), 100),
                    "role": r.get("role"),
                    "transaction_type": r.get("transaction_type"),
                    "shares": r.get("shares"),
                    "value_usd": r.get("value_usd"),
                    "holdings_after": r.get("holdings_after"),
                })
        except (KeyError, TypeError):
            return None


def insider_cluster_features(transactions: Sequence[Dict[str, Any]]
                             ) -> Dict[str, Any]:
    """Cluster features from point-in-time-filtered insider transactions.
    Only OPEN-MARKET purchases carry conviction weight (spec §66)."""
    buys = [t for t in transactions
            if t.get("transaction_type") == "open_market_purchase"]
    roles = {str(t.get("role", "")).upper() for t in buys}
    buyers = {t.get("insider_name") for t in buys if t.get("insider_name")}
    total_value = sum(t.get("value_usd") or 0 for t in buys)
    return {
        "open_market_buy_count": len(buys),
        "unique_insider_buyers": len(buyers),
        "insider_cluster": len(buyers) >= 3,
        "ceo_cfo_cluster": {"CEO", "CFO"} <= roles,
        "board_cluster": sum(1 for r in roles if "DIRECTOR" in r) >= 2,
        "total_buy_value_usd": total_value,
    }


# ── Institutional 13F (spec §67) ──────────────────────────────────────────────


class InstitutionalFilingSource(_AdapterSource):
    """13F positioning. POINT-IN-TIME RULE: availability is the FILING date
    (up to 45 days after quarter end), never the portfolio as-of date."""

    source_name = "institutional_13f"
    source_type = "government"
    collection_method = "official_api"
    typical_publication_lag_days = 45.0

    def normalize(self, r: Dict) -> Optional[ExternalEvent]:
        try:
            filing = str(r["filing_date"])
            return ExternalEvent(
                source_name=self.source_name, event_type="institutional_position",
                event_time=str(r["period_end"]),        # portfolio as-of
                publication_time=filing,                # availability = filing
                first_seen_time=r.get("first_seen", filing),
                symbols=[str(r["symbol"]).upper()],
                payload={
                    "institution": sanitize_external_text(
                        str(r.get("institution", "")), 120),
                    "shares": r.get("shares"),
                    "value_usd": r.get("value_usd"),
                    "change_shares": r.get("change_shares"),
                    "period_end": str(r["period_end"]),
                })
        except (KeyError, TypeError):
            return None


# ── Macro (spec §68-69) ───────────────────────────────────────────────────────


class MacroDataSource(_AdapterSource):
    """Macro series snapshots (yields, curve, USD, oil, gold, credit, vol
    indices, inflation, employment, liquidity proxies) as research features.
    Cross-asset relationships are ESTIMATED, never hard-coded."""

    source_name = "macro"
    source_type = "market"
    collection_method = "official_api"
    typical_publication_lag_days = 0.0

    SERIES = ("treasury_10y", "treasury_2y", "yield_curve_2s10s", "usd_index",
              "oil", "gold", "credit_spread_hy", "vix", "inflation_yoy",
              "unemployment", "fed_balance_sheet")

    def normalize(self, r: Dict) -> Optional[ExternalEvent]:
        try:
            pub = str(r.get("publication_time") or r["observation_time"])
            return ExternalEvent(
                source_name=self.source_name, event_type="macro_observation",
                event_time=str(r["observation_time"]), publication_time=pub,
                first_seen_time=r.get("first_seen", pub),
                payload={"series": r["series"], "value": r.get("value")})
        except (KeyError, TypeError):
            return None


def rolling_cross_asset_relationship(x: Sequence[float], y: Sequence[float],
                                     window: int = 60) -> Dict[str, Any]:
    """Rolling beta/correlation between two return series — relationships are
    measured and monitored for structural change, never assumed (spec §69)."""
    import numpy as np
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = min(len(xs), len(ys))
    if n < window * 2:
        return {"available": False}
    xs, ys = xs[-n:], ys[-n:]
    halves = []
    for seg_x, seg_y in ((xs[: n // 2], ys[: n // 2]), (xs[n // 2:], ys[n // 2:])):
        vx = np.var(seg_x)
        beta = float(np.cov(seg_y, seg_x)[0, 1] / vx) if vx > 0 else 0.0
        sx, sy = np.std(seg_x), np.std(seg_y)
        corr = float(np.corrcoef(seg_x, seg_y)[0, 1]) if sx > 0 and sy > 0 else 0.0
        halves.append({"beta": beta, "corr": corr})
    return {
        "available": True,
        "beta_first_half": halves[0]["beta"], "beta_second_half": halves[1]["beta"],
        "corr_first_half": halves[0]["corr"], "corr_second_half": halves[1]["corr"],
        "beta_shift": halves[1]["beta"] - halves[0]["beta"],
        "structural_change": abs(halves[1]["beta"] - halves[0]["beta"])
        > max(abs(halves[0]["beta"]) * 0.75, 0.25),
    }


def default_interface_sources() -> List[ExternalDataSource]:
    """All adapter interfaces, provider-less (UNAVAILABLE) by default."""
    return [OptionsIntelligenceSource(), ShortInterestSource(),
            EarningsRevisionSource(), SECFilingChangeSource(),
            InsiderTransactionSource(), InstitutionalFilingSource(),
            MacroDataSource()]
