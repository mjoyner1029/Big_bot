"""Concrete external-intelligence source adapters.

Implemented (API-first, spec §5):
  - USASpendingSource        — official api.usaspending.gov (no scraping)
  - CongressionalTradesSource — official disclosure feeds via pluggable
    fetcher; the availability timestamp is the DISCLOSURE time, never the
    transaction time (spec §12)
  - HyperliquidPublicSource  — official public info API
  - WalletIntelligenceEngine — public on-chain/derivatives wallet activity
    (features only, never copy-trades)

Plus: GovernmentRecipientResolver (name→ticker, confidence-scored),
member/trader skill models with shrinkage, and point-in-time feature builders.
All network paths fail soft; every record carries publication-lag metadata.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from data.external_intelligence import (
    AVAILABLE,
    UNAVAILABLE,
    ExternalDataSource,
    ExternalEvent,
    RawEventStore,
    sanitize_external_text,
    source_reliability_score,
)

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


def _http_post_json(url: str, payload: Dict, timeout: int = 10) -> Optional[Dict]:
    try:
        import requests
        resp = requests.post(url, json=payload, timeout=timeout,
                             headers={"User-Agent": "big-bot-research/1.0"})
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        logger.debug(f"http_post {url}: {e}")
        return None


def _http_get_json(url: str, params: Dict = None, timeout: int = 10) -> Optional[Any]:
    try:
        import requests
        resp = requests.get(url, params=params or {}, timeout=timeout,
                            headers={"User-Agent": "big-bot-research/1.0"})
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        logger.debug(f"http_get {url}: {e}")
        return None


# ── USAspending (spec §7-10) ─────────────────────────────────────────────────


class USASpendingSource(ExternalDataSource):
    """Official USAspending API adapter. Point-in-time: the feature becomes
    available at the record's publication (action/last-modified) date, never
    the award's effective date."""

    source_name = "usaspending"
    source_type = "government"
    capabilities = ["federal_awards"]
    collection_method = "official_api"
    typical_publication_lag_days = 14.0
    API = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

    def __init__(self, fetch_fn: Optional[Callable] = None) -> None:
        # fetch_fn injectable for tests / offline operation
        self._fetch_fn = fetch_fn

    def availability(self) -> str:
        try:
            import requests  # noqa: F401
            return AVAILABLE
        except ImportError:
            return UNAVAILABLE

    def terms_policy(self) -> str:
        return "official US government open-data API (api.usaspending.gov)"

    def fetch_since(self, since_iso: str) -> List[ExternalEvent]:
        raw = (self._fetch_fn(since_iso) if self._fetch_fn
               else self._fetch_api(since_iso))
        return [self.normalize(r) for r in raw if r]

    def _fetch_api(self, since_iso: str) -> List[Dict]:
        payload = {
            "filters": {
                "time_period": [{"start_date": since_iso[:10],
                                 "end_date": _utcnow()[:10]}],
                "award_type_codes": ["A", "B", "C", "D"],
            },
            "fields": ["Award ID", "Recipient Name", "Award Amount",
                       "Start Date", "End Date", "Awarding Agency",
                       "Funding Agency", "Last Modified Date", "Description",
                       "NAICS", "PSC"],
            "limit": 100, "page": 1,
        }
        data = _http_post_json(self.API, payload)
        return (data or {}).get("results", [])

    def normalize(self, r: Dict) -> Optional[ExternalEvent]:
        try:
            action_date = str(r.get("Start Date") or r.get("action_date") or "")
            publication = str(r.get("Last Modified Date")
                              or r.get("publication_date") or "")
            if not publication:
                # No verified publication timestamp → model the typical lag
                publication = self._lagged(action_date)
            return ExternalEvent(
                source_name=self.source_name,
                event_type="federal_award",
                event_time=action_date,
                publication_time=publication,
                # Official API with verified publication timestamp: a record
                # is observable from its publication (backfill-safe PIT)
                first_seen_time=r.get("first_seen", publication),
                payload={
                    "award_id": r.get("Award ID") or r.get("award_id"),
                    "recipient_name": sanitize_external_text(
                        str(r.get("Recipient Name") or r.get("recipient_name") or ""), 200),
                    "award_amount": float(r.get("Award Amount")
                                          or r.get("award_amount") or 0),
                    "awarding_agency": sanitize_external_text(
                        str(r.get("Awarding Agency") or ""), 120),
                    "funding_agency": sanitize_external_text(
                        str(r.get("Funding Agency") or ""), 120),
                    "naics": r.get("NAICS"),
                    "psc": r.get("PSC"),
                    "description": sanitize_external_text(
                        str(r.get("Description") or ""), 500),
                    "reliability": source_reliability_score(
                        official=True, latency_days=self.typical_publication_lag_days),
                },
            )
        except Exception as e:
            logger.debug(f"USASpending normalize: {e}")
            return None

    def _lagged(self, event_date: str) -> str:
        try:
            dt = datetime.fromisoformat(event_date[:10])
            return (dt + timedelta(days=self.typical_publication_lag_days)).isoformat()
        except ValueError:
            return _utcnow()


class GovernmentRecipientResolver:
    """Maps award recipients to public tickers with explicit confidence.
    Exact/normalized matching only — fuzzy matches never assumed correct."""

    _SUFFIXES = re.compile(
        r"\b(inc|corp|corporation|llc|ltd|co|company|holdings|group|plc|lp)\b\.?",
        re.IGNORECASE)

    def __init__(self, name_to_ticker: Optional[Dict[str, str]] = None) -> None:
        # Seed mapping; extend via configuration or a licensed entity dataset
        self._map = {self._norm(k): v for k, v in (name_to_ticker or {}).items()}

    @classmethod
    def _norm(cls, name: str) -> str:
        n = cls._SUFFIXES.sub("", (name or "").lower())
        return re.sub(r"[^a-z0-9 ]", "", n).strip()

    def resolve(self, recipient_name: str) -> Optional[Dict[str, Any]]:
        norm = self._norm(recipient_name)
        if not norm:
            return None
        if norm in self._map:
            return {"ticker": self._map[norm], "mapping_confidence": 0.95,
                    "method": "normalized_exact"}
        # Parent-prefix match (e.g. "X Subsidiary" → "X") — lower confidence
        for known, ticker in self._map.items():
            if norm.startswith(known + " ") or known.startswith(norm + " "):
                return {"ticker": ticker, "mapping_confidence": 0.7,
                        "method": "prefix"}
        return None   # never guess


def government_award_features(
    store: RawEventStore, ticker: str, as_of_iso: str,
    resolver: GovernmentRecipientResolver,
    market_cap: Optional[float] = None,
) -> Dict[str, Any]:
    """Point-in-time federal-award features for one ticker (spec §9).
    Uses only events published/observed at or before as_of."""
    events = store.events_available_at(as_of_iso, source_name="usaspending",
                                       event_type="federal_award")
    mine, mine_90d, mine_30d = [], [], []
    as_of = datetime.fromisoformat(as_of_iso.replace("Z", "+00:00"))
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    for e in events:
        mapping = resolver.resolve(e.payload.get("recipient_name", ""))
        if not mapping or mapping["ticker"] != ticker:
            continue
        if mapping["mapping_confidence"] < 0.7:
            continue
        mine.append(e)
        try:
            pub = datetime.fromisoformat(e.available_from.replace("Z", "+00:00"))
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            age = (as_of - pub).days
            if age <= 90:
                mine_90d.append(e)
            if age <= 30:
                mine_30d.append(e)
        except ValueError:
            continue
    amount_30d = sum(e.payload.get("award_amount", 0) for e in mine_30d)
    amount_90d = sum(e.payload.get("award_amount", 0) for e in mine_90d)
    return {
        "new_federal_award": bool(mine_30d),
        "award_amount_30d": amount_30d,
        "award_amount_90d": amount_90d,
        "award_amount_vs_market_cap": (amount_90d / market_cap
                                       if market_cap else None),
        "award_count_total": len(mine),
        "award_growth_30d_vs_90d": (amount_30d / (amount_90d / 3)
                                    if amount_90d > 0 else None),
    }


# ── Congressional trades (spec §11-15) ───────────────────────────────────────


class CongressionalTradesSource(ExternalDataSource):
    """Congressional disclosure adapter (official House/Senate feeds or a
    permitted aggregator via injected fetcher).

    CRITICAL (spec §12): transaction_date and disclosure_timestamp are stored
    separately, and the record's publication_time IS the disclosure time —
    research can never act on the trade before the public could know it.
    """

    source_name = "congressional_trades"
    source_type = "political"
    capabilities = ["member_trades"]
    collection_method = "official_dataset"
    typical_publication_lag_days = 30.0   # disclosures often lag by weeks

    def __init__(self, fetch_fn: Optional[Callable] = None) -> None:
        self._fetch_fn = fetch_fn

    def availability(self) -> str:
        return AVAILABLE if self._fetch_fn else UNAVAILABLE

    def terms_policy(self) -> str:
        return ("official House/Senate financial-disclosure feeds or a "
                "permitted aggregator; no ToS-violating scraping")

    def fetch_since(self, since_iso: str) -> List[ExternalEvent]:
        raw = self._fetch_fn(since_iso) if self._fetch_fn else []
        return [e for e in (self.normalize(r) for r in raw) if e]

    def normalize(self, r: Dict) -> Optional[ExternalEvent]:
        try:
            disclosure = str(r["disclosure_date"])
            return ExternalEvent(
                source_name=self.source_name,
                event_type="congress_trade",
                event_time=str(r["transaction_date"]),
                publication_time=disclosure,     # availability = DISCLOSURE
                first_seen_time=r.get("first_seen", disclosure),
                symbols=[str(r.get("ticker", "")).upper()] if r.get("ticker") else [],
                payload={
                    "member": sanitize_external_text(str(r.get("member", "")), 100),
                    "chamber": r.get("chamber"),
                    "transaction_type": r.get("transaction_type"),
                    "amount_range": r.get("amount_range"),
                    "owner": r.get("owner"),
                    "committees": r.get("committees") or [],
                    "transaction_date": str(r["transaction_date"]),
                    "disclosure_date": disclosure,
                    "reliability": source_reliability_score(
                        official=True, latency_days=30),
                },
            )
        except (KeyError, TypeError) as e:
            logger.debug(f"Congressional normalize skipped record: {e}")
            return None


def member_skill_scores(store: RawEventStore, as_of_iso: str,
                        excess_return_fn: Callable[[str, str, int], Optional[float]],
                        horizon_days: int = 60,
                        shrinkage_n: float = 20.0) -> Dict[str, Dict[str, Any]]:
    """Point-in-time member skill (spec §14): post-DISCLOSURE excess returns
    with shrinkage toward zero — 3 lucky trades never outrank 150 disclosures.

    excess_return_fn(ticker, from_iso, horizon_days) supplies historically
    observable post-disclosure excess return (None when unknown).
    """
    events = store.events_available_at(as_of_iso,
                                       source_name="congressional_trades",
                                       event_type="congress_trade")
    per_member: Dict[str, List[float]] = {}
    for e in events:
        if not e.symbols:
            continue
        if (e.payload.get("transaction_type") or "").lower() not in ("buy", "purchase"):
            continue
        ret = excess_return_fn(e.symbols[0], e.available_from, horizon_days)
        if ret is None:
            continue
        per_member.setdefault(e.payload.get("member", "unknown"), []).append(ret)

    out = {}
    for member, rets in per_member.items():
        n = len(rets)
        mean = sum(rets) / n
        shrunk = mean * (n / (n + shrinkage_n))   # shrinkage toward 0
        out[member] = {
            "n_disclosures": n,
            "raw_mean_excess": mean,
            "skill_score": max(0.0, min(100.0, 50.0 + shrunk * 1000)),
            "hit_rate": sum(1 for r in rets if r > 0) / n,
        }
    return out


def congressional_features(store: RawEventStore, ticker: str, as_of_iso: str,
                           skill: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
    """Point-in-time congressional features for one ticker (spec §13)."""
    as_of = datetime.fromisoformat(as_of_iso.replace("Z", "+00:00"))
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    events = store.events_available_at(as_of_iso,
                                       source_name="congressional_trades",
                                       event_type="congress_trade",
                                       symbol=ticker)
    buys_7, buys_30, sells_30 = [], [], []
    members_buying = set()
    for e in events:
        try:
            pub = datetime.fromisoformat(e.available_from.replace("Z", "+00:00"))
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        age = (as_of - pub).days
        side = (e.payload.get("transaction_type") or "").lower()
        if side in ("buy", "purchase"):
            if age <= 30:
                buys_30.append(e)
                members_buying.add(e.payload.get("member"))
            if age <= 7:
                buys_7.append(e)
        elif side in ("sell", "sale") and age <= 30:
            sells_30.append(e)
    skill = skill or {}
    weighted = sum(skill.get(e.payload.get("member"), {}).get("skill_score", 50.0)
                   for e in buys_30)
    return {
        "congress_buy_count_7d": len(buys_7),
        "congress_buy_count_30d": len(buys_30),
        "congress_sell_count_30d": len(sells_30),
        "net_congress_direction": len(buys_30) - len(sells_30),
        "unique_members_buying": len(members_buying),
        "clustered_buying": len(members_buying) >= 3,
        "skill_weighted_buying": weighted / max(len(buys_30), 1),
    }


# ── Crypto trader/wallet intelligence (spec §16-23, §71-72) ──────────────────


class HyperliquidPublicSource(ExternalDataSource):
    """Hyperliquid public info API adapter (positions/fills of public wallets).
    Observation latency is modeled — signals become available only after the
    bot could realistically have observed them."""

    source_name = "hyperliquid_public"
    source_type = "crypto"
    capabilities = ["trader_positions", "trader_fills"]
    collection_method = "official_api"
    typical_publication_lag_days = 0.0
    observation_latency_seconds: float = 60.0
    API = "https://api.hyperliquid.xyz/info"

    def __init__(self, wallets: Optional[List[str]] = None,
                 fetch_fn: Optional[Callable] = None) -> None:
        self.wallets = wallets or []
        self._fetch_fn = fetch_fn

    def availability(self) -> str:
        if self._fetch_fn:
            return AVAILABLE
        try:
            import requests  # noqa: F401
            return AVAILABLE if self.wallets else UNAVAILABLE
        except ImportError:
            return UNAVAILABLE

    def terms_policy(self) -> str:
        return "official Hyperliquid public info API"

    def fetch_since(self, since_iso: str) -> List[ExternalEvent]:
        raw = (self._fetch_fn(since_iso) if self._fetch_fn
               else self._fetch_api(since_iso))
        return [e for e in (self.normalize(r) for r in raw) if e]

    def _fetch_api(self, since_iso: str) -> List[Dict]:
        out = []
        for wallet in self.wallets[:20]:
            data = _http_post_json(self.API, {"type": "userFills", "user": wallet})
            for fill in (data or [])[:200]:
                fill["wallet"] = wallet
                out.append(fill)
        return out

    def normalize(self, r: Dict) -> Optional[ExternalEvent]:
        try:
            ts_ms = int(r.get("time") or 0)
            event_time = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) \
                .isoformat() if ts_ms else _utcnow()
            # Realistic observation latency (spec §71): the bot can act only
            # AFTER it could have seen the fill
            observed = (datetime.fromisoformat(event_time)
                        + timedelta(seconds=self.observation_latency_seconds)
                        ).isoformat()
            coin = str(r.get("coin", "")).upper()
            return ExternalEvent(
                source_name=self.source_name,
                event_type="trader_fill",
                event_time=event_time,
                publication_time=observed,
                first_seen_time=r.get("first_seen", observed),
                symbols=[f"{coin}-USD"] if coin else [],
                payload={
                    "wallet": str(r.get("wallet", ""))[:64],
                    "side": r.get("side"),
                    "size": float(r.get("sz") or 0),
                    "price": float(r.get("px") or 0),
                    "observed_price": float(r.get("observed_px") or r.get("px") or 0),
                    "reliability": source_reliability_score(official=True),
                },
            )
        except Exception as e:
            logger.debug(f"Hyperliquid normalize: {e}")
            return None


@dataclass
class TraderRecord:
    """Chronological per-trade returns of one public wallet/trader."""
    wallet: str
    returns: List[float]                 # fractional per-trade returns
    timestamps: List[str] = field(default_factory=list)
    funded_at: Optional[str] = None
    max_leverage: float = 1.0


def trader_skill_score(record: TraderRecord, oos_fraction: float = 0.4,
                       shrinkage_n: float = 30.0) -> Dict[str, Any]:
    """TraderSkillScore 0-100 (spec §18-19). Ranks by OUT-OF-SAMPLE
    persistence, shrunk by sample size; penalizes single-outlier PnL, tiny
    samples, and extreme leverage. Never lifetime PnL alone."""
    from core.validation_stats import (effective_sample_size,
                                       profit_concentration, sharpe_ratio)
    rets = record.returns
    n = len(rets)
    if n < 5:
        return {"skill_score": 0.0, "reason": "insufficient sample", "n": n}
    split = max(3, int(n * (1 - oos_fraction)))
    in_sample, oos = rets[:split], rets[split:]
    is_sharpe = sharpe_ratio(in_sample)
    oos_sharpe = sharpe_ratio(oos) if len(oos) >= 3 else 0.0
    persistence = (min(oos_sharpe / is_sharpe, 1.5)
                   if is_sharpe > 0 and oos_sharpe > 0 else 0.0)
    conc = profit_concentration(rets)["concentration_score"]
    ess = effective_sample_size(rets)
    shrink = ess / (ess + shrinkage_n)
    mean = sum(rets) / n
    base = max(0.0, min(1.0, 0.5 + mean * 20)) * shrink
    score = 100.0 * base * (0.4 + 0.6 * persistence) * (1.0 - 0.5 * conc)
    if record.max_leverage > 20:
        score *= 0.5     # extreme leverage penalty
    return {
        "skill_score": max(0.0, min(100.0, score)),
        "oos_persistence": persistence,
        "in_sample_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "profit_concentration": conc,
        "effective_sample": ess,
        "n": n,
    }


class WalletIntelligenceEngine:
    """Aggregates public wallet/trader activity into FEATURES (spec §20:
    never copy-trades). Skill scores decay with recency and are recomputed
    from rolling windows — nobody is permanently labeled GENIUS."""

    def __init__(self, store: RawEventStore) -> None:
        self.store = store
        self._skill: Dict[str, Dict[str, Any]] = {}

    def update_skill(self, records: Sequence[TraderRecord],
                     recency_half_life: int = 90) -> None:
        for r in records:
            base = trader_skill_score(r)
            # Recency decay: stale evidence loses authority (spec §70)
            if r.timestamps:
                try:
                    last = datetime.fromisoformat(
                        r.timestamps[-1].replace("Z", "+00:00"))
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=timezone.utc)
                    age = (datetime.now(timezone.utc) - last).days
                    base["skill_score"] *= 0.5 ** (age / recency_half_life)
                except ValueError:
                    pass
            self._skill[r.wallet] = base

    def skill_of(self, wallet: str) -> float:
        return self._skill.get(wallet, {}).get("skill_score", 0.0)

    def positioning_features(self, symbol: str, as_of_iso: str,
                             window_hours: float = 24.0) -> Dict[str, Any]:
        """Point-in-time, skill-weighted positioning features (spec §22)."""
        as_of = datetime.fromisoformat(as_of_iso.replace("Z", "+00:00"))
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        since = (as_of - timedelta(hours=window_hours)).isoformat()
        events = self.store.events_available_at(
            as_of_iso, source_name="hyperliquid_public",
            event_type="trader_fill", symbol=symbol, since_iso=since)
        long_p = short_p = 0.0
        for e in events:
            w = self.skill_of(e.payload.get("wallet", "")) / 100.0
            notional = e.payload.get("size", 0) * e.payload.get("observed_price", 0)
            side = str(e.payload.get("side", "")).upper()
            if side in ("B", "BUY", "LONG"):
                long_p += w * notional
            elif side in ("A", "S", "SELL", "SHORT"):
                short_p += w * notional
        total = long_p + short_p
        return {
            "top_trader_long_pressure": long_p,
            "top_trader_short_pressure": short_p,
            "skill_weighted_positioning": ((long_p - short_p) / total
                                           if total > 0 else 0.0),
            "observed_fills": len(events),
        }
