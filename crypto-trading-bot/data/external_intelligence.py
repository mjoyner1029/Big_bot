"""External Intelligence Engine — foundation layer.

Ingests legitimate external market-information sources, normalizes them into
TIME-SAFE structured records, and exposes point-in-time features to the
existing research framework. It NEVER places trades:

    External Sources → Source Adapters → Raw Event Store → Normalization
    → Point-in-Time Feature Store → ResearchCampaignRunner → Validation
    → Alpha Library

Core disciplines:
  - point-in-time: every record carries event_time, publication_time,
    first_seen_time, ingestion_time; research may only use information at
    max(publication_time, first_seen_time) + modeled latency
  - API-first collection; compliant web collection only where permitted
  - source provenance/versioning/reliability metadata on every record
  - external text is DATA, never instructions (prompt-injection defense)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

AVAILABLE = "AVAILABLE"
UNAVAILABLE = "UNAVAILABLE"
UNAVAILABLE_POLICY = "UNAVAILABLE_POLICY"   # collection not permitted
DEGRADED = "DEGRADED"


# ── Point-in-time record model (spec §38-39) ──────────────────────────────────


@dataclass
class ExternalEvent:
    """One normalized external record with full time-safety metadata."""
    source_name: str
    event_type: str
    event_time: str                  # when the event actually occurred
    publication_time: str            # when it became PUBLIC
    first_seen_time: str             # when THIS system first observed it
    payload: Dict[str, Any] = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    ingestion_time: str = field(default_factory=_utcnow)
    adapter_version: str = "1"
    schema_version: str = "1"
    record_hash: str = ""

    def __post_init__(self):
        if not self.record_hash:
            payload = json.dumps(
                {"s": self.source_name, "t": self.event_type,
                 "e": self.event_time, "p": self.payload}, sort_keys=True,
                default=str)
            self.record_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def available_from(self) -> str:
        """Earliest timestamp the bot COULD have known this information.
        Never event_time — always publication/observation based."""
        return max(self.publication_time, self.first_seen_time)


# ── Source adapter contract (spec §4) ─────────────────────────────────────────


@dataclass
class RateLimitPolicy:
    requests_per_minute: int = 30
    backoff_seconds: float = 2.0


class ExternalDataSource(ABC):
    """Contract every external source adapter implements."""

    source_name: str = "base"
    source_type: str = "generic"          # government | political | crypto | market
    capabilities: List[str] = []
    adapter_version: str = "1"
    schema_version: str = "1"
    # 'official_api' | 'official_dataset' | 'permitted_aggregator'
    # | 'permitted_web' | 'headless_browser'
    collection_method: str = "official_api"
    typical_publication_lag_days: float = 0.0   # publication-lag model (spec §39)

    @abstractmethod
    def availability(self) -> str:
        """AVAILABLE / UNAVAILABLE / UNAVAILABLE_POLICY / DEGRADED."""

    @abstractmethod
    def fetch_since(self, since_iso: str) -> List[ExternalEvent]:
        """Fetch normalized events observed since the given timestamp.
        Must fail soft (return []) — never raise into the research loop."""

    def health_check(self) -> Dict[str, Any]:
        return {"source": self.source_name, "availability": self.availability()}

    def rate_limit_policy(self) -> RateLimitPolicy:
        return RateLimitPolicy()

    def terms_policy(self) -> str:
        """Human-readable summary of why this collection method is permitted."""
        return "official public API"

    def point_in_time_semantics(self) -> str:
        return ("features available at max(publication_time, first_seen_time); "
                f"typical publication lag {self.typical_publication_lag_days}d")


# ── Raw event store (immutable, provenance-preserving) ────────────────────────

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS external_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name       TEXT NOT NULL,
    event_type        TEXT NOT NULL,
    event_time        TEXT NOT NULL,
    publication_time  TEXT NOT NULL,
    first_seen_time   TEXT NOT NULL,
    ingestion_time    TEXT NOT NULL,
    symbols           TEXT,
    payload           TEXT,
    adapter_version   TEXT,
    schema_version    TEXT,
    record_hash       TEXT UNIQUE
)
"""

_CREATE_HEALTH = """
CREATE TABLE IF NOT EXISTS source_health (
    source_name       TEXT PRIMARY KEY,
    last_success      TEXT,
    last_failure      TEXT,
    consecutive_failures INTEGER DEFAULT 0,
    schema_broken     INTEGER DEFAULT 0,
    disabled          INTEGER DEFAULT 0,
    notes             TEXT
)
"""


class RawEventStore:
    """Immutable, deduplicated store of normalized external events."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite") -> None:
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_EVENTS)
            conn.execute(_CREATE_HEALTH)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ee_source "
                         "ON external_events(source_name, publication_time)")
            conn.commit()

    def ingest(self, events: Sequence[ExternalEvent]) -> int:
        n = 0
        with sqlite3.connect(self.db_path) as conn:
            for e in events:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO external_events "
                        "(source_name, event_type, event_time, publication_time, "
                        " first_seen_time, ingestion_time, symbols, payload, "
                        " adapter_version, schema_version, record_hash) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (e.source_name, e.event_type, e.event_time,
                         e.publication_time, e.first_seen_time, e.ingestion_time,
                         json.dumps(e.symbols), json.dumps(e.payload, default=str),
                         e.adapter_version, e.schema_version, e.record_hash),
                    )
                    n += conn.total_changes and 1
                except sqlite3.Error as err:
                    logger.warning(f"RawEventStore ingest error: {err}")
            conn.commit()
        return n

    def events_available_at(self, as_of_iso: str,
                            source_name: Optional[str] = None,
                            symbol: Optional[str] = None,
                            event_type: Optional[str] = None,
                            since_iso: Optional[str] = None) -> List[ExternalEvent]:
        """POINT-IN-TIME query: only records whose publication/first-seen time
        is <= as_of. Backtests can never see information before it existed."""
        q = ("SELECT source_name, event_type, event_time, publication_time, "
             "first_seen_time, ingestion_time, symbols, payload, "
             "adapter_version, schema_version, record_hash "
             "FROM external_events WHERE MAX(publication_time, first_seen_time) <= ?")
        args: list = [as_of_iso]
        if source_name:
            q += " AND source_name=?"
            args.append(source_name)
        if event_type:
            q += " AND event_type=?"
            args.append(event_type)
        if since_iso:
            q += " AND MAX(publication_time, first_seen_time) >= ?"
            args.append(since_iso)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            symbols = json.loads(r[6] or "[]")
            if symbol and symbol not in symbols:
                continue
            out.append(ExternalEvent(
                source_name=r[0], event_type=r[1], event_time=r[2],
                publication_time=r[3], first_seen_time=r[4], ingestion_time=r[5],
                symbols=symbols, payload=json.loads(r[7] or "{}"),
                adapter_version=r[8], schema_version=r[9], record_hash=r[10]))
        return out

    # ── Source health (spec §42): schema breaks fail CLOSED ──────────────────

    def record_success(self, source_name: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO source_health (source_name, last_success, consecutive_failures) "
                "VALUES (?,?,0) ON CONFLICT(source_name) DO UPDATE SET "
                "last_success=excluded.last_success, consecutive_failures=0",
                (source_name, _utcnow()))
            conn.commit()

    def record_failure(self, source_name: str, schema_broken: bool = False,
                       note: str = "") -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO source_health (source_name, last_failure, "
                "consecutive_failures, schema_broken, disabled, notes) "
                "VALUES (?,?,1,?,?,?) ON CONFLICT(source_name) DO UPDATE SET "
                "last_failure=excluded.last_failure, "
                "consecutive_failures=consecutive_failures+1, "
                "schema_broken=MAX(schema_broken, excluded.schema_broken), "
                "disabled=MAX(disabled, excluded.disabled), notes=excluded.notes",
                (source_name, _utcnow(), int(schema_broken), int(schema_broken),
                 note))
            conn.commit()
        if schema_broken:
            logger.critical(
                f"Source '{source_name}' DISABLED: schema change detected — "
                "failing closed rather than ingesting corrupted data")

    def is_disabled(self, source_name: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT disabled FROM source_health WHERE source_name=?",
                (source_name,)).fetchone()
        return bool(row and row[0])


# ── Source reliability (spec §40) ─────────────────────────────────────────────


def source_reliability_score(
    *, official: bool, timestamp_reliability: float = 1.0,
    revision_risk: float = 0.0, missingness: float = 0.0,
    latency_days: float = 0.0,
) -> float:
    """0-1 reliability used as research metadata — low-confidence scraped data
    never silently receives official-API authority."""
    score = 0.5 + (0.3 if official else 0.0)
    score += 0.2 * max(0.0, min(timestamp_reliability, 1.0))
    score -= 0.2 * max(0.0, min(revision_risk, 1.0))
    score -= 0.2 * max(0.0, min(missingness, 1.0))
    score -= min(latency_days / 60.0, 0.2)
    return max(0.0, min(1.0, score))


# ── Compliant web collection (spec §6, §63-64) ───────────────────────────────


class CompliantWebCollector:
    """Permitted-webpage extraction ONLY. Identifies itself, rate limits,
    caches by content hash, and FAILS CLOSED on schema changes. Never bypasses
    authentication, CAPTCHA, or anti-bot controls."""

    def __init__(self, user_agent: str = "big-bot-research/1.0 (compliant collector)",
                 min_interval_seconds: float = 2.0) -> None:
        self.user_agent = user_agent
        self.min_interval_seconds = min_interval_seconds
        self._last_fetch: Dict[str, float] = {}
        self._etags: Dict[str, str] = {}
        self._hashes: Dict[str, str] = {}

    def fetch(self, url: str, permitted: bool = False,
              expected_schema_marker: Optional[str] = None) -> Optional[str]:
        if not permitted:
            logger.warning(f"CompliantWebCollector: '{url}' not marked permitted "
                           "— refusing (UNAVAILABLE_POLICY)")
            return None
        now = time.time()
        wait = self.min_interval_seconds - (now - self._last_fetch.get(url, 0))
        if wait > 0:
            time.sleep(wait)
        try:
            import requests
            headers = {"User-Agent": self.user_agent}
            if url in self._etags:
                headers["If-None-Match"] = self._etags[url]
            resp = requests.get(url, headers=headers, timeout=10)
            self._last_fetch[url] = time.time()
            if resp.status_code == 304:
                return None   # unchanged
            if resp.status_code != 200:
                return None
            if "ETag" in resp.headers:
                self._etags[url] = resp.headers["ETag"]
            text = resp.text
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            if self._hashes.get(url) == content_hash:
                return None   # duplicate content
            self._hashes[url] = content_hash
            # Schema change detection: fail closed (spec §64)
            if expected_schema_marker and expected_schema_marker not in text:
                logger.critical(
                    f"CompliantWebCollector: schema marker missing at {url} — "
                    "failing closed")
                raise SchemaChangedError(url)
            return text
        except SchemaChangedError:
            raise
        except Exception as e:
            logger.debug(f"CompliantWebCollector: {url}: {e}")
            return None


class SchemaChangedError(RuntimeError):
    pass


# ── Prompt-injection defense (spec §84-85) ────────────────────────────────────

_INJECTION_PATTERNS = re.compile(
    r"(ignore (all )?(previous|prior|above) (instructions|prompts)|"
    r"disregard (the )?(system|previous)|you are now|new instructions:|"
    r"execute|run the following|<script|javascript:|os\.system|subprocess)",
    re.IGNORECASE)


def sanitize_external_text(text: str, max_length: int = 2000) -> str:
    """External content is DATA, never instructions. Strips markup/control
    characters, truncates, and neutralizes instruction-like phrasing before
    any LLM exposure. Website text can never execute or direct trading."""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", text)                 # strip HTML
    cleaned = re.sub(r"[\x00-\x08\x0b-\x1f]", "", cleaned)  # control chars
    cleaned = _INJECTION_PATTERNS.sub("[NEUTRALIZED]", cleaned)
    return cleaned[:max_length]


def contains_injection_attempt(text: str) -> bool:
    return bool(_INJECTION_PATTERNS.search(text or ""))


# ── Source registry (spec §61) ────────────────────────────────────────────────


class ExternalSourceRegistry:
    """Configuration-driven registry of source adapters."""

    def __init__(self, event_store: Optional[RawEventStore] = None) -> None:
        self._sources: Dict[str, ExternalDataSource] = {}
        self.event_store = event_store

    def register(self, source: ExternalDataSource) -> None:
        self._sources[source.source_name] = source

    def get(self, name: str) -> Optional[ExternalDataSource]:
        return self._sources.get(name)

    def status_report(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "availability": s.availability(),
                "type": s.source_type,
                "collection_method": s.collection_method,
                "publication_lag_days": s.typical_publication_lag_days,
                "terms": s.terms_policy(),
                "disabled": (self.event_store.is_disabled(name)
                             if self.event_store else False),
            }
            for name, s in self._sources.items()
        }

    def refresh_all(self, since_iso: str) -> Dict[str, int]:
        """Fetch + ingest from every available source. Fail-soft per source."""
        counts: Dict[str, int] = {}
        for name, source in self._sources.items():
            if self.event_store and self.event_store.is_disabled(name):
                counts[name] = -1
                continue
            if source.availability() != AVAILABLE:
                counts[name] = 0
                continue
            try:
                events = source.fetch_since(since_iso)
                counts[name] = (self.event_store.ingest(events)
                                if self.event_store else len(events))
                if self.event_store:
                    self.event_store.record_success(name)
            except SchemaChangedError:
                if self.event_store:
                    self.event_store.record_failure(name, schema_broken=True)
                counts[name] = -1
            except Exception as e:
                logger.warning(f"Source '{name}' refresh failed: {e}")
                if self.event_store:
                    self.event_store.record_failure(name, note=str(e)[:200])
                counts[name] = 0
        return counts
