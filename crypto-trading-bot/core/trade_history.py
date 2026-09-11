"""Canonical trade-history access: ONE return formula, loud schema failures.

trade_memory canonical fields (core/trade_memory.py):
    size_dollars  — deployed dollar NOTIONAL (never multiply by entry_price)
    net_pnl       — dollars, NET of fees/slippage
    net_return_pct — percent (100 × fraction)

Canonical normalized return:
    net_return_fraction = net_pnl / size_dollars

Schema mismatches RAISE TradeMemorySchemaError — they never silently return
empty history that would make the portfolio look risk-free (spec §10-12).
"""
from __future__ import annotations

import logging
import sqlite3
import statistics as st
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

RETURN_NORMALIZATION_VERSION = "1.0.0"
SCHEMA_VERSION = 2

RESOLUTION_DAILY = "DAILY"
RESOLUTION_HOURLY = "HOURLY"
RESOLUTION_TRADE = "STRATEGY_WINDOW"


class TradeMemorySchemaError(RuntimeError):
    """TRADE_MEMORY_SCHEMA_MISMATCH — a production query referenced a column
    that does not exist. This is a code/schema bug, not empty data."""


class CoreAccountingInitializationError(RuntimeError):
    """Core accounting/risk infrastructure could not be initialized safely.
    The bot must FAIL CLOSED — never trade on untrusted records."""


@dataclass
class NormalizedTradeReturn:
    trade_id: int
    alpha_id: Optional[str]
    symbol: str
    exit_time: Optional[str]
    return_fraction: Optional[float]      # 0.01 == +1%
    pnl_usd: float
    notional_usd: float
    market_regime: str = "unknown"
    flags: Tuple[str, ...] = ()


def compute_trade_net_return(net_pnl: Optional[float],
                             size_dollars: Optional[float],
                             *, gross_capital_usd: Optional[float] = None
                             ) -> Optional[float]:
    """THE return formula. size_dollars is deployed notional (partial fills
    store FILLED notional at close). Multi-leg trades pass gross capital
    employed as the denominator. Shorts need no inversion — net_pnl is signed.
    Invalid/zero notional → None (never a division blow-up)."""
    denominator = gross_capital_usd if gross_capital_usd else size_dollars
    if net_pnl is None or denominator is None or denominator <= 0:
        return None
    return float(net_pnl) / float(denominator)


def _run_query(db_path: str, query: str, args: Sequence = ()) -> List[tuple]:
    """Distinguish EXPECTED EMPTY (missing table = no history yet) from
    SCHEMA FAILURE (missing column = bug) — the latter raises."""
    try:
        with sqlite3.connect(db_path) as conn:
            return conn.execute(query, list(args)).fetchall()
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "no such table" in msg:
            return []                       # legitimately no history yet
        logger.critical(f"TRADE_MEMORY_SCHEMA_MISMATCH: {e} — query: {query[:120]}")
        raise TradeMemorySchemaError(str(e)) from e


def fetch_attributed_returns(db_path: str,
                             alpha_id: Optional[str] = None
                             ) -> List[NormalizedTradeReturn]:
    """All closed attributed trades, normalized with the canonical formula."""
    q = ("SELECT tm.id, ta.alpha_id, tm.symbol, tm.exit_time, tm.net_pnl, "
         "tm.size_dollars, COALESCE(tm.market_regime,'unknown'), tm.entry_time "
         "FROM trade_attribution ta JOIN trade_memory tm "
         "ON tm.id = ta.trade_memory_id WHERE tm.exit_time IS NOT NULL ")
    args: List[Any] = []
    if alpha_id:
        q += "AND ta.alpha_id=? "
        args.append(alpha_id)
    q += "ORDER BY tm.exit_time"
    out = []
    for (tid, aid, symbol, exit_time, pnl, size, regime, entry_time) in \
            _run_query(db_path, q, args):
        flags: List[str] = []
        ret = compute_trade_net_return(pnl, size)
        if ret is None:
            flags.append("INVALID_NOTIONAL")
        elif abs(ret) > 10.0:
            flags.append("ABSURD_RETURN")   # >1000% — likely unit error
            ret = None
        if entry_time and exit_time and str(entry_time) > str(exit_time):
            flags.append("TIMESTAMPS_REVERSED")
            ret = None
        out.append(NormalizedTradeReturn(
            trade_id=tid, alpha_id=aid, symbol=symbol, exit_time=exit_time,
            return_fraction=ret, pnl_usd=float(pnl or 0.0),
            notional_usd=float(size or 0.0), market_regime=str(regime),
            flags=tuple(flags)))
    return out


def bucketed_alpha_returns(db_path: str,
                           resolution: str = RESOLUTION_DAILY
                           ) -> Dict[str, Dict[str, float]]:
    """alpha_id -> {bucket_timestamp: sum-weighted return}, built from
    canonical normalized returns. Buckets: DAILY (10 chars) or HOURLY (13)."""
    width = 13 if resolution == RESOLUTION_HOURLY else 10
    trades = fetch_attributed_returns(db_path)
    agg: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    for t in trades:
        if t.return_fraction is None or not t.alpha_id or not t.exit_time:
            continue
        bucket = str(t.exit_time)[:width]
        agg.setdefault(t.alpha_id, {}).setdefault(bucket, []).append(
            (t.return_fraction, t.notional_usd))
    out: Dict[str, Dict[str, float]] = {}
    for alpha_id, buckets in agg.items():
        out[alpha_id] = {}
        for bucket, pairs in buckets.items():
            total_notional = sum(n for _, n in pairs)
            if total_notional > 0:          # notional-weighted bucket return
                out[alpha_id][bucket] = sum(r * n for r, n in pairs) / total_notional
    return out


def choose_risk_resolution(db_path: str, *, min_hourly_trades: int = 60
                           ) -> str:
    """Median holding period + sample size decide the bucket resolution;
    intraday alphas with enough history get HOURLY buckets (spec §28-31)."""
    rows = _run_query(
        db_path,
        "SELECT tm.holding_hours FROM trade_attribution ta JOIN trade_memory tm "
        "ON tm.id = ta.trade_memory_id WHERE tm.exit_time IS NOT NULL "
        "AND tm.holding_hours IS NOT NULL")
    holds = [float(r[0]) for r in rows if r[0] is not None]
    if len(holds) >= min_hourly_trades and st.median(holds) < 4.0:
        return RESOLUTION_HOURLY
    return RESOLUTION_DAILY


# ── Database schema health (spec §13-14) ──────────────────────────────────────

REQUIRED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "trade_memory": ("id", "symbol", "entry_time", "exit_time", "entry_price",
                     "size_dollars", "quantity", "gross_pnl", "net_pnl",
                     "net_return_pct", "total_fees", "market_regime"),
    "trade_attribution": ("trade_memory_id", "alpha_id", "alpha_version",
                          "candidate_id"),
    "alpha_library": ("alpha_id", "lifecycle_state"),
}


class DatabaseSchemaHealthCheck:
    """Startup/runtime verification that critical modules' queries can run."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite") -> None:
        self.db_path = db_path

    def run(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {"ok": True, "tables": {}, "critical": []}
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version "
                "(id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER, "
                "updated_at TEXT)")
            conn.execute(
                "INSERT INTO schema_version (id, version, updated_at) "
                "VALUES (1, ?, datetime('now')) ON CONFLICT(id) DO NOTHING",
                (SCHEMA_VERSION,))
            for table, required in REQUIRED_COLUMNS.items():
                cols = {r[1] for r in conn.execute(
                    f"PRAGMA table_info({table})").fetchall()}
                if not cols:
                    report["tables"][table] = "missing (empty history OK)"
                    continue
                missing = [c for c in required if c not in cols]
                if missing:
                    report["ok"] = False
                    report["critical"].append(
                        f"TRADE_MEMORY_SCHEMA_MISMATCH: {table} missing {missing}")
                report["tables"][table] = ("ok" if not missing
                                           else f"missing columns: {missing}")
            row = conn.execute(
                "SELECT version FROM schema_version WHERE id=1").fetchone()
            report["schema_version"] = row[0] if row else None
        for line in report["critical"]:
            logger.critical(line)
        return report

    def bump_version(self, version: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE schema_version SET version=?, updated_at=datetime('now') "
                "WHERE id=1 AND version < ?", (version, version))


def migrate_legacy_position_size(db_path: str) -> Dict[str, int]:
    """Idempotent: legacy DBs with a position_size column get size_dollars
    backfilled ONLY where units are unambiguous (quantity×entry_price agrees);
    ambiguous rows are flagged for review, never guessed (spec §80-81)."""
    migrated = flagged = 0
    with sqlite3.connect(db_path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(trade_memory)")}
        if "position_size" not in cols:
            return {"migrated": 0, "flagged": 0, "note": "no legacy column"}
        if "needs_unit_review" not in cols:
            conn.execute("ALTER TABLE trade_memory ADD COLUMN "
                         "needs_unit_review INTEGER DEFAULT 0")
        rows = conn.execute(
            "SELECT id, position_size, quantity, entry_price FROM trade_memory "
            "WHERE (size_dollars IS NULL OR size_dollars=0) "
            "AND position_size IS NOT NULL").fetchall()
        for tid, legacy, qty, price in rows:
            if qty and price and abs(legacy - qty * price) / max(qty * price, 1e-9) < 0.05:
                conn.execute("UPDATE trade_memory SET size_dollars=? WHERE id=?",
                             (legacy, tid))     # legacy value was dollars
                migrated += 1
            else:
                conn.execute("UPDATE trade_memory SET needs_unit_review=1 "
                             "WHERE id=?", (tid,))
                flagged += 1
    return {"migrated": migrated, "flagged": flagged}


# ── Data-integrity monitor (spec §76-78) ──────────────────────────────────────


def data_integrity_scan(db_path: str) -> Dict[str, Any]:
    """Flags suspicious rows for review — never deletes anything."""
    issues: List[Dict[str, Any]] = []
    rows = _run_query(
        db_path,
        "SELECT id, symbol, size_dollars, net_pnl, entry_time, exit_time, "
        "position_id FROM trade_memory WHERE exit_time IS NOT NULL")
    seen_positions: Dict[Any, int] = {}
    for tid, symbol, size, pnl, entry, exit_, pos_id in rows:
        if size is None or size <= 0:
            issues.append({"trade_id": tid, "flag": "MISSING_OR_ZERO_NOTIONAL"})
            continue
        ret = compute_trade_net_return(pnl, size)
        if ret is not None and abs(ret) > 10.0:
            issues.append({"trade_id": tid, "flag": "ABSURD_RETURN",
                           "detail": f"{ret:+.0%} — POSSIBLE_UNIT_ERROR"})
        if entry and exit_ and str(entry) > str(exit_):
            issues.append({"trade_id": tid, "flag": "TIMESTAMPS_REVERSED"})
        if pos_id is not None:
            if pos_id in seen_positions:
                issues.append({"trade_id": tid, "flag": "POSSIBLE_DUPLICATE",
                               "detail": f"position {pos_id} also trade "
                                         f"{seen_positions[pos_id]}"})
            seen_positions[pos_id] = tid
    if issues:
        logger.critical(f"DATA INTEGRITY: {len(issues)} flagged rows "
                        f"(first: {issues[0]})")
    return {"flagged": len(issues), "issues": issues,
            "scanned": len(rows),
            "normalization_version": RETURN_NORMALIZATION_VERSION}
