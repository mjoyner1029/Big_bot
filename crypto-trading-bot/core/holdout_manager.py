"""Final holdout protection.

Splits history into DISCOVERY / VALIDATION / HOLDOUT periods and enforces
that the holdout segment is only evaluated ONCE after parameters are frozen.
If the strategy changes after holdout inspection, the old result is
invalidated and a fresh untouched period (or forward evidence) is required.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

_CREATE = """
CREATE TABLE IF NOT EXISTS holdout_registry (
    strategy_id          TEXT NOT NULL,
    holdout_start        TEXT NOT NULL,
    holdout_end          TEXT NOT NULL,
    strategy_version     TEXT NOT NULL,
    access_count         INTEGER DEFAULT 0,
    first_access         TEXT,
    version_at_access    TEXT,
    result               TEXT,
    invalidated          INTEGER DEFAULT 0,
    invalidated_reason   TEXT,
    created_at           TEXT,
    PRIMARY KEY (strategy_id, holdout_start, holdout_end)
)
"""


@dataclass
class DataSplit:
    discovery: Tuple[int, int]
    validation: Tuple[int, int]
    holdout: Tuple[int, int]


def three_way_split(n: int, discovery_frac: float = 0.6,
                    validation_frac: float = 0.2) -> DataSplit:
    """Chronological DISCOVERY / VALIDATION / HOLDOUT index ranges."""
    d_end = int(n * discovery_frac)
    v_end = int(n * (discovery_frac + validation_frac))
    return DataSplit(
        discovery=(0, d_end),
        validation=(d_end, v_end),
        holdout=(v_end, n),
    )


class HoldoutManager:
    """Controls access to the final holdout period."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite") -> None:
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE)
            conn.commit()

    def register(self, strategy_id: str, holdout_start: str, holdout_end: str,
                 strategy_version: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO holdout_registry "
                "(strategy_id, holdout_start, holdout_end, strategy_version, created_at) "
                "VALUES (?,?,?,?,?)",
                (strategy_id, holdout_start, holdout_end, strategy_version, _utcnow()),
            )
            conn.commit()

    def can_access(self, strategy_id: str, holdout_start: str, holdout_end: str) -> Tuple[bool, str]:
        row = self._get(strategy_id, holdout_start, holdout_end)
        if row is None:
            return False, "holdout not registered — register before evaluation"
        if row["invalidated"]:
            return False, f"holdout invalidated: {row['invalidated_reason']}"
        if row["access_count"] >= 1:
            return False, "holdout already consumed (single evaluation allowed)"
        return True, "ok"

    def record_access(self, strategy_id: str, holdout_start: str, holdout_end: str,
                      strategy_version: str, result: Dict) -> bool:
        """Consume the holdout. Returns False if access is not allowed."""
        ok, reason = self.can_access(strategy_id, holdout_start, holdout_end)
        if not ok:
            logger.warning(f"HoldoutManager: access denied for {strategy_id}: {reason}")
            return False
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE holdout_registry SET access_count=access_count+1, "
                "first_access=COALESCE(first_access, ?), version_at_access=?, result=? "
                "WHERE strategy_id=? AND holdout_start=? AND holdout_end=?",
                (_utcnow(), strategy_version, json.dumps(result, default=str),
                 strategy_id, holdout_start, holdout_end),
            )
            conn.commit()
        return True

    def on_strategy_changed(self, strategy_id: str, new_version: str) -> int:
        """Invalidate consumed holdout results if the strategy changed after
        inspecting them. Returns number of invalidated records."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE holdout_registry SET invalidated=1, "
                "invalidated_reason='strategy version changed to ' || ? || ' after holdout access' "
                "WHERE strategy_id=? AND access_count >= 1 AND invalidated=0 "
                "AND version_at_access IS NOT NULL AND version_at_access != ?",
                (new_version, strategy_id, new_version),
            )
            conn.commit()
            n = cur.rowcount
        if n:
            logger.warning(
                f"HoldoutManager: invalidated {n} holdout result(s) for {strategy_id}; "
                "a new untouched period or forward-test evidence is required"
            )
        return n

    def status(self, strategy_id: str) -> list:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM holdout_registry WHERE strategy_id=?", (strategy_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def _get(self, strategy_id: str, holdout_start: str, holdout_end: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM holdout_registry WHERE strategy_id=? AND holdout_start=? AND holdout_end=?",
                (strategy_id, holdout_start, holdout_end),
            ).fetchone()
        return dict(row) if row else None
