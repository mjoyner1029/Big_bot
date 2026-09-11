"""Trade attribution + continuous learning loop.

Every executed or paper trade is attributed to:
    alpha_id / alpha_version / candidate_id / signal_id /
    meta_model_version / feature_version / execution_model_version

so realized outcomes can update alpha metrics, edge health and EV-model
training labels. Retraining is threshold-gated, never per-trade.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

_CREATE = """
CREATE TABLE IF NOT EXISTS trade_attribution (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_memory_id         INTEGER,
    position_id             INTEGER,
    alpha_id                TEXT NOT NULL,
    alpha_version           TEXT DEFAULT '1',
    candidate_id            TEXT,
    signal_id               TEXT,
    meta_model_version      TEXT,
    feature_version         TEXT,
    execution_model_version TEXT,
    recorded_at             TEXT NOT NULL
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_ta_alpha ON trade_attribution(alpha_id)",
    "CREATE INDEX IF NOT EXISTS idx_ta_tm    ON trade_attribution(trade_memory_id)",
]


class TradeAttributionStore:
    def __init__(self, db_path: str = "data/trade_memory.sqlite") -> None:
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE)
            for idx in _INDEXES:
                conn.execute(idx)
            conn.commit()

    def record(
        self,
        alpha_id: str,
        *,
        alpha_version: str = "1",
        trade_memory_id: Optional[int] = None,
        position_id: Optional[int] = None,
        candidate_id: Optional[str] = None,
        signal_id: Optional[str] = None,
        meta_model_version: Optional[str] = None,
        feature_version: Optional[str] = None,
        execution_model_version: Optional[str] = None,
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO trade_attribution (trade_memory_id, position_id, "
                "alpha_id, alpha_version, candidate_id, signal_id, "
                "meta_model_version, feature_version, execution_model_version, "
                "recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (trade_memory_id, position_id, alpha_id, alpha_version,
                 candidate_id, signal_id, meta_model_version, feature_version,
                 execution_model_version, _utcnow()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def for_alpha(self, alpha_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trade_attribution WHERE alpha_id=? ORDER BY recorded_at",
                (alpha_id,),
            ).fetchall()
        return [dict(r) for r in rows]


class LearningLoop:
    """Post-close learning: update alpha realized metrics + edge health and
    decide (threshold-gated) whether the EV model should retrain."""

    def __init__(
        self,
        db_path: str = "data/trade_memory.sqlite",
        alpha_library=None,
        retrain_min_new_samples: int = 25,
        retrain_min_hours: float = 24.0,
    ) -> None:
        self.db_path = db_path
        self.alpha_library = alpha_library
        self.retrain_min_new_samples = retrain_min_new_samples
        self.retrain_min_hours = retrain_min_hours
        self._samples_since_retrain = 0
        self._last_retrain: Optional[datetime] = None

    def on_trade_closed(self, alpha_id: str, net_return: float,
                        drift_detected: bool = False) -> Dict[str, Any]:
        """Update alpha evidence; return retrain decision (never auto per-trade)."""
        self._samples_since_retrain += 1
        self._update_alpha_metrics(alpha_id)

        should_retrain = False
        reasons = []
        if self._samples_since_retrain >= self.retrain_min_new_samples:
            should_retrain = True
            reasons.append(f"new_samples={self._samples_since_retrain}")
        if drift_detected:
            should_retrain = True
            reasons.append("drift_detected")
        if should_retrain and self._last_retrain is not None:
            hours = (datetime.now(timezone.utc) - self._last_retrain).total_seconds() / 3600
            if hours < self.retrain_min_hours and not drift_detected:
                should_retrain = False
                reasons.append(f"cooldown ({hours:.1f}h < {self.retrain_min_hours}h)")
        return {"retrain": should_retrain, "reasons": reasons}

    def mark_retrained(self) -> None:
        self._samples_since_retrain = 0
        self._last_retrain = datetime.now(timezone.utc)

    def _update_alpha_metrics(self, alpha_id: str) -> None:
        """Refresh the alpha's realized evidence and edge health score."""
        if self.alpha_library is None:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT tm.net_return_pct FROM trade_attribution ta "
                    "JOIN trade_memory tm ON tm.id = ta.trade_memory_id "
                    "WHERE ta.alpha_id=? AND tm.net_return_pct IS NOT NULL "
                    "ORDER BY tm.exit_time DESC LIMIT 100",
                    (alpha_id,),
                ).fetchall()
        except sqlite3.OperationalError:
            return
        rets = [float(r[0]) / 100.0 for r in rows]
        if not rets:
            return
        n = len(rets)
        wins = sum(1 for r in rets if r > 0)
        expectancy = sum(rets) / n
        # Edge health: recent realized expectancy scaled into [0,1]
        recent = rets[:20]
        recent_exp = sum(recent) / len(recent)
        health = max(0.0, min(1.0, 0.5 + recent_exp * 50))
        try:
            self.alpha_library.update_evidence(
                alpha_id,
                sample_size=n,
                net_expectancy=expectancy,
                win_rate=wins / n,
                edge_health_score=health,
            )
        except Exception as e:
            logger.warning(f"LearningLoop: alpha metric update failed: {e}")
