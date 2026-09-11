"""Meta-alpha layer: which alpha FAMILIES work in the current market state.

    MetaAlphaModel    : regime-conditioned E[alpha return], shadow-first
    EdgeSurvivalModel : P(edge stays economically useful) + decay actions
    CrowdingMonitor   : timing-shift / crowding warnings

None of these bypass Alpha validation; they only modulate capital among
ALREADY-validated alphas (spec §41-50).
"""
from __future__ import annotations

import logging
import math
import sqlite3
import statistics as st
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


# ── MetaAlphaModel (spec §41-45) ──────────────────────────────────────────────


class MetaAlphaModel:
    """Empirical regime-conditioned family performance with shrinkage.

    NOT a direction predictor. It estimates, per alpha (or family):
        E[alpha return | current regime], P(positive), downside
    from realized attributed returns, and produces a regime_fit multiplier
    for allocation. New versions run SHADOW first (spec §45).
    """

    def __init__(self, shrinkage_n: int = 30, mode: str = "shadow") -> None:
        self.shrinkage_n = shrinkage_n
        self.mode = mode                              # 'shadow' | 'active'
        # (alpha_or_family, regime) -> list of returns
        self._history: Dict[Tuple[str, str], List[float]] = {}
        self._shadow_log: List[Dict[str, Any]] = []

    def observe(self, alpha_id: str, regime: str, net_return: float) -> None:
        self._history.setdefault((alpha_id, regime), []).append(net_return)

    def _overall(self, alpha_id: str) -> List[float]:
        out: List[float] = []
        for (a, _), rets in self._history.items():
            if a == alpha_id:
                out.extend(rets)
        return out

    def predict(self, alpha_id: str, regime: str) -> Dict[str, Any]:
        """Regime-conditioned expectation, shrunk toward the alpha's overall
        mean so thin regime samples don't produce wild estimates."""
        cond = self._history.get((alpha_id, regime), [])
        overall = self._overall(alpha_id)
        if not overall:
            return {"regime_fit": 1.0, "expected_alpha_ev": None,
                    "confidence": 0.0, "n_conditional": 0}
        base_mean = st.mean(overall)
        n = len(cond)
        k = self.shrinkage_n
        cond_mean = st.mean(cond) if cond else base_mean
        shrunk = (n * cond_mean + k * base_mean) / (n + k)
        p_positive = (sum(1 for r in cond if r > 0) + k * 0.5) / (n + k) if n else 0.5
        downside = min(cond) if cond else (min(overall) if overall else 0.0)
        # regime_fit: multiplicative capital modifier in [0.25, 1.5]
        if base_mean > 0:
            fit = shrunk / base_mean
        else:
            fit = 1.0 if shrunk >= base_mean else 0.5
        fit = max(0.25, min(1.5, fit))
        return {
            "regime_fit": fit,
            "expected_alpha_ev": shrunk,
            "p_positive": p_positive,
            "expected_downside": downside,
            "confidence": min(n / (n + k), 0.95),
            "n_conditional": n,
        }

    def rank_alphas(self, alpha_ids: Sequence[str], regime: str
                    ) -> List[Tuple[str, float]]:
        scored = [(a, self.predict(a, regime).get("expected_alpha_ev") or 0.0)
                  for a in alpha_ids]
        return sorted(scored, key=lambda t: t[1], reverse=True)

    def shadow_compare(self, deterministic_choice: Sequence[str],
                       regime: str) -> Dict[str, Any]:
        """In shadow mode, log what the meta-model WOULD have chosen; the
        deterministic selection remains authoritative until promotion."""
        meta_rank = [a for a, _ in self.rank_alphas(list(deterministic_choice), regime)]
        entry = {"at": _utcnow(), "regime": regime,
                 "deterministic": list(deterministic_choice),
                 "meta_ranking": meta_rank, "mode": self.mode}
        self._shadow_log.append(entry)
        return entry

    def regime_fit_for_allocation(self, alpha_id: str, regime: str) -> float:
        """Only modifies sizing when promoted to active; shadow returns 1.0."""
        if self.mode != "active":
            return 1.0
        return self.predict(alpha_id, regime)["regime_fit"]


class MetaShadowStore:
    """Persists shadow predictions + realized outcomes so the promotion gate
    runs on durable forward evidence (spec §56-57)."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite") -> None:
        import sqlite3
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta_shadow_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    alpha_id TEXT, regime TEXT,
                    predicted_ev REAL, confidence REAL, rank INTEGER,
                    realized_return REAL, resolved_at TEXT
                )""")

    def record_prediction(self, alpha_id: str, regime: str,
                          predicted_ev: Optional[float],
                          confidence: Optional[float],
                          rank: int = 0) -> int:
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO meta_shadow_predictions (recorded_at, alpha_id, "
                "regime, predicted_ev, confidence, rank) VALUES (?,?,?,?,?,?)",
                (_utcnow(), alpha_id, regime, predicted_ev, confidence, rank))
            return int(cur.lastrowid)

    def resolve(self, prediction_id: int, realized_return: float) -> None:
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE meta_shadow_predictions SET realized_return=?, "
                "resolved_at=? WHERE id=?",
                (realized_return, _utcnow(), prediction_id))

    def resolved_pairs(self) -> List[Tuple[float, float]]:
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT predicted_ev, realized_return FROM "
                "meta_shadow_predictions WHERE realized_return IS NOT NULL "
                "AND predicted_ev IS NOT NULL").fetchall()
        return [(r[0], r[1]) for r in rows]


# ── EdgeSurvivalModel (spec §46-48) ───────────────────────────────────────────


@dataclass
class SurvivalAssessment:
    alpha_id: str
    survival_probability: float
    recommended_state: str                # HEALTHY/WATCH/DEGRADED/PAUSED
    capital_multiplier: float
    signals: Dict[str, float] = field(default_factory=dict)


class EdgeSurvivalModel:
    """P(edge remains economically useful over the next horizon), from
    rolling EV level+slope, frequency and friction trends."""

    def __init__(self, watch_threshold: float = 0.6,
                 degraded_threshold: float = 0.4,
                 paused_threshold: float = 0.25) -> None:
        self.watch_threshold = watch_threshold
        self.degraded_threshold = degraded_threshold
        self.paused_threshold = paused_threshold

    def assess(self, alpha_id: str, *, recent_returns: Sequence[float],
               older_returns: Sequence[float],
               slippage_trend: float = 0.0,
               frequency_ratio: float = 1.0,
               correlation_change: float = 0.0) -> SurvivalAssessment:
        """recent vs older attributed net returns → survival probability.
        slippage_trend: recent/older slippage − 1 (positive = worsening).
        frequency_ratio: recent/expected signal frequency."""
        prob = 0.8
        signals: Dict[str, float] = {}
        recent_mean = st.mean(recent_returns) if recent_returns else 0.0
        older_mean = st.mean(older_returns) if older_returns else 0.0
        signals["recent_ev"] = recent_mean
        signals["older_ev"] = older_mean

        if older_mean > 0:
            decay = (older_mean - recent_mean) / older_mean   # 0 = stable
            signals["ev_decay"] = decay
            prob -= 0.5 * max(min(decay, 1.5), 0.0)
        if recent_mean < 0:
            prob -= 0.25
        prob -= 0.15 * max(slippage_trend, 0.0)
        prob -= 0.10 * max(correlation_change, 0.0)
        if frequency_ratio < 0.5:
            prob -= 0.10
        if len(recent_returns) < 10:
            # thin evidence — never rush to kill an edge on a tiny sample
            prob = max(prob, self.watch_threshold)
        prob = round(max(0.0, min(1.0, prob)), 6)

        if prob >= self.watch_threshold:
            state, mult = "HEALTHY", 1.0
        elif prob > self.degraded_threshold:
            state, mult = "WATCH", 0.6
        elif prob >= self.paused_threshold:
            state, mult = "DEGRADED", 0.3
        else:
            state, mult = "PAUSED", 0.0
        return SurvivalAssessment(alpha_id=alpha_id, survival_probability=prob,
                                  recommended_state=state,
                                  capital_multiplier=mult, signals=signals)


# ── CrowdingMonitor (spec §49-50) ─────────────────────────────────────────────


class CrowdingMonitor:
    """Detects an edge getting crowded/front-run: post-signal return shrinks
    while pre-signal move grows, spreads worsen, capacity shrinks."""

    def assess(self, alpha_id: str, *,
               post_signal_returns_early: Sequence[float],
               post_signal_returns_recent: Sequence[float],
               pre_signal_moves_early: Optional[Sequence[float]] = None,
               pre_signal_moves_recent: Optional[Sequence[float]] = None,
               spread_trend: float = 0.0,
               min_samples: int = 10) -> Dict[str, Any]:
        flags: List[str] = []
        detail: Dict[str, float] = {}
        if (len(post_signal_returns_early) >= min_samples
                and len(post_signal_returns_recent) >= min_samples):
            early = st.mean(post_signal_returns_early)
            recent = st.mean(post_signal_returns_recent)
            detail["post_signal_early"] = early
            detail["post_signal_recent"] = recent
            if early > 0 and recent < early * 0.5:
                flags.append("POST_SIGNAL_RETURN_SHRINKING")
        if (pre_signal_moves_early and pre_signal_moves_recent
                and len(pre_signal_moves_early) >= min_samples
                and len(pre_signal_moves_recent) >= min_samples):
            pre_early = st.mean(pre_signal_moves_early)
            pre_recent = st.mean(pre_signal_moves_recent)
            detail["pre_signal_early"] = pre_early
            detail["pre_signal_recent"] = pre_recent
            if pre_recent > pre_early * 1.5 and pre_recent > 0:
                flags.append("PRE_SIGNAL_MOVE_INCREASING")
        if spread_trend > 0.25:
            flags.append("SPREAD_WORSENING")
        if ("POST_SIGNAL_RETURN_SHRINKING" in flags
                and "PRE_SIGNAL_MOVE_INCREASING" in flags):
            flags.append("EDGE_TIMING_SHIFT")   # profit migrating earlier
        return {
            "alpha_id": alpha_id,
            "crowded": bool(flags),
            "flags": flags,
            "detail": detail,
            "recommended_action": ("re_research_timing"
                                   if "EDGE_TIMING_SHIFT" in flags else
                                   "reduce_capital" if flags else "none"),
        }
