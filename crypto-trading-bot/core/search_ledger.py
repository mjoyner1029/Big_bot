"""Global research accounting: search ledger, research ROI, source ROI.

Every hypothesis ever tested is counted here, per source/family/detector, so
FDR/DSR/complexity penalties always know the TRUE search breadth — history is
never reset because a hypothesis came from a different source (spec §21-24,
§82-83).
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


class SearchLedger:
    """Persistent, append-only accounting of research search breadth."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite") -> None:
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    campaign_id TEXT,
                    source TEXT NOT NULL,          -- PRICE/TEMPORAL/GOVERNMENT/...
                    family TEXT,
                    detector TEXT,
                    hypotheses INTEGER NOT NULL,
                    parameter_variants INTEGER DEFAULT 0,
                    interaction_depth INTEGER DEFAULT 1
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    family TEXT,
                    stage TEXT NOT NULL,   -- generated/fdr/validated/paper/forward
                    count INTEGER NOT NULL,
                    compute_seconds REAL DEFAULT 0,
                    forward_pnl REAL DEFAULT 0
                )""")

    def record_search(self, source: str, hypotheses: int, *,
                      campaign_id: Optional[str] = None,
                      family: Optional[str] = None,
                      detector: Optional[str] = None,
                      parameter_variants: int = 0,
                      interaction_depth: int = 1) -> None:
        if hypotheses <= 0:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO search_ledger (recorded_at, campaign_id, source, "
                "family, detector, hypotheses, parameter_variants, interaction_depth) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (_utcnow(), campaign_id, source, family, detector,
                 int(hypotheses), int(parameter_variants), int(interaction_depth)))

    def record_outcome(self, source: str, stage: str, count: int, *,
                       family: Optional[str] = None,
                       compute_seconds: float = 0.0,
                       forward_pnl: float = 0.0) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO research_outcomes (recorded_at, source, family, "
                "stage, count, compute_seconds, forward_pnl) VALUES (?,?,?,?,?,?,?)",
                (_utcnow(), source, family, stage, int(count),
                 float(compute_seconds), float(forward_pnl)))

    def total_breadth(self, source: Optional[str] = None) -> int:
        """Lifetime hypothesis count — the number FDR corrections must know."""
        with sqlite3.connect(self.db_path) as conn:
            if source:
                row = conn.execute(
                    "SELECT COALESCE(SUM(hypotheses),0) FROM search_ledger "
                    "WHERE source=?", (source,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT COALESCE(SUM(hypotheses),0) FROM search_ledger").fetchone()
        return int(row[0])

    def breadth_by_source(self) -> Dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT source, SUM(hypotheses) FROM search_ledger "
                "GROUP BY source").fetchall()
        return {r[0]: int(r[1]) for r in rows}

    def report(self) -> Dict[str, Any]:
        by_source = self.breadth_by_source()
        return {"by_source": by_source, "total": sum(by_source.values())}


# ── Research ROI (spec §23-24) ────────────────────────────────────────────────


def research_roi_scores(ledger: SearchLedger, *,
                        min_observations: int = 200,
                        shrinkage_hypotheses: int = 2000) -> Dict[str, Dict[str, Any]]:
    """Per research family/source: value produced per unit search.

    Uses shrinkage toward the global mean so low-frequency families
    (e.g. government spending) are not abandoned after tiny samples.
    """
    breadth = ledger.breadth_by_source()
    with sqlite3.connect(ledger.db_path) as conn:
        rows = conn.execute(
            "SELECT source, stage, SUM(count), SUM(compute_seconds), SUM(forward_pnl) "
            "FROM research_outcomes GROUP BY source, stage").fetchall()
    stages: Dict[str, Dict[str, float]] = {}
    for source, stage, count, secs, pnl in rows:
        s = stages.setdefault(source, {"compute_seconds": 0.0, "forward_pnl": 0.0})
        s[stage] = s.get(stage, 0.0) + count
        s["compute_seconds"] += secs or 0.0
        s["forward_pnl"] += pnl or 0.0

    # Global base rate for shrinkage
    total_h = max(sum(breadth.values()), 1)
    total_valid = sum(s.get("validated", 0) for s in stages.values())
    base_rate = total_valid / total_h

    out: Dict[str, Dict[str, Any]] = {}
    for source, n_hyp in breadth.items():
        s = stages.get(source, {})
        validated = s.get("validated", 0)
        # shrunk validation rate: (validated + base_rate*k) / (n + k)
        k = shrinkage_hypotheses
        shrunk_rate = (validated + base_rate * k) / (n_hyp + k)
        mature = n_hyp >= min_observations
        out[source] = {
            "hypotheses": n_hyp,
            "validated": validated,
            "paper": s.get("paper", 0),
            "raw_validation_rate": validated / n_hyp if n_hyp else 0.0,
            "shrunk_validation_rate": shrunk_rate,
            "forward_pnl": s.get("forward_pnl", 0.0),
            "compute_seconds": s.get("compute_seconds", 0.0),
            "mature": mature,
            "research_roi_score": shrunk_rate * (1.0 if mature else 0.5),
            "verdict": ("insufficient_sample" if not mature else
                        "productive" if shrunk_rate > base_rate else "below_average"),
        }
    return out


def allocate_research_budget(roi: Dict[str, Dict[str, Any]],
                             total_budget: int) -> Dict[str, int]:
    """Split a hypothesis budget across families proportional to shrunk ROI,
    with a floor so no family starves (spec §24)."""
    if not roi:
        return {}
    floor = max(1, total_budget // (len(roi) * 5))
    weights = {s: max(v["research_roi_score"], 1e-6) for s, v in roi.items()}
    wsum = sum(weights.values())
    alloc = {s: max(floor, int(total_budget * w / wsum)) for s, w in weights.items()}
    return alloc


# ── Source ROI (spec §82-83) ──────────────────────────────────────────────────


def source_roi_report(ledger: SearchLedger,
                      source_costs_monthly: Optional[Dict[str, float]] = None
                      ) -> Dict[str, Dict[str, Any]]:
    """Economics per data source. Reports; never auto-cancels anything."""
    roi = research_roi_scores(ledger)
    costs = source_costs_monthly or {}
    out = {}
    for source, stats in roi.items():
        cost = costs.get(source, 0.0)
        value = stats["forward_pnl"]
        out[source] = {
            **stats,
            "monthly_cost_usd": cost,
            "net_value_usd": value - cost,
            "economics": ("free_source" if cost == 0 else
                          "positive" if value > cost else
                          "poor_source_economics" if stats["mature"] else
                          "insufficient_sample"),
        }
    return out


# ── Information value test (spec §81, tests §108-109) ─────────────────────────


def information_value_test(
    base_oos_returns: Sequence[float],
    augmented_oos_returns: Sequence[float],
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """BASE vs BASE+FEATURE-FAMILY on out-of-sample returns. The feature
    family earns authority only if the improvement is real, not noise."""
    import random
    import statistics as st
    a, b = list(base_oos_returns), list(augmented_oos_returns)
    if len(a) < 10 or len(b) < 10:
        return {"incremental_value": 0.0, "significant": False,
                "note": "insufficient_sample"}

    def sharpe(x):
        sd = st.pstdev(x)
        return st.mean(x) / sd if sd > 0 else 0.0

    delta_ev = st.mean(b) - st.mean(a)
    delta_sharpe = sharpe(b) - sharpe(a)
    # bootstrap the EV delta
    rng = random.Random(seed)
    wins = 0
    for _ in range(n_bootstrap):
        ra = [rng.choice(a) for _ in range(len(a))]
        rb = [rng.choice(b) for _ in range(len(b))]
        if st.mean(rb) > st.mean(ra):
            wins += 1
    p_improvement = wins / n_bootstrap
    significant = delta_ev > 0 and p_improvement >= 0.95
    return {
        "incremental_ev": delta_ev,
        "incremental_sharpe": delta_sharpe,
        "p_improvement": p_improvement,
        "significant": significant,
        "incremental_value": delta_ev if significant else 0.0,
    }
