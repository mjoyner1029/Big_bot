"""Synchronized portfolio return paths + production correlation matrices.

Replaces "append separate alpha histories" with a TIME-ALIGNED matrix:

    timestamp | Alpha_A | Alpha_B | Alpha_C
    t1        |  0.002  | -0.001  |  0.000

Alignment policy (spec §15): an alpha with no closed trade on a timestamp
contributes 0.0 (flat — no position, no P&L). Returns are never forward-filled
from market data. Ruin/drawdown simulation runs on the PORTFOLIO PATH via
stationary block bootstrap, so simultaneous losses are preserved.
"""
from __future__ import annotations

import logging
import random
import sqlite3
import statistics as st
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PORTFOLIO_RISK_VERSION = "2.0.0"
CORRELATION_MODEL_VERSION = "2.0.0"


# ── Synchronized return matrix (spec §14-17) ──────────────────────────────────


def build_return_matrix(
    alpha_returns: Dict[str, Dict[str, float]],
) -> Tuple[List[str], Dict[str, List[float]]]:
    """alpha_id -> {timestamp: return} → (sorted timestamps, aligned columns).
    Missing (inactive) entries are 0.0 by policy."""
    timestamps = sorted({t for by_t in alpha_returns.values() for t in by_t})
    matrix = {a: [by_t.get(t, 0.0) for t in timestamps]
              for a, by_t in alpha_returns.items()}
    return timestamps, matrix


def portfolio_return_series(
    matrix: Dict[str, List[float]],
    weights: Optional[Dict[str, float]] = None,
    weight_history: Optional[List[Dict[str, float]]] = None,
) -> List[float]:
    """R_p,t = Σ w_i,t · R_i,t. Uses per-timestamp weights when the
    allocation history exists, else static weights (documented fallback)."""
    if not matrix:
        return []
    n = len(next(iter(matrix.values())))
    out = []
    for t in range(n):
        if weight_history is not None and t < len(weight_history):
            w_t = weight_history[t]
        else:
            w_t = weights or {a: 1.0 / len(matrix) for a in matrix}
        out.append(sum(w_t.get(a, 0.0) * col[t] for a, col in matrix.items()))
    return out


# ── Portfolio path simulation (spec §18-21) ───────────────────────────────────


def simulate_portfolio_paths(
    portfolio_returns: Sequence[float], *,
    n_sims: int = 1000, horizon: Optional[int] = None,
    block_size: int = 5, seed: int = 42,
    ruin_drawdowns: Sequence[float] = (0.5, 0.8),
    operational_minimum_frac: Optional[float] = None,
) -> Dict[str, Any]:
    """Stationary block bootstrap over the SYNCHRONIZED portfolio path —
    temporal dependence and simultaneous losses preserved (spec §19)."""
    rets = list(portfolio_returns)
    if len(rets) < 20:
        return {"available": False, "note": "insufficient_sample",
                "risk_of_ruin": {str(d): 1.0 for d in ruin_drawdowns}}
    rng = random.Random(seed)
    n = len(rets)
    horizon = horizon or max(n, 250)
    block = max(2, min(block_size, n // 4))
    max_dds: List[float] = []
    terminal: List[float] = []
    ruined = {d: 0 for d in ruin_drawdowns}
    op_ruined = 0
    log_growths: List[float] = []
    for _ in range(n_sims):
        equity = peak = 1.0
        dd_max = 0.0
        steps = 0
        while steps < horizon:
            start = rng.randrange(0, n - block + 1)
            for r in rets[start:start + block]:
                equity *= max(1.0 + r, 0.0)
                peak = max(peak, equity)
                dd_max = max(dd_max, 1.0 - equity / peak if peak > 0 else 1.0)
                steps += 1
                if steps >= horizon or equity <= 0:
                    break
            if equity <= 0:
                break
        for d in ruin_drawdowns:
            if dd_max >= d:
                ruined[d] += 1
        if operational_minimum_frac and equity <= operational_minimum_frac:
            op_ruined += 1
        max_dds.append(dd_max)
        terminal.append(equity)
        log_growths.append(
            (0.0 if equity <= 0 else __import__("math").log(equity)) / horizon)
    max_dds.sort()
    terminal.sort()
    k = max(int(0.05 * len(terminal)), 1)
    return {
        "available": True,
        "version": PORTFOLIO_RISK_VERSION,
        "reproducibility": {"seed": seed, "n_sims": n_sims, "horizon": horizon,
                            "block_size": block, "sample_length": n},
        "risk_of_ruin": {str(d): ruined[d] / n_sims for d in ruin_drawdowns},
        "operational_ruin": (op_ruined / n_sims
                             if operational_minimum_frac else None),
        "max_drawdown_p50": max_dds[len(max_dds) // 2],
        "max_drawdown_p95": max_dds[int(0.95 * len(max_dds)) - 1],
        "terminal_wealth_p5": terminal[k - 1],
        "terminal_wealth_p50": terminal[len(terminal) // 2],
        "expected_shortfall_5pct": st.mean(terminal[:k]) - 1.0,
        "expected_log_growth": st.mean(log_growths),
        "worst_path_terminal": terminal[0],
    }


# ── Downside / stress correlation matrices (spec §23-27) ──────────────────────


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    from core.strategy_correlation import _pearson as p
    return p(list(xs), list(ys))


def downside_correlation_matrix(
    matrix: Dict[str, List[float]], *, percentile: float = 0.3,
    min_overlap: int = 10,
) -> Dict[Tuple[str, str], float]:
    """Correlation conditional on the PORTFOLIO (equal-weight) return being in
    its worst `percentile` of timestamps — measures who loses together."""
    alphas = list(matrix)
    if len(alphas) < 2:
        return {}
    n = len(matrix[alphas[0]])
    port = [st.mean(matrix[a][t] for a in alphas) for t in range(n)]
    k = max(int(n * percentile), min_overlap)
    worst = sorted(range(n), key=lambda t: port[t])[:k]
    out: Dict[Tuple[str, str], float] = {}
    for i, a in enumerate(alphas):
        for b in alphas[i + 1:]:
            c = _pearson([matrix[a][t] for t in worst],
                         [matrix[b][t] for t in worst])
            if c is not None:
                out[(a, b)] = c
    return out


def stress_correlation_matrix(
    matrix: Dict[str, List[float]],
    stress_mask: Optional[Sequence[bool]] = None, *,
    vol_multiple: float = 2.0, min_stress_samples: int = 8,
) -> Dict[Tuple[str, str], float]:
    """Correlation during STRESS timestamps: caller-provided mask (market
    drawdown / liquidation windows) or endogenous high-vol detection
    (|portfolio return| > vol_multiple × rolling std)."""
    alphas = list(matrix)
    if len(alphas) < 2:
        return {}
    n = len(matrix[alphas[0]])
    if stress_mask is None:
        port = [st.mean(matrix[a][t] for a in alphas) for t in range(n)]
        sd = st.pstdev(port) or 1e-9
        stress_mask = [abs(r) > vol_multiple * sd for r in port]
    idx = [t for t in range(n) if stress_mask[t]]
    if len(idx) < min_stress_samples:
        return {}
    out: Dict[Tuple[str, str], float] = {}
    for i, a in enumerate(alphas):
        for b in alphas[i + 1:]:
            c = _pearson([matrix[a][t] for t in idx],
                         [matrix[b][t] for t in idx])
            if c is not None:
                out[(a, b)] = c
    return out


def effective_cluster_count(
    matrix: Dict[str, List[float]],
    downside: Optional[Dict[Tuple[str, str], float]] = None,
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """Nominal alpha count vs EFFECTIVE independent clusters, using the max of
    ordinary and downside correlation (spec §27, §64)."""
    alphas = list(matrix)
    downside = downside or {}
    parent = {a: a for a in alphas}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(alphas):
        for b in alphas[i + 1:]:
            ordinary = _pearson(matrix[a], matrix[b]) or 0.0
            down = downside.get((a, b)) or downside.get((b, a)) or 0.0
            if max(ordinary, down) >= threshold:
                parent[find(a)] = find(b)
    clusters: Dict[str, List[str]] = {}
    for a in alphas:
        clusters.setdefault(find(a), []).append(a)
    return {"nominal_alphas": len(alphas),
            "effective_clusters": len(clusters),
            "clusters": list(clusters.values()),
            "version": CORRELATION_MODEL_VERSION}


# ── Production loader + refresh policy (spec §23, §25-26) ─────────────────────


class ProductionCorrelationService:
    """Builds synchronized daily attributed-return matrices from the
    attribution store; refreshes on schedule/new-data thresholds, never
    per tick. Persists matrices for reproducibility."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite",
                 min_new_trades: int = 10,
                 min_correlation_observations: int = 10,
                 shrinkage_k: int = 20) -> None:
        self.db_path = db_path
        self.min_new_trades = min_new_trades
        self.min_correlation_observations = min_correlation_observations
        self.shrinkage_k = shrinkage_k
        self.risk_resolution: str = "DAILY"
        self._cache: Optional[Dict[str, Any]] = None
        self._cached_trade_count = 0
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS correlation_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    computed_at TEXT NOT NULL,
                    version TEXT, n_alphas INTEGER, n_timestamps INTEGER,
                    ordinary_json TEXT, downside_json TEXT, stress_json TEXT
                )""")

    def daily_alpha_returns(self, resolution: Optional[str] = None
                            ) -> Dict[str, Dict[str, float]]:
        """Canonical normalized returns (net_pnl / size_dollars), bucketed at
        the chosen risk resolution. Schema failures raise — an empty risk
        history from a broken query must never look like zero risk."""
        from core.trade_history import (
            bucketed_alpha_returns,
            choose_risk_resolution,
        )
        self.risk_resolution = resolution or choose_risk_resolution(self.db_path)
        return bucketed_alpha_returns(self.db_path,
                                      resolution=self.risk_resolution)

    def _trade_count(self) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                return conn.execute(
                    "SELECT COUNT(*) FROM trade_attribution").fetchone()[0]
        except sqlite3.OperationalError:
            return 0

    def matrices(self, force: bool = False) -> Dict[str, Any]:
        """Cached; recomputed only when enough NEW trades arrived (spec §25)."""
        count = self._trade_count()
        if (not force and self._cache is not None
                and count - self._cached_trade_count < self.min_new_trades):
            return self._cache
        returns = self.daily_alpha_returns()
        timestamps, matrix = build_return_matrix(returns)
        # Min-sample policy + shrinkage toward 0: tiny overlapping samples
        # must not produce extreme ±1.0 estimates (spec §24-25)
        ordinary: Dict[Tuple[str, str], float] = {}
        alphas = list(matrix)
        for i, a in enumerate(alphas):
            for b in alphas[i + 1:]:
                overlap = sum(1 for t in range(len(timestamps))
                              if matrix[a][t] != 0.0 and matrix[b][t] != 0.0)
                if overlap < self.min_correlation_observations:
                    continue                       # conservative: no estimate
                c = _pearson(matrix[a], matrix[b])
                if c is not None:
                    ordinary[(a, b)] = c * overlap / (overlap + self.shrinkage_k)
        downside = downside_correlation_matrix(matrix)
        stress = stress_correlation_matrix(matrix)
        result = {
            "timestamps": timestamps, "matrix": matrix,
            "ordinary": ordinary, "downside": downside, "stress": stress,
            "clusters": effective_cluster_count(matrix, downside)
            if matrix else {"nominal_alphas": 0, "effective_clusters": 0},
            "version": CORRELATION_MODEL_VERSION,
            "risk_resolution": self.risk_resolution,
        }
        self._cache = result
        self._cached_trade_count = count
        try:
            import json
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO correlation_snapshots (computed_at, version, "
                    "n_alphas, n_timestamps, ordinary_json, downside_json, "
                    "stress_json) VALUES (datetime('now'),?,?,?,?,?,?)",
                    (CORRELATION_MODEL_VERSION, len(alphas), len(timestamps),
                     json.dumps({f"{a}|{b}": v for (a, b), v in ordinary.items()}),
                     json.dumps({f"{a}|{b}": v for (a, b), v in downside.items()}),
                     json.dumps({f"{a}|{b}": v for (a, b), v in stress.items()})))
        except sqlite3.OperationalError:
            pass
        return result

    def populate_allocator(self, allocator) -> None:
        """Feed production ordinary + downside (max with stress) matrices into
        the PortfolioAllocator before allocation (spec §26)."""
        m = self.matrices()
        allocator.alpha_correlations = dict(m["ordinary"])
        combined = dict(m["downside"])
        for pair, v in m["stress"].items():
            combined[pair] = max(combined.get(pair, 0.0), v)
        allocator.downside_correlations = combined
