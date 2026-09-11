"""Opportunity economics: half-life, frequency, capacity, dollar alpha,
capital-time efficiency (spec §33-38).

These measure the ECONOMIC scale of an edge; they never override
risk-adjusted validation — they inform sizing and prioritization.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.portfolio_robustness import estimate_alpha_capacity, return_on_capital_time

logger = logging.getLogger(__name__)


# ── Opportunity half-life (spec §33-34) ───────────────────────────────────────


@dataclass
class HalfLifeProfile:
    horizons: List[float]                 # in bars/days as supplied
    mean_returns: List[float]             # cumulative expected return per horizon
    peak_horizon: Optional[float]
    peak_return: Optional[float]
    half_life: Optional[float]            # horizon where edge decays to half of peak
    optimal_exit_horizon: Optional[float]


class OpportunityHalfLifeEstimator:
    """From per-signal forward returns at multiple horizons, estimate where
    the edge peaks and how fast it decays."""

    def estimate(self, horizon_returns: Dict[float, Sequence[float]],
                 min_samples: int = 15) -> Optional[HalfLifeProfile]:
        """horizon_returns: horizon -> list of per-signal cumulative returns."""
        horizons = sorted(h for h, rets in horizon_returns.items()
                          if len(rets) >= min_samples)
        if len(horizons) < 2:
            return None
        means = [sum(horizon_returns[h]) / len(horizon_returns[h]) for h in horizons]
        peak_i = max(range(len(means)), key=lambda i: means[i])
        peak_h, peak_r = horizons[peak_i], means[peak_i]
        half_life = None
        if peak_r > 0:
            for h, m in zip(horizons[peak_i:], means[peak_i:]):
                if m <= peak_r / 2:
                    half_life = h
                    break
        return HalfLifeProfile(
            horizons=list(horizons), mean_returns=means,
            peak_horizon=peak_h, peak_return=peak_r,
            half_life=half_life,
            optimal_exit_horizon=peak_h,
        )


# ── Frequency (spec §35) ──────────────────────────────────────────────────────


def signal_frequency(signal_times: Sequence[str],
                     observation_days: float) -> Dict[str, float]:
    n = len(signal_times)
    if observation_days <= 0:
        return {"per_day": 0.0, "per_month": 0.0, "per_year": 0.0, "observed": n}
    per_day = n / observation_days
    return {"per_day": per_day, "per_month": per_day * 21,
            "per_year": per_day * 252, "observed": n}


# ── Expected dollar alpha (spec §37) ──────────────────────────────────────────


def expected_dollar_alpha(*, expected_net_return: float,
                          practical_capacity_usd: float,
                          signals_per_year: float) -> Dict[str, float]:
    """Economic SCALE measure — never a replacement for risk-adjusted rank.
    A tiny-capacity 80bps edge can matter less than a deep 20bps edge."""
    per_signal = expected_net_return * practical_capacity_usd
    return {
        "dollar_alpha_per_signal": per_signal,
        "dollar_alpha_per_year": per_signal * signals_per_year,
    }


# ── Capital-time efficiency (spec §38) ────────────────────────────────────────


def capital_time_efficiency(*, expected_net_return: float,
                            holding_days: float,
                            margin_multiplier: float = 1.0,
                            max_drawdown: Optional[float] = None) -> Optional[float]:
    """Edge per unit capital per unit time, penalized for margin usage and
    strategy drawdown depth."""
    base = return_on_capital_time(expected_net_return,
                                  capital_required=1.0, holding_days=holding_days)
    if base is None:
        return None
    base /= max(margin_multiplier, 1.0)
    if max_drawdown and max_drawdown > 0:
        base *= 1.0 / (1.0 + max_drawdown)
    return base


# ── Empirical signal decay (spec: half-life must be LEARNED, not assumed) ────

HALF_LIFE_MODEL_VERSION = "2.0.0"


def empirical_decay_curve(
    horizon_returns: Dict[float, Sequence[float]], *,
    min_signals: int = 15, n_boot: int = 200, seed: int = 7,
) -> Optional[Dict[str, Any]]:
    """From per-signal forward CUMULATIVE returns at multiple horizons:
    peak_time, half_life (first horizon where remaining alpha ≤ 50% of peak),
    95%-decay time, bootstrap CI on the half-life, and a stability flag.
    Tiny samples return None — never trusted."""
    import random
    horizons = sorted(h for h, r in horizon_returns.items()
                      if len(r) >= min_signals)
    if len(horizons) < 3:
        return None
    n = min(len(horizon_returns[h]) for h in horizons)

    def curve_stats(sample_idx: Optional[List[int]] = None):
        means = []
        for h in horizons:
            rets = list(horizon_returns[h])[:n]
            if sample_idx is not None:
                rets = [rets[i] for i in sample_idx]
            means.append(sum(rets) / len(rets))
        peak_i = max(range(len(means)), key=lambda i: means[i])
        peak_t, peak_r = horizons[peak_i], means[peak_i]
        half = None
        decay95 = None
        if peak_r > 0:
            for h, m in zip(horizons[peak_i:], means[peak_i:]):
                if half is None and m <= peak_r / 2:
                    half = h
                if decay95 is None and m <= peak_r * 0.05:
                    decay95 = h
        return peak_t, peak_r, half, decay95, means

    peak_t, peak_r, half, decay95, means = curve_stats()
    rng = random.Random(seed)
    boot_halves = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        _, _, bh, _, _ = curve_stats(idx)
        if bh is not None:
            boot_halves.append(bh)
    boot_halves.sort()
    ci = (boot_halves[int(0.05 * len(boot_halves))],
          boot_halves[int(0.95 * len(boot_halves)) - 1]) if len(boot_halves) >= 20 \
        else (None, None)
    stable = (ci[0] is not None and half is not None
              and ci[1] <= (half or 0) * 4)
    return {
        "peak_time": peak_t, "peak_return": peak_r,
        "half_life": half, "decay_95": decay95,
        "curve": dict(zip(horizons, means)),
        "n_signals": n, "ci_low": ci[0], "ci_high": ci[1],
        "stable": stable, "source": "EMPIRICAL",
        "version": HALF_LIFE_MODEL_VERSION,
    }


class HalfLifeStore:
    """Persists learned decay curves per alpha (spec §92)."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite") -> None:
        import sqlite3
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS half_life_estimates (
                    alpha_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    peak_time REAL, half_life REAL, decay_95 REAL,
                    n_signals INTEGER, ci_low REAL, ci_high REAL,
                    stable INTEGER, source TEXT, version TEXT, curve_json TEXT
                )""")

    def save(self, alpha_id: str, estimate: Dict[str, Any]) -> None:
        import json
        import sqlite3
        from datetime import datetime, timezone
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO half_life_estimates (alpha_id, updated_at, peak_time, "
                "half_life, decay_95, n_signals, ci_low, ci_high, stable, source, "
                "version, curve_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(alpha_id) DO UPDATE SET updated_at=excluded.updated_at, "
                "peak_time=excluded.peak_time, half_life=excluded.half_life, "
                "decay_95=excluded.decay_95, n_signals=excluded.n_signals, "
                "ci_low=excluded.ci_low, ci_high=excluded.ci_high, "
                "stable=excluded.stable, source=excluded.source, "
                "version=excluded.version, curve_json=excluded.curve_json",
                (alpha_id, datetime.now(timezone.utc).isoformat(),
                 estimate.get("peak_time"), estimate.get("half_life"),
                 estimate.get("decay_95"), estimate.get("n_signals"),
                 estimate.get("ci_low"), estimate.get("ci_high"),
                 int(bool(estimate.get("stable"))), estimate.get("source"),
                 estimate.get("version"), json.dumps(estimate.get("curve", {}))))

    def get(self, alpha_id: str) -> Optional[Dict[str, Any]]:
        import json
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT peak_time, half_life, decay_95, n_signals, ci_low, "
                "ci_high, stable, source, version, curve_json "
                "FROM half_life_estimates WHERE alpha_id=?", (alpha_id,)).fetchone()
        if not row:
            return None
        return {"peak_time": row[0], "half_life": row[1], "decay_95": row[2],
                "n_signals": row[3], "ci_low": row[4], "ci_high": row[5],
                "stable": bool(row[6]), "source": row[7], "version": row[8],
                "curve": json.loads(row[9] or "{}")}


# ── Combined opportunity economics ────────────────────────────────────────────


def opportunity_economics(*, expected_net_return: float,
                          adv_usd: Optional[float],
                          holding_days: float,
                          signals_per_year: float,
                          volatility_daily: float = 0.02,
                          max_drawdown: Optional[float] = None) -> Dict[str, Any]:
    cap = estimate_alpha_capacity(
        adv_usd=adv_usd, expected_net_return=expected_net_return,
        signal_frequency_per_day=signals_per_year / 252.0,
        holding_days=holding_days)
    capacity_usd = cap.get("estimated_alpha_capacity_usd", 0.0)
    dollars = expected_dollar_alpha(
        expected_net_return=expected_net_return,
        practical_capacity_usd=capacity_usd,
        signals_per_year=signals_per_year)
    return {
        **cap, **dollars,
        "capital_time_efficiency": capital_time_efficiency(
            expected_net_return=expected_net_return, holding_days=holding_days,
            max_drawdown=max_drawdown),
        "holding_days": holding_days,
        "signals_per_year": signals_per_year,
    }
