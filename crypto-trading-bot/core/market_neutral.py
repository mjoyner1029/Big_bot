"""Market-neutral alpha research: pairs, cointegration, basket relative value.

Implemented as DiscoveryDetector subclasses so market-neutral hypotheses flow
through the SAME campaign FDR/validation funnel as everything else
(spec §53-56). Relationships are validated for cointegration/stationarity and
spread half-life — correlation alone is never sufficient.
"""
from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.discovery_detectors import (
    DiscoveredHypothesis,
    DiscoveryDetector,
    SearchLimits,
)

logger = logging.getLogger(__name__)


def engle_granger_lite(x: pd.Series, y: pd.Series) -> Optional[Dict[str, float]]:
    """OLS hedge ratio + residual stationarity via variance-ratio and
    AR(1) coefficient. No statsmodels dependency; conservative thresholds."""
    n = min(len(x), len(y))
    if n < 120:
        return None
    xv = x.iloc[-n:].to_numpy(dtype=float)
    yv = y.iloc[-n:].to_numpy(dtype=float)
    if np.std(xv) <= 0 or np.std(yv) <= 0:
        return None
    beta = float(np.cov(yv, xv)[0, 1] / np.var(xv))
    spread = yv - beta * xv
    spread = spread - spread.mean()
    # AR(1) on spread: rho < ~0.97 suggests mean reversion at daily frequency
    s0, s1 = spread[:-1], spread[1:]
    denom = float(np.dot(s0, s0))
    if denom <= 0:
        return None
    rho = float(np.dot(s0, s1) / denom)
    if not (-1.0 < rho < 1.0):
        return None
    half_life = math.log(0.5) / math.log(abs(rho)) if 0 < abs(rho) < 1 else float("inf")
    # structural-break guard: first-half vs second-half spread mean shift
    mid = len(spread) // 2
    shift = abs(spread[:mid].mean() - spread[mid:].mean())
    sd = float(spread.std()) or 1e-9
    return {
        "beta": beta,
        "rho": rho,
        "half_life_bars": half_life,
        "spread_std": sd,
        "mean_shift_sigmas": shift / sd,
        "stationary": rho < 0.97 and half_life < 60 and shift / sd < 1.0,
    }


class PairsRelativeValueDetector(DiscoveryDetector):
    """Cointegrated-pair mean reversion within sectors/asset classes.

    Search discipline: candidate pairs limited to same sector (equities) or
    same asset class (crypto), capped; entries at predeclared z-score grid —
    thresholds are NEVER derived from one observed extreme (spec §16).
    """

    family = "MARKET_NEUTRAL"
    Z_GRID = (1.0, 1.5, 2.0)          # predeclared, not data-mined per pair

    def __init__(self, limits: Optional[SearchLimits] = None, tracker=None,
                 max_pairs: int = 60, sector_of=None) -> None:
        super().__init__(limits, tracker)
        self.max_pairs = max_pairs
        self._sector_of = sector_of    # symbol -> sector (None = same-class pairing)

    def _candidate_pairs(self, symbols: List[str]) -> List[Tuple[str, str]]:
        if self._sector_of:
            by_sector: Dict[str, List[str]] = {}
            for s in symbols:
                sec = self._sector_of(s)
                if sec:
                    by_sector.setdefault(sec, []).append(s)
            pairs = [p for group in by_sector.values()
                     for p in combinations(sorted(group), 2)]
        else:
            pairs = list(combinations(sorted(symbols), 2))
        return pairs[: self.max_pairs]

    def scan(self, data: Dict[str, pd.DataFrame]) -> List[DiscoveredHypothesis]:
        closes = {s: pd.to_numeric(df["close"], errors="coerce").dropna()
                  for s, df in data.items() if "close" in df.columns}
        symbols = [s for s, c in closes.items() if len(c) >= 120]
        out: List[DiscoveredHypothesis] = []
        for a, b in self._candidate_pairs(symbols):
            ca, cb = closes[a].align(closes[b], join="inner")
            stats = engle_granger_lite(np.log(ca), np.log(cb))
            if not stats or not stats["stationary"]:
                continue
            spread = (np.log(cb) - stats["beta"] * np.log(ca))
            spread = (spread - spread.mean()) / (spread.std() or 1e-9)
            hold = max(2, min(int(stats["half_life_bars"]), 20))
            for z in self.Z_GRID:
                # long the cheap leg when spread is stretched, converge in `hold`
                entries = spread[spread <= -z].index
                rets = []
                times = []
                for t in entries:
                    loc = spread.index.get_loc(t)
                    if loc + hold >= len(spread):
                        continue
                    rets.append(float(spread.iloc[loc + hold] - spread.iloc[loc])
                                * stats["spread_std"])
                    times.append(t)
                if not rets:
                    continue
                h = self._emit(
                    symbol=f"{b}/{a}", subfamily="pair_reversion",
                    direction="long",
                    conditions=[{"feature": "market_regime", "op": "!=",
                                 "value": "PANIC"}],
                    returns=pd.Series(rets, index=pd.DatetimeIndex(times))
                    if times else rets,
                    holding_bars=hold,
                    description=f"cointegrated pair {b}~{a} beta={stats['beta']:.2f} "
                                f"z<=-{z} HL={stats['half_life_bars']:.0f}",
                    pair=(a, b), hedge_beta=stats["beta"], entry_z=z,
                    half_life_bars=stats["half_life_bars"],
                    market_neutral=True)
                if h:
                    out.append(h)
        return out


class BasketRelativeValueDetector(DiscoveryDetector):
    """Single instrument vs its peer basket: mean reversion of relative
    return extremes at a predeclared percentile grid (spec §56)."""

    family = "MARKET_NEUTRAL"
    PCT_GRID = (0.05, 0.10)           # bottom-decile underperformance vs basket

    def __init__(self, limits: Optional[SearchLimits] = None, tracker=None,
                 peer_basket_fn=None) -> None:
        super().__init__(limits, tracker)
        self._peers = peer_basket_fn   # symbol -> list of peer symbols

    def scan(self, data: Dict[str, pd.DataFrame]) -> List[DiscoveredHypothesis]:
        if self._peers is None:
            return []
        closes = {s: pd.to_numeric(df["close"], errors="coerce").dropna()
                  for s, df in data.items() if "close" in df.columns}
        out: List[DiscoveredHypothesis] = []
        for symbol, close in closes.items():
            if len(close) < 120:
                continue
            peers = [p for p in (self._peers(symbol) or []) if p in closes][:15]
            if len(peers) < 4:
                continue
            basket = pd.concat([closes[p].pct_change() for p in peers], axis=1) \
                .mean(axis=1)
            rel = close.pct_change().rolling(5).sum() - basket.rolling(5).sum()
            rel = rel.dropna()
            if len(rel) < 60:
                continue
            for pct in self.PCT_GRID:
                thresh = rel.quantile(pct)
                entries = rel[rel <= thresh].index
                fwd = close.pct_change(5).shift(-5) - basket.rolling(5).sum().shift(-5)
                rets, times = [], []
                for t in entries:
                    v = fwd.get(t)
                    if v is not None and not pd.isna(v):
                        rets.append(float(v))
                        times.append(t)
                if not rets:
                    continue
                h = self._emit(
                    symbol=symbol, subfamily="basket_reversion",
                    direction="long",
                    conditions=[{"feature": "sector_relative_return", "op": "<",
                                 "value": round(float(thresh), 4)}],
                    returns=pd.Series(rets, index=pd.DatetimeIndex(times)),
                    holding_bars=5,
                    description=f"{symbol} vs {len(peers)}-peer basket, "
                                f"bottom {pct:.0%} relative 5d return",
                    peers=peers, percentile=pct, market_neutral=True)
                if h:
                    out.append(h)
        return out
