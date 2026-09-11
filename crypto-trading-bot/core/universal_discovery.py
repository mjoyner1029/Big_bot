"""Universal discovery layer — finds what nobody told it to look for.

    CrossSectionalOpportunityScanner : multi-horizon absolute/relative/risk-
                                       adjusted percentile ranks + momentum
                                       acceleration across the whole universe
    BenchmarkResolver                : per-instrument benchmark sets
    MarketChangePointDetector        : CUSUM + rolling distribution divergence
    MarketOutlierDetector            : robust-z / MAD outliers on any features
    ReturnDecompositionEngine        : WHERE returns are earned (overnight vs
                                       intraday, weekdays, month turns) —
                                       independently surfaces Micron-style
                                       overnight anomalies universe-wide

Outputs are ANOMALIES and HYPOTHESES, never trades (AnomalyScore ≠
OpportunityScore). No hard-coded tickers anywhere.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Benchmark resolution (spec §27) ───────────────────────────────────────────


class BenchmarkResolver:
    """Relevant benchmarks per instrument: market + sector ETF + asset-class
    index. Uses InstrumentUniverse metadata — no named-ticker special cases."""

    def __init__(self, universe=None) -> None:
        if universe is None:
            from core.instruments import InstrumentUniverse
            universe = InstrumentUniverse()
        self.universe = universe

    def benchmarks_for(self, symbol: str) -> List[str]:
        inst = self.universe.get(symbol)
        if inst is None:
            return ["SPY"] if "-" not in symbol else ["BTC-USD"]
        if inst.asset_class == "crypto":
            return [b for b in ("BTC-USD", "ETH-USD") if b != symbol]
        out = ["SPY", "QQQ"]
        if inst.sector_etf and inst.sector_etf != symbol:
            out.append(inst.sector_etf)
        return out

    def peer_basket(self, symbol: str) -> List[str]:
        inst = self.universe.get(symbol)
        if inst is None or not inst.sector:
            return []
        return [s for s in self.universe.symbols_in_sector(inst.sector)
                if s != symbol]


# ── Cross-sectional scanner (spec §25-29) ─────────────────────────────────────

HORIZONS = (1, 5, 20, 60, 120, 252)


@dataclass
class CrossSectionalRow:
    symbol: str
    returns: Dict[int, float] = field(default_factory=dict)          # horizon -> return
    return_percentiles: Dict[int, float] = field(default_factory=dict)
    market_relative: Dict[int, float] = field(default_factory=dict)
    sector_relative: Dict[int, float] = field(default_factory=dict)
    risk_adjusted_20d: Optional[float] = None
    momentum_acceleration_score: Optional[float] = None
    volume_acceleration: Optional[float] = None
    volatility_change: Optional[float] = None
    anomaly_flags: List[str] = field(default_factory=list)


class CrossSectionalOpportunityScanner:
    """Ranks the whole universe every research interval — unexpected
    outperformers surface as percentile extremes, not named tickers."""

    def __init__(self, benchmark_resolver: Optional[BenchmarkResolver] = None,
                 extreme_percentile: float = 0.99,
                 min_abs_move: float = 0.02) -> None:
        self.benchmarks = benchmark_resolver or BenchmarkResolver()
        self.extreme_percentile = extreme_percentile
        self.min_abs_move = min_abs_move   # rank extremes must also be
        # economically meaningful — a flat market has a top percentile too

    def scan(self, data: Dict[str, pd.DataFrame]) -> List[CrossSectionalRow]:
        rows: Dict[str, CrossSectionalRow] = {}
        for symbol, df in data.items():
            close = pd.to_numeric(df["close"], errors="coerce").dropna()
            if len(close) < 30:
                continue
            row = CrossSectionalRow(symbol=symbol)
            for h in HORIZONS:
                if len(close) > h:
                    row.returns[h] = float(close.iloc[-1] / close.iloc[-1 - h] - 1)
            rets20 = close.pct_change().iloc[-20:]
            vol = float(rets20.std())
            if vol > 0 and 20 in row.returns:
                row.risk_adjusted_20d = row.returns[20] / (vol * math.sqrt(20))
            # Momentum acceleration: short-horizon strength vs long (spec §29)
            if 20 in row.returns and 60 in row.returns:
                short_rate = row.returns[20] / 20
                long_rate = row.returns[60] / 60
                row.momentum_acceleration_score = short_rate - long_rate
            if "volume" in df.columns:
                v = pd.to_numeric(df["volume"], errors="coerce").dropna()
                if len(v) > 40:
                    recent, base = float(v.iloc[-5:].mean()), float(v.iloc[-40:-5].mean())
                    row.volume_acceleration = recent / base if base > 0 else None
            r_now = float(close.pct_change().iloc[-10:].std())
            r_before = float(close.pct_change().iloc[-60:-10].std()) if len(close) > 60 else None
            if r_before:
                row.volatility_change = r_now / r_before
            rows[symbol] = row

        # Market/sector-relative returns
        for symbol, row in rows.items():
            for bench in self.benchmarks.benchmarks_for(symbol):
                b = rows.get(bench)
                if b is None:
                    continue
                target = (row.market_relative if bench in ("SPY", "QQQ", "BTC-USD",
                                                           "ETH-USD")
                          else row.sector_relative)
                for h in HORIZONS:
                    if h in row.returns and h in b.returns:
                        target[h] = row.returns[h] - b.returns[h]

        # Cross-sectional percentile ranks (spec §28)
        for h in HORIZONS:
            vals = sorted((r.returns[h], s) for s, r in rows.items()
                          if h in r.returns)
            n = len(vals)
            for rank, (_, s) in enumerate(vals):
                rows[s].return_percentiles[h] = (rank + 1) / n if n else 0.5

        # Flag extremes — this is how unexpected outperformers are found
        for row in rows.values():
            for h in (20, 60):
                pct = row.return_percentiles.get(h)
                ret = row.returns.get(h, 0.0)
                if pct is None or abs(ret) < self.min_abs_move:
                    continue
                if pct >= self.extreme_percentile and ret > 0:
                    row.anomaly_flags.append(f"EXTREME_RELATIVE_OUTPERFORMANCE_{h}D")
                if pct <= 1 - self.extreme_percentile and ret < 0:
                    row.anomaly_flags.append(f"EXTREME_UNDERPERFORMANCE_{h}D")
            if (row.momentum_acceleration_score or 0) > 0 and \
                    row.return_percentiles.get(20, 0) > 0.95 and \
                    row.returns.get(20, 0.0) >= self.min_abs_move:
                row.anomaly_flags.append("MOMENTUM_ACCELERATION")
        return list(rows.values())


# ── Change-point detection (spec §30-31) ──────────────────────────────────────


@dataclass
class ChangePoint:
    symbol: str
    metric: str
    index: int
    timestamp: Optional[str]
    magnitude: float
    method: str


class MarketChangePointDetector:
    """CUSUM mean-shift + rolling volatility-divergence change points."""

    def __init__(self, threshold_sigmas: float = 6.0, drift: float = 0.5) -> None:
        self.threshold_sigmas = threshold_sigmas
        self.drift = drift

    def detect(self, symbol: str, df: pd.DataFrame) -> List[ChangePoint]:
        out: List[ChangePoint] = []
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close) < 60:
            return out
        rets = close.pct_change().dropna()
        out += self._cusum(symbol, rets, "returns_mean", df)
        vol = rets.rolling(10).std().dropna()
        out += self._cusum(symbol, vol.diff().dropna(), "volatility", df)
        if "volume" in df.columns:
            v = pd.to_numeric(df["volume"], errors="coerce").dropna()
            if len(v) > 60:
                out += self._cusum(symbol, v.pct_change().dropna(), "volume", df)
        return out

    def _cusum(self, symbol: str, series: pd.Series, metric: str,
               df: pd.DataFrame) -> List[ChangePoint]:
        values = series.to_numpy(dtype=float)
        mu, sigma = float(np.mean(values)), float(np.std(values))
        if sigma <= 0:
            return []
        z = (values - mu) / sigma
        pos = neg = 0.0
        points = []
        for i, x in enumerate(z):
            pos = max(0.0, pos + x - self.drift)
            neg = min(0.0, neg + x + self.drift)
            if pos > self.threshold_sigmas or neg < -self.threshold_sigmas:
                ts = str(series.index[i]) if hasattr(series.index, "__getitem__") else None
                points.append(ChangePoint(symbol=symbol, metric=metric, index=i,
                                          timestamp=ts,
                                          magnitude=max(pos, -neg),
                                          method="CUSUM"))
                pos = neg = 0.0
        return points


# ── General outlier engine (spec §32-33) ──────────────────────────────────────


@dataclass
class Outlier:
    symbol: str
    feature: str
    value: float
    robust_z: float
    kind: str


class MarketOutlierDetector:
    """Robust-z (median/MAD) outliers over any feature matrix — price/volume/
    volatility/correlation/external-intelligence features alike."""

    def __init__(self, z_threshold: float = 4.0) -> None:
        self.z_threshold = z_threshold

    def detect(self, feature_matrix: Dict[str, Dict[str, float]]) -> List[Outlier]:
        """feature_matrix: symbol -> {feature_name: value}."""
        by_feature: Dict[str, List[Tuple[str, float]]] = {}
        for symbol, feats in feature_matrix.items():
            for name, value in feats.items():
                if value is None or not math.isfinite(value):
                    continue
                by_feature.setdefault(name, []).append((symbol, value))
        out: List[Outlier] = []
        for name, pairs in by_feature.items():
            if len(pairs) < 8:
                continue
            values = np.array([v for _, v in pairs], dtype=float)
            med = float(np.median(values))
            mad = float(np.median(np.abs(values - med)))
            scale = mad * 1.4826 if mad > 0 else float(np.std(values)) or 1e-9
            for symbol, value in pairs:
                z = (value - med) / scale
                if abs(z) >= self.z_threshold:
                    out.append(Outlier(symbol=symbol, feature=name, value=value,
                                       robust_z=z,
                                       kind="high" if z > 0 else "low"))
        return out


# ── Return decomposition (spec §34-35) ────────────────────────────────────────


@dataclass
class ReturnDecomposition:
    symbol: str
    total_log_return: float
    overnight_contribution: float        # fraction of total log return
    intraday_contribution: float
    weekday_contributions: Dict[str, float]
    month_end_contribution: float
    extreme_flags: List[str] = field(default_factory=list)


class ReturnDecompositionEngine:
    """Asks WHERE returns are earned. An asset whose long-run appreciation is
    ≥ the configured share overnight (close→open) gets flagged and becomes an
    automatic research hypothesis — universe-wide, no named tickers."""

    def __init__(self, extreme_share: float = 0.8, min_t_stat: float = 2.0) -> None:
        self.extreme_share = extreme_share
        self.min_t_stat = min_t_stat

    def decompose(self, symbol: str, df: pd.DataFrame) -> Optional[ReturnDecomposition]:
        if not {"open", "close"} <= set(df.columns) or len(df) < 60:
            return None
        o = pd.to_numeric(df["open"], errors="coerce")
        c = pd.to_numeric(df["close"], errors="coerce")
        prev_c = c.shift(1)
        overnight = np.log(o / prev_c).replace([np.inf, -np.inf], np.nan).dropna()
        intraday = np.log(c / o).replace([np.inf, -np.inf], np.nan).dropna()
        total = float(overnight.sum() + intraday.sum())
        if abs(total) < 1e-9:
            return None
        on_share = float(overnight.sum()) / total
        in_share = float(intraday.sum()) / total

        weekday: Dict[str, float] = {}
        if isinstance(df.index, pd.DatetimeIndex):
            daily = np.log(c / prev_c).dropna()
            for day, grp in daily.groupby(daily.index.strftime("%A").str.upper()):
                weekday[day] = float(grp.sum()) / total
        month_end = 0.0
        if isinstance(df.index, pd.DatetimeIndex):
            daily = np.log(c / prev_c).dropna()
            mask = daily.index.day >= 28
            month_end = float(daily[mask].sum()) / total

        flags = []

        def _t_stat(series: pd.Series) -> float:
            s = float(series.std())
            n = len(series)
            if s <= 0 or n < 20:
                return 0.0
            return float(series.mean()) / (s / math.sqrt(n))

        # Shares alone are unstable when total return is small — require the
        # component itself to be STATISTICALLY significant, not just dominant
        if total > 0 and on_share >= self.extreme_share and \
                _t_stat(overnight) >= self.min_t_stat:
            flags.append("EXTREME_OVERNIGHT_CONTRIBUTION")
        if total > 0 and in_share >= self.extreme_share and \
                _t_stat(intraday) >= self.min_t_stat:
            flags.append("EXTREME_INTRADAY_CONTRIBUTION")
        if isinstance(df.index, pd.DatetimeIndex):
            daily = np.log(c / prev_c).dropna()
            for day, share in weekday.items():
                if total > 0 and share >= self.extreme_share:
                    grp = daily[daily.index.strftime("%A").str.upper() == day]
                    if _t_stat(grp) >= self.min_t_stat:
                        flags.append(f"EXTREME_{day}_CONTRIBUTION")

        return ReturnDecomposition(
            symbol=symbol, total_log_return=total,
            overnight_contribution=on_share,
            intraday_contribution=in_share,
            weekday_contributions=weekday,
            month_end_contribution=month_end,
            extreme_flags=flags,
        )
