"""Centralized Feature Registry.

Alphas request features by validated name instead of recomputing indicators
independently. All features are computed strictly from data up to and
including the decision bar (timestamp-safe, no future leakage) and cached per
(symbol, data-signature).
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_VERSION = "1.0.0"


def _last(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    if s.empty:
        return None
    v = float(s.iloc[-1])
    return v if math.isfinite(v) else None


def _close(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["close"], errors="coerce")


def _f_price(df):
    return _last(_close(df))


def _f_returns(df, bars: int):
    c = _close(df).dropna()
    if len(c) <= bars:
        return None
    prev = float(c.iloc[-1 - bars])
    return (float(c.iloc[-1]) - prev) / prev if prev else None


def _f_volume(df):
    if "volume" not in df.columns:
        return None
    return _last(pd.to_numeric(df["volume"], errors="coerce"))


def _f_relative_volume_20d(df):
    if "volume" not in df.columns:
        return None
    v = pd.to_numeric(df["volume"], errors="coerce").dropna()
    if len(v) < 21:
        return None
    base = float(v.iloc[-21:-1].mean())
    return float(v.iloc[-1]) / base if base > 0 else None


def _f_atr(df, period: int = 14):
    if not {"high", "low", "close"} <= set(df.columns):
        return None
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = _close(df)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return _last(tr.ewm(alpha=1 / period, min_periods=period).mean())


def _f_atr_pct(df):
    atr = _f_atr(df)
    price = _f_price(df)
    return atr / price if atr is not None and price else None


def _f_rsi(df, period: int = 14):
    c = _close(df)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    rs_series = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs_series))
    return _last(rsi)


def _f_macd(df):
    c = _close(df)
    macd = c.ewm(span=12, min_periods=12, adjust=False).mean() - \
        c.ewm(span=26, min_periods=26, adjust=False).mean()
    return _last(macd)


def _f_sma(df, period: int):
    return _last(_close(df).rolling(period, min_periods=period).mean())


def _f_distance_from_sma(df, period: int):
    sma = _f_sma(df, period)
    price = _f_price(df)
    return (price - sma) / sma if sma and price else None


def _f_realized_volatility(df, bars: int = 20):
    c = _close(df).dropna()
    if len(c) <= bars:
        return None
    rets = c.pct_change().iloc[-bars:]
    return float(rets.std())


def _f_gap_pct(df):
    """Open of last bar vs previous close."""
    if "open" not in df.columns or len(df) < 2:
        return None
    o = pd.to_numeric(df["open"], errors="coerce")
    c = _close(df)
    prev_close = float(c.iloc[-2]) if not math.isnan(float(c.iloc[-2])) else None
    last_open = float(o.iloc[-1]) if not math.isnan(float(o.iloc[-1])) else None
    if not prev_close or last_open is None:
        return None
    return (last_open - prev_close) / prev_close


def _f_day_of_week(df):
    if isinstance(df.index, pd.DatetimeIndex) and len(df):
        return df.index[-1].strftime("%A").upper()
    return None


def _f_time_of_day(df):
    if isinstance(df.index, pd.DatetimeIndex) and len(df):
        return df.index[-1].hour
    return None


def _f_bar_range_pct(df):
    """Intrabar (high-low)/mid range — liquidity-risk proxy, NOT bid/ask spread."""
    if not {"high", "low"} <= set(df.columns) or not len(df):
        return None
    hi = float(pd.to_numeric(df["high"], errors="coerce").iloc[-1])
    lo = float(pd.to_numeric(df["low"], errors="coerce").iloc[-1])
    mid = (hi + lo) / 2
    return (hi - lo) / mid if mid > 0 else None


def _f_dollar_volume_24h(df):
    if "volume" not in df.columns or len(df) < 1:
        return None
    v = pd.to_numeric(df["volume"], errors="coerce").iloc[-24:]
    p = _f_price(df)
    return float(v.sum()) * p if p else None


class FeatureRegistry:
    """Validated, cached, leakage-safe feature computation.

    Context features (market_regime, funding_rate, earnings_days_away, ...)
    that cannot be derived from OHLCV are supplied via ``context`` and simply
    passed through — they are still validated names.
    """

    _COMPUTED: Dict[str, Callable] = {
        "price": _f_price,
        "returns_1d": lambda df: _f_returns(df, 1),
        "returns_5d": lambda df: _f_returns(df, 5),
        "returns_20d": lambda df: _f_returns(df, 20),
        "volume": _f_volume,
        "relative_volume_20d": _f_relative_volume_20d,
        "atr": _f_atr,
        "atr_pct": _f_atr_pct,
        "rsi": _f_rsi,
        "macd": _f_macd,
        "sma_20": lambda df: _f_sma(df, 20),
        "sma_50": lambda df: _f_sma(df, 50),
        "sma_200": lambda df: _f_sma(df, 200),
        "distance_from_sma_20": lambda df: _f_distance_from_sma(df, 20),
        "distance_from_sma_50": lambda df: _f_distance_from_sma(df, 50),
        "distance_from_sma_200": lambda df: _f_distance_from_sma(df, 200),
        "realized_volatility": _f_realized_volatility,
        "gap_pct": _f_gap_pct,
        "day_of_week": _f_day_of_week,
        "time_of_day": _f_time_of_day,
        "bar_range_pct": _f_bar_range_pct,
        "dollar_volume_24h": _f_dollar_volume_24h,
    }

    # Context-only features (provided externally; never fabricated)
    _CONTEXT = frozenset({
        "market_regime", "sector", "sector_relative_return",
        "market_relative_return", "earnings_days_away", "funding_rate",
        "open_interest", "basis", "asset_class",
        # External intelligence (point-in-time, from the RawEventStore)
        "new_federal_award", "award_amount_vs_market_cap", "award_growth_30d",
        "congress_buy_count_7d", "congress_buy_count_30d",
        "net_congress_direction", "unique_members_buying", "clustered_buying",
        "skill_weighted_buying", "committee_relevance_score",
        "top_trader_long_pressure", "top_trader_short_pressure",
        "skill_weighted_positioning",
        # Universal discovery outputs
        "returns_percentile_20d", "returns_percentile_60d",
        "momentum_acceleration_score", "overnight_contribution",
        "change_point_detected",
    })

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.version = FEATURE_VERSION

    @classmethod
    def known_features(cls) -> List[str]:
        return sorted(set(cls._COMPUTED) | cls._CONTEXT)

    @classmethod
    def is_valid_feature(cls, name: str) -> bool:
        return name in cls._COMPUTED or name in cls._CONTEXT

    def compute(
        self,
        symbol: str,
        df: pd.DataFrame,
        names: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute the requested features from data up to the last bar.

        Unknown names raise ValueError (validated registry — no arbitrary
        feature strings). Missing context features resolve to None.
        """
        context = context or {}
        names = names or list(self._COMPUTED)
        for n in names:
            if not self.is_valid_feature(n):
                raise ValueError(f"Unknown feature '{n}' — not in FeatureRegistry")

        work = df.copy()
        work.columns = [str(c).lower() for c in work.columns]
        key = f"{symbol}:{self._signature(work)}"
        cached = self._cache.get(key, {})

        out: Dict[str, Any] = {}
        for n in names:
            if n in self._CONTEXT:
                out[n] = context.get(n)
                continue
            if n in cached:
                out[n] = cached[n]
                continue
            try:
                value = self._COMPUTED[n](work)
            except Exception as e:
                logger.debug(f"FeatureRegistry: {n} failed for {symbol}: {e}")
                value = None
            cached[n] = value
            out[n] = value

        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = cached
        return out

    @staticmethod
    def _signature(df: pd.DataFrame) -> str:
        if not len(df):
            return "empty"
        tail = df["close"].iloc[-5:] if "close" in df.columns else df.iloc[-5:, 0]
        payload = f"{len(df)}:{df.index[-1]}:{list(pd.to_numeric(tail, errors='coerce'))}"
        return hashlib.md5(payload.encode()).hexdigest()[:12]
