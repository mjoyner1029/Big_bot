"""EMA trend-follow strategy.

Uses a 9/21/50 EMA stack to identify trending markets.  A BUY is triggered
when the fast EMA crosses above the mid EMA while price is above the slow
EMA.  SELL is the mirror.  ADX-style trend strength is approximated via the
EMA spread width.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from core.signal_flipper import AssetClass, Signal, SignalType
from core.strategy_base import StrategyBase
from core.strategy_registry import StrategyRegistry


@StrategyRegistry.register("ema_trend_follow")
class EMATrendFollowStrategy(StrategyBase):
    """9/21/50 EMA golden/death cross with trend-strength filter."""

    name = "ema_trend_follow"

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config)
        self.fast_ema    = int(self._cfg("ema_fast",  9))
        self.mid_ema     = int(self._cfg("ema_mid",  21))
        self.slow_ema    = int(self._cfg("ema_slow", 50))
        self.min_spread_pct = float(self._cfg("ema_min_spread_pct", 0.002))  # TUNED: 0.003→0.002 (easier)
        self.atr_stop_mult  = float(self._cfg("ema_atr_stop_mult",  1.5))

    def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Signal:
        df = data.get("df")
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return self._no_trade(symbol, reason="Missing or invalid DataFrame")

        close_col = self._col(df, ["close", "Close"])
        high_col  = self._col(df, ["high",  "High"])
        low_col   = self._col(df, ["low",   "Low"])
        if close_col is None:
            return self._no_trade(symbol, reason="Missing close column")

        if len(df) < self.slow_ema + 5:
            return self._no_trade(symbol, reason=f"Insufficient data ({len(df)} < {self.slow_ema + 5})")

        close = pd.to_numeric(df[close_col], errors="coerce")
        high  = pd.to_numeric(df[high_col],  errors="coerce") if high_col else close
        low   = pd.to_numeric(df[low_col],   errors="coerce") if low_col  else close

        ema_f = close.ewm(span=self.fast_ema, adjust=False).mean()
        ema_m = close.ewm(span=self.mid_ema,  adjust=False).mean()
        ema_s = close.ewm(span=self.slow_ema, adjust=False).mean()

        if len(ema_f.dropna()) < 2:
            return self._no_trade(symbol, reason="Not enough EMA history")

        price    = float(close.dropna().iloc[-1])
        ef_now   = float(ema_f.dropna().iloc[-1])
        em_now   = float(ema_m.dropna().iloc[-1])
        es_now   = float(ema_s.dropna().iloc[-1])
        ef_prev  = float(ema_f.dropna().iloc[-2])
        em_prev  = float(ema_m.dropna().iloc[-2])

        spread_pct = abs(ef_now - es_now) / es_now if es_now else 0.0
        atr_val = self._atr_value(high, low, close)

        golden_cross = (ef_prev <= em_prev) and (ef_now > em_now) and (price > es_now)
        death_cross  = (ef_prev >= em_prev) and (ef_now < em_now) and (price < es_now)

        if golden_cross and spread_pct >= self.min_spread_pct:
            stop   = price - (self.atr_stop_mult * atr_val) if atr_val else price * 0.97
            target = price + 1.5 * (price - stop)
            conf   = min(88.0, 55.0 + spread_pct * 5000)
            return Signal(
                symbol=symbol,
                signal=SignalType.BUY,
                confidence=conf,
                entry=round(price, 6),
                stop_loss=round(stop, 6),
                targets=[round(price + (price - stop), 6), round(target, 6)],
                strategy_name=self.name,
                reason=f"BUY: EMA golden cross (fast={ef_now:.2f} > mid={em_now:.2f}), spread={spread_pct:.3%}",
                asset_class=self._asset_class(symbol),
                metadata={"ema_fast": round(ef_now, 4), "ema_mid": round(em_now, 4),
                          "ema_slow": round(es_now, 4), "spread_pct": round(spread_pct, 5)},
            )

        if death_cross and spread_pct >= self.min_spread_pct:
            stop   = price + (self.atr_stop_mult * atr_val) if atr_val else price * 1.03
            target = price - 2.5 * (stop - price)
            conf   = min(88.0, 55.0 + spread_pct * 5000)
            return Signal(
                symbol=symbol,
                signal=SignalType.SELL,
                confidence=conf,
                entry=round(price, 6),
                stop_loss=round(stop, 6),
                targets=[round(price - (stop - price), 6), round(target, 6)],
                strategy_name=self.name,
                reason=f"SELL: EMA death cross (fast={ef_now:.2f} < mid={em_now:.2f}), spread={spread_pct:.3%}",
                asset_class=self._asset_class(symbol),
                metadata={"ema_fast": round(ef_now, 4), "ema_mid": round(em_now, 4),
                          "ema_slow": round(es_now, 4), "spread_pct": round(spread_pct, 5)},
            )

        return self._no_trade(
            symbol,
            reason=f"No EMA cross: fast={ef_now:.4f}, mid={em_now:.4f}, slow={es_now:.4f}",
        )

    @staticmethod
    def _col(df: pd.DataFrame, names: list) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        return None

    @staticmethod
    def _atr_value(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        clean = atr.dropna()
        return float(clean.iloc[-1]) if not clean.empty else 0.0

    @staticmethod
    def _asset_class(symbol: str) -> AssetClass:
        u = symbol.upper()
        if "-" in u or u.endswith(("USDT", "USDC")):
            return AssetClass.CRYPTO
        return AssetClass.STOCK
