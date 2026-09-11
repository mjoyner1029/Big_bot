"""Mean-reversion Z-score strategy.

Computes the rolling Z-score of price relative to its N-day moving average.
Extreme deviations (|Z| > threshold) in mean-reverting regimes signal entry
against the move.  RSI confirms the extremity.

Works on stocks, ETFs — less reliable on trending crypto but included with
a conservative threshold.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from core.signal_flipper import AssetClass, Signal, SignalType
from core.strategy_base import StrategyBase
from core.strategy_registry import StrategyRegistry


@StrategyRegistry.register("mean_reversion_zscore")
class MeanReversionZScoreStrategy(StrategyBase):
    """Rolling Z-score mean-reversion entry."""

    name = "mean_reversion_zscore"

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config)
        self.ma_period     = int(self._cfg("zscore_ma_period",   20))
        self.std_period    = int(self._cfg("zscore_std_period",  20))
        self.z_entry       = float(self._cfg("zscore_z_entry",   2.0))  # GOLDILOCKS: 11 trades/year
        self.rsi_oversold  = float(self._cfg("zscore_rsi_os",   45.0))  # GOLDILOCKS
        self.rsi_overbought = float(self._cfg("zscore_rsi_ob",  55.0))  # GOLDILOCKS
        self.atr_stop_mult = float(self._cfg("zscore_atr_stop",  1.5))

    def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Signal:
        df = data.get("df")
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return self._no_trade(symbol, reason="Missing or invalid DataFrame")

        close_col = self._col(df, ["close", "Close"])
        high_col  = self._col(df, ["high",  "High"])
        low_col   = self._col(df, ["low",   "Low"])
        if close_col is None:
            return self._no_trade(symbol, reason="Missing close column")

        min_rows = max(self.ma_period, self.std_period) + 14 + 5
        if len(df) < min_rows:
            return self._no_trade(symbol, reason=f"Insufficient data ({len(df)} < {min_rows})")

        close = pd.to_numeric(df[close_col], errors="coerce")
        high  = pd.to_numeric(df[high_col],  errors="coerce") if high_col else close
        low   = pd.to_numeric(df[low_col],   errors="coerce") if low_col  else close

        ma  = close.rolling(self.ma_period).mean()
        std = close.rolling(self.std_period).std(ddof=1)
        std = std.replace(0, 1e-12)
        zscore = (close - ma) / std

        rsi = self._rsi(close)
        atr_val = self._atr_value(high, low, close)

        price     = float(close.dropna().iloc[-1])
        z_now     = float(zscore.dropna().iloc[-1]) if not zscore.dropna().empty else 0.0
        rsi_now   = float(rsi.dropna().iloc[-1])    if not rsi.dropna().empty    else 50.0

        if abs(z_now) < self.z_entry:
            return self._no_trade(
                symbol,
                reason=f"Z-score {z_now:.2f} below entry threshold {self.z_entry:.1f}",
            )

        if z_now <= -self.z_entry and rsi_now <= self.rsi_oversold:
            stop   = price - (self.atr_stop_mult * atr_val) if atr_val else price * 0.97
            ma_val = float(ma.dropna().iloc[-1]) if not ma.dropna().empty else price
            target = ma_val  # revert to mean
            conf   = min(85.0, 50.0 + abs(z_now) * 6 + (self.rsi_oversold - rsi_now) * 0.3)
            return Signal(
                symbol=symbol,
                signal=SignalType.BUY,
                confidence=conf,
                entry=round(price, 6),
                stop_loss=round(stop, 6),
                targets=[round(target, 6)],
                strategy_name=self.name,
                reason=f"BUY: Z-score oversold; z={z_now:.2f}, rsi={rsi_now:.1f}",
                asset_class=self._asset_class(symbol),
                metadata={"zscore": round(z_now, 3), "rsi": round(rsi_now, 1), "ma": round(ma_val, 4)},
            )

        if z_now >= self.z_entry and rsi_now >= self.rsi_overbought:
            stop   = price + (self.atr_stop_mult * atr_val) if atr_val else price * 1.03
            ma_val = float(ma.dropna().iloc[-1]) if not ma.dropna().empty else price
            target = ma_val
            conf   = min(85.0, 50.0 + abs(z_now) * 6 + (rsi_now - self.rsi_overbought) * 0.3)
            return Signal(
                symbol=symbol,
                signal=SignalType.SELL,
                confidence=conf,
                entry=round(price, 6),
                stop_loss=round(stop, 6),
                targets=[round(target, 6)],
                strategy_name=self.name,
                reason=f"SELL: Z-score overbought; z={z_now:.2f}, rsi={rsi_now:.1f}",
                asset_class=self._asset_class(symbol),
                metadata={"zscore": round(z_now, 3), "rsi": round(rsi_now, 1), "ma": round(ma_val, 4)},
            )

        return self._no_trade(
            symbol,
            reason=f"Z-score {z_now:.2f} extreme but RSI {rsi_now:.1f} not confirming",
        )

    @staticmethod
    def _col(df: pd.DataFrame, names: list) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        return None

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_g = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_l = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs    = avg_g / avg_l.replace(0, 1e-12)
        return 100 - (100 / (1 + rs))

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
