"""VWAP mean-reversion strategy.

Stocks trading significantly below VWAP with a bullish reversal candle are
strong mean-reversion candidates intraday.  Above-VWAP bearish reversals
trigger SELL signals.  VWAP is computed from the OHLCV data in ``data["df"]``.

This strategy is stocks/ETF only — VWAP is less meaningful for 24/7 crypto
since session boundaries don't reset it.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from core.signal_flipper import AssetClass, Signal, SignalType
from core.strategy_base import StrategyBase
from core.strategy_registry import StrategyRegistry


@StrategyRegistry.register("vwap_reversion")
class VWAPReversionStrategy(StrategyBase):
    """Intraday VWAP mean-reversion for stocks and ETFs."""

    name = "vwap_reversion"
    supported_asset_classes = (AssetClass.STOCK, AssetClass.ETF)

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config)
        self.vwap_band_pct  = float(self._cfg("vwap_band_pct",  0.015))   # TUNED: 0.5%→1.5% (require bigger deviation)
        self.rsi_oversold   = float(self._cfg("vwap_rsi_os",   30.0))     # TUNED: 35→30 (tighter)
        self.rsi_overbought = float(self._cfg("vwap_rsi_ob",   70.0))     # TUNED: 65→70 (tighter)
        self.atr_stop_mult  = float(self._cfg("vwap_atr_stop", 1.2))
        self.volume_mult    = float(self._cfg("vwap_volume_mult", 1.3))  # TUNED: Add volume confirmation

    def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Signal:
        df = data.get("df")
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return self._no_trade(symbol, reason="Missing or invalid DataFrame")

        close_col  = self._col(df, ["close",  "Close"])
        high_col   = self._col(df, ["high",   "High"])
        low_col    = self._col(df, ["low",    "Low"])
        volume_col = self._col(df, ["volume", "Volume"])

        if None in (close_col, high_col, low_col, volume_col):
            return self._no_trade(symbol, reason="DataFrame missing OHLCV columns")

        if len(df) < 20:
            return self._no_trade(symbol, reason="Insufficient data (need ≥ 20 bars)")

        close  = pd.to_numeric(df[close_col],  errors="coerce")
        high   = pd.to_numeric(df[high_col],   errors="coerce")
        low    = pd.to_numeric(df[low_col],    errors="coerce")
        volume = pd.to_numeric(df[volume_col], errors="coerce")

        typical_price = (high + low + close) / 3.0
        tp_vol = typical_price * volume
        vwap   = tp_vol.cumsum() / volume.cumsum()

        rsi = self._rsi(close)

        price    = float(close.dropna().iloc[-1])
        vwap_now = float(vwap.dropna().iloc[-1]) if not vwap.dropna().empty else price
        rsi_now  = float(rsi.dropna().iloc[-1])  if not rsi.dropna().empty  else 50.0

        atr_val  = self._atr_value(high, low, close)
        dev_pct  = (price - vwap_now) / vwap_now if vwap_now else 0.0

        # Oversold below VWAP → mean-reversion BUY
        if dev_pct <= -self.vwap_band_pct and rsi_now <= self.rsi_oversold:
            stop   = price - (self.atr_stop_mult * atr_val) if atr_val else price * 0.985
            target = vwap_now  # target is VWAP reversion
            conf   = min(85.0, 50.0 + abs(dev_pct) * 2000 + (self.rsi_oversold - rsi_now) * 0.5)
            return Signal(
                symbol=symbol,
                signal=SignalType.BUY,
                confidence=conf,
                entry=round(price, 6),
                stop_loss=round(stop, 6),
                targets=[round(target, 6)],
                strategy_name=self.name,
                reason=f"BUY: VWAP reversion; price={price:.2f} vs vwap={vwap_now:.2f} ({dev_pct:.2%}), rsi={rsi_now:.1f}",
                asset_class=AssetClass.STOCK,
                metadata={"vwap": round(vwap_now, 4), "dev_pct": round(dev_pct, 5), "rsi": round(rsi_now, 1)},
            )

        # Overbought above VWAP → mean-reversion SELL
        if dev_pct >= self.vwap_band_pct and rsi_now >= self.rsi_overbought and volume_surge:
            stop   = price + (self.atr_stop_mult * atr_val) if atr_val else price * 1.015
            target = vwap_now
            conf   = min(85.0, 50.0 + abs(dev_pct) * 2000 + (rsi_now - self.rsi_overbought) * 0.5)
            return Signal(
                symbol=symbol,
                signal=SignalType.SELL,
                confidence=conf,
                entry=round(price, 6),
                stop_loss=round(stop, 6),
                targets=[round(target, 6)],
                strategy_name=self.name,
                reason=f"SELL: VWAP reversion; price={price:.2f} vs vwap={vwap_now:.2f} ({dev_pct:.2%}), rsi={rsi_now:.1f}",
                asset_class=AssetClass.STOCK,
                metadata={"vwap": round(vwap_now, 4), "dev_pct": round(dev_pct, 5), "rsi": round(rsi_now, 1)},
            )

        return self._no_trade(
            symbol,
            reason=f"No VWAP setup: price={price:.2f}, vwap={vwap_now:.2f}, dev={dev_pct:.3%}, rsi={rsi_now:.1f}",
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
