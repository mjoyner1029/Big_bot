"""N-day breakout strategy with volume confirmation.

The strategy looks for a close beyond the recent high/low range with
supporting volume and uses ATR-scaled stops and targets.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from core.signal_flipper import AssetClass, Signal, SignalType
from core.strategy_base import StrategyBase
from core.strategy_registry import StrategyRegistry


@StrategyRegistry.register("breakout")
class BreakoutStrategy(StrategyBase):
    """N-day breakout with volume confirmation."""

    name = "breakout"

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config)
        self.lookback_days  = self._cfg("breakout_lookback_days", 20)
        self.volume_mult    = self._cfg("breakout_volume_mult",    1.2)  # TUNED: 1.5→1.2 (easier to trigger)
        self.atr_multiplier = self._cfg("breakout_atr_multiplier", 1.5)
        self.rsi_period     = self._cfg("breakout_rsi_period", 14)
        self.rsi_midline    = self._cfg("breakout_rsi_midline", 45)  # TUNED: 50→45 (easier to trigger)

    def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Signal:
        df = data.get("df")
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return self._no_trade(symbol, reason="Missing or invalid DataFrame in data['df']")

        close_col = self._resolve_col(df, ["close", "Close"])
        high_col = self._resolve_col(df, ["high", "High"])
        low_col = self._resolve_col(df, ["low", "Low"])
        volume_col = self._resolve_col(df, ["volume", "Volume"])

        if None in (close_col, high_col, low_col):
            return self._no_trade(symbol, reason="DataFrame missing OHLC columns")

        min_rows = self.lookback_days + 2
        if len(df) < min_rows:
            return self._no_trade(symbol, reason=f"Insufficient rows ({len(df)} < {min_rows}) for breakout lookback")

        work = df.copy()
        close = pd.to_numeric(work[close_col], errors="coerce")
        high = pd.to_numeric(work[high_col], errors="coerce")
        low = pd.to_numeric(work[low_col], errors="coerce")
        volume = pd.to_numeric(work[volume_col], errors="coerce") if volume_col else None

        price = float(close.dropna().iloc[-1])
        recent_high = float(high.shift(1).rolling(self.lookback_days).max().dropna().iloc[-1])
        recent_low = float(low.shift(1).rolling(self.lookback_days).min().dropna().iloc[-1])
        atr = self._atr(high, low, close)
        atr_value = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else None
        rsi_series = self._rsi(close, self.rsi_period)
        rsi_value = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else None

        volume_ok = True
        volume_value = None
        volume_avg = None
        if volume is not None:
            volume_value = float(volume.dropna().iloc[-1])
            volume_avg_series = volume.shift(1).rolling(self.lookback_days).mean().dropna()
            if volume_avg_series.empty:
                return self._no_trade(symbol, reason="Could not compute volume confirmation")
            volume_avg = float(volume_avg_series.iloc[-1])
            volume_ok = volume_value >= (volume_avg * self.volume_mult)

        if price > recent_high and volume_ok and (rsi_value is not None and rsi_value > self.rsi_midline):
            stop_loss = price - ((atr_value or price * 0.02) * self.atr_multiplier)
            if stop_loss >= price:
                stop_loss = price * 0.98
            target_1 = price + (price - stop_loss)
            target_2 = price + 2 * (price - stop_loss)
            confidence = self._confidence(price, recent_high, side="buy", volume_ok=volume_ok)
            return Signal(
                symbol=symbol,
                signal=SignalType.BUY,
                confidence=confidence,
                entry=round(price, 6),
                stop_loss=round(stop_loss, 6),
                targets=[round(target_1, 6), round(target_2, 6)],
                strategy_name=self.name,
                reason=f"BUY: close {price:.4f} broke above {self.lookback_days}-day high {recent_high:.4f}",
                asset_class=self._infer_asset_class(symbol),
                metadata={
                    "recent_high": round(recent_high, 6),
                    "recent_low": round(recent_low, 6),
                    "atr": round(atr_value, 6) if atr_value is not None else None,
                    "volume": round(volume_value, 3) if volume_value is not None else None,
                    "volume_avg": round(volume_avg, 3) if volume_avg is not None else None,
                    "volume_confirmed": volume_ok,
                    "rsi": round(rsi_value, 3) if rsi_value is not None else None,
                },
            )

        if price < recent_low and volume_ok and (rsi_value is not None and rsi_value < self.rsi_midline):
            stop_loss = price + ((atr_value or price * 0.02) * self.atr_multiplier)
            if stop_loss <= price:
                stop_loss = price * 1.02
            target_1 = price - (stop_loss - price)
            target_2 = price - 2 * (stop_loss - price)
            confidence = self._confidence(price, recent_low, side="sell", volume_ok=volume_ok)
            return Signal(
                symbol=symbol,
                signal=SignalType.SELL,
                confidence=confidence,
                entry=round(price, 6),
                stop_loss=round(stop_loss, 6),
                targets=[round(target_1, 6), round(target_2, 6)],
                strategy_name=self.name,
                reason=f"SELL: close {price:.4f} broke below {self.lookback_days}-day low {recent_low:.4f}",
                asset_class=self._infer_asset_class(symbol),
                metadata={
                    "recent_high": round(recent_high, 6),
                    "recent_low": round(recent_low, 6),
                    "atr": round(atr_value, 6) if atr_value is not None else None,
                    "volume": round(volume_value, 3) if volume_value is not None else None,
                    "volume_avg": round(volume_avg, 3) if volume_avg is not None else None,
                    "volume_confirmed": volume_ok,
                    "rsi": round(rsi_value, 3) if rsi_value is not None else None,
                },
            )

        return self._no_trade(
            symbol,
            reason=(
                f"Conditions not met: price={price:.4f}, high={recent_high:.4f}, "
                f"low={recent_low:.4f}, volume_confirmed={volume_ok}, rsi={rsi_value}"
            ),
        )

    @staticmethod
    def _resolve_col(df: pd.DataFrame, choices) -> str | None:
        for col in choices:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-12)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _confidence(price: float, level: float, side: str, volume_ok: bool) -> float:
        breakout_strength = abs(price - level) / max(level, 1e-12)
        base = 60.0 + min(20.0, breakout_strength * 400)
        if volume_ok:
            base += 10.0
        if side == "sell":
            base += 2.0
        return round(min(95.0, max(55.0, base)), 1)

    @staticmethod
    def _infer_asset_class(symbol: str) -> AssetClass:
        upper = symbol.upper()
        if "-" in upper or upper.endswith(("USDT", "USDC")):
            return AssetClass.CRYPTO
        return AssetClass.STOCK
