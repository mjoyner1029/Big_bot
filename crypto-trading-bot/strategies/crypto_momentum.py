"""Crypto Momentum Strategy.

🔧 FIXED (July 23, 2026):
- Removed hardcoded 2% stop loss cap
- Now uses thresholds.py for stops and targets
- Respects our 4-5% stop, 2.5-3.5% target fixes

Generates BUY/SELL signals for crypto assets based on RSI momentum combined
with a trend filter (price vs SMA) and optional volume confirmation.

Signal logic
------------
BUY  : RSI(14) is above ``rsi_bull`` (default 55) AND rising over the last
       ``lookback`` bars AND price is above its SMA(20).
SELL : RSI(14) is below ``rsi_bear`` (default 45) AND falling over the last
       ``lookback`` bars AND price is below its SMA(20).

Unlike a strict single-bar crossover, this looks for *sustained momentum* so
it fires on trending markets without needing an exact threshold-crossing in the
most recent bar.

Designed for crypto assets (symbols containing "-" or ending in USDT/BUSD/USD).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from core.signal_flipper import AssetClass, Signal, SignalType
from core.strategy_base import StrategyBase
from core.strategy_registry import StrategyRegistry
from strategies.thresholds import get_trade_thresholds  # 🔧 ADDED


@StrategyRegistry.register("crypto_momentum")
class CryptoMomentumStrategy(StrategyBase):
    """RSI-crossover momentum strategy for crypto assets."""

    name = "crypto_momentum"
    supported_asset_classes = (AssetClass.CRYPTO,)

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.rsi_period: int = int(self._cfg("crypto_rsi_period", 14))
        self.sma_period: int = int(self._cfg("crypto_sma_period", 20))
        self.rsi_bull: float = float(self._cfg("crypto_rsi_bull", 53.0))  # GOLDILOCKS: 21 trades/year
        self.rsi_bear: float = float(self._cfg("crypto_rsi_bear", 47.0))  # GOLDILOCKS
        self.lookback: int = int(self._cfg("crypto_momentum_lookback", 3))  # GOLDILOCKS
        self.atr_stop_mult: float = float(self._cfg("crypto_atr_stop_mult", 1.5))
        self.min_bars: int = max(self.rsi_period, self.sma_period) + self.lookback + 2

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Signal:
        """Return a BUY, SELL, or NO_TRADE signal for *symbol*.

        Expected data keys
        ------------------
        df : pd.DataFrame  OHLCV with DatetimeIndex (required)
        """
        # Only apply to crypto symbols
        if not self._is_crypto(symbol):
            return self._no_trade(symbol, reason="Not a crypto symbol — crypto_momentum only trades crypto")

        df = data.get("df")
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return self._no_trade(symbol, reason="Missing or invalid DataFrame in data['df']")

        close_col = self._resolve_col(df, ["close", "Close"])
        high_col = self._resolve_col(df, ["high", "High"])
        low_col = self._resolve_col(df, ["low", "Low"])
        volume_col = self._resolve_col(df, ["volume", "Volume"])

        if close_col is None:
            return self._no_trade(symbol, reason="DataFrame missing close price column")

        if len(df) < self.min_bars:
            return self._no_trade(
                symbol,
                reason=f"Insufficient data ({len(df)} bars, need {self.min_bars})",
            )

        close = pd.to_numeric(df[close_col], errors="coerce")
        if close.isna().iloc[-1]:
            return self._no_trade(symbol, reason="Latest close price is NaN")

        # ── Indicators ────────────────────────────────────────────────────

        rsi = self._rsi(close, self.rsi_period)
        sma = close.rolling(self.sma_period).mean()

        rsi_now = self._last_valid(rsi)
        rsi_anchor = self._nth_valid(rsi, self.lookback + 3)  # RSI N bars ago
        sma_now = self._last_valid(sma)
        price = self._last_valid(close)

        if None in (rsi_now, rsi_anchor, sma_now, price):
            return self._no_trade(symbol, reason="Could not compute RSI/SMA values")

        # ── Volume (informational — not a hard gate) ───────────────────────

        vol_note = ""
        if volume_col is not None:
            vol = pd.to_numeric(df[volume_col], errors="coerce")
            vol_now = self._last_valid(vol)
            vol_ma = self._last_valid(vol.rolling(self.sma_period).mean())
            if vol_now is not None and vol_ma is not None and vol_ma > 0:
                ratio = vol_now / vol_ma
                vol_note = f" vol_ratio={ratio:.2f}"

        # ── ATR for stop-loss ─────────────────────────────────────────────

        atr_value: Optional[float] = None
        if high_col is not None and low_col is not None:
            high = pd.to_numeric(df[high_col], errors="coerce")
            low = pd.to_numeric(df[low_col], errors="coerce")
            atr_series = self._atr(high, low, close, period=14)
            atr_value = self._last_valid(atr_series)

        # ── Signal logic ──────────────────────────────────────────────────
        # BUY: RSI is above bull threshold AND has been rising over lookback window
        rsi_rising = rsi_now > rsi_anchor
        rsi_falling = rsi_now < rsi_anchor
        price_above_sma = price > sma_now
        price_below_sma = price < sma_now

        # ── Calculate confidence (needed for thresholds) ──────────────────
        confidence = self._confidence(rsi_now, side="buy")  # Calculate early

        # BUY: RSI above bull zone and trending up, price above SMA
        if rsi_now >= self.rsi_bull and rsi_rising and price_above_sma:
            # 🔧 FIX: Use thresholds.py instead of hardcoded values
            thresholds = get_trade_thresholds(
                entry_price=price,
                confidence=confidence,
                side="buy",
                asset_type="crypto",
                atr=atr_value
            )
            
            stop_loss = thresholds['stop_loss_price']
            target1 = thresholds['take_profit_price']
            
            # 🔧 FIX: Stretch target - extend beyond main target, not based on stop
            # For BUY: target should be ABOVE entry
            target_dist = target1 - price  # How far above entry is target1
            target2 = target1 + (target_dist * 0.5)  # 50% beyond main target
            
            return Signal(
                symbol=symbol,
                signal=SignalType.BUY,
                confidence=confidence,
                entry=round(price, 6),
                stop_loss=round(stop_loss, 6),
                targets=[round(target1, 6), round(target2, 6)],
                strategy_name=self.name,
                asset_class=AssetClass.CRYPTO,
                reason=(
                    f"BUY momentum: RSI={rsi_now:.1f} > {self.rsi_bull} and rising "
                    f"({rsi_anchor:.1f}→{rsi_now:.1f} over {self.lookback}h), "
                    f"price({price:.4f}) > SMA{self.sma_period}({sma_now:.4f}){vol_note} | "
                    f"🔧 Using thresholds.py: SL={thresholds['stop_loss_pct']*100:.1f}% TP={thresholds['take_profit_pct']*100:.1f}%"
                ),
            )

        # SELL: RSI below bear zone and trending down, price below SMA
        if rsi_now <= self.rsi_bear and rsi_falling and price_below_sma:
            confidence = self._confidence(rsi_now, side="sell")
            
            # 🔧 FIX: Use thresholds.py instead of hardcoded values
            thresholds = get_trade_thresholds(
                entry_price=price,
                confidence=confidence,
                side="sell",
                asset_type="crypto",
                atr=atr_value
            )
            
            stop_loss = thresholds['stop_loss_price']
            target1 = thresholds['take_profit_price']
            
            # Optional: Add a stretch target
            risk_dist = stop_loss - price
            target2 = price - risk_dist * 2.0  # Aggressive stretch
            
            return Signal(
                symbol=symbol,
                signal=SignalType.SELL,
                confidence=confidence,
                entry=round(price, 6),
                stop_loss=round(stop_loss, 6),
                targets=[round(target1, 6), round(target2, 6)],
                strategy_name=self.name,
                asset_class=AssetClass.CRYPTO,
                reason=(
                    f"SELL momentum: RSI={rsi_now:.1f} < {self.rsi_bear} and falling "
                    f"({rsi_anchor:.1f}→{rsi_now:.1f} over {self.lookback}h), "
                    f"price({price:.4f}) < SMA{self.sma_period}({sma_now:.4f}){vol_note} | "
                    f"🔧 Using thresholds.py: SL={thresholds['stop_loss_pct']*100:.1f}% TP={thresholds['take_profit_pct']*100:.1f}%"
                ),
            )

        # No signal — explain the closest miss
        trend = "above" if price_above_sma else "below"
        rsi_dir = "rising" if rsi_rising else "falling"
        reason = (
            f"No signal: RSI={rsi_now:.1f} ({rsi_dir}, anchor={rsi_anchor:.1f}), "
            f"price {trend} SMA{self.sma_period}({sma_now:.4f}){vol_note}"
        )
        return self._no_trade(symbol, reason=reason)

    # ── Static helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _is_crypto(symbol: str) -> bool:
        return "-" in symbol or symbol.upper().endswith(("USDT", "BUSD", "USD"))

    @staticmethod
    def _resolve_col(df: pd.DataFrame, choices: list) -> Optional[str]:
        for col in choices:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, 1e-12)
        return 100 - (100 / (1 + rs))

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
    def _last_valid(series: pd.Series) -> Optional[float]:
        valid = series.dropna()
        return float(valid.iloc[-1]) if not valid.empty else None

    @staticmethod
    def _nth_valid(series: pd.Series, n: int = 2) -> Optional[float]:
        """Return the n-th value from the end of non-NaN values (1 = last)."""
        valid = series.dropna()
        return float(valid.iloc[-n]) if len(valid) >= n else None

    def _confidence(self, rsi: float, side: str) -> float:
        """Confidence scaled by RSI distance from the crossover threshold.
        
        Returns: 55-80 (percentage score, not decimal!)
        """
        if side == "buy":
            strength = min(1.0, (rsi - self.rsi_bull) / 20.0)
        else:
            strength = min(1.0, (self.rsi_bear - rsi) / 20.0)
        base = 55.0 + strength * 25.0   # range [55, 80]
        return round(min(80.0, max(55.0, base)), 1)
