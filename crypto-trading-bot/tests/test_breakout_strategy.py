"""Unit tests for breakout strategy RSI + volume confirmation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.signal_flipper import SignalType
from strategies.breakout import BreakoutStrategy


class TestBreakoutStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = BreakoutStrategy(
            config={
                "breakout_lookback_days": 10,
                "breakout_volume_mult": 1.2,
                "breakout_rsi_midline": 50,
            }
        )

    @staticmethod
    def _df(kind: str) -> pd.DataFrame:
        base = [100.0 + 0.2 * i for i in range(15)]
        if kind == "buy":
            close = base + [110.0]
        elif kind == "sell":
            close = [110.0 - 0.3 * i for i in range(15)] + [98.0]
        elif kind == "no_rsi":
            close = [110.0 - 0.6 * i for i in range(15)] + [112.0]
        else:
            close = base + [101.0]

        high = [c + 1.0 for c in close]
        low = [c - 1.0 for c in close]
        vol = [1000.0] * 15 + [2200.0]

        return pd.DataFrame({
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        })

    def test_buy_requires_rsi_above_midline(self):
        signal = self.strategy.generate_signal("NVDA", {"df": self._df("buy")})
        self.assertEqual(signal.signal, SignalType.BUY)

    def test_sell_requires_rsi_below_midline(self):
        signal = self.strategy.generate_signal("NVDA", {"df": self._df("sell")})
        self.assertEqual(signal.signal, SignalType.SELL)

    def test_breakout_rejected_when_rsi_not_confirmed(self):
        strict = BreakoutStrategy(
            config={
                "breakout_lookback_days": 10,
                "breakout_volume_mult": 1.2,
                "breakout_rsi_midline": 80,
            }
        )
        signal = strict.generate_signal("NVDA", {"df": self._df("no_rsi")})
        self.assertEqual(signal.signal, SignalType.NO_TRADE)
        self.assertIn("rsi", signal.reason.lower())


if __name__ == "__main__":
    unittest.main()
