"""Statistical Arbitrage Strategy — regression-based pairs trading.

Uses linear regression to model the relationship between correlated assets,
then trades deviations from the predicted relationship. More sophisticated
than simple pairs trading because it accounts for time-varying spreads.

Strategy:
  1. Run rolling regression: asset1_price = β * asset2_price + α
  2. Calculate residual (actual - predicted)
  3. Trade when residual exceeds threshold (z-score)
  4. Market-neutral: hedge ratio determined by β coefficient

This works even when correlation changes over time, making it more robust
than static ratio-based pairs trading.

Position sizing: Use β as hedge ratio for market neutrality.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from core.signal_flipper import Signal, SignalType, AssetClass
from core.strategy_base import StrategyBase
from core.strategy_registry import StrategyRegistry

logger = logging.getLogger(__name__)


@StrategyRegistry.register("statistical_arbitrage")
class StatisticalArbitrageStrategy(StrategyBase):
    """Statistical arbitrage using rolling OLS regression.
    
    Parameters
    ----------
    pair_symbols : tuple
        The two symbols to pair trade
    lookback_days : int
        Rolling window for regression (default 60)
    entry_z_score : float
        Z-score threshold to enter (default 2.0)
    exit_z_score : float
        Z-score to take profit (default 0.5)
    min_correlation : float
        Minimum correlation required (default 0.7)
    min_r_squared : float
        Minimum R² for regression fit quality (default 0.5)
    """

    def __init__(
        self,
        pair_symbols: tuple[str, str] = ("NVDA", "AMD"),
        lookback_days: int = 60,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
        min_correlation: float = 0.7,
        min_r_squared: float = 0.5,
    ) -> None:
        self.pair_symbols = pair_symbols
        self.lookback_days = lookback_days
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.min_correlation = min_correlation
        self.min_r_squared = min_r_squared
        self.name = f"stat_arb_{pair_symbols[0]}_{pair_symbols[1]}"

    def generate_signal(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        """Generate statistical arbitrage signals using regression residuals."""
        if context is None or "pair_data" not in context:
            return self._no_trade(symbol, "Missing pair_data in context")

        pair_data = context["pair_data"]
        sym1, sym2 = self.pair_symbols

        if sym1 not in pair_data or sym2 not in pair_data:
            return self._no_trade(symbol, f"Missing data for {sym1} or {sym2}")

        df1 = pair_data[sym1]
        df2 = pair_data[sym2]

        if df1.empty or df2.empty or len(df1) < self.lookback_days or len(df2) < self.lookback_days:
            return self._no_trade(symbol, f"Insufficient data (need {self.lookback_days})")

        # Merge and align data
        df_merged = pd.DataFrame({
            "y": df1["close"],  # Dependent variable (sym1)
            "x": df2["close"],  # Independent variable (sym2)
        }).dropna()

        if len(df_merged) < self.lookback_days:
            return self._no_trade(symbol, "Insufficient aligned data")

        # Rolling OLS regression
        y = df_merged["y"].values[-self.lookback_days:]
        x = df_merged["x"].values[-self.lookback_days:]

        # Add intercept column
        X = np.column_stack([np.ones(len(x)), x])
        
        try:
            # Compute regression: y = α + β*x + ε
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
        except Exception as e:
            return self._no_trade(symbol, f"Regression failed: {e}")

        # Quality checks
        if r_squared < self.min_r_squared:
            return self._no_trade(
                symbol,
                f"Poor fit: R²={r_squared:.3f} < {self.min_r_squared}"
            )

        correlation = df_merged["y"].rolling(self.lookback_days).corr(df_merged["x"]).iloc[-1]
        if pd.isna(correlation) or correlation < self.min_correlation:
            return self._no_trade(
                symbol,
                f"Low correlation: {correlation:.3f} < {self.min_correlation}"
            )

        # Calculate residuals (actual - predicted)
        df_merged["predicted"] = intercept + slope * df_merged["x"]
        df_merged["residual"] = df_merged["y"] - df_merged["predicted"]

        # Z-score of residual
        residual_mean = df_merged["residual"].rolling(self.lookback_days).mean().iloc[-1]
        residual_std = df_merged["residual"].rolling(self.lookback_days).std().iloc[-1]

        if pd.isna(residual_std) or residual_std == 0:
            return self._no_trade(symbol, "Invalid residual statistics")

        current_residual = df_merged["residual"].iloc[-1]
        z_score = (current_residual - residual_mean) / residual_std

        price1 = float(df_merged["y"].iloc[-1])
        price2 = float(df_merged["x"].iloc[-1])
        hedge_ratio = slope  # β coefficient determines hedge ratio

        # Entry logic
        if z_score > self.entry_z_score:
            # Residual too high = sym1 overvalued relative to sym2
            # Short sym1, long sym2 (hedge ratio adjusted)
            confidence = self._calculate_confidence(abs(z_score), r_squared, correlation)
            
            return Signal(
                symbol=f"{sym1}/{sym2}",
                signal=SignalType.SELL,  # Short sym1
                confidence=confidence,
                entry=round(price1, 6),
                stop_loss=round(price1 * 1.03, 6),
                targets=[round(price1 * 0.99, 6), round(price1 * 0.97, 6)],
                strategy_name=self.name,
                reason=f"StatArb: SHORT {sym1}, LONG {sym2} | z={z_score:.2f}, β={hedge_ratio:.3f}",
                asset_class=AssetClass.STOCK,
                metadata={
                    "pair_symbols": self.pair_symbols,
                    "z_score": round(z_score, 3),
                    "hedge_ratio": round(hedge_ratio, 4),
                    "r_squared": round(r_squared, 3),
                    "correlation": round(correlation, 3),
                    "residual": round(current_residual, 6),
                    "price1": round(price1, 6),
                    "price2": round(price2, 6),
                    "long_symbol": sym2,
                    "short_symbol": sym1,
                    "intercept": round(intercept, 6),
                    "slope": round(slope, 6),
                },
            )

        elif z_score < -self.entry_z_score:
            # Residual too low = sym1 undervalued relative to sym2
            # Long sym1, short sym2
            confidence = self._calculate_confidence(abs(z_score), r_squared, correlation)
            
            return Signal(
                symbol=f"{sym1}/{sym2}",
                signal=SignalType.BUY,  # Long sym1
                confidence=confidence,
                entry=round(price1, 6),
                stop_loss=round(price1 * 0.97, 6),
                targets=[round(price1 * 1.01, 6), round(price1 * 1.03, 6)],
                strategy_name=self.name,
                reason=f"StatArb: LONG {sym1}, SHORT {sym2} | z={z_score:.2f}, β={hedge_ratio:.3f}",
                asset_class=AssetClass.STOCK,
                metadata={
                    "pair_symbols": self.pair_symbols,
                    "z_score": round(z_score, 3),
                    "hedge_ratio": round(hedge_ratio, 4),
                    "r_squared": round(r_squared, 3),
                    "correlation": round(correlation, 3),
                    "residual": round(current_residual, 6),
                    "price1": round(price1, 6),
                    "price2": round(price2, 6),
                    "long_symbol": sym1,
                    "short_symbol": sym2,
                    "intercept": round(intercept, 6),
                    "slope": round(slope, 6),
                },
            )

        return self._no_trade(
            symbol,
            f"Residual within range: z={z_score:.2f}",
            metadata={
                "z_score": round(z_score, 3),
                "r_squared": round(r_squared, 3),
                "hedge_ratio": round(hedge_ratio, 4),
            }
        )

    def _calculate_confidence(
        self, z_score: float, r_squared: float, correlation: float
    ) -> float:
        """Confidence based on z-score, fit quality, and correlation."""
        # Z-score contribution (0-30 points)
        z_conf = min(30.0, (z_score - self.entry_z_score) * 10)
        
        # R² fit quality (0-20 points)
        r2_conf = (r_squared - self.min_r_squared) * 40  # 0.5-1.0 → 0-20 points
        
        # Correlation strength (0-15 points)
        corr_conf = (correlation - self.min_correlation) * 50
        
        confidence = 55.0 + z_conf + r2_conf + corr_conf
        return round(min(90.0, max(55.0, confidence)), 1)

    def _no_trade(
        self,
        symbol: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Signal:
        """Helper for NO_TRADE signals."""
        return Signal(
            symbol=symbol,
            signal=SignalType.NO_TRADE,
            strategy_name=self.name,
            reason=reason,
            metadata=metadata or {}
        )


# Statistical arbitrage basket — high-quality correlated pairs
STAT_ARB_PAIRS = [
    ("NVDA", "AMD"),       # High correlation, same sector
    ("BTC-USD", "ETH-USD"),  # Crypto majors
    ("JPM", "BAC"),        # Large banks
    ("XLF", "KBE"),        # Financial ETFs
    ("GLD", "GDX"),        # Gold vs gold miners
    ("USO", "XLE"),        # Oil vs energy sector
]
