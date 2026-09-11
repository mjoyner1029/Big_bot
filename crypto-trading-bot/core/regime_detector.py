#!/usr/bin/env python3
"""
Market Regime Detector - Identifies current market conditions
and adapts strategy selection accordingly

Regimes:
- TRENDING_UP: Strong uptrend
- TRENDING_DOWN: Strong downtrend  
- RANGING: Sideways/choppy
- VOLATILE: High volatility, any direction
- QUIET: Low volatility, tight range
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from enum import Enum


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"


class RegimeDetector:
    """Detects market regime from price data"""
    
    def __init__(self):
        self.regime = None
        self.confidence = 0.0
        
    def detect_regime(self, df: pd.DataFrame) -> Dict:
        """
        Detect current market regime
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with regime, confidence, and metrics
        """
        if len(df) < 50:
            return {
                'regime': MarketRegime.RANGING,
                'confidence': 0.5,
                'metrics': {}
            }
        
        # Calculate indicators
        close = df['close']
        high = df['high']
        low = df['low']
        
        # 1. Trend strength (ADX-like)
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        
        trend_direction = 1 if sma_20.iloc[-1] > sma_50.iloc[-1] else -1
        trend_strength = abs((sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] * 100)
        
        # 2. Volatility (ATR as % of price)
        atr = self._calculate_atr(df, period=14)
        atr_pct = (atr.iloc[-1] / close.iloc[-1]) * 100
        
        # 3. Range vs Trend
        price_range = (high.rolling(20).max().iloc[-1] - low.rolling(20).min().iloc[-1])
        price_range_pct = (price_range / close.iloc[-1]) * 100
        
        # 4. Recent momentum
        momentum_5d = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100)
        momentum_20d = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100)
        
        # Detect regime
        regime = None
        confidence = 0.0
        
        # TRENDING_UP: Strong uptrend
        if (trend_direction > 0 and trend_strength > 2.0 and 
            momentum_5d > 1.0 and momentum_20d > 3.0):
            regime = MarketRegime.TRENDING_UP
            confidence = min(trend_strength / 5.0, 1.0)
        
        # TRENDING_DOWN: Strong downtrend
        elif (trend_direction < 0 and trend_strength > 2.0 and 
              momentum_5d < -1.0 and momentum_20d < -3.0):
            regime = MarketRegime.TRENDING_DOWN
            confidence = min(trend_strength / 5.0, 1.0)
        
        # VOLATILE: High ATR regardless of direction
        elif atr_pct > 3.0:
            regime = MarketRegime.VOLATILE
            confidence = min(atr_pct / 5.0, 1.0)
        
        # QUIET: Low volatility, tight range
        elif atr_pct < 1.0 and price_range_pct < 3.0:
            regime = MarketRegime.QUIET
            confidence = 1.0 - (atr_pct / 2.0)
        
        # RANGING: Default for choppy/sideways
        else:
            regime = MarketRegime.RANGING
            confidence = 0.7
        
        self.regime = regime
        self.confidence = confidence
        
        return {
            'regime': regime,
            'confidence': confidence,
            'metrics': {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'atr_pct': atr_pct,
                'price_range_pct': price_range_pct,
                'momentum_5d': momentum_5d,
                'momentum_20d': momentum_20d,
            }
        }
    
    def get_optimal_strategies(self, regime: MarketRegime) -> List[str]:
        """
        Get best strategies for current regime
        
        Returns:
            List of strategy names suited for this regime
        """
        strategy_map = {
            MarketRegime.TRENDING_UP: [
                'CryptoMomentum',
                'EMATrendFollow',
                'MomentumFactor',
                'Breakout',
            ],
            MarketRegime.TRENDING_DOWN: [
                'CryptoMomentum',  # Can short
                'MomentumFactor',  # Can short
                'RSIDivergence',   # Catch reversals
            ],
            MarketRegime.RANGING: [
                'MeanReversion',
                'VWAPReversion',
                'RSIDivergence',
                'DividendGrowth',  # Works in any condition
            ],
            MarketRegime.VOLATILE: [
                'Breakout',
                'CryptoMomentum',
                'LeveragedRecovery',  # Catches bounces
            ],
            MarketRegime.QUIET: [
                'DividendGrowth',     # Only winner today!
                'MeanReversion',      # Scalp small moves
                'VWAPReversion',      # Tight ranges
                'MACDCrossover',      # Catch small swings
            ],
        }
        
        return strategy_map.get(regime, [
            'DividendGrowth',  # Works everywhere
            'MeanReversion',
        ])
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr


def get_regime_detector() -> RegimeDetector:
    """Get singleton regime detector instance"""
    return RegimeDetector()


if __name__ == "__main__":
    # Test regime detection
    from data.fetcher import fetch_latest_market_data
    
    print("\n" + "="*70)
    print("REGIME DETECTOR TEST")
    print("="*70)
    
    for symbol in ['BTC-USD', 'SPY', 'NVDA']:
        print(f"\n📊 {symbol}")
        print("-"*70)
        
        df = fetch_latest_market_data(symbol)
        if df is not None and len(df) >= 50:
            detector = get_regime_detector()
            result = detector.detect_regime(df)
            
            regime = result['regime'].value
            conf = result['confidence']
            metrics = result['metrics']
            
            print(f"  Regime: {regime.upper()} (confidence: {conf:.1%})")
            print(f"  Trend: {metrics['trend_strength']:.2f}% ({'UP' if metrics['trend_direction'] > 0 else 'DOWN'})")
            print(f"  Volatility: {metrics['atr_pct']:.2f}%")
            print(f"  Momentum 5d: {metrics['momentum_5d']:+.2f}%")
            print(f"  Momentum 20d: {metrics['momentum_20d']:+.2f}%")
            
            strategies = detector.get_optimal_strategies(result['regime'])
            print(f"  Best strategies: {', '.join(strategies)}")
    
    print("\n" + "="*70)
