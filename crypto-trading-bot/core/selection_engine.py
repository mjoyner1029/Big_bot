"""Selection Engine - Quality filter combining pro trader methods + Kronos evaluation.

Inspired by:
- Rayner Teo: Market structure + Area of value + Entry trigger
- Warrior Trading: MACD + Volume profile + Candlestick quality
- Humbled Trader: Risk/Reward validation
- Kronos Foundation Model: Multi-horizon probabilistic evaluation

The selection engine scores every trade 0-100 and rejects anything below threshold.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from core.signal_flipper import Signal, SignalType
from core.kronos_predictor import BigBotKronosPredictor, is_kronos_available

logger = logging.getLogger(__name__)


def get_signal_direction(signal):
    """Extract direction from Signal object (handles both old and new format)."""
    if hasattr(signal, 'signal'):
        sig = signal.signal
        if sig == SignalType.BUY or sig == 'BUY':
            return 'BUY'
        elif sig == SignalType.SELL or sig == 'SELL':
            return 'SELL'
    elif hasattr(signal, 'direction'):
        return signal.direction
    return 'NEUTRAL'


class TradeQualityScore:
    """Rayner Teo + Warrior Trading quality scoring system.
    
    Scores 0-100 based on:
    - Market structure (30 pts): Trend alignment
    - Area of value (25 pts): Near support/resistance
    - Entry trigger (20 pts): Candlestick pattern quality
    - Volume profile (15 pts): Buy/sell pressure
    - MACD confirmation (10 pts): Momentum alignment
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def score_setup(self, symbol: str, signal: Signal, data: pd.DataFrame) -> Dict:
        """Calculate setup quality score (0-100).
        
        Returns:
            {
                'score': int (0-100),
                'market_structure': int (0-30),
                'area_of_value': int (0-25),
                'entry_trigger': int (0-20),
                'volume_profile': int (0-15),
                'macd': int (0-10),
                'breakdown': str,
                'passed': bool
            }
        """
        scores = {
            'market_structure': self._score_market_structure(signal, data),
            'area_of_value': self._score_area_of_value(signal, data),
            'entry_trigger': self._score_entry_trigger(signal, data),
            'volume_profile': self._score_volume_profile(signal, data),
            'macd': self._score_macd(signal, data)
        }
        
        total = sum(scores.values())
        
        # Generate breakdown
        breakdown = " | ".join([f"{k}: {v}" for k, v in scores.items()])
        
        return {
            'score': total,
            'passed': total >= 60,  # Minimum threshold
            'breakdown': breakdown,
            **scores
        }
    
    def _score_market_structure(self, signal: Signal, data: pd.DataFrame) -> int:
        """Rayner Teo: Check trend alignment (0-30 pts).
        
        Rules:
        - LONG in uptrend: 30
        - SHORT in downtrend: 30
        - LONG in downtrend: 0
        - SHORT in uptrend: 0
        - Choppy/ranging: 15
        """
        try:
            # Calculate 20 EMA and 50 EMA
            ema_20 = data['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = data['close'].ewm(span=50).mean().iloc[-1]
            current_price = data['close'].iloc[-1]
            
            # Determine trend
            is_uptrend = ema_20 > ema_50 and current_price > ema_20
            is_downtrend = ema_20 < ema_50 and current_price < ema_20
            
            # Score based on alignment
            if get_signal_direction(signal) == 'BUY':
                if is_uptrend:
                    return 30  # Perfect alignment
                elif is_downtrend:
                    return 0   # Counter-trend (bad)
                else:
                    return 15  # Ranging (marginal)
            else:  # SHORT
                if is_downtrend:
                    return 30
                elif is_uptrend:
                    return 0
                else:
                    return 15
                    
        except Exception as e:
            self.logger.warning(f"Market structure check failed: {e}")
            return 10  # Neutral score on error
    
    def _score_area_of_value(self, signal: Signal, data: pd.DataFrame) -> int:
        """Rayner Teo: Check if price near support/resistance (0-25 pts).
        
        Rules:
        - Within 1% of recent swing high/low: 25
        - Within 2%: 20
        - Within 3%: 15
        - >3% away from key levels: 0-10
        """
        try:
            current_price = data['close'].iloc[-1]
            
            # Find recent swing highs/lows (last 50 bars)
            window = data.iloc[-50:]
            swing_high = window['high'].max()
            swing_low = window['low'].min()
            
            # Calculate distance to nearest level
            dist_to_high = abs(current_price - swing_high) / swing_high
            dist_to_low = abs(current_price - swing_low) / swing_low
            nearest_dist = min(dist_to_high, dist_to_low)
            
            # Score based on distance
            if nearest_dist < 0.01:  # Within 1%
                return 25
            elif nearest_dist < 0.02:  # Within 2%
                return 20
            elif nearest_dist < 0.03:  # Within 3%
                return 15
            elif nearest_dist < 0.05:  # Within 5%
                return 10
            else:
                return 5  # Too far from key levels
                
        except Exception as e:
            self.logger.warning(f"Area of value check failed: {e}")
            return 10
    
    def _score_entry_trigger(self, signal: Signal, data: pd.DataFrame) -> int:
        """Rayner Teo: Candlestick pattern quality (0-20 pts).
        
        Checks:
        - Bullish engulfing / hammer for LONG: +20
        - Bearish engulfing / shooting star for SHORT: +20
        - Strong body (>50% of range): +15
        - Weak body (<30% of range): +5
        - No pattern: +10 (neutral)
        """
        try:
            # Get last 2 candles
            candle1 = data.iloc[-2]
            candle2 = data.iloc[-1]
            
            # Calculate body percentage
            body = abs(candle2['close'] - candle2['open'])
            full_range = candle2['high'] - candle2['low']
            body_pct = body / full_range if full_range > 0 else 0
            
            # Check for patterns
            if get_signal_direction(signal) == 'BUY':
                # Bullish engulfing
                is_engulfing = (candle1['close'] < candle1['open'] and
                               candle2['close'] > candle2['open'] and
                               candle2['close'] > candle1['open'])
                
                # Hammer (long lower wick, small body at top)
                lower_wick = candle2['open'] - candle2['low']
                is_hammer = (lower_wick > 2 * body and
                            candle2['close'] > candle2['open'])
                
                if is_engulfing or is_hammer:
                    return 20
                elif body_pct > 0.5:
                    return 15  # Strong bullish candle
                elif body_pct < 0.3:
                    return 5   # Weak candle
                else:
                    return 10  # Neutral
                    
            else:  # SHORT
                # Bearish engulfing
                is_engulfing = (candle1['close'] > candle1['open'] and
                               candle2['close'] < candle2['open'] and
                               candle2['close'] < candle1['open'])
                
                # Shooting star (long upper wick, small body at bottom)
                upper_wick = candle2['high'] - candle2['close']
                is_shooting_star = (upper_wick > 2 * body and
                                   candle2['close'] < candle2['open'])
                
                if is_engulfing or is_shooting_star:
                    return 20
                elif body_pct > 0.5:
                    return 15
                elif body_pct < 0.3:
                    return 5
                else:
                    return 10
                    
        except Exception as e:
            self.logger.warning(f"Entry trigger check failed: {e}")
            return 10
    
    def _score_volume_profile(self, signal: Signal, data: pd.DataFrame) -> int:
        """Warrior Trading: Buy/sell pressure analysis (0-15 pts).
        
        Checks last 3 bars:
        - <20% red volume: 15 (clean)
        - 20-30% red volume: 10 (acceptable)
        - 30-50% red volume: 5 (marginal)
        - >50% red volume: 0 (too much selling)
        """
        try:
            # Get last 3 bars
            recent = data.iloc[-3:]
            
            # Calculate red/green volume
            red_volume = 0
            green_volume = 0
            
            for idx, row in recent.iterrows():
                if row['close'] < row['open']:
                    red_volume += row['volume']
                else:
                    green_volume += row['volume']
            
            total_volume = red_volume + green_volume
            red_pct = red_volume / total_volume if total_volume > 0 else 0.5
            
            # Score based on selling pressure
            if get_signal_direction(signal) == 'BUY':
                if red_pct < 0.20:
                    return 15  # Clean
                elif red_pct < 0.30:
                    return 10  # Acceptable
                elif red_pct < 0.50:
                    return 5   # Marginal
                else:
                    return 0   # Too much selling
            else:  # SHORT - inverse logic
                if red_pct > 0.80:
                    return 15
                elif red_pct > 0.70:
                    return 10
                elif red_pct > 0.50:
                    return 5
                else:
                    return 0
                    
        except Exception as e:
            self.logger.warning(f"Volume profile check failed: {e}")
            return 5
    
    def _score_macd(self, signal: Signal, data: pd.DataFrame) -> int:
        """Warrior Trading: MACD confirmation (0-10 pts).
        
        Rules:
        - MACD > 0 AND rising for LONG: 10
        - MACD < 0 AND falling for SHORT: 10
        - MACD opposite direction: 0
        - MACD neutral: 5
        """
        try:
            # Calculate MACD
            exp1 = data['close'].ewm(span=12).mean()
            exp2 = data['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9).mean()
            
            macd_val = macd.iloc[-1]
            macd_prev = macd.iloc[-2]
            macd_rising = macd_val > macd_prev
            
            # Score based on alignment
            if get_signal_direction(signal) == 'BUY':
                if macd_val > 0 and macd_rising:
                    return 10  # Perfect
                elif macd_val > 0:
                    return 7   # Positive but not rising
                elif macd_rising:
                    return 5   # Rising but still negative
                else:
                    return 0   # Both negative
            else:  # SHORT
                if macd_val < 0 and not macd_rising:
                    return 10
                elif macd_val < 0:
                    return 7
                elif not macd_rising:
                    return 5
                else:
                    return 0
                    
        except Exception as e:
            self.logger.warning(f"MACD check failed: {e}")
            return 5


class RiskRewardValidator:
    """Humbled Trader: Risk/Reward ratio validation.
    
    Ensures every trade has minimum 2:1 R:R before entry.
    """
    
    def __init__(self, min_rr_ratio: float = 2.0):
        self.min_rr_ratio = min_rr_ratio
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, signal: Signal, entry_price: float, 
                stop_loss: float, target: float) -> Tuple[bool, str, float]:
        """Check if trade meets minimum R:R ratio.
        
        Returns:
            (passed, reason, actual_rr)
        """
        try:
            risk = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            
            if risk == 0:
                return False, "Invalid stop loss (risk = 0)", 0.0
            
            rr_ratio = reward / risk
            
            if rr_ratio >= self.min_rr_ratio:
                return True, f"R:R {rr_ratio:.1f}:1 acceptable", rr_ratio
            else:
                return False, f"R:R {rr_ratio:.1f}:1 below minimum {self.min_rr_ratio}:1", rr_ratio
                
        except Exception as e:
            self.logger.error(f"R:R validation failed: {e}")
            return False, f"R:R check error: {e}", 0.0


class KronosEvaluator:
    """Kronos Foundation Model evaluation - multi-horizon probabilistic scoring.
    
    Uses Kronos's approach:
    - Multiple prediction horizons (5min, 15min, 1hr)
    - Probability-weighted confidence
    - Combines with quality score for final decision
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and is_kronos_available()
        self.predictor = None
        
        if self.enabled:
            try:
                self.predictor = BigBotKronosPredictor(model_size='small', device='cpu')
                logger.info("KronosEvaluator initialized successfully")
            except Exception as e:
                logger.warning(f"Kronos initialization failed: {e}")
                self.enabled = False
        else:
            logger.info("KronosEvaluator disabled (Kronos not available)")
    
    def evaluate(self, symbol: str, signal: Signal, data: pd.DataFrame) -> Dict:
        """Multi-horizon Kronos evaluation.
        
        Returns:
            {
                'kronos_score': int (0-20),  # Bonus points for quality score
                'horizons': {
                    '5min': {...},
                    '15min': {...},
                    '1hr': {...}
                },
                'consensus': str,
                'confidence': float
            }
        """
        if not self.enabled or self.predictor is None:
            return {
                'kronos_score': 0,
                'horizons': {},
                'consensus': 'NEUTRAL',
                'confidence': 0.0,
                'error': 'Kronos not available'
            }
        
        try:
            # Multi-horizon predictions
            horizons = {
                '5min': self._predict_horizon(data, pred_len=1),   # 5 minutes
                '15min': self._predict_horizon(data, pred_len=3),  # 15 minutes
                '1hr': self._predict_horizon(data, pred_len=12)    # 1 hour
            }
            
            # Calculate consensus
            consensus, confidence = self._calculate_consensus(horizons, signal)
            
            # Score (0-20 bonus points)
            kronos_score = self._calculate_score(consensus, confidence, signal)
            
            return {
                'kronos_score': kronos_score,
                'horizons': horizons,
                'consensus': consensus,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Kronos evaluation failed: {e}")
            return {
                'kronos_score': 0,
                'horizons': {},
                'consensus': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _predict_horizon(self, data: pd.DataFrame, pred_len: int) -> Dict:
        """Get prediction for specific horizon."""
        result = self.predictor.get_directional_signal(
            data,
            lookback=min(400, len(data)),
            pred_len=pred_len
        )
        return result
    
    def _calculate_consensus(self, horizons: Dict, signal: Signal) -> Tuple[str, float]:
        """Calculate multi-horizon consensus."""
        votes = {'LONG': 0, 'SHORT': 0, 'NEUTRAL': 0}
        total_confidence = 0
        valid_horizons = 0
        
        for horizon, result in horizons.items():
            if 'signal' in result:
                votes[result['signal']] += 1
                total_confidence += result.get('confidence', 0.0)
                valid_horizons += 1
        
        if valid_horizons == 0:
            return 'NEUTRAL', 0.0
        
        # Determine consensus
        consensus = max(votes, key=votes.get)
        avg_confidence = total_confidence / valid_horizons
        
        return consensus, avg_confidence
    
    def _calculate_score(self, consensus: str, confidence: float, signal: Signal) -> int:
        """Calculate Kronos bonus score (0-20 pts).
        
        Rules:
        - Consensus aligns with signal + high confidence: 20
        - Consensus aligns + medium confidence: 15
        - Consensus aligns + low confidence: 10
        - Neutral consensus: 5
        - Consensus contradicts signal: 0
        """
        signal_direction = 'LONG' if get_signal_direction(signal) == 'BUY' else 'SHORT'
        
        if consensus == signal_direction:
            if confidence > 0.75:
                return 20  # Strong agreement
            elif confidence > 0.60:
                return 15  # Moderate agreement
            else:
                return 10  # Weak agreement
        elif consensus == 'NEUTRAL':
            return 5  # No opinion
        else:
            return 0  # Contradiction


class SelectionEngine:
    """Master selection engine - combines all quality filters + Kronos evaluation.
    
    Total score: 0-120 points
    - Quality filters: 0-100 (market structure, area of value, trigger, volume, MACD)
    - Kronos bonus: 0-20 (multi-horizon consensus)
    
    Thresholds:
    - 80+: Full position size (high quality)
    - 60-79: Half position size (marginal quality)
    - <60: Reject trade (low quality)
    """
    
    def __init__(self, enable_kronos: bool = True, min_rr_ratio: float = 2.0):
        self.quality_scorer = TradeQualityScore()
        self.rr_validator = RiskRewardValidator(min_rr_ratio)
        self.kronos_evaluator = KronosEvaluator(enabled=enable_kronos)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate_trade(self, symbol: str, signal: Signal, data: pd.DataFrame,
                      entry_price: float, stop_loss: float, target: float) -> Dict:
        """Complete trade evaluation - quality + R:R + Kronos.
        
        Returns:
            {
                'approved': bool,
                'total_score': int (0-120),
                'quality_score': int (0-100),
                'kronos_score': int (0-20),
                'rr_ratio': float,
                'rr_passed': bool,
                'position_size_multiplier': float (0, 0.5, or 1.0),
                'reason': str,
                'breakdown': Dict
            }
        """
        # Step 1: Quality score (0-100)
        quality_result = self.quality_scorer.score_setup(symbol, signal, data)
        quality_score = quality_result['score']
        
        # Step 2: R:R validation (hard requirement)
        rr_passed, rr_reason, rr_ratio = self.rr_validator.validate(
            signal, entry_price, stop_loss, target
        )
        
        # Step 3: Kronos evaluation (0-20 bonus)
        kronos_result = self.kronos_evaluator.evaluate(symbol, signal, data)
        kronos_score = kronos_result['kronos_score']
        
        # Total score
        total_score = quality_score + kronos_score
        
        # Decision logic
        if not rr_passed:
            # Auto-reject if R:R too low
            approved = False
            size_multiplier = 0.0
            reason = f"Rejected: {rr_reason}"
        elif total_score >= 80:
            # High quality - full size
            approved = True
            size_multiplier = 1.0
            reason = f"Approved: High quality ({total_score}/120)"
        elif total_score >= 60:
            # Marginal quality - half size
            approved = True
            size_multiplier = 0.5
            reason = f"Approved: Marginal quality ({total_score}/120), reduced size"
        else:
            # Low quality - reject
            approved = False
            size_multiplier = 0.0
            reason = f"Rejected: Low quality ({total_score}/120)"
        
        # Log decision
        self.logger.info(
            f"[{symbol}] Selection: {reason} | "
            f"Quality: {quality_score}/100 | "
            f"Kronos: {kronos_score}/20 | "
            f"R:R: {rr_ratio:.1f}:1"
        )
        
        return {
            'approved': approved,
            'total_score': total_score,
            'quality_score': quality_score,
            'kronos_score': kronos_score,
            'rr_ratio': rr_ratio,
            'rr_passed': rr_passed,
            'position_size_multiplier': size_multiplier,
            'reason': reason,
            'breakdown': {
                'quality': quality_result,
                'kronos': kronos_result,
                'rr': {'passed': rr_passed, 'ratio': rr_ratio, 'reason': rr_reason}
            }
        }
