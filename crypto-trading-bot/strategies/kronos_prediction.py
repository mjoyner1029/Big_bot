"""Kronos Foundation Model prediction strategy.

Uses Kronos (AAAI 2026) foundation model trained on 45+ exchanges to predict
next N candlesticks and generate trading signals.
"""
from typing import Dict, Any
import logging

from core.strategy_base import StrategyBase
from core.signal_flipper import Signal, SignalType
from core.strategy_registry import StrategyRegistry
from core.kronos_predictor import BigBotKronosPredictor, is_kronos_available

logger = logging.getLogger(__name__)

@StrategyRegistry.register("kronos_prediction")
class KronosPredictionStrategy(StrategyBase):
    """Foundation model predictions using Kronos.
    
    Predicts next 12 candles (1 hour for 5-min timeframe) and generates
    signals based on expected price movement.
    """
    
    name = "kronos_prediction"
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Configuration
        self.model_size = self._cfg('kronos_model_size', 'small')
        self.min_confidence = self._cfg('kronos_min_confidence', 0.60)
        self.pred_horizon = self._cfg('kronos_pred_horizon', 12)  # 12 candles = 1 hour
        self.lookback = self._cfg('kronos_lookback', 400)
        
        # Lazy-load predictor (expensive)
        self._predictor = None
        
        # Check availability
        if not is_kronos_available():
            logger.error("Kronos not available - strategy will not generate signals")
    
    @property
    def predictor(self):
        """Lazy-load Kronos predictor."""
        if self._predictor is None:
            if not is_kronos_available():
                return None
            
            logger.info(f"Loading Kronos {self.model_size} model (first use)...")
            self._predictor = BigBotKronosPredictor(
                model_size=self.model_size,
                device='cpu'  # Can change to 'cuda' if GPU available
            )
            logger.info("✅ Kronos model loaded")
        
        return self._predictor
    
    def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Signal:
        """Generate signal using Kronos predictions."""
        
        # Check if Kronos is available
        if not is_kronos_available():
            return self._no_trade(
                symbol,
                reason="Kronos not available (missing dependencies)"
            )
        
        # Get DataFrame
        df = data.get('df')
        if df is None or len(df) < self.lookback:
            return self._no_trade(
                symbol,
                reason=f"Insufficient data for Kronos (<{self.lookback} candles)"
            )
        
        try:
            # Get Kronos prediction
            kronos_signal = self.predictor.get_directional_signal(
                df,
                lookback=self.lookback,
                pred_len=self.pred_horizon
            )
            
            # Check for errors
            if 'error' in kronos_signal:
                return self._no_trade(
                    symbol,
                    reason=f"Kronos error: {kronos_signal['error']}"
                )
            
            # Check confidence
            if kronos_signal['confidence'] < self.min_confidence:
                return self._no_trade(
                    symbol,
                    reason=f"Kronos confidence {kronos_signal['confidence']:.1%} < {self.min_confidence:.1%}"
                )
            
            # Convert to Signal
            price = float(df['close'].iloc[-1])
            target = kronos_signal['target_price']
            predicted_change = kronos_signal['predicted_change']
            
            # Determine signal type
            if kronos_signal['signal'] == 'LONG':
                signal_type = SignalType.BUY
                stop_loss = price * 0.97  # 3% stop loss
            elif kronos_signal['signal'] == 'SHORT':
                signal_type = SignalType.SELL
                stop_loss = price * 1.03  # 3% stop loss
            else:
                return self._no_trade(
                    symbol,
                    reason="Kronos: NEUTRAL signal"
                )
            
            # Create signal
            confidence_pct = kronos_signal['confidence'] * 100
            
            return Signal(
                symbol=symbol,
                signal=signal_type,
                confidence=confidence_pct,
                entry=price,                    # 🔧 FIX: entry not entry_price
                stop_loss=stop_loss,
                targets=[target],               # 🔧 FIX: targets list not take_profit
                reason=f"Kronos predicts {predicted_change:+.2%} move in {self.pred_horizon} candles",
                strategy_name=self.name,
                asset_class=self._asset_class(symbol)  # 🔧 FIX: Add asset class
            )
            
        except Exception as e:
            logger.error(f"Kronos prediction failed for {symbol}: {e}", exc_info=True)
            return self._no_trade(
                symbol,
                reason=f"Kronos error: {str(e)}"
            )
    
    @staticmethod
    def _asset_class(symbol: str):
        """Determine asset class from symbol."""
        from core.signal_flipper import AssetClass
        u = symbol.upper()
        if "-" in u or u.endswith(("USDT", "USDC", "USD")):
            return AssetClass.CRYPTO
        return AssetClass.STOCK
