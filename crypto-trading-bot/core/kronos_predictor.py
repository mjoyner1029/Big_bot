"""Kronos Foundation Model integration for Big Bot.

Wrapper around Kronos model to provide candlestick predictions for crypto trading.

Kronos path resolution order:
  1. KRONOS_PATH environment variable
  2. Sibling directory ../Kronos (relative to this repo)
  3. Disabled gracefully if not found
"""
import os
import sys
import logging
from typing import Optional, Dict
from datetime import timedelta

import pandas as pd  # always available regardless of Kronos

# ── Kronos path resolution (no hardcoded machine paths) ──────────────────────
def _resolve_kronos_path() -> Optional[str]:
    # 1. Explicit env var
    env_path = os.environ.get('KRONOS_PATH')
    if env_path and os.path.isdir(env_path):
        return env_path
    # 2. Relative sibling: <repo_root>/../Kronos
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sibling = os.path.join(os.path.dirname(repo_root), 'Kronos')
    if os.path.isdir(sibling):
        return sibling
    # 3. Direct sibling of repo root (common monorepo layout)
    sibling2 = os.path.join(repo_root, '..', 'Kronos')
    if os.path.isdir(os.path.normpath(sibling2)):
        return os.path.normpath(sibling2)
    return None

_kronos_path = _resolve_kronos_path()
if _kronos_path and _kronos_path not in sys.path:
    sys.path.insert(0, _kronos_path)

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    import torch
    KRONOS_AVAILABLE = True
except ImportError as e:
    KRONOS_AVAILABLE = False
    _import_error = str(e)

logger = logging.getLogger(__name__)

class BigBotKronosPredictor:
    """Wrapper around Kronos for Big Bot integration.
    
    Provides candlestick predictions for crypto trading using the Kronos
    foundation model trained on 45+ global exchanges.
    """
    
    def __init__(self, model_size='small', device='cpu'):
        """Initialize Kronos predictor.
        
        Args:
            model_size: 'mini', 'small', or 'base'
            device: 'cpu' or 'cuda'
        """
        if not KRONOS_AVAILABLE:
            raise ImportError(f"Kronos not available: {_import_error}")
        
        self.model_size = model_size
        self.device = device
        
        # Lazy load (expensive operation)
        self._tokenizer = None
        self._model = None
        self._predictor = None
        
        logger.info(f"KronosPredictor initialized (model_size={model_size}, device={device})")
    
    @property
    def tokenizer(self):
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            logger.info("Loading Kronos tokenizer...")
            self._tokenizer = KronosTokenizer.from_pretrained(
                "NeoQuasar/Kronos-Tokenizer-base"
            )
        return self._tokenizer
    
    @property
    def model(self):
        """Lazy-load model."""
        if self._model is None:
            logger.info(f"Loading Kronos-{self.model_size} model...")
            self._model = Kronos.from_pretrained(
                f"NeoQuasar/Kronos-{self.model_size}"
            )
            # Move to device
            self._model = self._model.to(self.device)
        return self._model
    
    @property  
    def predictor(self):
        """Lazy-load predictor."""
        if self._predictor is None:
            self._predictor = KronosPredictor(
                self.model,
                self.tokenizer,
                max_context=512
            )
        return self._predictor
    
    def predict_next_candles(self, df: pd.DataFrame, lookback: int = 400, 
                            pred_len: int = 12, verbose: bool = False) -> Optional[pd.DataFrame]:
        """Predict next N candles from historical data.
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Historical candles to use (max 512)
            pred_len: Number of candles to predict
            verbose: Enable verbose logging
            
        Returns:
            DataFrame with predicted OHLCV or None on error
        """
        try:
            # Convert to Kronos format
            kronos_df = self._convert_to_kronos_format(df)
            
            if len(kronos_df) < lookback:
                logger.warning(f"Insufficient data: {len(kronos_df)} < {lookback}")
                return None
            
            # Prepare input data
            x_df = kronos_df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume', 'amount']]
            x_timestamp = kronos_df.iloc[:lookback]['timestamps']
            
            # Generate future timestamps (5-minute intervals)
            y_timestamp = self._generate_future_timestamps(
                x_timestamp.iloc[-1],
                pred_len,
                interval='5min'
            )
            
            # Get prediction
            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=verbose
            )
            
            return pred_df
            
        except Exception as e:
            logger.error(f"Kronos prediction failed: {e}")
            return None
    
    def get_directional_signal(self, df: pd.DataFrame, 
                               lookback: int = 400,
                               pred_len: int = 12) -> Dict:
        """Get LONG/SHORT/NEUTRAL signal from prediction.
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Historical candles to use
            pred_len: Candles to predict ahead (12 = 1 hour for 5-min)
            
        Returns:
            Dictionary with signal, confidence, predicted_change, target_price
        """
        # Get prediction
        pred_df = self.predict_next_candles(df, lookback, pred_len)
        
        if pred_df is None:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'predicted_change': 0.0,
                'target_price': None,
                'error': 'Prediction failed'
            }
        
        # Calculate expected price movement
        current_price = float(df['close'].iloc[-1])
        predicted_price = float(pred_df['close'].iloc[-1])  # pred_len candles ahead
        
        price_change_pct = (predicted_price - current_price) / current_price
        
        # Thresholds
        LONG_THRESHOLD = 0.015   # >1.5% up
        SHORT_THRESHOLD = -0.015  # >1.5% down
        
        # Generate signal
        if price_change_pct > LONG_THRESHOLD:
            signal = 'LONG'
            confidence = min(abs(price_change_pct) * 50, 0.95)
        elif price_change_pct < SHORT_THRESHOLD:
            signal = 'SHORT'
            confidence = min(abs(price_change_pct) * 50, 0.95)
        else:
            signal = 'NEUTRAL'
            confidence = 0.5
        
        return {
            'signal': signal,
            'confidence': confidence,
            'predicted_change': price_change_pct,
            'target_price': predicted_price
        }
    
    def _convert_to_kronos_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Big Bot DataFrame to Kronos format."""
        kronos_df = df.copy()
        
        # Ensure lowercase column names
        kronos_df.columns = [c.lower() for c in kronos_df.columns]
        
        # Ensure timestamps column
        if 'timestamp' in kronos_df.columns and 'timestamps' not in kronos_df.columns:
            kronos_df['timestamps'] = pd.to_datetime(kronos_df['timestamp'])
        elif 'timestamps' not in kronos_df.columns:
            # Check if index is DatetimeIndex
            if isinstance(kronos_df.index, pd.DatetimeIndex):
                kronos_df['timestamps'] = kronos_df.index
            else:
                # Generate timestamps if missing (5-min intervals)
                kronos_df['timestamps'] = pd.date_range(
                    end=pd.Timestamp.now(),
                    periods=len(kronos_df),
                    freq='5min'
                )
        else:
            # Timestamps column exists but might need conversion
            if isinstance(kronos_df['timestamps'], pd.DatetimeIndex):
                pass  # Already datetime
            else:
                kronos_df['timestamps'] = pd.to_datetime(kronos_df['timestamps'])
        
        # Add 'amount' if missing (volume * close)
        if 'amount' not in kronos_df.columns:
            # 🔧 FIX: Use pandas Series multiplication directly (no numpy conversion)
            kronos_df['amount'] = kronos_df['volume'] * kronos_df['close']
        
        # Ensure required columns exist
        required = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
        for col in required:
            if col not in kronos_df.columns:
                logger.error(f"Missing required column: {col}")
                raise ValueError(f"Missing column: {col}")
        
        return kronos_df[required]
    
    def _generate_future_timestamps(self, last_ts, 
                                   n_periods: int, interval: str = '5min') -> pd.Series:
        """Generate future timestamps for predictions."""
        # Handle both pd.Timestamp and pd.Series
        if isinstance(last_ts, pd.Series):
            last_ts = last_ts.iloc[-1]
        elif not isinstance(last_ts, pd.Timestamp):
            last_ts = pd.Timestamp(last_ts)
            
        return pd.Series(pd.date_range(
            start=last_ts + pd.Timedelta(interval),
            periods=n_periods,
            freq=interval
        ))

# Convenience function for checking availability
def is_kronos_available() -> bool:
    """Check if Kronos is available for import."""
    return KRONOS_AVAILABLE
