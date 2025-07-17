import logging
from typing import Optional, Dict, Any
from indicators.ta_indicators import add_ta_indicators
from models.price_predictor import predict_price_movement
from models.sentiment_model import analyze_sentiment
from strategies.thresholds import get_trade_thresholds
from config.config import CONFIG
import pandas as pd

def generate_trade_signal(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Generate a trade signal based on TA, ML, and sentiment.
    Args:
        df: DataFrame with historical price data.
    Returns:
        Trade signal dict or None.
    """
    try:
        df = add_ta_indicators(df)
        recent = df.iloc[-1]

        ta_score = 1 if (recent.get("rsi", 100) < 30 and recent.get("macd", -1) > 0) else 0
        ml_score = predict_price_movement(df)
        sentiment_score = analyze_sentiment("Bitcoin is trending up!")  # placeholder text

        combined_confidence = 0.4 * ta_score + 0.4 * ml_score + 0.2 * sentiment_score
        logging.info(f"Confidence: {combined_confidence:.2f}")

        if combined_confidence >= CONFIG["confidence_threshold"]:
            entry_price = recent["Close"]
            thresholds = get_trade_thresholds(entry_price, combined_confidence)
            return {
                "symbol": CONFIG.get("symbol", "ETH-USD"),
                "side": "buy",
                "entry_price": entry_price,
                "confidence": combined_confidence,
                **thresholds
            }

        return None
    except Exception as e:
        logging.error(f"Error generating trade signal: {e}")
        return None
