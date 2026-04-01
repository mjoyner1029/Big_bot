"""
Shared ML utilities: feature engineering, train/test splitting,
model persistence, and normalisation helpers.
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from indicators.ta_indicators import add_ta_indicators
from config.config import CONFIG


# ── Feature columns used by the price-prediction model ────────────
FEATURE_COLS: List[str] = [
    "rsi", "stoch_k", "stoch_d", "williams_r", "roc",
    "macd", "macd_signal", "macd_hist",
    "ema9", "ema20", "ema50", "sma50",
    "adx", "adx_pos", "adx_neg",
    "bb_pctb", "bb_width", "atr",
    "obv", "mfi", "adi",
    "close_pct_change", "close_log_return",
    "high_low_range", "close_to_ema20", "close_to_sma50",
]


def build_features(df: pd.DataFrame, look_ahead: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and binary label y from raw OHLCV data.

    Label:  1 if Close is higher `look_ahead` bars from now, else 0.

    Returns:
        (X, y) — aligned, NaN-dropped feature DataFrame and label Series.
    """
    df = add_ta_indicators(df)

    # Target: price went up over the next `look_ahead` bars
    df["target"] = (df["Close"].shift(-look_ahead) > df["Close"]).astype(int)

    # Keep only feature columns that actually exist in df
    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=available + ["target"])

    X = df[available].copy()
    y = df["target"].copy()
    return X, y


def train_test_split_time(
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-aware train/test split (no shuffling — preserves temporal order).
    """
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def save_model(model, path: Optional[str] = None, symbol: Optional[str] = None) -> str:
    """Pickle a model to disk. If *symbol* is given, path is auto-generated."""
    if path is None:
        base_dir = CONFIG.get("model_dir", "models/saved")
        if symbol:
            safe = symbol.replace("/", "_").replace("-", "_")
            path = os.path.join(base_dir, f"xgb_{safe}.pkl")
        else:
            path = CONFIG.get("price_model_path", "models/saved/xgb_price_model.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {path}")
    return path


def load_model(path: Optional[str] = None, symbol: Optional[str] = None):
    """Load a pickled model from disk. Returns None if file missing."""
    if path is None:
        base_dir = CONFIG.get("model_dir", "models/saved")
        if symbol:
            safe = symbol.replace("/", "_").replace("-", "_")
            path = os.path.join(base_dir, f"xgb_{safe}.pkl")
        else:
            path = CONFIG.get("price_model_path", "models/saved/xgb_price_model.pkl")
    if not os.path.exists(path):
        logging.warning(f"Model file not found: {path}")
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    logging.info(f"Model loaded from {path}")
    return model


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series to 0-1 range."""
    rng = series.max() - series.min()
    if rng == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / rng


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise every column in a DataFrame."""
    return df.apply(normalize_series)
