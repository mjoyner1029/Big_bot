"""
XGBoost-based price-direction predictor.

- `train_model()`      – train from scratch on a DataFrame of OHLCV.
- `predict_price_movement()` – return a confidence score (0-1) for an up-move.
"""
import logging
import os
import numpy as np
import pandas as pd
from typing import Optional
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from models.model_utils import (
    build_features,
    train_test_split_time,
    save_model,
    load_model,
    FEATURE_COLS,
)
from indicators.ta_indicators import add_ta_indicators
from config.config import CONFIG


def train_model(
    df: pd.DataFrame,
    save: bool = True,
    test_ratio: float = 0.2,
    symbol: Optional[str] = None,
) -> XGBClassifier:
    """
    Train an XGBoost classifier to predict whether the next bar will be up.

    Args:
        df:         Raw OHLCV DataFrame (will have TA indicators added).
        save:       Whether to persist the trained model to disk.
        test_ratio: Fraction of data reserved for evaluation.
        symbol:     Ticker for per-symbol model persistence.
    Returns:
        Trained XGBClassifier.
    """
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y, test_ratio=test_ratio)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logging.info(f"[PricePredictor] {symbol or 'default'} Test accuracy: {acc:.4f}")
    logging.info(f"[PricePredictor]\n{classification_report(y_test, preds, zero_division=0)}")

    if save:
        save_model(model, symbol=symbol)

    return model


def predict_price_movement(df: pd.DataFrame, symbol: Optional[str] = None) -> Optional[float]:
    """
    Predict the probability that the next bar moves UP.

    Loads the persisted XGBoost model (trains one on-the-fly if none exists)
    and returns a score in [0, 1].

    Args:
        df:     Raw OHLCV DataFrame (at least ~60 rows for indicators to warm up).
        symbol: Ticker — used to load the symbol-specific model.
    Returns:
        Float in [0, 1] — probability of an up-move, or None if prediction fails.
    """
    try:
        model = load_model(symbol=symbol)
        if model is None:
            logging.info(f"[PricePredictor] No saved model for {symbol or 'default'} — training from current data")
            if len(df) < 100:
                logging.warning(f"[PricePredictor] Insufficient data ({len(df)} rows) to train — skipping")
                return None
            model = train_model(df, save=True, symbol=symbol)

        # Prepare the latest row as input
        df_ind = add_ta_indicators(df)
        available = [c for c in FEATURE_COLS if c in df_ind.columns]
        if len(available) < len(FEATURE_COLS) // 2:
            logging.warning(f"[PricePredictor] Only {len(available)}/{len(FEATURE_COLS)} features available — skipping")
            return None
        latest = df_ind[available].iloc[[-1]].copy()

        # Fill missing values with column medians from the full dataset
        for col in available:
            if latest[col].isna().any():
                col_median = df_ind[col].median()
                latest[col] = latest[col].fillna(col_median)
                logging.debug(f"[PricePredictor] Filled NaN in '{col}' with median {col_median:.4f}")

        prob = model.predict_proba(latest)[0][1]  # probability of class 1 (up)
        logging.info(f"[PricePredictor] Up-probability: {prob:.4f}")
        return float(prob)

    except Exception as e:
        logging.error(f"Price prediction failed: {e}", exc_info=True)
        return None
