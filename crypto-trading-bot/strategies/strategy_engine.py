"""Strategy engine — evaluates TA, ML, sentiment, and (optionally) Claude
to produce buy / sell / hold signals for any asset (crypto or stock).
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import pandas as pd

from indicators.ta_indicators import add_ta_indicators, get_latest_indicator_snapshot
from models.price_predictor import predict_price_movement
from models.sentiment_model import analyze_sentiment
from strategies.thresholds import get_trade_thresholds
from config.config import CONFIG, is_crypto


# ── TA scoring ────────────────────────────────────────────────────

def _compute_ta_score(row: pd.Series) -> Optional[float]:
    """
    Multi-factor TA score in [0, 1].
    Aggregates RSI, MACD, Bollinger %B, ADX, and Stochastic signals.
    Only scores indicators that are actually present — returns None
    if fewer than 2 indicators are available.
    """
    points = 0.0
    total = 0.0

    # RSI
    rsi = row.get("rsi")
    if rsi is not None and not pd.isna(rsi):
        if rsi < 30:          points += 1.0    # oversold → bullish
        elif rsi < 45:        points += 0.6
        elif rsi > 70:        points += 0.0    # overbought → bearish
        elif rsi > 55:        points += 0.3
        else:                 points += 0.5
        total += 1.0

    # MACD histogram
    macd_hist = row.get("macd_hist")
    if macd_hist is not None and not pd.isna(macd_hist):
        if macd_hist > 0:     points += 1.0
        elif macd_hist > -0.5: points += 0.4
        total += 1.0

    # Bollinger %B (0 = lower band, 1 = upper band)
    bb = row.get("bb_pctb")
    if bb is not None and not pd.isna(bb):
        if bb < 0.2:          points += 0.9    # near lower band
        elif bb < 0.4:        points += 0.6
        elif bb > 0.8:        points += 0.1
        else:                 points += 0.5
        total += 1.0

    # ADX trend strength + direction
    adx = row.get("adx")
    adx_pos = row.get("adx_pos")
    adx_neg = row.get("adx_neg")
    if adx is not None and not pd.isna(adx):
        adx_pos = adx_pos if (adx_pos is not None and not pd.isna(adx_pos)) else 0
        adx_neg = adx_neg if (adx_neg is not None and not pd.isna(adx_neg)) else 0
        if adx > 25 and adx_pos > adx_neg:
            points += 1.0
        elif adx > 25 and adx_neg > adx_pos:
            points += 0.0
        else:
            points += 0.5
        total += 1.0

    # Stochastic %K vs %D crossover
    stoch_k = row.get("stoch_k")
    stoch_d = row.get("stoch_d")
    if stoch_k is not None and not pd.isna(stoch_k):
        stoch_d = stoch_d if (stoch_d is not None and not pd.isna(stoch_d)) else stoch_k
        if stoch_k < 20 and stoch_k > stoch_d:
            points += 1.0
        elif stoch_k > 80 and stoch_k < stoch_d:
            points += 0.0
        else:
            points += 0.5
        total += 1.0

    if total < 2:
        logging.warning("[Strategy] Fewer than 2 TA indicators available — cannot score")
        return None

    return points / total


def _compute_sell_ta_score(row: pd.Series) -> Optional[float]:
    """
    Sell-side TA score in [0, 1].  Higher = more reason to sell.
    Only scores indicators that are actually present — returns None
    if fewer than 2 indicators are available.
    """
    points = 0.0
    total = 0.0

    rsi = row.get("rsi")
    if rsi is not None and not pd.isna(rsi):
        if rsi > 70:          points += 1.0
        elif rsi > 60:        points += 0.6
        else:                 points += 0.2
        total += 1.0

    macd_hist = row.get("macd_hist")
    if macd_hist is not None and not pd.isna(macd_hist):
        if macd_hist < 0:     points += 1.0
        elif macd_hist < 0.5: points += 0.5
        total += 1.0

    bb = row.get("bb_pctb")
    if bb is not None and not pd.isna(bb):
        if bb > 0.9:          points += 1.0
        elif bb > 0.7:        points += 0.6
        else:                 points += 0.2
        total += 1.0

    stoch_k = row.get("stoch_k")
    stoch_d = row.get("stoch_d")
    if stoch_k is not None and not pd.isna(stoch_k):
        stoch_d = stoch_d if (stoch_d is not None and not pd.isna(stoch_d)) else stoch_k
        if stoch_k > 80 and stoch_k < stoch_d:
            points += 1.0
        else:
            points += 0.3
        total += 1.0

    if total < 2:
        logging.warning("[Strategy] Fewer than 2 sell-side TA indicators available — cannot score")
        return None

    return points / total


# ── Signal generation ─────────────────────────────────────────────

def generate_trade_signal(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate a trade signal (buy, sell, or None) for a single asset.

    Scoring pipeline:
      1. Technical analysis (multi-indicator)
      2. XGBoost up-probability
      3. Sentiment (VADER + optional Claude)
      4. Claude TA-interpretation (optional)
      5. Weighted blend → confidence
      6. Claude trade-validation gate (optional)

    Args:
        df:     OHLCV DataFrame (≥60 rows recommended).
        symbol: Ticker string — used for sentiment queries and
                asset-class–aware thresholds.
    Returns:
        Trade signal dict or None.
    """
    symbol = symbol or CONFIG.get("symbol", "ETH-USD")

    try:
        df = add_ta_indicators(df)
        recent = df.iloc[-1]

        # ── 1. TA scores ─────────────────────────────────────────
        buy_ta = _compute_ta_score(recent)
        sell_ta = _compute_sell_ta_score(recent)

        if buy_ta is None or sell_ta is None:
            logging.warning(f"[Strategy] {symbol} — insufficient TA data, skipping")
            return None

        # ── 2. ML prediction (probability of up-move) ───────────
        ml_score = predict_price_movement(df, symbol=symbol)

        # ── 3. Sentiment ─────────────────────────────────────────
        sentiment_raw = analyze_sentiment(symbol=symbol)
        # Rescale [-1, 1] → [0, 1] for blending (None if unavailable)
        sentiment_score = ((sentiment_raw + 1.0) / 2.0) if sentiment_raw is not None else None

        # ── 4. LLM TA interpretation (optional) ─────────────────
        llm_buy_boost = None
        use_llm = CONFIG.get("use_llm") and CONFIG.get("anthropic_api_key")
        indicator_snapshot = {}
        if use_llm:
            try:
                from models.llm_analyst import llm_interpret_indicators
                indicator_snapshot = get_latest_indicator_snapshot(df)
                interpretation = llm_interpret_indicators(indicator_snapshot, symbol=symbol)
                if interpretation:
                    bias = interpretation.get("bias", "neutral")
                    conf = float(interpretation.get("confidence", 0.5))
                    if bias == "bullish":
                        llm_buy_boost = 0.5 + conf * 0.5
                    elif bias == "bearish":
                        llm_buy_boost = 0.5 - conf * 0.5
                    else:
                        llm_buy_boost = 0.5
            except Exception as e:
                logging.warning(f"[Strategy] LLM interpretation skipped: {e}")

        # ── 5. Weighted blend (only include available components) ─
        components_buy = []
        components_sell = []
        weights = []

        # TA — always present at this point
        components_buy.append(buy_ta)
        components_sell.append(sell_ta)
        weights.append(CONFIG["ta_weight"])

        # ML — include only if prediction succeeded
        if ml_score is not None:
            components_buy.append(ml_score)
            components_sell.append(1 - ml_score)
            weights.append(CONFIG["ml_weight"])
        else:
            logging.info(f"[Strategy] {symbol} — ML prediction unavailable, excluded from blend")

        # Sentiment — include only if data was available
        if sentiment_score is not None:
            components_buy.append(sentiment_score)
            components_sell.append(1 - sentiment_score)
            weights.append(CONFIG["sentiment_weight"])
        else:
            logging.info(f"[Strategy] {symbol} — Sentiment unavailable, excluded from blend")

        # LLM — include only if call succeeded
        if llm_buy_boost is not None:
            components_buy.append(llm_buy_boost)
            components_sell.append(1 - llm_buy_boost)
            weights.append(CONFIG["llm_weight"])

        total_w = sum(weights)
        if total_w == 0:
            logging.warning(f"[Strategy] {symbol} — all components unavailable")
            return None

        buy_confidence = sum(c * w for c, w in zip(components_buy, weights)) / total_w
        sell_confidence = sum(c * w for c, w in zip(components_sell, weights)) / total_w

        logging.info(
            f"[Strategy] {symbol}  buy_conf={buy_confidence:.3f}  "
            f"sell_conf={sell_confidence:.3f}  "
            f"(TA_buy={buy_ta:.2f} TA_sell={sell_ta:.2f} "
            f"ML={ml_score if ml_score is not None else 'N/A'} "
            f"sent={sentiment_score if sentiment_score is not None else 'N/A'} "
            f"llm={llm_buy_boost if llm_buy_boost is not None else 'N/A'})"
        )

        threshold = CONFIG["confidence_threshold"]
        entry_price = float(recent["Close"])

        # ── Decide: buy, sell, or hold ───────────────────────────
        side = None
        confidence = 0.0
        if buy_confidence >= threshold and buy_confidence > sell_confidence:
            side = "buy"
            confidence = buy_confidence
        elif sell_confidence >= threshold and sell_confidence > buy_confidence:
            side = "sell"
            confidence = sell_confidence

        if side is None:
            logging.info(f"[Strategy] {symbol} — no signal (hold)")
            return None

        asset_type = "crypto" if is_crypto(symbol) else "stock"

        # Pass real ATR for volatility-adjusted TP/SL when available
        atr_val = recent.get("atr")
        atr_arg = float(atr_val) if (atr_val is not None and not pd.isna(atr_val)) else None
        thresholds = get_trade_thresholds(entry_price, confidence,
                                          side=side, asset_type=asset_type,
                                          atr=atr_arg)

        signal: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "asset_type": asset_type,
            "entry_price": entry_price,
            "confidence": round(confidence, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **thresholds,
        }

        # ── 6. LLM validation gate (optional) ───────────────────
        if use_llm:
            try:
                from models.llm_analyst import llm_validate_trade
                if not indicator_snapshot:
                    indicator_snapshot = get_latest_indicator_snapshot(df)
                validation = llm_validate_trade(signal, indicator_snapshot, symbol=symbol)
                if not validation.get("approved", True):
                    logging.info(
                        f"[Strategy] Claude REJECTED trade: "
                        f"{validation.get('reasoning', 'no reason')}"
                    )
                    return None
                adj_conf = validation.get("adjusted_confidence")
                if adj_conf is not None:
                    signal["confidence"] = round(float(adj_conf), 4)
                    logging.info(f"[Strategy] Claude adjusted confidence → {signal['confidence']}")
            except Exception as e:
                logging.warning(f"[Strategy] LLM validation skipped: {e}")

        logging.info(f"[Strategy] Signal → {signal['side'].upper()} {symbol} @ {entry_price:.2f}  conf={signal['confidence']}")
        return signal

    except Exception as e:
        logging.error(f"[Strategy] Error generating trade signal for {symbol}: {e}", exc_info=True)
        return None
