"""Backtesting engine — simulates the full trading pipeline on historical data.

Supports:
  • Multi-asset backtests
  • Configurable commission / slippage
  • TP / SL exit simulation at each bar
  • Equity curve generation for metrics module
  • LLM-in-the-loop mode (with disk cache to avoid redundant API calls)
"""
import hashlib
import json
import logging
import os
from typing import Callable, Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from config.config import CONFIG
from indicators.ta_indicators import add_ta_indicators
from strategies.thresholds import get_trade_thresholds


# ── LLM response cache for backtesting ───────────────────────────

_LLM_CACHE_DIR = os.path.join(CONFIG.get("model_dir", "models/saved"), "llm_bt_cache")


def _cache_key(symbol: str, bar_date: str, indicator_snapshot: dict) -> str:
    """Deterministic hash of the inputs so repeated backtests reuse LLM calls."""
    raw = json.dumps({"s": symbol, "d": bar_date, "ind": indicator_snapshot}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _llm_cache_get(key: str) -> Optional[dict]:
    path = os.path.join(_LLM_CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _llm_cache_set(key: str, value: dict) -> None:
    os.makedirs(_LLM_CACHE_DIR, exist_ok=True)
    path = os.path.join(_LLM_CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w") as f:
            json.dump(value, f)
    except Exception:
        pass


def run_backtest(
    strategy_fn: Callable[[pd.DataFrame, str], Optional[Dict[str, Any]]],
    historical_df: pd.DataFrame,
    symbol: str = "BTC-USD",
    initial_capital: Optional[float] = None,
    fee_pct: Optional[float] = None,
    warmup: int = 60,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """
    Run a backtest over *historical_df* using *strategy_fn* to generate signals.

    Args:
        strategy_fn:     (df_slice, symbol) -> signal dict or None
        historical_df:   Full OHLCV DataFrame.
        symbol:          Ticker being tested.
        initial_capital: Starting cash (default from config).
        fee_pct:         Commission per trade as a fraction (default from config).
        warmup:          Minimum number of bars before the strategy is invoked.
        use_llm:         When True, LLM calls are included (with caching).
                         When False (default), LLM is suppressed for speed.
    Returns:
        Dict containing trades list, equity_curve Series, and summary stats.
    """
    capital = initial_capital or CONFIG["capital"]
    fee = fee_pct if fee_pct is not None else CONFIG.get("backtest_fee_pct", 0.001)

    # Temporarily control LLM usage
    original_llm = CONFIG.get("use_llm", False)
    if not use_llm:
        CONFIG["use_llm"] = False

    cash = capital
    position: Optional[Dict[str, Any]] = None
    trades: List[Dict[str, Any]] = []
    equity_curve: List[float] = []

    df = add_ta_indicators(historical_df).dropna().reset_index(drop=True)
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    for i in range(warmup, len(df)):
        current_close = float(closes[i])
        current_high = float(highs[i])
        current_low = float(lows[i])

        # ── Check open position for TP / SL hits ─────────────────
        if position is not None:
            side = position["side"]
            tp = position["take_profit_price"]
            sl = position["stop_loss_price"]
            qty = position["qty"]
            entry = position["entry_price"]

            # Trailing stop update
            trail_pct = position.get("trailing_stop_pct", 0)
            if trail_pct > 0:
                if side == "buy":
                    new_sl = current_high * (1 - trail_pct)
                    if new_sl > sl:
                        position["stop_loss_price"] = new_sl
                        sl = new_sl
                else:
                    new_sl = current_low * (1 + trail_pct)
                    if new_sl < sl:
                        position["stop_loss_price"] = new_sl
                        sl = new_sl

            hit = None
            exit_price = current_close

            if side == "buy":
                if current_high >= tp:
                    hit, exit_price = "tp_hit", tp
                elif current_low <= sl:
                    hit, exit_price = "sl_hit", sl
            else:
                if current_low <= tp:
                    hit, exit_price = "tp_hit", tp
                elif current_high >= sl:
                    hit, exit_price = "sl_hit", sl

            if hit:
                proceeds = qty * exit_price * (1 - fee)
                if side == "buy":
                    pnl = proceeds - position["cost"]
                else:
                    pnl = position["cost"] - proceeds
                cash += proceeds
                trades.append({
                    **position,
                    "exit_price": round(exit_price, 4),
                    "pnl": round(pnl, 2),
                    "result": hit,
                    "exit_bar": i,
                })
                position = None

        # ── Generate signal (only if flat) ───────────────────────
        if position is None:
            df_slice = df.iloc[:i + 1]
            try:
                signal = strategy_fn(df_slice, symbol)
            except Exception as e:
                logging.debug(f"[Backtest] Strategy error at bar {i}: {e}")
                signal = None

            # ── Optional LLM validation gate (cached) ────────────
            if signal and use_llm and CONFIG.get("anthropic_api_key"):
                try:
                    from indicators.ta_indicators import get_latest_indicator_snapshot
                    from models.llm_analyst import llm_validate_trade
                    ind_snap = get_latest_indicator_snapshot(df_slice)
                    bar_date = str(df_slice.index[-1]) if hasattr(df_slice.index[-1], 'isoformat') else str(i)
                    ck = _cache_key(symbol, bar_date, ind_snap)
                    cached = _llm_cache_get(ck)
                    if cached is not None:
                        validation = cached
                    else:
                        validation = llm_validate_trade(signal, ind_snap, symbol=symbol) or {}
                        _llm_cache_set(ck, validation)

                    if not validation.get("approved", True):
                        logging.debug(f"[Backtest-LLM] Claude rejected signal at bar {i}")
                        signal = None
                    elif validation.get("adjusted_confidence") is not None:
                        signal["confidence"] = float(validation["adjusted_confidence"])
                except Exception as e:
                    logging.debug(f"[Backtest-LLM] LLM validation error at bar {i}: {e}")

            if signal and cash > 0:
                entry_price = current_close
                side = signal.get("side", "buy")
                confidence = signal.get("confidence", 0.5)
                asset_type = signal.get("asset_type", "crypto")

                # Compute TP/SL from thresholds module if not in signal
                if "stop_loss_price" not in signal or "take_profit_price" not in signal:
                    thresh = get_trade_thresholds(
                        entry_price, confidence, side=side, asset_type=asset_type,
                    )
                    if "stop_loss_price" not in signal:
                        signal["stop_loss_price"] = thresh["stop_loss_price"]
                    if "take_profit_price" not in signal:
                        signal["take_profit_price"] = thresh["take_profit_price"]
                    if "trailing_stop_pct" not in signal:
                        signal["trailing_stop_pct"] = thresh["trailing_stop_pct"]

                sl_price = signal["stop_loss_price"]
                distance = abs(entry_price - sl_price)
                if distance < entry_price * 0.001:
                    # SL too close — use threshold-derived distance
                    thresh = get_trade_thresholds(
                        entry_price, confidence, side=side, asset_type=asset_type,
                    )
                    distance = abs(entry_price - thresh["stop_loss_price"])

                equity = cash  # no open position
                risk_amt = equity * CONFIG["risk_per_trade_pct"]
                qty = risk_amt / distance
                max_qty = (equity * CONFIG["max_position_pct"]) / entry_price
                qty = min(qty, max_qty)
                cost = qty * entry_price * (1 + fee)

                if cost <= cash:
                    cash -= cost
                    position = {
                        "symbol": symbol,
                        "side": side,
                        "entry_price": round(entry_price, 4),
                        "qty": round(qty, 6),
                        "cost": round(cost, 2),
                        "take_profit_price": signal["take_profit_price"],
                        "stop_loss_price": signal["stop_loss_price"],
                        "trailing_stop_pct": signal.get("trailing_stop_pct", 0),
                        "confidence": confidence,
                        "entry_bar": i,
                    }

        # Record equity
        pos_value = position["qty"] * current_close if position else 0
        equity_curve.append(cash + pos_value)

    # Close any remaining position at the last close
    if position is not None:
        final_close = float(closes[-1])
        proceeds = position["qty"] * final_close * (1 - fee)
        pnl = proceeds - position["cost"] if position["side"] == "buy" else position["cost"] - proceeds
        cash += proceeds
        trades.append({
            **position,
            "exit_price": round(final_close, 4),
            "pnl": round(pnl, 2),
            "result": "end_of_data",
            "exit_bar": len(df) - 1,
        })

    equity_series = pd.Series(equity_curve, name="equity")

    # Restore original LLM setting
    CONFIG["use_llm"] = original_llm

    from backtest.metrics import compute_backtest_metrics
    metrics = compute_backtest_metrics(trades, equity_series, initial_capital=capital)

    logging.info(
        f"[Backtest] {symbol}: {len(trades)} trades  "
        f"Return={metrics['total_return_pct']:.2f}%  "
        f"Sharpe={metrics['sharpe_ratio']:.2f}  "
        f"MaxDD={metrics['max_drawdown_pct']:.2f}%  "
        f"WinRate={metrics['win_rate']*100:.1f}%"
    )

    return {
        "symbol": symbol,
        "trades": trades,
        "equity_curve": equity_series,
        "metrics": metrics,
    }
