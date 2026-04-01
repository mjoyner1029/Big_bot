"""
Multi-asset trading bot — crypto & stocks.

Run modes:
  python main.py              → continuous trading loop (paper or live)
  python main.py --once       → single pass then exit
  python main.py --backtest   → backtest all symbols and print metrics
  python main.py --train      → train ML + RL models from latest data
  python main.py --pretrain   → pre-train on maximum historical data

Production loop:
  1. WebSocket price feed provides real-time prices (with yfinance fallback).
  2. LLM watcher runs in the background, scoring every symbol every 15 min.
  3. Each cycle: reconcile broker positions → check TP/SL → generate signals
     → execute trades → periodic rebalancing.
"""
import argparse
import logging
import os
import sys
import time
from typing import Dict, Optional

import pandas as pd

from config.config import CONFIG, get_all_symbols, is_crypto
from data.fetcher import fetch_latest_market_data, fetch_multiple_symbols
from strategies.strategy_engine import generate_trade_signal
from trading.trade_executor import execute_trade, check_open_positions, get_portfolio

# ── Logging setup ─────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CONFIG.get("bot_log_path", "logs/bot.log")),
        logging.StreamHandler(),
    ],
)


# ── Subsystem lazy-init helpers ───────────────────────────────────

_feed_manager = None
_llm_watcher = None
_cycle_count = 0


def _start_subsystems() -> None:
    """Start WebSocket feeds and LLM watcher (best-effort, errors logged)."""
    global _feed_manager, _llm_watcher

    # ── WebSocket real-time price feed ────────────────────────────
    if CONFIG.get("use_websocket", True):
        try:
            from data.websocket_feed import get_feed_manager
            _feed_manager = get_feed_manager()
            symbols = get_all_symbols()
            _feed_manager.start(symbols)
            logging.info("[Main] WebSocket price feed started")
        except Exception as e:
            logging.warning(f"[Main] WebSocket feed failed to start (will use yfinance): {e}")
            _feed_manager = None

    # ── LLM continuous watcher ────────────────────────────────────
    if CONFIG.get("use_llm") and CONFIG.get("anthropic_api_key"):
        try:
            from models.llm_watcher import get_watcher
            _llm_watcher = get_watcher()
            _llm_watcher.start()
            logging.info("[Main] LLM watcher started")
        except Exception as e:
            logging.warning(f"[Main] LLM watcher failed to start: {e}")
            _llm_watcher = None


def _stop_subsystems() -> None:
    """Gracefully stop background subsystems."""
    if _feed_manager:
        try:
            _feed_manager.stop()
        except Exception:
            pass
    if _llm_watcher:
        try:
            _llm_watcher.stop()
        except Exception:
            pass


def _get_current_prices(symbols) -> Dict[str, float]:
    """Get current prices — prefer WebSocket, fall back to yfinance OHLCV."""
    prices: Dict[str, float] = {}

    # Try WebSocket first
    if _feed_manager:
        ws_prices = _feed_manager.get_latest_prices()
        for sym in symbols:
            if sym in ws_prices:
                prices[sym] = ws_prices[sym]

    # Fall back to yfinance for any missing symbols
    missing = [s for s in symbols if s not in prices]
    if missing:
        for sym in missing:
            df = fetch_latest_market_data(ticker=sym)
            if df is not None and not df.empty:
                prices[sym] = float(df["Close"].iloc[-1])

    return prices


# ── Core loop iteration ──────────────────────────────────────────

def run_one_cycle() -> None:
    """
    Execute one full cycle:
      1. Reconcile positions with broker (live mode only)
      2. Fetch data for every symbol on the watchlist
      3. Check open positions against current prices (TP/SL/trailing)
      4. Generate signals for symbols we don't already hold
      5. Execute any resulting trades
      6. Periodic rebalancing based on LLM watcher opinions
    """
    global _cycle_count
    _cycle_count += 1

    symbols = get_all_symbols()
    logging.info(f"── Cycle {_cycle_count} start ── {len(symbols)} symbols ──")

    portfolio = get_portfolio()

    # 0. Position reconciliation (live mode only)
    if not CONFIG.get("use_paper_trading", True):
        try:
            from trading.reconciler import reconcile
            reconcile(portfolio)
        except Exception as e:
            logging.warning(f"[Main] Reconciliation error: {e}")

    # 1. Fetch market data (for TA indicators — always need full OHLCV)
    market_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_latest_market_data(ticker=sym)
        if df is not None and not df.empty:
            market_data[sym] = df
        else:
            logging.warning(f"Skipping {sym} — no data")

    if not market_data:
        logging.warning("No market data fetched for any symbol — skipping cycle")
        return

    # 2. Build a current-price map (prefer real-time WebSocket prices)
    current_prices = _get_current_prices(symbols)
    # Fill any still-missing from OHLCV
    for sym, df in market_data.items():
        if sym not in current_prices:
            current_prices[sym] = float(df["Close"].iloc[-1])

    # 3. Check open positions (TP/SL/trailing)
    check_open_positions(current_prices)

    # 4. Generate signals for symbols without open positions
    held_symbols = {p["symbol"] for p in portfolio.open_positions}

    for sym, df in market_data.items():
        if sym in held_symbols:
            logging.info(f"[Main] {sym} — already holding, skip signal generation")
            continue

        signal = generate_trade_signal(df, symbol=sym)

        # Incorporate LLM watcher opinion if available
        if signal and _llm_watcher:
            try:
                opinion = _llm_watcher.get_opinion(sym)
                if opinion:
                    llm_bias = opinion.get("bias", "neutral")
                    llm_score = opinion.get("score", 0)
                    # Reject signals that strongly contradict the LLM watcher
                    if signal["side"] == "buy" and llm_bias == "bearish" and llm_score < -0.5:
                        logging.info(f"[Main] LLM watcher bearish on {sym} (score={llm_score:.2f}) — suppressing buy")
                        continue
                    elif signal["side"] == "sell" and llm_bias == "bullish" and llm_score > 0.5:
                        logging.info(f"[Main] LLM watcher bullish on {sym} (score={llm_score:.2f}) — suppressing sell")
                        continue
            except Exception as e:
                logging.debug(f"[Main] LLM watcher opinion error for {sym}: {e}")

        if signal:
            execute_trade(signal)
        else:
            logging.info(f"[Main] {sym} — no signal")

    # 5. Periodic rebalancing (every N cycles)
    rebalance_interval = CONFIG.get("rebalance_interval_cycles", 12)  # default every 12 cycles (~1hr at 5min)
    if rebalance_interval > 0 and _cycle_count % rebalance_interval == 0:
        _run_rebalance(portfolio, current_prices)

    # 6. Log portfolio summary
    summary = portfolio.summary(current_prices)
    logging.info(
        f"── Cycle {_cycle_count} end ── equity=${summary['equity']:.2f}  "
        f"cash=${summary['cash']:.2f}  "
        f"open={summary['open_positions']}  "
        f"PnL=${summary['total_realised_pnl']:.2f}  "
        f"win_rate={summary['win_rate']*100:.1f}%"
    )


def _run_rebalance(portfolio, current_prices: Dict[str, float]) -> None:
    """Collect LLM watcher opinions and rebalance portfolio accordingly."""
    if not _llm_watcher:
        return

    try:
        # Build target allocations from LLM opinions
        target_allocs: Dict[str, float] = {}
        symbols = get_all_symbols()
        bullish_syms = []

        for sym in symbols:
            opinion = _llm_watcher.get_opinion(sym)
            if opinion and opinion.get("bias") == "bullish" and opinion.get("score", 0) > 0.3:
                bullish_syms.append((sym, opinion["score"]))

        if not bullish_syms:
            logging.info("[Main] No bullish symbols from LLM watcher — skipping rebalance")
            return

        # Equal-weight allocation among bullish symbols (capped at max_position_pct)
        max_pct = CONFIG.get("max_position_pct", 0.25)
        n = len(bullish_syms)
        per_sym = min(1.0 / n, max_pct)
        for sym, score in bullish_syms:
            target_allocs[sym] = per_sym

        orders = portfolio.rebalance(target_allocs, current_prices)
        for order in orders:
            signal = {
                "symbol": order["symbol"],
                "side": order["side"],
                "entry_price": order["current_price"],
                "asset_type": "crypto" if is_crypto(order["symbol"]) else "stock",
                "confidence": 0.6,
                "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            execute_trade(signal)

        if orders:
            logging.info(f"[Main] Rebalanced: executed {len(orders)} orders")

    except Exception as e:
        logging.warning(f"[Main] Rebalance error: {e}")


# ── Backtest mode ─────────────────────────────────────────────────

def run_backtest_mode(use_llm: bool = False) -> None:
    """Backtest the strategy on every symbol in the watchlist."""
    from backtest.backtester import run_backtest

    if not use_llm:
        original_llm = CONFIG.get("use_llm", False)
        CONFIG["use_llm"] = False

    symbols = get_all_symbols()

    for sym in symbols:
        logging.info(f"[Backtest] Fetching data for {sym} …")
        df = fetch_latest_market_data(ticker=sym, period="1y", interval="1h")
        if df is None or len(df) < 100:
            logging.warning(f"[Backtest] Not enough data for {sym}, skipping")
            continue

        def strategy_fn(df_slice, symbol=sym):
            return generate_trade_signal(df_slice, symbol=symbol)

        result = run_backtest(strategy_fn, df, symbol=sym, use_llm=use_llm)
        m = result["metrics"]
        logging.info(
            f"[Backtest] {sym} DONE: "
            f"{m['total_trades']} trades | "
            f"Return={m['total_return_pct']:.2f}% | "
            f"Sharpe={m['sharpe_ratio']:.2f} | "
            f"MaxDD={m['max_drawdown_pct']:.2f}% | "
            f"WinRate={m['win_rate']*100:.1f}% | "
            f"Final=${m['final_equity']:.2f}"
        )

    if not use_llm:
        CONFIG["use_llm"] = original_llm  # restore


# ── Train mode ────────────────────────────────────────────────────

def run_train_mode() -> None:
    """Train ML (XGBoost) and RL models from the latest data."""
    from models.price_predictor import train_model
    from models.reinforce_trainer import train_reinforcement_model

    symbols = get_all_symbols()
    for sym in symbols:
        logging.info(f"[Train] Fetching data for {sym} …")
        df = fetch_latest_market_data(ticker=sym, period="1y", interval="1h")
        if df is None or len(df) < 200:
            logging.warning(f"[Train] Not enough data for {sym}")
            continue

        logging.info(f"[Train] Training XGBoost for {sym} …")
        train_model(df, save=True, symbol=sym)

        logging.info(f"[Train] Training RL agent for {sym} …")
        train_reinforcement_model(df, episodes=50, save=True, symbol=sym)

    logging.info("[Train] All models trained")


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-asset trading bot")
    parser.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    parser.add_argument("--backtest", action="store_true", help="Run backtests on all symbols")
    parser.add_argument("--backtest-llm", action="store_true", help="Run backtests WITH LLM (cached)")
    parser.add_argument("--train", action="store_true", help="Train ML/RL models")
    parser.add_argument("--pretrain", action="store_true", help="Pre-train on max historical data")
    args = parser.parse_args()

    if args.backtest or args.backtest_llm:
        run_backtest_mode(use_llm=args.backtest_llm)
        return

    if args.train:
        run_train_mode()
        return

    if args.pretrain:
        # Delegate to the pretrain script
        from scripts.pretrain import main as pretrain_main
        pretrain_main()
        return

    # ── Start subsystems for live/paper loop ─────────────────────
    _start_subsystems()

    if args.once:
        run_one_cycle()
        _stop_subsystems()
        return

    # Continuous loop
    interval = CONFIG.get("loop_interval_seconds", 300)
    logging.info(f"Starting continuous trading loop (interval={interval}s)")
    logging.info(f"Asset class: {CONFIG['asset_class']}")
    logging.info(f"Paper trading: {CONFIG['use_paper_trading']}")
    logging.info(f"LLM enabled: {CONFIG.get('use_llm', False)}")
    logging.info(f"WebSocket feed: {'active' if _feed_manager else 'inactive (yfinance fallback)'}")
    logging.info(f"LLM watcher: {'active' if _llm_watcher else 'inactive'}")
    logging.info(f"Symbols: {get_all_symbols()}")

    while True:
        try:
            run_one_cycle()
        except KeyboardInterrupt:
            logging.info("Interrupted — saving state and shutting down")
            get_portfolio().save_state()
            _stop_subsystems()
            sys.exit(0)
        except Exception as e:
            logging.exception(f"Cycle error: {e}")

        logging.info(f"Sleeping {interval}s until next cycle …")
        time.sleep(interval)


if __name__ == "__main__":
    main()
