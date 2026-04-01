#!/usr/bin/env python3
"""Pre-train ML models on maximum available historical data.

Usage:
    python scripts/pretrain.py                  # all symbols (watchlist)
    python scripts/pretrain.py --all            # entire universe (580+ symbols)
    python scripts/pretrain.py --symbols BTC-USD ETH-USD AAPL
    python scripts/pretrain.py --crypto-only
    python scripts/pretrain.py --stocks-only

What this does:
  1. For each symbol, fetches the MAXIMUM available history from yfinance.
     • Stocks: as far back as yfinance/Yahoo has data (S&P 500 back to ~1927,
       many large-caps back to the 1970s–1990s).
     • Crypto: since the exchange listing (BTC since ~2014 on Yahoo).
  2. Adds technical indicators.
  3. Trains an XGBoost price-direction classifier per symbol.
  4. Trains a Q-learning RL agent per symbol.
  5. Saves all models to models/saved/.

Historical data note:
  The user requested "400+ years of stock data".  Modern electronic market data
  only goes back ~100 years (NYSE 1927-ish via Yahoo).  This script fetches the
  absolute maximum available ("max" period) which gives the longest history
  yfinance can deliver for each ticker.  For crypto, data starts around 2014.
"""
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import pandas as pd

# ── Project imports ───────────────────────────────────────────────
# Ensure project root is importable
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config.config import CONFIG, get_all_symbols, is_crypto
from data.fetcher import fetch_latest_market_data
from models.price_predictor import train_model as train_xgb
from models.reinforce_trainer import train_reinforcement_model as train_rl
from indicators.ta_indicators import add_ta_indicators

# ── Logging ───────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/pretrain.log"),
        logging.StreamHandler(),
    ],
)


# ── Data fetching with maximum history ────────────────────────────

def fetch_max_history(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch the longest available OHLCV history for a symbol.

    Uses daily bars for training (hourly is capped at ~2 years on yfinance).
    Falls back to progressively shorter periods if 'max' fails.
    """
    import yfinance as yf

    # Daily bars with maximum look-back
    for period in ["max", "10y", "5y", "2y", "1y"]:
        try:
            df = yf.download(symbol, period=period, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df is not None and len(df) >= 200:
                logging.info(f"[Pretrain] {symbol}: fetched {len(df)} daily bars (period={period})")
                return df
        except Exception as e:
            logging.debug(f"[Pretrain] {symbol} period={period} failed: {e}")
            continue

    logging.warning(f"[Pretrain] {symbol}: could not fetch sufficient daily data")
    return None


# ── Per-symbol training pipeline ──────────────────────────────────

def pretrain_symbol(symbol: str, rl_episodes: int = 100) -> dict:
    """Fetch max history and train both XGBoost + RL for one symbol.

    Returns a summary dict with rows, accuracy, etc.
    """
    result = {"symbol": symbol, "rows": 0, "xgb_ok": False, "rl_ok": False, "error": None}

    try:
        df = fetch_max_history(symbol)
        if df is None or len(df) < 200:
            result["error"] = f"Insufficient data ({len(df) if df is not None else 0} rows)"
            return result

        result["rows"] = len(df)

        # ── XGBoost ──────────────────────────────────────────────
        try:
            train_xgb(df, save=True, symbol=symbol)
            result["xgb_ok"] = True
        except Exception as e:
            logging.error(f"[Pretrain] {symbol} XGBoost failed: {e}")
            result["error"] = f"XGBoost: {e}"

        # ── RL agent ─────────────────────────────────────────────
        try:
            train_rl(df, episodes=rl_episodes, save=True, symbol=symbol)
            result["rl_ok"] = True
        except Exception as e:
            logging.error(f"[Pretrain] {symbol} RL failed: {e}")
            result["error"] = f"RL: {e}"

    except Exception as e:
        result["error"] = str(e)
        logging.error(f"[Pretrain] {symbol} failed: {e}", exc_info=True)

    return result


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train ML models on max historical data")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to train")
    parser.add_argument("--all", action="store_true", help="Use the full 580+ symbol universe")
    parser.add_argument("--crypto-only", action="store_true", help="Only crypto symbols")
    parser.add_argument("--stocks-only", action="store_true", help="Only stock symbols")
    parser.add_argument("--rl-episodes", type=int, default=100, help="RL training episodes per symbol")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers (1=sequential)")
    args = parser.parse_args()

    # ── Determine symbol list ────────────────────────────────────
    if args.symbols:
        symbols = args.symbols
    elif args.all:
        from config.symbols import get_all_available_symbols
        symbols = get_all_available_symbols()
    elif args.crypto_only:
        if args.all:
            from config.symbols import CRYPTO_SYMBOLS
            symbols = CRYPTO_SYMBOLS
        else:
            symbols = CONFIG["crypto_watchlist"]
    elif args.stocks_only:
        if args.all:
            from config.symbols import STOCK_SYMBOLS
            symbols = STOCK_SYMBOLS
        else:
            symbols = CONFIG["stock_watchlist"]
    else:
        symbols = get_all_symbols()

    logging.info(f"[Pretrain] Starting pre-training for {len(symbols)} symbols")
    logging.info(f"[Pretrain] RL episodes per symbol: {args.rl_episodes}")
    logging.info(f"[Pretrain] Models will be saved to: {CONFIG.get('model_dir', 'models/saved')}/")

    os.makedirs(CONFIG.get("model_dir", "models/saved"), exist_ok=True)
    start_time = time.time()

    results = []

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(pretrain_symbol, sym, args.rl_episodes): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    r = future.result()
                    results.append(r)
                    status = "OK" if r["xgb_ok"] and r["rl_ok"] else f"PARTIAL ({r['error']})"
                    logging.info(f"[Pretrain] {sym}: {r['rows']} rows — {status}")
                except Exception as e:
                    logging.error(f"[Pretrain] {sym} raised: {e}")
                    results.append({"symbol": sym, "rows": 0, "xgb_ok": False, "rl_ok": False, "error": str(e)})
    else:
        for i, sym in enumerate(symbols, 1):
            logging.info(f"[Pretrain] ({i}/{len(symbols)}) {sym} …")
            r = pretrain_symbol(sym, args.rl_episodes)
            results.append(r)
            status = "OK" if r["xgb_ok"] and r["rl_ok"] else f"PARTIAL ({r['error']})"
            logging.info(f"[Pretrain] {sym}: {r['rows']} rows — {status}")

    elapsed = time.time() - start_time

    # ── Summary ──────────────────────────────────────────────────
    total = len(results)
    xgb_ok = sum(1 for r in results if r["xgb_ok"])
    rl_ok = sum(1 for r in results if r["rl_ok"])
    total_rows = sum(r["rows"] for r in results)
    failed = [r for r in results if not r["xgb_ok"] or not r["rl_ok"]]

    logging.info("=" * 60)
    logging.info(f"[Pretrain] COMPLETE in {elapsed:.1f}s")
    logging.info(f"[Pretrain] Total symbols: {total}")
    logging.info(f"[Pretrain] Total data rows: {total_rows:,}")
    logging.info(f"[Pretrain] XGBoost models trained: {xgb_ok}/{total}")
    logging.info(f"[Pretrain] RL agents trained: {rl_ok}/{total}")

    if failed:
        logging.warning(f"[Pretrain] {len(failed)} symbols had issues:")
        for r in failed[:20]:
            logging.warning(f"  {r['symbol']}: {r['error']}")
        if len(failed) > 20:
            logging.warning(f"  … and {len(failed) - 20} more")

    logging.info("=" * 60)


if __name__ == "__main__":
    main()
