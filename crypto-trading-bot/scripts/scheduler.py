"""Unified Scheduler — Crypto 24/7 + Stock Market-Hours.

Two independent trading loops share this single entry point:

  CRYPTO LOOP
    • Runs continuously, 24 h / 7 d (markets never close for crypto)
    • Refreshes the watchlist every CRYPTO_WATCHLIST_REFRESH_CYCLES cycles
      using the live CoinGecko discovery in data/trending_crypto.py
    • Default cycle interval: same as bot's configured loop interval

  STOCK LOOP
    • Only executes cycles when NYSE/NASDAQ is open (9:30 AM–4:00 PM ET,
      Mon–Fri, excluding holidays handled by pytz + our holiday list)
    • Logs a wait message outside market hours instead of busy-sleeping
    • Runs market-open check every STOCK_OPEN_POLL_SECONDS seconds when closed
    • Default cycle interval: same as bot's configured loop interval

Both loops call ``run_one_cycle_for()`` which temporarily patches
``CONFIG["asset_class"]`` and ``CONFIG["crypto_watchlist"]`` /
``CONFIG["stock_watchlist"]`` so the existing main.py machinery executes
without modification.

Usage::

    # Run both loops (default)
    python scripts/scheduler.py

    # Run crypto loop only
    python scripts/scheduler.py --mode crypto

    # Run stock loop only
    python scripts/scheduler.py --mode stocks

    # Dry-run: print status then exit
    python scripts/scheduler.py --status
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import List

import pytz

# ── Bootstrap path so we can import from project root ────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.config import CONFIG, get_loop_interval
from strategies.market_hours import is_market_open, get_market_hours_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scheduler")

# ── Tunable constants ─────────────────────────────────────────────────────────

# How often to re-fetch the live crypto watchlist (every N crypto cycles)
CRYPTO_WATCHLIST_REFRESH_CYCLES: int = int(os.getenv("CRYPTO_WATCHLIST_REFRESH", "4"))

# Max symbols in dynamic crypto watchlist
CRYPTO_MAX_SYMBOLS: int = int(os.getenv("CRYPTO_MAX_SYMBOLS", "30"))

# How often to poll market-open status when stocks are closed (seconds)
STOCK_OPEN_POLL_SECONDS: int = int(os.getenv("STOCK_OPEN_POLL", "60"))

# Time zone for market hours display
ET = pytz.timezone("America/New_York")


# ── Cycle runner ──────────────────────────────────────────────────────────────

def run_one_cycle_for(asset_class: str, watchlist: List[str]) -> None:
    """Execute one full trading cycle for the given asset class.

    Temporarily patches CONFIG so main.run_one_cycle() operates on the
    correct assets, then restores CONFIG.
    """
    original_asset_class = CONFIG.get("asset_class")
    original_crypto = list(CONFIG.get("crypto_watchlist", []))
    original_stock = list(CONFIG.get("stock_watchlist", []))

    try:
        CONFIG["asset_class"] = asset_class
        if asset_class == "crypto":
            CONFIG["crypto_watchlist"] = watchlist
        else:
            CONFIG["stock_watchlist"] = watchlist

        # Import here to ensure CONFIG patch is visible to main
        from main import run_one_cycle
        run_one_cycle()

    except Exception as exc:
        logger.exception("[Scheduler] Cycle error for %s: %s", asset_class, exc)
    finally:
        CONFIG["asset_class"] = original_asset_class
        CONFIG["crypto_watchlist"] = original_crypto
        CONFIG["stock_watchlist"] = original_stock


# ── Crypto 24/7 loop ──────────────────────────────────────────────────────────

def crypto_loop(interval: int) -> None:
    """Run the crypto trading loop indefinitely, refreshing the watchlist
    every *CRYPTO_WATCHLIST_REFRESH_CYCLES* cycles."""
    from data.trending_crypto import get_live_crypto_watchlist, invalidate_cache

    cycle = 0
    watchlist: List[str] = []

    logger.info("[CryptoLoop] Starting 24/7 crypto loop (interval=%ds, max_symbols=%d)", interval, CRYPTO_MAX_SYMBOLS)

    while True:
        # Refresh watchlist periodically
        if cycle % CRYPTO_WATCHLIST_REFRESH_CYCLES == 0:
            logger.info("[CryptoLoop] Refreshing live crypto watchlist …")
            invalidate_cache()
            try:
                watchlist = get_live_crypto_watchlist(max_symbols=CRYPTO_MAX_SYMBOLS)
                logger.info("[CryptoLoop] Watchlist (%d symbols): %s", len(watchlist), watchlist[:10])
            except Exception as exc:
                logger.warning("[CryptoLoop] Watchlist refresh failed: %s — using previous list", exc)
                if not watchlist:
                    watchlist = list(CONFIG.get("crypto_watchlist", []))

        cycle += 1
        logger.info("[CryptoLoop] ── Cycle %d ── %d symbols ──", cycle, len(watchlist))

        if watchlist:
            run_one_cycle_for("crypto", watchlist)
        else:
            logger.warning("[CryptoLoop] Empty watchlist — skipping cycle")

        logger.info("[CryptoLoop] Sleeping %ds …", interval)
        time.sleep(interval)


# ── Stock market-hours loop ───────────────────────────────────────────────────

def stock_loop(interval: int) -> None:
    """Run the stock trading loop only when NYSE/NASDAQ is open."""
    # Use a representative stock ticker for market-hours checks
    _SENTINEL_SYMBOL = "AAPL"

    stock_watchlist = list(CONFIG.get("stock_watchlist", []))
    cycle = 0

    logger.info("[StockLoop] Starting stock loop (interval=%ds, symbols=%d)", interval, len(stock_watchlist))

    while True:
        datetime.now(ET)

        if is_market_open(_SENTINEL_SYMBOL):
            cycle += 1
            logger.info("[StockLoop] ── Cycle %d ── NYSE open ── %d symbols ──", cycle, len(stock_watchlist))
            run_one_cycle_for("stocks", stock_watchlist)
            logger.info("[StockLoop] Sleeping %ds …", interval)
            time.sleep(interval)
        else:
            # Market closed — log status and wait before re-polling
            summary = get_market_hours_summary([_SENTINEL_SYMBOL])
            logger.info("[StockLoop] Market closed. %s  (polling every %ds)", summary, STOCK_OPEN_POLL_SECONDS)
            time.sleep(STOCK_OPEN_POLL_SECONDS)


# ── Status helper ─────────────────────────────────────────────────────────────

def print_status() -> None:
    """Print current market status and watchlists, then exit."""
    from data.trending_crypto import get_live_crypto_watchlist

    print("\n" + "=" * 70)
    print("SCHEDULER STATUS")
    print("=" * 70)

    now_et = datetime.now(ET)
    print(f"Current time (ET): {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    market_open = is_market_open("AAPL")
    print(f"Stock market open: {'YES' if market_open else 'NO'}")
    print(get_market_hours_summary(["AAPL"]))

    print("\nFetching live crypto watchlist …")
    try:
        wl = get_live_crypto_watchlist(max_symbols=CRYPTO_MAX_SYMBOLS)
        print(f"Crypto watchlist ({len(wl)} symbols): {wl}")
    except Exception as exc:
        print(f"[WARN] Could not fetch live watchlist: {exc}")

    static_stocks = CONFIG.get("stock_watchlist", [])
    print(f"\nStatic stock watchlist ({len(static_stocks)} symbols): {static_stocks[:15]}")

    print("=" * 70)


# ── Training data stats helper ────────────────────────────────────────────────

def print_training_stats() -> None:
    """Print statistics about collected LLM training data."""
    try:
        from models.training_data_collector import get_stats
        stats = get_stats()
        print("\n── Training Data ─────────────────────────────")
        print(f"  Total examples : {stats['total']}")
        print(f"  Wins / Losses  : {stats['wins']} / {stats['losses']}")
        print(f"  Win rate       : {stats.get('win_rate', 0):.1%}")
        print(f"  Symbols seen   : {stats.get('symbols', 0)}")
        print(f"  Pending open   : {stats.get('pending_open', 0)}")
        print(f"  Storage file   : {stats.get('file', 'N/A')}")
        print("─" * 46)
    except Exception as exc:
        print(f"[Training data] N/A ({exc})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified scheduler: crypto 24/7 + stock market-hours",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["both", "crypto", "stocks"],
        default="both",
        help="Which loop(s) to run (default: both)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print market status and watchlists then exit",
    )
    parser.add_argument(
        "--export-training",
        action="store_true",
        help="Export collected training data as fine-tuning JSONL then exit",
    )
    parser.add_argument(
        "--import-memory",
        action="store_true",
        help="Import existing memory store into training data then exit",
    )
    args = parser.parse_args()

    # ── One-shot commands ─────────────────────────────────────────
    if args.status:
        print_status()
        print_training_stats()
        return

    if args.export_training:
        from models.training_data_collector import export_finetuning_dataset
        out = export_finetuning_dataset()
        print(f"Fine-tuning dataset written → {out}")
        return

    if args.import_memory:
        from models.training_data_collector import import_from_memory_store
        n = import_from_memory_store()
        print(f"Imported {n} records from memory store")
        return

    # ── Start subsystems (WebSocket, LLM watcher, etc.) ──────────
    try:
        from main import _start_subsystems
        _start_subsystems()
        logger.info("[Scheduler] Subsystems started")
    except Exception as exc:
        logger.warning("[Scheduler] Subsystem startup error (non-fatal): %s", exc)

    interval = get_loop_interval()
    logger.info("[Scheduler] Loop interval = %ds", interval)

    threads: List[threading.Thread] = []

    if args.mode in ("both", "crypto"):
        t_crypto = threading.Thread(
            target=crypto_loop,
            args=(interval,),
            name="CryptoLoop",
            daemon=True,
        )
        threads.append(t_crypto)
        t_crypto.start()
        logger.info("[Scheduler] Crypto 24/7 loop started")

    if args.mode in ("both", "stocks"):
        t_stock = threading.Thread(
            target=stock_loop,
            args=(interval,),
            name="StockLoop",
            daemon=True,
        )
        threads.append(t_stock)
        t_stock.start()
        logger.info("[Scheduler] Stock market-hours loop started")

    if not threads:
        logger.error("[Scheduler] No loops started — check --mode argument")
        sys.exit(1)

    logger.info("[Scheduler] Running.  Press Ctrl+C to stop.")
    try:
        while True:
            # Keep main thread alive; daemon threads die when main exits
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("[Scheduler] Interrupted — shutting down")
        try:
            from main import _stop_subsystems
            _stop_subsystems()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()
