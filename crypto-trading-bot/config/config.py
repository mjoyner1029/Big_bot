import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    # ── Asset classes & symbols ───────────────────────────────────
    # See config/symbols.py for the FULL universe (hundreds of tickers).
    # These watchlists are the *active* subset that get auto-refreshed.
    "asset_class": "both",              # "crypto", "stocks", or "both"
    "crypto_watchlist": [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    ],
    "stock_watchlist": [
        "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA",
        "SPY", "QQQ", "JPM",
    ],
    "symbol": "ETH-USD",               # default single-symbol for quick runs
    "period": "3mo",
    "interval": "1h",

    # ── Capital & risk management ─────────────────────────────────
    "capital": float(os.getenv("TRADING_CAPITAL", "0")),  # set via env or leave 0 to pull from broker
    "risk_per_trade_pct": 0.02,         # risk 2% of capital per trade
    "max_open_positions": 5,
    "max_position_pct": 0.25,           # max 25% of capital in one position

    # ── Exchange / broker settings ────────────────────────────────
    "exchange": "coinbase",
    "use_paper_trading": True,

    # Crypto exchange (ccxt-compatible)
    "coinbase_api_key": os.getenv("COINBASE_API_KEY", ""),
    "coinbase_api_secret": os.getenv("COINBASE_API_SECRET", ""),
    "coinbase_passphrase": os.getenv("COINBASE_PASSPHRASE", ""),

    # Stock broker (Alpaca — paper & live)
    "alpaca_api_key": os.getenv("ALPACA_API_KEY", ""),
    "alpaca_api_secret": os.getenv("ALPACA_API_SECRET", ""),
    "alpaca_base_url": os.getenv(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    ),

    # ── Strategy thresholds ───────────────────────────────────────
    "confidence_threshold": 0.55,
    "ta_weight": 0.30,
    "ml_weight": 0.30,
    "sentiment_weight": 0.15,
    "llm_weight": 0.25,                # Claude's opinion weight

    # ── Claude / Anthropic LLM ────────────────────────────────────
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    "anthropic_model": "claude-sonnet-4-20250514",
    "use_llm": True,                    # set False to skip all Claude calls

    # ── Sentiment data sources ────────────────────────────────────
    "news_api_key": os.getenv("NEWS_API_KEY", ""),
    "news_query_terms_crypto": ["crypto", "bitcoin", "ethereum", "solana"],
    "news_query_terms_stocks": ["stock market", "S&P 500", "NASDAQ", "earnings"],
    "sentiment_lookback_hours": 24,

    # ── ML model paths ────────────────────────────────────────────
    "model_dir": "models/saved",
    "price_model_path": "models/saved/xgb_price_model.pkl",
    "rl_model_path": "models/saved/rl_agent.pkl",

    # ── Scheduler / loop ──────────────────────────────────────────
    "loop_interval_seconds": 300,       # 5 minutes between iterations

    # ── WebSocket & real-time feeds ───────────────────────────────
    "use_websocket": True,              # use WebSocket for real-time prices
    "websocket_reconnect_max": 5,       # max reconnect attempts

    # ── LLM watcher (background market monitoring) ────────────────
    "llm_watcher_interval": 900,        # 15 minutes between LLM scans
    "llm_watcher_max_symbols": 30,      # max symbols per LLM scan batch

    # ── Position reconciliation & rebalancing ─────────────────────
    "reconciliation_enabled": True,     # sync with broker each cycle
    "rebalance_enabled": True,          # periodic portfolio rebalancing
    "rebalance_interval_cycles": 12,    # rebalance every 12 cycles (~1hr)

    # ── Logging & notifications ───────────────────────────────────
    "trade_log_path": "logs/trade_log.csv",
    "bot_log_path": "logs/bot.log",
    "state_path": "logs/bot_state.json",
    "enable_notifications": False,
    "notification_webhook": os.getenv("DISCORD_WEBHOOK", ""),

    # ── Backtesting ───────────────────────────────────────────────
    "backtest_start": "2025-01-01",
    "backtest_end": "2026-03-31",
    "backtest_fee_pct": 0.001,          # 0.1% per trade
}


def get_all_symbols() -> list:
    """Return the active watchlist based on the configured asset class.

    This is the *small* set auto-refreshed by the main loop and dashboard.
    For the full universe of selectable symbols, use config.symbols.
    """
    ac = CONFIG["asset_class"]
    if ac == "crypto":
        return CONFIG["crypto_watchlist"]
    elif ac == "stocks":
        return CONFIG["stock_watchlist"]
    return CONFIG["crypto_watchlist"] + CONFIG["stock_watchlist"]


def is_crypto(symbol: str) -> bool:
    """Heuristic: crypto tickers contain '-' (BTC-USD) or end with common bases."""
    return "-" in symbol or symbol.endswith(("USDT", "BUSD", "USD"))


def get_news_terms_for(symbol: str) -> list:
    """Return relevant search terms for a symbol."""
    if is_crypto(symbol):
        return CONFIG["news_query_terms_crypto"]
    return CONFIG["news_query_terms_stocks"] + [symbol]
