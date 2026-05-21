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
        # ── HIGH GROWTH / MOMENTUM (Primary Focus) ──────────────────
        "NVDA",  # NVIDIA (AI chips - massive growth)
        "TSLA",  # Tesla (EV, energy, AI)
        "META",  # Meta (AI, metaverse)
        "AAPL",  # Apple (ecosystem, services)
        "GOOGL", # Google (AI, cloud, search)
        "MSFT",  # Microsoft (cloud, AI, enterprise)
        "AMZN",  # Amazon (cloud, e-commerce)
        
        # ── AI & SEMICONDUCTORS (High Volatility = High Returns) ────
        "AMD",   # AMD (AI chips, data center)
        "AVGO",  # Broadcom (networking, AI)
        "QCOM",  # Qualcomm (mobile, automotive)
        "ANET",  # Arista Networks (data center)
        
        # ── CYBERSECURITY (Strong Growth Sector) ────────────────────
        "CRWD",  # CrowdStrike (endpoint security)
        "PANW",  # Palo Alto Networks (network security)
        "ZS",    # Zscaler (cloud security)
        "FTNT",  # Fortinet (enterprise security)
        
        # ── DEFENSE (Gov Contract Edge) ─────────────────────────────
        "LMT",   # Lockheed Martin (stable, dividends)
        "RTX",   # Raytheon (aerospace, defense)
        "KTOS",  # Kratos (small cap, high beta)
        "PLTR",  # Palantir (AI, data, gov contracts)
        
        # ── CLOUD / SAAS (High Margin Business) ─────────────────────
        "DDOG",  # Datadog (monitoring, observability)
        "NET",   # Cloudflare (edge computing, security)
        "SNOW",  # Snowflake (data cloud)
        
        # ── MARKET INDEXES (Hedging & Benchmark) ────────────────────
        "SPY",   # S&P 500
        "QQQ",   # Nasdaq 100
    ],
    "symbol": "ETH-USD",               # default single-symbol for quick runs
    "period": "3mo",
    "interval": "1h",

    # ── Capital & risk management ─────────────────────────────────
    "capital": float(os.getenv("TRADING_CAPITAL", "0")),  # set via env or leave 0 to pull from broker
    
    # Mode-specific risk parameters (auto-selected by trading_mode)
    # Format: {conservative, balanced, aggressive, claude_hf}
    "risk_per_trade_pct": 0.02,         # risk 2% of capital per trade (balanced default)
    "risk_per_trade_conservative": 0.01,   # 1% risk
    "risk_per_trade_balanced": 0.02,       # 2% risk
    "risk_per_trade_aggressive": 0.03,     # 3% risk
    "risk_per_trade_claude_hf": 0.005,     # 0.5% risk (high frequency = smaller positions)
    
    "max_open_positions": 5,
    "max_open_positions_conservative": 3,
    "max_open_positions_balanced": 5,
    "max_open_positions_aggressive": 8,
    "max_open_positions_claude_hf": 15,     # Claude agent: high frequency needs more slots
    
    "max_position_pct": 0.25,           # max 25% of capital in one position
    "max_position_pct_conservative": 0.15,
    "max_position_pct_balanced": 0.25,
    "max_position_pct_aggressive": 0.35,
    "max_position_pct_claude_hf": 0.10,     # smaller positions for HF

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
    "ta_weight": 0.30,                 # Technical analysis
    "ml_weight": 0.30,                 # Machine learning predictions
    "sentiment_weight": 0.20,          # News/social sentiment  
    "llm_weight": 0.20,                # Claude's analysis (when enabled)
    "gov_contract_weight": 1.0,        # Gov contracts (equal weight - just another edge)

    # ── Claude / Anthropic LLM ────────────────────────────────────
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    "anthropic_model": "claude-3-5-sonnet-20241022",
    "use_llm": False,                  # DISABLED - set True once API key is working
    
    # ── Autonomous Trading (Full Self-Direction) ──────────────────
    # Inspired by Claude's 48hr autonomous experiment: +1,322% with no human intervention
    "enable_autonomous_learning": True,  # Learn from every trade outcome
    "autonomous_reflection_interval": 12,  # Self-reflect every N cycles
    "enable_world_events_analysis": False,  # DISABLED - requires LLM
    "world_events_analysis_interval": 4,   # Refresh world events every N cycles

    # ── Sentiment data sources ────────────────────────────────────
    "news_api_key": os.getenv("NEWS_API_KEY", ""),
    "news_query_terms_crypto": ["crypto", "bitcoin", "ethereum", "solana"],
    "news_query_terms_stocks": ["stock market", "S&P 500", "NASDAQ", "earnings"],
    "sentiment_lookback_hours": 24,

    # ── Government Contract Monitoring (USAspending.gov) ───────────
    # One edge among many - contracts add conviction to existing signals
    "enable_gov_contracts": True,           # Enable gov contract monitoring
    "min_contract_amount": 50_000_000,      # $50M minimum (quality over quantity)
    "max_contract_amount": 500_000_000,     # $500M max
    "gov_contract_lookback_days": 7,        # Check last 7 days

    # ── ML model paths ────────────────────────────────────────────
    "model_dir": "models/saved",
    "price_model_path": "models/saved/xgb_price_model.pkl",
    "rl_model_path": "models/saved/rl_agent.pkl",

    # ── Trading Mode (inspired by Claude's 48hr autonomous experiment) ───
    # Modes: "conservative" | "balanced" | "aggressive" | "claude_hf"
    # claude_hf = Claude High-Frequency (~108 trades/hour capability)
    "trading_mode": "balanced",
    
    # ── Scheduler / loop ──────────────────────────────────────────
    "loop_interval_seconds": 300,       # 5 minutes between iterations (balanced mode)
    # Mode-specific intervals (override above if trading_mode set):
    "loop_interval_conservative": 600,  # 10 min
    "loop_interval_balanced": 300,      # 5 min
    "loop_interval_aggressive": 120,    # 2 min
    "loop_interval_claude_hf": 30,      # 30 sec → ~120 opportunities/hour

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

    # ── SMS Alerts (Twilio) ───────────────────────────────────────
    "twilio_account_sid": os.getenv("TWILIO_ACCOUNT_SID", ""),
    "twilio_auth_token": os.getenv("TWILIO_AUTH_TOKEN", ""),
    "twilio_from_number": os.getenv("TWILIO_FROM_NUMBER", ""),
    "alert_phone_number": os.getenv("ALERT_PHONE_NUMBER", ""),

    # ── Logging & notifications ───────────────────────────────────
    "trade_log_path": "logs/trade_log.csv",
    "bot_log_path": "logs/bot.log",
    "state_path": "logs/bot_state.json",
    "cost_log_path": "logs/cost_tracker.json",  # Track API & transaction costs
    "enable_notifications": False,
    "notification_webhook": os.getenv("DISCORD_WEBHOOK", ""),
    
    # Transaction cost tracking (Claude covered costs with +1,322% return)
    "track_transaction_costs": True,
    "track_api_costs": True,
    "api_cost_per_call": {
        "anthropic_input_1k": 0.003,    # Claude Sonnet pricing per 1K tokens
        "anthropic_output_1k": 0.015,
        "openai_gpt4_1k": 0.03,
    },
    "exchange_fee_pct": 0.001,  # 0.1% typical for most exchanges
    "slippage_estimate_pct": 0.0005,  # 0.05% estimated slippage

    # ── Backtesting ───────────────────────────────────────────────
    "backtest_start": "2025-01-01",
    "backtest_end": "2026-03-31",
    "backtest_fee_pct": 0.001,          # 0.1% per trade

    # ── Discipline layer (professional risk controls) ─────────────

    # ── PRODUCTION SAFETY CONTROLS (CRITICAL) ─────────────────────
    # Circuit breakers and emergency protection
    "max_daily_drawdown_pct": 0.03,     # 3% daily drawdown = halt trading for day
    "max_total_loss_pct": 0.20,         # 20% total loss = emergency halt
    "rapid_loss_threshold": 5,          # 5 consecutive losses triggers pause
    "rapid_loss_window_sec": 600,       # Within 10 minutes
    "rapid_loss_pause_sec": 3600,       # Pause for 1 hour after rapid losses
    
    # Order validation and position limits
    "max_single_trade_value": 5000,     # Max $5k per trade ($10k cap → $5k per trade)
    "min_trade_value": 10,              # Min $10 to avoid dust trades  
    "max_position_value": 10000,        # Max $10k in any single position
    "order_validation_enabled": True,   # Enable pre-execution validation
    
    # Rate limiting (prevent API bans)
    "rate_limiting_enabled": True,      # Enable rate limiting (CRITICAL)
    "retry_on_failure": True,           # Retry failed API calls with backoff
    "max_api_retries": 3,               # Maximum retry attempts
    
    # Emergency controls
    "emergency_stop_file": "EMERGENCY_STOP",  # Create this file to halt bot
    "pause_trading_file": "PAUSE_TRADING",    # Create this file to pause
    
    # Correlation / cluster exposure limits
    "max_cluster_exposure_pct": 0.40,   # max 40% of equity in one correlated group
    "max_asset_class_exposure_pct": 0.60,  # max 60% net in crypto or stocks
    "max_same_direction_per_cluster": 2,   # max 2 longs (or 2 shorts) in same cluster
    "correlation_warning_threshold": 0.75, # warn if correlation > 0.75

    # Loss-streak adaptive behavior
    "max_consecutive_losses_pause": 4,  # hard pause after 4 consecutive losses
    "loss_streak_reduce_at": 2,         # start reducing size after 2 consecutive losses

    # Overtrading protection (mode-aware)
    "max_trades_per_day": 20,           # hard cap on daily executions (balanced)
    "max_trades_per_day_conservative": 10,
    "max_trades_per_day_balanced": 20,
    "max_trades_per_day_aggressive": 50,
    "max_trades_per_day_claude_hf": 500,  # Claude did 5,200 in 48hrs = ~108/hr avg

    # Volume / liquidity
    "min_volume_ratio": 0.20,           # reject if volume < 20% of 20-bar median

    # Partial exit / scaling out
    "enable_scale_out": True,           # sell partial at TP1, trail the rest
    "scale_out_at_pct": 0.50,           # TP1 = 50% of the way to full TP
    "scale_out_fraction": 0.50,         # sell 50% of position at TP1

    # Multi-timeframe confirmation
    "use_multi_timeframe": True,        # use daily TF to confirm hourly signals
    "mtf_counter_trend_penalty": 0.08,  # confidence penalty for counter-trend signals
    "mtf_aligned_boost": 0.03,          # confidence boost for trend-aligned signals

    # Session / market hours
    "enforce_market_hours": True,       # block stock trades outside market hours
    "flatten_overnight": True,          # close stock positions before market close
    
    # ── INTEGRATED FEATURES FROM SOLANA BOT (Battle-Tested) ──────
    
    # Kill Switch (Market Crash Protection) - Asset-Specific
    "kill_switch_enabled": True,        # Enable automatic crash protection
    
    # Crypto thresholds (more volatile, higher thresholds)
    "kill_market_symbol_crypto": "BTC-USD",  # Primary crypto indicator
    "kill_4h_drop_pct_crypto": 6.0,     # Close all if -6% in 4 hours
    "kill_24h_drop_pct_crypto": 10.0,   # Close all if -10% in 24 hours
    
    # Stock thresholds (less volatile, tighter thresholds)
    "kill_market_symbol_stock": "SPY",  # Primary stock indicator (S&P 500)
    "kill_4h_drop_pct_stock": 4.0,      # Close all if -4% in 4 hours
    "kill_24h_drop_pct_stock": 7.0,     # Close all if -7% in 24 hours
    
    "kill_cooldown_hours": 24,          # Wait 24h before re-entering after kill
    "kill_portfolio_drawdown_pct": 15.0,  # Kill on 15% total portfolio drawdown
    
    # Position Health Monitoring
    "position_health_monitor_enabled": True,  # Enable automatic position auditing
    "loss_watchdog_threshold_pct": -2.0,      # Flag positions losing > 2%
    "loss_watchdog_strikes": 3,               # Close after 3 consecutive underwater checks
    "stale_position_strikes": 4,              # Close after 4 out-of-range checks
    "max_position_hold_hours": 48,            # Max hold time without profit
    "stale_position_pnl_threshold": 0.5,      # Min % profit to avoid stale flag
    "trailing_stop_enabled": True,            # Enable trailing stops
    "trailing_stop_pct": 50,                  # Give back 50% of best gain
    "auto_take_profit_enabled": False,        # Disable auto TP (use signal TP/SL)
    "take_profit_threshold_pct": 10.0,        # Auto TP at +10% (if enabled)
    
    # Profit Pile Accounting
    "profit_reinvest_pct": 60,                # 60% reinvest, 40% to pile
    "profit_withdrawal_threshold": 1000,      # Suggest withdrawal at $1000
    
    # State Management
    "state_dir": "state",                     # Directory for state files
    "state_backup_retention_hours": 72,       # Keep backups for 72 hours
    
    # Macro Snapshot & Regime Detection - Asset-Specific
    "macro_monitoring_enabled": True,         # Enable market regime detection
    "macro_refresh_interval_min": 15,         # Refresh macro every 15 min
    
    # Crypto regime thresholds (more volatile)
    "regime_crisis_threshold_crypto": 10,     # % drop to trigger crisis mode
    "regime_volatile_threshold_crypto": 5,    # % swing to trigger volatile mode
    "regime_bull_threshold_crypto": 2,        # % rise for bull signals
    "regime_bear_threshold_crypto": -2,       # % drop for bear signals
    
    # Stock regime thresholds (less volatile)
    "regime_crisis_threshold_stock": 7,       # % drop to trigger crisis mode
    "regime_volatile_threshold_stock": 3,     # % swing to trigger volatile mode
    "regime_bull_threshold_stock": 1.5,       # % rise for bull signals
    "regime_bear_threshold_stock": -1.5,      # % drop for bear signals
    
    "regime_rsi_floor": 40,                   # Suppress longs when RSI < 40 + downtrend
    "suppress_longs_in_bear": True,           # Suppress longs in strong bear market
    
    # Market Hours Management (Stock-Specific)
    "allow_extended_hours_trading": False,    # Pre-market and after-hours
    "flatten_before_close_minutes": 30,       # Close positions 30min before close
    "min_minutes_before_close": 60,           # Don't open new positions within 60min of close
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


def get_mode_config(param_name: str) -> any:
    """Get mode-specific configuration value based on current trading_mode.
    
    Example: get_mode_config('risk_per_trade_pct') returns the risk % for current mode.
    Falls back to base parameter if mode-specific version doesn't exist.
    """
    mode = CONFIG.get("trading_mode", "balanced")
    mode_key = f"{param_name}_{mode}"
    
    if mode_key in CONFIG:
        return CONFIG[mode_key]
    return CONFIG.get(param_name)


def get_loop_interval() -> int:
    """Get the appropriate loop interval based on trading mode."""
    mode = CONFIG.get("trading_mode", "balanced")
    return CONFIG.get(f"loop_interval_{mode}", CONFIG["loop_interval_seconds"])


def get_trading_mode_info() -> dict:
    """Get comprehensive info about current trading mode."""
    mode = CONFIG.get("trading_mode", "balanced")
    return {
        "mode": mode,
        "loop_interval": get_loop_interval(),
        "risk_per_trade": get_mode_config("risk_per_trade_pct"),
        "max_positions": get_mode_config("max_open_positions"),
        "max_position_size": get_mode_config("max_position_pct"),
        "max_trades_per_day": get_mode_config("max_trades_per_day"),
        "description": {
            "conservative": "Low frequency, 1% risk, max stability",
            "balanced": "Standard mode, 2% risk, proven parameters",
            "aggressive": "Higher frequency, 3% risk, active trading",
            "claude_hf": "High-frequency mode inspired by Claude's +1,322% autonomous run",
        }.get(mode, "Unknown mode"),
    }
