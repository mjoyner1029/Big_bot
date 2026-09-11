import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Development policy (not a trading control) ─────────────────────────────
# The architecture is FROZEN. Future feature work must begin from a MEASURED
# failure (paper calibration error, execution error, risk gap, data coverage
# gap) — never from "this feature sounds useful". See paper_evidence reports.
ARCHITECTURE_PHASE = "STABLE"

# Sustained-PAPER measurement mode: prediction capture, outcome resolution,
# decay measurement, calibration + health reports. Never promotes to live.
PAPER_EVIDENCE_MODE = os.getenv("PAPER_EVIDENCE_MODE", "true").lower() == "true"


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() == "true"


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_list(key: str, default: list[str]) -> list[str]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return values or default

CONFIG = {
    # ── Asset classes & symbols ───────────────────────────────────
    # See config/symbols.py for the FULL universe (hundreds of tickers).
    # These watchlists are the *active* subset that get auto-refreshed.
    "asset_class": os.getenv("ASSET_CLASS", "both"),              # "crypto", "stocks", or "both"
    
    # Dead symbols (delisted, malformed data, persistent fetch failures)
    "quarantined_symbols": _env_list("QUARANTINED_SYMBOLS", [
        "BTW-USD", "S-USD", "SIREN-USD", "CARDS-USD", "LAB-USD"
    ]),
    "crypto_watchlist": _env_list("CRYPTO_WATCHLIST", [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    ]),
    "stock_watchlist": _env_list("STOCK_WATCHLIST", [
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

        # ── STRATEGY-SPECIFIC WATCHLISTS ────────────────────────────
        "OKLO",  # Energy + AI infrastructure
        "SMR",   # Nuclear energy
        "TE",    # Energy/industrial infrastructure proxy
        "EOSE",  # Grid storage
        "IONQ",  # Quantum computing
        "RGTI",  # Quantum computing
        "QBTS",  # Quantum computing
        "TTWO",  # GTA VI event catalyst
        "UPRO",  # Leveraged recovery
        "TQQQ",  # Leveraged recovery
        "SPXL",  # Leveraged recovery
        "MSTR",  # Dividend/treasury and BTC treasury factor
        "MARA",  # Dividend/treasury and BTC treasury factor
        "RIOT",  # Dividend/treasury and BTC treasury factor
        "STRV",  # Dividend treasury watchlist
        "STRF",  # Dividend treasury watchlist
        
        # ── MARKET INDEXES (Hedging & Benchmark) ────────────────────
        "SPY",   # S&P 500
        "QQQ",   # Nasdaq 100
    ]),
    "symbol": "ETH-USD",               # default single-symbol for quick runs
    "period": "1mo",   # 30-day window — stays within Alpaca IEX's ~40-day history
    "interval": "1h",

    # ── Capital & risk management ─────────────────────────────────
    "capital": float(os.getenv("TRADING_CAPITAL", "2000")),  # 14-day test capital
    
    # Mode-specific risk parameters (auto-selected by trading_mode)
    # Format: {conservative, balanced, aggressive, claude_hf}
    "risk_per_trade_pct": _env_float("RISK_PER_TRADE_PCT", 0.02),         # risk 2% of capital per trade (balanced default)
    "risk_per_trade_conservative": 0.01,   # 1% risk
    "risk_per_trade_balanced": 0.02,       # 2% risk
    "risk_per_trade_aggressive": 0.03,     # 3% risk
    "risk_per_trade_claude_hf": 0.005,     # 0.5% risk (high frequency = smaller positions)
    "high_conviction_min_confidence": _env_float("HIGH_CONVICTION_MIN_CONFIDENCE", 0.80),
    "high_conviction_risk_per_trade_pct": _env_float("HIGH_CONVICTION_RISK_PER_TRADE_PCT", 0.015),
    "very_high_conviction_min_confidence": _env_float("VERY_HIGH_CONVICTION_MIN_CONFIDENCE", 0.90),
    "very_high_conviction_risk_per_trade_pct": _env_float("VERY_HIGH_CONVICTION_RISK_PER_TRADE_PCT", 0.025),
    
    "max_open_positions": _env_int("MAX_OPEN_POSITIONS", 3),
    "max_open_positions_conservative": 3,
    "max_open_positions_balanced": 5,
    "max_open_positions_aggressive": 8,
    "max_open_positions_claude_hf": 3,     # Conservative for paper trading validation
    
    "max_position_pct": _env_float("MAX_POSITION_PCT", 0.12),           # max 12% of capital in one position
    "max_position_pct_conservative": 0.10,
    "max_position_pct_balanced": 0.12,
    "max_position_pct_aggressive": 0.12,
    "max_position_pct_claude_hf": 0.10,     # smaller positions for HF
    # Hard absolute dollar cap per trade — overrides % sizing to protect against
    # micro-cap/low-liquidity tokens where % sizing can still produce huge notional.
    # Set MAX_TRADE_VALUE_USD env var to override (default $500 per trade).
    "max_trade_value_usd": _env_float("MAX_TRADE_VALUE_USD", 500.0),

    # ── Exchange / broker settings ────────────────────────────────
    "exchange": "coinbase",
    "use_paper_trading": True,
    # Explicit live-trading gates (must be set in environment before any live run)
    "ENABLE_LIVE_TRADING": os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true",
    "ALLOW_LIVE_TRADING": os.getenv("ALLOW_LIVE_TRADING", "false").lower() == "true",
    "LIVE_TRADING_ENABLED": os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true",

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

    # ── Dynamic watchlist discovery ───────────────────────────────
    # Crypto: merge CoinGecko trending + meme-momentum coins every cycle (cached 15 min)
    "enable_dynamic_crypto": _env_bool("ENABLE_DYNAMIC_CRYPTO", True),
    "max_dynamic_crypto": _env_int("MAX_DYNAMIC_CRYPTO", 50),   # cap on live crypto symbols
    # Stocks: scan Yahoo Finance screeners for penny/meme/momentum plays (cached 15 min)
    "enable_penny_stocks": _env_bool("ENABLE_PENNY_STOCKS", True),
    "max_penny_stocks": _env_int("MAX_PENNY_STOCKS", 30),       # cap on scanned stock symbols
    "penny_stock_max_price": _env_float("PENNY_STOCK_MAX_PRICE", 20.0),  # max price filter

    # ── Catalyst swing holds ──────────────────────────────────────
    # Identifies stocks with known upcoming catalysts (product launches,
    # gov contracts, FDA approvals, etc.) and holds them until the thesis plays out.
    "enable_catalyst_swing": _env_bool("ENABLE_CATALYST_SWING", True),
    "swing_min_magnitude": _env_float("SWING_MIN_MAGNITUDE", 55.0),   # 0-100 threshold
    "swing_trailing_stop_pct": _env_float("SWING_TRAILING_STOP_PCT", 0.07),  # 7% trailing
    "max_swing_positions": _env_int("MAX_SWING_POSITIONS", 3),  # max concurrent swing holds

    # ── Options trading ───────────────────────────────────────────
    # Buys calls (bullish) or puts (bearish) when a stock signal is high-conviction.
    # Max risk per options trade = options_risk_pct of equity (default 2%).
    "enable_options_trading": _env_bool("ENABLE_OPTIONS_TRADING", True),
    "options_min_confidence": _env_float("OPTIONS_MIN_CONFIDENCE", 72.0),
    "options_risk_pct": _env_float("OPTIONS_RISK_PCT", 0.02),          # 2% of equity max loss
    "options_expiry_days": _env_int("OPTIONS_EXPIRY_DAYS", 21),        # ~3 weeks out
    "options_profit_target_mult": _env_float("OPTIONS_PROFIT_TARGET_MULT", 2.5),  # 150% gain

    # ── GEX (Gamma Exposure) Analysis ─────────────────────────────
    # Regime detection via options market maker positioning (Deribit for crypto).
    # Boosts/reduces strategies based on volatility regime.
    "enable_gex_analysis": _env_bool("ENABLE_GEX_ANALYSIS", True),

    # ── Short Selling ─────────────────────────────────────────────
    # Allows strategies to generate SELL signals for bearish setups.
    # Increases win rate by capturing downside moves in bear/sideways markets.
    "enable_short_selling": _env_bool("ENABLE_SHORT_SELLING", True),
    "short_min_confidence": _env_float("SHORT_MIN_CONFIDENCE", 0.62),      # Higher than longs (0.55)
    "short_max_positions": _env_int("SHORT_MAX_POSITIONS", 3),             # Max concurrent shorts
    "short_max_hold_days": _env_int("SHORT_MAX_HOLD_DAYS", 7),             # Close shorts within 1 week
    "short_stop_loss_pct": _env_float("SHORT_STOP_LOSS_PCT", 0.03),        # Tighter SL (3% vs 2% for longs)
    "short_blacklist": _env_list("SHORT_BLACKLIST", ["GME", "AMC", "BBBY"]),  # Avoid meme stocks (squeeze risk)
    "short_max_interest_pct": _env_float("SHORT_MAX_INTEREST_PCT", 20.0),  # Skip heavily shorted stocks
    "short_max_borrow_fee_pct": _env_float("SHORT_MAX_BORROW_FEE_PCT", 5.0),  # Skip expensive borrows

    # ── Pairs Trading & Statistical Arbitrage ─────────────────────
    # Market-neutral long/short strategies that profit from mean reversion.
    "enable_pairs_trading": _env_bool("ENABLE_PAIRS_TRADING", True),
    "pairs_lookback_days": _env_int("PAIRS_LOOKBACK_DAYS", 60),
    "pairs_entry_z_score": _env_float("PAIRS_ENTRY_Z_SCORE", 2.0),        # 2 std devs
    "pairs_exit_z_score": _env_float("PAIRS_EXIT_Z_SCORE", 0.5),
    "pairs_min_correlation": _env_float("PAIRS_MIN_CORRELATION", 0.7),
    "enable_stat_arb": _env_bool("ENABLE_STAT_ARB", True),
    "stat_arb_min_r_squared": _env_float("STAT_ARB_MIN_R_SQUARED", 0.5),  # Regression fit quality

    # ── Strategy thresholds ───────────────────────────────────────
    "confidence_threshold": _env_float("CONFIDENCE_THRESHOLD", 0.62),
    # Blacklist underperforming strategies (disable for current run)
    # Updated 2026-06-24: Removed blacklist to test all strategies in 14-day test
    "strategy_blacklist": _env_list("STRATEGY_BLACKLIST", []),
    # Paper-training performance targets used by readiness checks.
    # TARGET_AVERAGE_GAIN_PCT is interpreted as net total return over the
    # evaluated trade window (e.g. 0.30 = +30%).
    # Win rate is secondary to return; 45% reflects a high-R:R strategy.
    "target_win_rate": _env_float("TARGET_WIN_RATE", 0.65),
    "target_average_gain_pct": _env_float("TARGET_AVERAGE_GAIN_PCT", 0.50),
    # Confidence-to-size mapping for position sizing.
    # Multiplier is 1.0 at confidence_threshold, scales down toward min below
    # threshold, and scales up toward max above threshold.
    "confidence_position_sizing_enabled": _env_bool("CONFIDENCE_POSITION_SIZING_ENABLED", True),
    "confidence_size_min_mult": _env_float("CONFIDENCE_SIZE_MIN_MULT", 0.60),
    "confidence_size_max_mult": _env_float("CONFIDENCE_SIZE_MAX_MULT", 1.40),
    "confidence_size_min_mult_conservative": 0.75,
    "confidence_size_max_mult_conservative": 1.20,
    "confidence_size_min_mult_balanced": 0.60,
    "confidence_size_max_mult_balanced": 1.40,
    "confidence_size_min_mult_aggressive": 0.50,
    "confidence_size_max_mult_aggressive": 1.60,
    "confidence_size_min_mult_claude_hf": 0.85,
    "confidence_size_max_mult_claude_hf": 1.15,
    # approval_threshold: composite ScoreBundle score (0-100) required to
    # approve a DecisionEngine signal.  With neutral fundamentals and a 65%
    # technical signal the score is ~56.5.  Default 55.0 for paper trading;
    # raise to 70+ for live to require genuinely elevated signals.
    "approval_threshold": _env_float("APPROVAL_THRESHOLD", 55.0),
    "approval_threshold_live": _env_float("APPROVAL_THRESHOLD_LIVE", 70.0),  # Higher bar for live trading
    "min_technical_score": _env_float("MIN_TECHNICAL_SCORE", 20.0),  # Require minimum technical conviction
    # min_risk_reward: minimum reward-to-risk ratio for a signal to be approved.
    # Crypto signals at box boundaries often have R:R 1.0-1.5; 2.0 is too strict
    # for paper training. Raise to 1.5+ for live trading.
    "min_risk_reward": _env_float("MIN_RISK_REWARD", 1.8),
    "ta_weight": 0.30,                 # Technical analysis
    "ml_weight": 0.30,                 # Machine learning predictions
    "sentiment_weight": 0.20,          # News/social sentiment  
    "llm_weight": 0.20,                # Claude's analysis (when enabled)
    "gov_contract_weight": 1.0,        # Gov contracts (equal weight - just another edge)
    "alt_data_weight": 0.18,           # Unified alt-data edge (Capitol/OpenInsider/FOMO/USAspending)

    # ── Claude / Anthropic LLM ────────────────────────────────────
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    "anthropic_model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    "use_llm": _env_bool("USE_LLM", True),
    
    # ── Autonomous Trading (Full Self-Direction) ──────────────────
    # Inspired by Claude's 48hr autonomous experiment: +1,322% with no human intervention
    "enable_autonomous_learning": True,  # Learn from every trade outcome
    "autonomous_reflection_interval": 12,  # Self-reflect every N cycles
    "enable_world_events_analysis": _env_bool("ENABLE_WORLD_EVENTS_ANALYSIS", True),
    "world_events_analysis_interval": 30,   # Refresh world events every N cycles (reduced from 4 to save API calls)
    "autonomous_pause_enabled": _env_bool("AUTONOMOUS_PAUSE_ENABLED", False),  # Allow autonomous pause (disable for paper trading)

    # ── Sentiment data sources ────────────────────────────────────
    "news_api_key": os.getenv("NEWS_API_KEY", ""),
    "news_query_terms_crypto": ["crypto", "bitcoin", "ethereum", "solana"],
    "news_query_terms_stocks": ["stock market", "S&P 500", "NASDAQ", "earnings"],
    "sentiment_lookback_hours": 24,

    # ── Government Contract Monitoring (USAspending.gov) ───────────
    # One edge among many - contracts add conviction to existing signals
    "enable_gov_contracts": _env_bool("ENABLE_GOV_CONTRACTS", False),  # Enable gov contract monitoring
    "min_contract_amount": 50_000_000,      # $50M minimum (quality over quantity)
    "max_contract_amount": 500_000_000,     # $500M max
    "gov_contract_lookback_days": 7,        # Check last 7 days

    # ── Unified Alternative-Data Intelligence ──────────────────────
    # Uses public disclosure/social sources with latency-aware weighting.
    "enable_alt_data_intel": _env_bool("ENABLE_ALT_DATA_INTEL", True),

    # Source switches
    "enable_capitol_trades_signals": True,
    "enable_openinsider_signals": True,
    "enable_fomo_signals": _env_bool("ENABLE_FOMO_SIGNALS", True),
    "fomo_crypto_only": _env_bool("FOMO_CRYPTO_ONLY", True),
    "enable_fomo_manual_snapshot": _env_bool("ENABLE_FOMO_MANUAL_SNAPSHOT", True),
    "fomo_manual_snapshot_path": os.getenv("FOMO_MANUAL_SNAPSHOT_PATH", ""),

    # Source endpoints
    "capitol_trades_url": "https://www.capitoltrades.com/trades",
    "openinsider_url": "http://openinsider.com/latest-insider-trading",
    "fomo_signal_endpoint": os.getenv("FOMO_SIGNAL_ENDPOINT", ""),
    "fomo_family_url": os.getenv("FOMO_FAMILY_URL", "https://fomo.family"),

    # Source-level blend inside alt-data edge
    "alt_weight_usaspending": 0.25,
    "alt_weight_capitoltrades": 0.25,
    "alt_weight_openinsider": 0.40,
    "alt_weight_fomo": 0.10,

    # Parser/cache tuning
    "capitol_trades_cache_sec": 1800,
    "openinsider_cache_sec": 900,
    "fomo_cache_sec": 120,
    "fomo_manual_cache_sec": _env_int("FOMO_MANUAL_CACHE_SEC", 60),
    "fomo_manual_max_age_sec": _env_int("FOMO_MANUAL_MAX_AGE_SEC", 900),
    "openinsider_notional_scale": 5_000_000.0,

    # ── ML model paths ────────────────────────────────────────────
    "model_dir": "models/saved",
    "price_model_path": "models/saved/xgb_price_model.pkl",
    "rl_model_path": "models/saved/rl_agent.pkl",

    # ── Trading Mode (inspired by Claude's 48hr autonomous experiment) ───
    # Modes: "conservative" | "balanced" | "aggressive" | "claude_hf"
    # claude_hf = Claude High-Frequency (~108 trades/hour capability)
    "trading_mode": os.getenv("TRADING_MODE", "balanced"),
    
    # ── Scheduler / loop ──────────────────────────────────────────
    "loop_interval_seconds": _env_int("LOOP_INTERVAL_SECONDS", 300),       # 5 minutes between iterations (balanced mode)
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
    "trade_log_path": os.getenv("TRADE_LOG_PATH", "logs/trade_log.csv"),
    "bot_log_path": os.getenv("BOT_LOG_PATH", "logs/bot.log"),
    "state_path": os.getenv("STATE_PATH", "state/14day_test_state.json"),
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
    # Paper-trading friction model (basis points = 1/100 of a percent)
    "paper_slippage_bps": 5,    # 5 bps adverse fill on entry (0.05 %)
    "paper_commission_bps": 10, # 10 bps round-trip commission (0.10 %)

    # ── Backtesting ───────────────────────────────────────────────
    "backtest_start": "2025-01-01",
    "backtest_end": "2026-03-31",
    "backtest_fee_pct": 0.001,          # 0.1% per trade

    # ── Overnight Edge Scanner (research-only, paper-trading default) ─────
    "overnight_edge": {
        "enabled": _env_bool("OVERNIGHT_EDGE_ENABLED", True),
        "paper_only": _env_bool("OVERNIGHT_EDGE_PAPER_ONLY", True),
        "allow_live": _env_bool("OVERNIGHT_EDGE_ALLOW_LIVE", False),
        "min_history_days": _env_int("OVERNIGHT_EDGE_MIN_HISTORY_DAYS", 252),
        "min_price": _env_float("OVERNIGHT_EDGE_MIN_PRICE", 5.0),
        "min_avg_daily_dollar_volume": _env_float("OVERNIGHT_EDGE_MIN_DOLLAR_VOLUME", 20_000_000.0),
        "min_observations": _env_int("OVERNIGHT_EDGE_MIN_OBSERVATIONS", 126),
        "min_net_overnight_cagr": _env_float("OVERNIGHT_EDGE_MIN_NET_CAGR", 0.0),
        "min_sharpe": _env_float("OVERNIGHT_EDGE_MIN_SHARPE", 1.0),
        "min_profit_factor": _env_float("OVERNIGHT_EDGE_MIN_PROFIT_FACTOR", 1.15),
        "max_drawdown": _env_float("OVERNIGHT_EDGE_MAX_DRAWDOWN", 0.50),
        "min_out_of_sample_sharpe": _env_float("OVERNIGHT_EDGE_MIN_OOS_SHARPE", 0.0),
        "max_top_5_profit_concentration": _env_float("OVERNIGHT_EDGE_MAX_TOP5_CONC", 0.40),
        "max_top_10_profit_concentration": _env_float("OVERNIGHT_EDGE_MAX_TOP10_CONC", 0.60),
        "max_portfolio_overnight_exposure_pct": _env_float("OVERNIGHT_EDGE_MAX_PORTFOLIO_EXPOSURE", 0.20),
        "max_single_overnight_position_pct": _env_float("OVERNIGHT_EDGE_MAX_SINGLE_POSITION", 0.10),
        "lookbacks": [20, 60, 126, 252, 504],
        "transaction_cost_bps": _env_float("OVERNIGHT_EDGE_TX_COST_BPS", 10.0),
        "entry_slippage_bps": _env_float("OVERNIGHT_EDGE_ENTRY_SLIPPAGE_BPS", 15.0),
        "exit_slippage_bps": _env_float("OVERNIGHT_EDGE_EXIT_SLIPPAGE_BPS", 25.0),
        "close_entry_window_minutes": _env_int("OVERNIGHT_EDGE_ENTRY_WINDOW_MIN", 15),
        "open_exit_window_minutes": _env_int("OVERNIGHT_EDGE_EXIT_WINDOW_MIN", 15),
        "live_trading_disabled_reason": "Overnight edge strategy remains paper-only until validation and risk checks are complete.",
    },

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

    # ── Adaptive Portfolio Intelligence Layer ─────────────────────
    # New 9-regime classifier + dynamic strategy weighting + voting
    "adaptive_intelligence_enabled": _env_bool("ADAPTIVE_INTELLIGENCE_ENABLED", True),

    # VotingEngine: minimum weighted signal score (0-100) to approve a trade
    "min_weighted_signal_score": _env_float("MIN_WEIGHTED_SIGNAL_SCORE", 75.0),

    # Performance targets (used by PerformanceReporter and WalkForwardValidator)
    "target_cagr": 0.30,           # 30 % annual return
    "target_max_drawdown": 0.15,   # ≤ 15 % max drawdown
    "target_profit_factor": 2.0,   # ≥ 2.0 profit factor
    "target_sharpe": 2.0,          # ≥ 2.0 Sharpe ratio
    "target_win_rate_ail": 0.55,   # ≥ 55 % win rate

    # TradeMemory database path
    "trade_memory_db": os.getenv("TRADE_MEMORY_DB", "data/trade_memory.sqlite"),

    # ConsequenceLearner: minimum closed trades before adjusting reliability scores
    "consequence_min_sample_size": _env_int("CONSEQUENCE_MIN_SAMPLE_SIZE", 20),

    # StrategyPerformanceTracker: min trades to include a strategy in stats
    "strategy_stats_min_trades": _env_int("STRATEGY_STATS_MIN_TRADES", 5),

    # CorrelationManager: max positions from any one cluster
    "max_per_cluster": _env_int("MAX_PER_CLUSTER", 2),

    # Adaptive position sizing risk ladder (as fraction of portfolio)
    "adaptive_risk_score_70_80": 0.0025,   # 0.25 % risk for score 70-80
    "adaptive_risk_score_80_90": 0.0050,   # 0.50 % risk for score 80-90
    "adaptive_risk_score_90_plus": 0.0100, # 1.00 % risk for score 90+

    # Walk-forward validation windows
    "wf_train_days": _env_int("WF_TRAIN_DAYS", 60),
    "wf_validate_days": _env_int("WF_VALIDATE_DAYS", 30),
    "wf_n_windows": _env_int("WF_N_WINDOWS", 6),
}


def get_all_symbols() -> list:
    """Return the active watchlist based on the configured asset class.

    When dynamic discovery is enabled (default), the crypto side is built live
    from CoinGecko (trending + meme-momentum + top-volume), cached for 15 min.
    The stock side always starts from the static watchlist; penny-stock/meme
    stock scanning is applied per-cycle inside run_one_cycle().
    """
    ac = CONFIG["asset_class"]

    if ac in ("crypto", "both"):
        if CONFIG.get("enable_dynamic_crypto", True):
            try:
                from data.trending_crypto import get_live_crypto_watchlist
                crypto_syms = get_live_crypto_watchlist(
                    max_symbols=int(CONFIG.get("max_dynamic_crypto", 50)),
                    include_meme=True,
                    include_trending=True,
                    fallback_static=True,   # always include static baseline
                )
            except Exception:
                crypto_syms = list(CONFIG["crypto_watchlist"])
        else:
            crypto_syms = list(CONFIG["crypto_watchlist"])
    else:
        crypto_syms = []

    if ac in ("stocks", "both"):
        stock_syms = list(CONFIG["stock_watchlist"])
    else:
        stock_syms = []

    # Combine and filter quarantined symbols
    all_syms = []
    if ac == "crypto":
        all_syms = crypto_syms
    elif ac == "stocks":
        all_syms = stock_syms
    else:
        all_syms = crypto_syms + stock_syms
    
    # Filter out quarantined (dead/delisted) symbols
    quarantined = set(CONFIG.get("quarantined_symbols", []))
    filtered = [s for s in all_syms if s not in quarantined]
    
    if quarantined:
        removed = set(all_syms) & quarantined
        if removed:
            logger.info(f"[Watchlist] Filtered {len(removed)} quarantined symbols: {sorted(removed)}")
    
    return filtered


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
