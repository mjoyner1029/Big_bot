# Multi-Asset Trading Bot (Crypto & Stocks)

An automated, AI-powered trading bot that trades **crypto** and **US equities** using a layered decision pipeline: technical analysis, machine learning, sentiment analysis, and Claude LLM validation.

---

## Architecture

```
main.py                  ← entry point (continuous loop / one-shot / backtest / train)
├── dashboard/
│   ├── app.py               ← Streamlit GUI (charts, trade panel, chat, logs)
│   └── components.py        ← Plotly chart builders (candlestick, indicators, equity)
├── config/config.py     ← centralised settings, API keys from .env
├── data/fetcher.py      ← yFinance market data fetcher
├── indicators/ta_indicators.py   ← 25+ TA indicators (RSI, MACD, BB, ADX …)
├── models/
│   ├── model_utils.py         ← feature engineering, model persistence
│   ├── price_predictor.py     ← XGBoost price-direction classifier
│   ├── sentiment_model.py     ← VADER + Claude sentiment scoring
│   ├── llm_analyst.py         ← Claude-powered TA interpretation & trade validation
│   └── reinforce_trainer.py   ← Q-learning RL trading agent
├── strategies/
│   ├── strategy_engine.py     ← weighted signal blend → buy / sell / hold
│   └── thresholds.py          ← adaptive TP / SL / trailing-stop levels
├── trading/
│   ├── trade_executor.py      ← routes orders (paper / crypto-live / stock-live)
│   ├── paper_trader.py        ← simulated fills
│   └── portfolio_manager.py   ← capital, positions, risk gates, JSON state
├── backtest/
│   ├── backtester.py          ← full historical simulation engine
│   └── metrics.py             ← Sharpe, Sortino, drawdown, win-rate, etc.
└── logs/
    ├── trade_logger.py        ← CSV audit trail
    ├── trade_log.csv
    ├── bot.log
    └── bot_state.json         ← persisted portfolio state
```

## Signal Pipeline

Each cycle, for every symbol without an open position:

1. **Technical Analysis** — multi-indicator score (RSI, MACD, Bollinger %B, ADX, Stochastic)
2. **ML Prediction** — XGBoost model outputs probability of an up-move (per-symbol model)
3. **Sentiment** — VADER scores on live headlines (NewsAPI / Google News RSS), optionally blended 60/40 with Claude deep analysis
4. **LLM TA Interpretation** — Claude reads the raw indicator snapshot and returns a bias + confidence
5. **Weighted Blend** — configurable weights (default: TA 30%, ML 30%, Sentiment 15%, LLM 25%)
6. **LLM Validation Gate** — Claude reviews the final signal and can reject or adjust confidence
7. **Threshold Calculation** — asset-class-aware TP/SL/trailing-stop (crypto bands are wider)
8. **Execution** — paper trade or live order via ccxt (crypto) / Alpaca (stocks)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env` and fill in your keys:

```bash
cp .env .env.local   # edit .env directly — it's gitignored by default
```

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Optional | Claude LLM analysis |
| `NEWS_API_KEY` | Optional | NewsAPI headlines (falls back to Google News RSS) |
| `COINBASE_API_KEY` / `SECRET` | Only for live crypto | ccxt exchange credentials |
| `ALPACA_API_KEY` / `SECRET` | Only for live stocks | Alpaca broker credentials |
| `DISCORD_WEBHOOK` | Optional | Trade notifications |

### 3. Run

```bash
# Paper-trade continuously (default — no real money)
python main.py

# Single cycle then exit
python main.py --once

# Backtest all symbols on 1-year of hourly data
python main.py --backtest

# Train XGBoost + RL models for all symbols
python main.py --train

# Launch the GUI dashboard
streamlit run dashboard/app.py
```

## Dashboard (GUI)

The Streamlit dashboard provides a visual interface with **5 tabs**:

| Tab | Features |
|---|---|
| **📊 Dashboard** | Equity curve, allocation donut, KPI metrics, open positions table, recent trades |
| **📈 Charts** | Candlestick charts with EMA/Bollinger overlays, RSI/MACD/Stochastic sub-panels, symbol & interval dropdowns |
| **💰 Trade** | Manual buy/sell with crypto & stock dropdowns, auto-signal or manual confidence, quantity input |
| **💬 Chat** | Conversational interface to Claude — ask about markets, your portfolio, or strategy (portfolio-context-aware) |
| **📋 Logs** | Trade log CSV viewer, bot log tail, raw JSON state viewer, config inspector |

Launch it with:
```bash
streamlit run dashboard/app.py
```

## Configuration

All settings live in `config/config.py`. Key options:

| Setting | Default | Description |
|---|---|---|
| `asset_class` | `"both"` | `"crypto"`, `"stocks"`, or `"both"` |
| `crypto_symbols` | `["BTC-USD","ETH-USD","SOL-USD"]` | Crypto watchlist |
| `stock_symbols` | `["AAPL","MSFT","NVDA","TSLA","AMZN","GOOG"]` | Stock watchlist |
| `capital` | `500` | Starting capital (USD) |
| `risk_per_trade_pct` | `0.02` | Risk 2% of equity per trade |
| `max_open_positions` | `5` | Max simultaneous positions |
| `max_position_pct` | `0.25` | Max 25% of equity in one position |
| `use_paper_trading` | `True` | **Set to False for live trading** |
| `use_llm` | `True` | Enable Claude analysis (gracefully skips if no API key) |
| `confidence_threshold` | `0.55` | Minimum blend score to trigger a trade |
| `loop_interval_seconds` | `300` | Seconds between cycles |

## Models

### XGBoost Price Predictor
- 26 engineered features from the TA indicator suite
- Binary classification: will the next bar close higher?
- Per-symbol models saved to `models/saved/xgb_<SYMBOL>.pkl`
- Auto-trains from available data if no model found

### Q-Learning RL Agent
- Tabular Q-learning with discretised state (RSI bins × MACD sign × EMA bins)
- 3 actions: hold / buy / sell
- Per-symbol agents saved to `models/saved/rl_<SYMBOL>.pkl`

### Claude LLM Analyst
- **Sentiment** — scores headlines on [-1, +1] (bearish → bullish)
- **TA Interpretation** — reads indicator snapshot, returns bias + confidence
- **Trade Validation** — reviews proposed trades, can reject or adjust confidence
- All prompts are asset-class–aware (adapts for crypto vs. stock)

## Execution Venues

| Mode | Crypto | Stocks |
|---|---|---|
| Paper | Simulated fills + CSV log | Simulated fills + CSV log |
| Live | [ccxt](https://github.com/ccxt/ccxt) (Coinbase default) | [Alpaca](https://alpaca.markets/) REST API |

## Risk Management

- **Fixed-fractional sizing** — position size = risk_amount / distance_to_stop
- **Max position cap** — no single trade exceeds `max_position_pct` of equity
- **Duplicate guard** — won't open a second position in the same symbol
- **Trailing stop** — tightens SL toward the market (60% of initial SL distance)
- **Portfolio state persistence** — JSON file survives restarts

## Backtesting

The backtester simulates the full pipeline on historical hourly data:
- Configurable commission (default 0.1%)
- TP/SL/trailing-stop exits checked at every bar using High/Low
- Equity curve generation
- LLM calls are automatically disabled during backtests for speed
- Metrics: Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor, expectancy

## Project Status

This is a functional trading system suitable for **paper trading and experimentation**. Before using real money:

- Thoroughly backtest your target symbols and time periods
- Validate model performance on out-of-sample data
- Start with minimal capital on paper mode
- Monitor logs and trade_log.csv closely
- Understand that past performance does not guarantee future results

## License

Private / personal use.
