# Crypto Trading Bot V3

LLM-orchestrated adaptive trading bot using Anthropic Claude + Kronos predictions.

## Quick Start

```bash
./START.sh          # Paper trading (default)
./run_tests.sh      # Testing menu (backtest / paper / live)
```

## Monitor

```bash
tail -f logs/bot.log
```

## Stop

```bash
pkill -f paper_trade_v3.py
```

## Entry Points

```
ultimate_bot_v3_llm.py   # Core bot (LLM-orchestrated)
paper_trade_v3.py        # Paper trading wrapper
live_test_v3.py          # Live trading (real money)
backtest_v3.py           # Backtest on historical data
```

## Workflow

1. LLM selects strategies based on market regime
2. Kronos model evaluates trade setups
3. Bot executes based on LLM + Kronos consensus
4. End-of-day learning adapts future behaviour

## Config

All settings in `.env`. API keys required: `ANTHROPIC_API_KEY`, `ALPACA_API_KEY`, `ALPACA_API_SECRET`.
