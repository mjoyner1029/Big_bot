# Solana Bot Integration Analysis

## Executive Summary

The Solana DLMM bot has **proven production value** (running successfully with real money). This document outlines how to integrate its battle-tested safety mechanisms into our current crypto trading bot.

## Key Features to Incorporate

### 1. **Kill Switch (Critical Safety)**
**What it does**: Automatically closes all positions when market crashes beyond thresholds
- SOL -6% in 4 hours → close everything, 24h cooldown
- SOL -10% in 24 hours → close everything, 24h cooldown
- Prevents catastrophic losses during flash crashes

**Why it matters**: 
- Saved the Solana bot from major drawdowns during volatile periods
- Automatic, no human intervention needed
- Cooldown prevents premature re-entry during continued volatility

**Implementation**: New module `strategies/killswitch.py`

### 2. **Position Health Monitor**
**What it does**: Continuously audits open positions and closes losers automatically
- IL (Impermanent Loss) Watchdog: closes when losses > gains for N consecutive checks
- Stale Range Detection: closes positions that drift out of profitable range
- Automatic fee claiming when accrued > threshold

**Why it matters**:
- The v1 bot lost money when positions sat underwater indefinitely
- Automatic position management means bot doesn't need constant monitoring
- Prevents "zombie" positions that tie up capital without earning

**Implementation**: New module `trading/position_health.py`

### 3. **Profit Pile Accounting**
**What it does**: Sacred accounting for realized profits, separate from working capital
- Fees/profits split: 60% reinvest, 40% pile (configurable)
- Bot NEVER dips into profit pile for trading
- Tracks lifetime earnings vs current positions

**Why it matters**:
- Clear separation between "money at risk" vs "money earned"
- Psychological benefit: pile only grows, never shrinks
- Accurate P&L tracking across bot lifetime

**Implementation**: Enhanced `trading/portfolio_manager.py`

### 4. **Robust State Persistence**
**What it does**: 
- Atomic writes (tmp file → rename) prevent corruption
- Timestamped backups kept for 72 hours
- State recovery from backup or exchange data

**Why it matters**:
- V1 bot lost state when file was corrupted during write
- Backups saved the day multiple times during testing
- Can survive crashes, power loss, disk issues

**Implementation**: Enhanced `logs/trade_logger.py` + new `config/state_manager.py`

### 5. **Macro Snapshot & Regime Detection**
**What it does**:
- Periodic market condition analysis (RSI, multi-timeframe % changes)
- Regime filter: suppresses aggressive positions during downtrends
- Market-aware position sizing

**Why it matters**:
- The v2 bot adapts strategy based on market conditions
- Prevents fighting the trend (rule 6)
- Better risk management during different market phases

**Implementation**: New module `strategies/macro_snapshot.py`

### 6. **Rotator/Tick Pattern**
**What it does**: 
- Structured execution cycle: macro → audit → hedge → new positions → digest
- Clean separation of concerns
- Predictable, debuggable flow

**Why it matters**:
- Makes the main loop much cleaner and easier to debug
- Each step has clear responsibilities
- Easier to add new features without breaking existing logic

**Implementation**: Refactor `main.py` to use tick-based execution

## Implementation Priority

### Phase 1: Critical Safety Features (Do First)
1. **Kill Switch** - Prevents catastrophic losses
2. **Atomic State Persistence** - Prevents data corruption
3. **Position Health Monitor** - Automatic loss prevention

### Phase 2: Enhanced Features
4. **Profit Pile Accounting** - Better P&L tracking
5. **Macro Snapshot** - Market regime awareness

### Phase 3: Architecture Improvements
6. **Rotator Pattern** - Cleaner main loop structure

## Key Differences to Consider

### Solana Bot vs Current Bot

| Feature | Solana Bot | Current Bot | Action |
|---------|-----------|-------------|--------|
| Trading Venue | Meteora DLMM (LP) | CEX spot + derivatives | Adapt kill switch thresholds |
| Position Type | Liquidity pools | Spot + short | Adapt health monitoring |
| Price Source | CoinGecko | yfinance + WebSocket | Use existing sources |
| Alerts | iMessage | Logs | Keep logs, add optional alerts |
| Language | TypeScript | Python | Port logic to Python |
| LLM Integration | None | Claude agent | Combine both approaches |

## Configuration Additions Needed

```python
# Kill switch settings
KILL_4H_DROP_PCT = 6.0          # Close all if -6% in 4h
KILL_24H_DROP_PCT = 10.0        # Close all if -10% in 24h  
KILL_COOLDOWN_HOURS = 24        # Wait 24h before re-entering

# Position health
IL_WATCHDOG_STRIKES = 3         # Close if underwater for 3 checks
STALE_POSITION_STRIKES = 4      # Close if out of profit range for 4 checks
FEE_CLAIM_THRESHOLD_USD = 1.0   # Claim fees when > $1

# Profit pile
PROFIT_REINVEST_PCT = 60        # 60% back to trading, 40% to pile

# State management
STATE_BACKUP_RETENTION_HOURS = 72
STATE_DIR = "state/"

# Macro monitoring
MACRO_REFRESH_INTERVAL = 15     # Minutes
REGIME_RSI_FLOOR = 45           # Suppress aggressive when RSI < 45 AND trend negative
```

## Preserved Features (Don't Break)

✅ Keep all existing features:
- Autonomous agent with LLM decision making
- World events analysis
- Sentiment analysis
- TA indicators
- Paper trading mode
- WebSocket price feeds
- Velocity optimizer
- Cost tracker
- Session management

🎯 Goal: **Combine** the safety/robustness of the Solana bot with the intelligence of the current bot.

## Expected Outcomes

After integration:
1. **Safer**: Kill switch and health monitoring prevent large losses
2. **More Robust**: Atomic writes and backups prevent data loss
3. **Better Tracking**: Profit pile shows true lifetime performance
4. **Smarter**: Macro awareness + LLM analysis = better decisions
5. **Production Ready**: Battle-tested patterns from working bot

## Migration Strategy

1. **Backwards Compatible**: New features default to disabled
2. **Gradual Rollout**: Enable one feature at a time in paper mode
3. **Testing**: Run paper trading for 24-48h before going live
4. **Monitoring**: Watch logs for new health checks and kill switch triggers

## Files to Create/Modify

### New Files
- `strategies/killswitch.py` - Market crash detection
- `trading/position_health.py` - Position auditing
- `strategies/macro_snapshot.py` - Market regime analysis
- `config/state_manager.py` - Atomic writes + backups

### Modified Files
- `main.py` - Add kill switch and health checks to main loop
- `trading/portfolio_manager.py` - Add profit pile accounting
- `config/config.py` - Add new configuration options
- `trading/trade_executor.py` - Integrate position health monitoring
- `logs/trade_logger.py` - Enhanced state persistence

---

**Next Steps**: Implement Phase 1 features (Kill Switch, State Persistence, Position Health)
# 1. Just set capital in .env:
TRADING_CAPITAL=1000

# 2. Set asset class in config/config.py:
"asset_class": "stocks",  # Instead of "both"

# 3. Test:
python main.py --once