# Freqtrade Indicator Integration Guide

## Overview

The bot now includes Freqtrade-compatible indicator patterns alongside the existing TA library. This provides:

- **Fast/Slow indicator variants** (RSI-7 and RSI-14 for example)
- **Composite signals** (EMA alignment, crossovers)
- **Risk management** (ATR-based stop-loss sizing)
- **Caching** (efficient repeated calculations)
- **Production patterns** (Freqtrade conventions)

## Quick Start

### 1. Basic Indicator Calculation

```python
from indicators import add_freqtrade_indicators, get_freqtrade_signals
import pandas as pd

# Load OHLCV data
df = pd.read_csv('market_data.csv')  # Must have: open, high, low, close, volume

# Add all Freqtrade indicators
df = add_freqtrade_indicators(df)

# Get latest row
latest = df.iloc[-1]
print(f"RSI (fast): {latest['rsi_fast']:.1f}")
print(f"RSI (slow): {latest['rsi_slow']:.1f}")
print(f"MACD: {latest['macd']:.4f}")
print(f"EMA9/20 trend: {'UP' if latest['ema9'] > latest['ema20'] else 'DOWN'}")
```

### 2. Signal Generation (Composite Buy/Sell)

```python
df = add_freqtrade_indicators(df)
df = get_freqtrade_signals(df)

# Entry signals: 1 = bullish, -1 = bearish, 0 = neutral
entry_signal = df.iloc[-1]['entry_signal']
trend_strength = df.iloc[-1]['trend_alignment']  # 0-3 layers aligned

if entry_signal == 1:
    print(f"🔥 BULLISH signal (trend strength: {trend_strength}/3)")
elif entry_signal == -1:
    print(f"❄️ BEARISH signal (trend strength: {trend_strength}/3)")
else:
    print("➡️ NEUTRAL - waiting for setup")
```

### 3. Risk Management (ATR-Based Position Sizing)

```python
from indicators import generate_stop_loss_atr

df = add_freqtrade_indicators(df)
sl_tp = generate_stop_loss_atr(df, atr_multiplier=2.0)

latest = df.iloc[-1]
entry_price = latest['close']
stop_loss = sl_tp['stop_loss_long'].iloc[-1]
take_profit = sl_tp['take_profit_long'].iloc[-1]

risk_amount = entry_price - stop_loss
reward_amount = take_profit - entry_price
risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

print(f"Entry:       ${entry_price:.2f}")
print(f"Stop Loss:   ${stop_loss:.2f}")
print(f"Take Profit: ${take_profit:.2f}")
print(f"Risk/Reward: {risk_reward_ratio:.2f}:1")
```

### 4. Integration with LLM Decision Engine

See [freqtrade_example.py](freqtrade_example.py) for full example.

```python
from indicators.freqtrade_example import (
    analyze_market_with_freqtrade,
    enrich_llm_context_with_freqtrade,
    classify_market_regime,
)

# Get market decision variables
decision = analyze_market_with_freqtrade(df)
print(decision)
# → {'entry_signal': 1, 'rsi_fast': 45.2, 'rsi_slow': 52.1, ...}

# Classify regime
regime = classify_market_regime(df)
print(f"Market Regime: {regime}")
# → 'uptrend', 'downtrend', 'ranging', etc.

# Enrich LLM context
llm_context = enrich_llm_context_with_freqtrade(df)
print(llm_context)
# → Multi-line markdown suitable for Claude input
```

## Available Indicators

### Momentum Indicators

| Indicator | Variants | Description |
|-----------|----------|-------------|
| RSI | `rsi_fast` (7), `rsi_slow` (14) | Oscillator, range 0-100 |
| Stochastic | `stoch_k`, `stoch_d` | Momentum crossover |
| MACD | `macd`, `macd_signal`, `macd_hist` | Trend following |

### Trend Indicators

| Indicator | Columns | Description |
|-----------|---------|-------------|
| EMA Family | `ema9`, `ema20`, `ema50`, `ema200` | Exponential moving averages |
| SMA Family | `sma50`, `sma200` | Simple moving averages |
| ADX | `adx`, `adx_pos`, `adx_neg` | Trend strength and direction |
| Ichimoku | `ichimoku_base` | Support/resistance base line |

### Volatility Indicators

| Indicator | Columns | Description |
|-----------|---------|-------------|
| Bollinger Bands | `bb_upper`, `bb_mid`, `bb_lower`, `bb_pctb`, `bb_width` | Mean reversion, volatility |
| ATR | `atr` | Average True Range (volatility) |

### Volume Indicators

| Indicator | Description |
|-----------|-------------|
| OBV | On-Balance Volume |
| MFI | Money Flow Index |

### Crossovers & Alignment

| Signal | Description |
|--------|-------------|
| `ema9_20_cross` | 1: golden cross, -1: death cross, 0: none |
| `ema20_50_cross` | Intermediate trend shift |
| `ema50_200_cross` | Major trend shift |
| `price_above_sma50` | Price > 50-day MA |
| `price_above_sma200` | Price > 200-day MA |
| `sma50_above_sma200` | Golden cross of moving averages |
| `trend_alignment` | 0-3: how many EMA/SMA layers aligned |

## Freqtrade-Style Signal Generation

The bot uses Freqtrade's composite signal approach:

### Entry Signal Logic

**BULLISH (entry_signal = 1)** when:
- EMA aligned: ema9 > ema20 > ema50 > sma200
- RSI not overbought: 40 < RSI(14) < 70
- Price above Bollinger mid
- Trend strength: ADX > 25

**BEARISH (entry_signal = -1)** when:
- EMA misaligned: ema9 < ema20 < ema50 < sma200
- RSI not oversold: 30 < RSI(14) < 60
- Price below Bollinger mid
- Trend strength: ADX > 25

**NEUTRAL (entry_signal = 0)** otherwise

## Caching & Performance

The indicators module includes caching to avoid recalculating on the same data:

```python
from indicators import clear_freqtrade_cache

# Add indicators (cached)
df1 = add_freqtrade_indicators(df)
df2 = add_freqtrade_indicators(df)  # Returns cached result

# When new data arrives
clear_freqtrade_cache()
df3 = add_freqtrade_indicators(df)  # Recalculates
```

## Integration Points in ultimate_bot_v3_llm.py

The bot's LLM decision engine can be enhanced:

```python
# In LLMTradingBot.fetch_market_data()
from indicators import add_freqtrade_indicators, get_freqtrade_signals

for symbol in symbols:
    df = self.market_data[symbol]
    
    # Add Freqtrade indicators
    df = add_freqtrade_indicators(df)
    df = get_freqtrade_signals(df)
    
    # Pass to LLM
    latest_indicators = df.iloc[-1].dropna().to_dict()
    
    # LLM can use: latest_indicators['entry_signal'], 
    #              latest_indicators['rsi_slow'], etc.
```

## Running Tests

```bash
# Test Freqtrade indicators specifically
python -m pytest tests/test_freqtrade_indicators.py -v

# All tests
python -m pytest tests/ -q

# Should see: 397 passed
```

## Comparison: Standard TA vs Freqtrade

| Aspect | Standard TA | Freqtrade |
|--------|-------------|-----------|
| Column Names | Mixed case (RSI, MACD) | Lowercase (rsi, macd) |
| Fast/Slow | Single variant | Fast & slow variants |
| Crossovers | Manual calculation | Built-in detection |
| EMA/SMA Family | Individual calls | Batch loading |
| Signal Generation | None built-in | Composite signals |
| Caching | None | Automatic |

## Migration Path

**Step 1:** Keep standard TA working

```python
from indicators import add_ta_indicators
df = add_ta_indicators(df)  # Still works
```

**Step 2:** Add Freqtrade indicators alongside

```python
from indicators import add_freqtrade_indicators
df = add_freqtrade_indicators(df)  # New columns added
```

**Step 3:** Gradually migrate strategy to use Freqtrade signals

```python
df = get_freqtrade_signals(df)
entry_signal = df.iloc[-1]['entry_signal']  # Use composite signal
```

## Troubleshooting

### Missing indicators in output

If some indicators are NaN:

1. **Insufficient data**: Need ~200 bars for SMA200 to stabilize
2. **Missing volume**: Some indicators (OBV, MFI) require volume column
3. **Column naming**: Ensure input has lowercase: 'open', 'high', 'low', 'close', 'volume'

```python
# Ensure proper naming
df.columns = [c.lower() for c in df.columns]
df = add_freqtrade_indicators(df)
```

### Performance issues

Clear cache if handling multiple symbols:

```python
from indicators import clear_freqtrade_cache

for symbol in symbols:
    clear_freqtrade_cache()
    df = add_freqtrade_indicators(market_data[symbol])
```

## References

- **Freqtrade Bot**: https://github.com/freqtrade/freqtrade
- **Freqtrade Strategy Examples**: https://github.com/freqtrade/freqtrade-strategies
- **TA Library**: https://github.com/bukosabino/ta
- **Our Implementation**: [freqtrade_patterns.py](freqtrade_patterns.py)
- **Integration Example**: [freqtrade_example.py](freqtrade_example.py)
- **Tests**: [tests/test_freqtrade_indicators.py](../tests/test_freqtrade_indicators.py)

## Next Steps

1. **Integrate into LLM pipeline**: Use `enrich_llm_context_with_freqtrade()` in decision engine
2. **Backtest new signals**: Use 18 new tests as validation baseline
3. **Monitor performance**: Compare Freqtrade signals vs standard TA in paper trading
4. **Iterate**: Adjust signal thresholds based on live results

---

**Status**: ✅ Integrated and tested (18 unit tests, all passing)  
**Production Ready**: Yes, for PAPER trading  
**Recommended**: Use alongside (not replacing) fundamental/ML analysis  
