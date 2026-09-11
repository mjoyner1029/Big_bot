"""
Freqtrade Indicators Integration Example

This shows how to use the Freqtrade-style indicators in the trading bot pipeline.
"""
import pandas as pd
from indicators import add_freqtrade_indicators, get_freqtrade_signals

# ════════════════════════════════════════════════════════════════════════════
# USAGE IN STRATEGY
# ════════════════════════════════════════════════════════════════════════════

def analyze_market_with_freqtrade(df: pd.DataFrame) -> dict:
    """
    Example: Use Freqtrade indicators in the LLM decision engine.
    
    Input:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)
    
    Output:
        dict with entry/exit signals and market regime
    """
    
    # Step 1: Add all Freqtrade indicators (60+ columns)
    df = add_freqtrade_indicators(df)
    
    # Step 2: Generate composite entry/exit signals
    df = get_freqtrade_signals(df)
    
    # Get latest row (most recent bar)
    latest = df.iloc[-1]
    
    # Step 3: Extract key decision variables
    decision = {
        # Entry signal
        "entry_signal": int(latest.get('entry_signal', 0)),  # 1, -1, or 0
        "trend_alignment": int(latest.get('trend_alignment', 0)),  # 0-3
        
        # Momentum
        "rsi_fast": latest.get('rsi_fast'),        # 7-period RSI (reaction)
        "rsi_slow": latest.get('rsi_slow'),        # 14-period RSI (confirmation)
        "rsi_overbought": latest.get('rsi_slow', 50) > 70,
        "rsi_oversold": latest.get('rsi_slow', 50) < 30,
        
        # Trend
        "ema_aligned": (
            latest.get('ema9', 0) > latest.get('ema20', 0) > 
            latest.get('ema50', 0) > latest.get('sma200', 0)
        ),
        "price_above_sma50": latest.get('price_above_sma50', 0) == 1,
        "price_above_sma200": latest.get('price_above_sma200', 0) == 1,
        
        # Volatility & Risk
        "atr": latest.get('atr', 0),
        "stop_loss_long": latest.get('stop_loss_long', 0),
        "stop_loss_short": latest.get('stop_loss_short', 0),
        "take_profit_long": latest.get('take_profit_long', 0),
        "take_profit_short": latest.get('take_profit_short', 0),
        
        # Bollinger Bands (mean reversion setup)
        "price_near_bb_upper": latest.get('bb_pctb', 0.5) > 0.8,
        "price_near_bb_lower": latest.get('bb_pctb', 0.5) < 0.2,
        "bb_width": latest.get('bb_width', 0),
        
        # Volume (if available)
        "obv": latest.get('obv'),
        "mfi": latest.get('mfi'),
        
        # Crossovers (momentum shifts)
        "ema9_20_cross": int(latest.get('ema9_20_cross', 0)),  # 1=golden, -1=death, 0=none
        "ema20_50_cross": int(latest.get('ema20_50_cross', 0)),
        "ema50_200_cross": int(latest.get('ema50_200_cross', 0)),
    }
    
    return decision


# ════════════════════════════════════════════════════════════════════════════
# FREQTRADE PATTERN: Signal Generation
# ════════════════════════════════════════════════════════════════════════════

def classify_market_regime(df: pd.DataFrame) -> str:
    """
    Classify market into regimes using Freqtrade patterns.
    
    Returns: 'strong_uptrend', 'uptrend', 'ranging', 'downtrend', 'strong_downtrend'
    """
    latest = df.iloc[-1]
    
    # Trend strength: ADX
    adx = latest.get('adx', 25)
    
    # Trend direction: EMA/SMA alignment + price positioning
    aligned_bullish = (
        latest.get('ema9', 0) > latest.get('ema20', 0) > 
        latest.get('ema50', 0) > latest.get('sma200', 0)
    )
    aligned_bearish = not aligned_bullish
    
    # Determine regime
    if aligned_bullish and adx > 30:
        return 'strong_uptrend'
    elif aligned_bullish and adx > 20:
        return 'uptrend'
    elif aligned_bearish and adx > 30:
        return 'strong_downtrend'
    elif aligned_bearish and adx > 20:
        return 'downtrend'
    else:
        return 'ranging'


# ════════════════════════════════════════════════════════════════════════════
# FREQTRADE PATTERN: Risk Management via ATR
# ════════════════════════════════════════════════════════════════════════════

def calculate_position_size_atr(
    capital: float,
    entry_price: float,
    stop_loss_price: float,
    risk_pct: float = 0.02,
    atr: float = 100.0
) -> float:
    """
    Calculate position size using Freqtrade's ATR-based risk sizing.
    
    Risk = (capital * risk_pct) / distance_to_stop
    
    This ensures consistent risk per trade regardless of volatility.
    """
    risk_amount = capital * risk_pct
    distance_to_stop = abs(entry_price - stop_loss_price)
    
    if distance_to_stop <= 0:
        return 0.0
    
    position_size = risk_amount / distance_to_stop
    return position_size


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION POINT: LLM Decision Engine
# ════════════════════════════════════════════════════════════════════════════

def enrich_llm_context_with_freqtrade(df: pd.DataFrame) -> str:
    """
    Generate a market summary using Freqtrade indicators for Claude LLM input.
    
    This enriches the LLM's context window with structured technical analysis.
    """
    decision = analyze_market_with_freqtrade(df)
    regime = classify_market_regime(df)
    latest = df.iloc[-1]
    
    summary = f"""
TECHNICAL ANALYSIS (Freqtrade Patterns)
═══════════════════════════════════════

Market Regime: {regime}
Trend Alignment: {decision['trend_alignment']}/3 layers (stronger = more aligned)

MOMENTUM
  RSI(7):  {decision['rsi_fast']:.1f} (fast reaction)
  RSI(14): {decision['rsi_slow']:.1f} (confirmation) {'[OVERBOUGHT]' if decision['rsi_overbought'] else ''} {'[OVERSOLD]' if decision['rsi_oversold'] else ''}
  
TREND
  EMA Aligned: {decision['ema_aligned']}
  Price above SMA50:  {decision['price_above_sma50']}
  Price above SMA200: {decision['price_above_sma200']}
  EMA9/20 Cross: {['None', 'Golden', 'Death'][decision['ema9_20_cross'] + 1]}

VOLATILITY & RISK
  ATR: ${decision['atr']:.2f}
  Stop Loss (Long):  ${decision['stop_loss_long']:.2f}
  Stop Loss (Short): ${decision['stop_loss_short']:.2f}
  Take Profit (Long):  ${decision['take_profit_long']:.2f}
  Take Profit (Short): ${decision['take_profit_short']:.2f}

ENTRY SIGNAL
  Composite: {['Neutral', 'BULLISH', 'BEARISH'][decision['entry_signal'] + 1]}

Context: Use this technical setup as confirmation for your decision.
Do NOT trade on technicals alone — combine with fundamental analysis, risk/reward, and position sizing.
"""
    return summary.strip()


# ════════════════════════════════════════════════════════════════════════════
# EXAMPLE WORKFLOW
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: Load sample market data
    # df = pd.read_csv('BTC-USD-15m.csv')  # OHLCV format
    
    # Add Freqtrade indicators
    # df = add_freqtrade_indicators(df)
    
    # Get decision variables
    # decision = analyze_market_with_freqtrade(df)
    # print(decision)
    
    # Enrich LLM context
    # context = enrich_llm_context_with_freqtrade(df)
    # print(context)
    
    # Pass to Claude for decision
    # response = claude.messages.create(
    #     model="claude-sonnet-4-5",
    #     messages=[{
    #         "role": "user",
    #         "content": f"Based on this market analysis:\n{context}\nShould I enter a long position?"
    #     }]
    # )
    
    pass
