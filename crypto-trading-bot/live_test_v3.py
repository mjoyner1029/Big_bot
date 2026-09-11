#!/usr/bin/env python3
"""
LIVE TRADING - Real money, real trades
ONLY run this after successful backtesting and paper trading!
"""
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Validate ALL required keys for live trading
REQUIRED_VARS = ['ANTHROPIC_API_KEY', 'ALPACA_API_KEY', 'ALPACA_API_SECRET']
missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
if missing:
    print(f"❌ ERROR: Missing required environment variables: {missing}")
    print(f"   Live trading requires all API keys configured in .env")
    sys.exit(1)

# Safety check: Require explicit confirmation
def safety_check():
    """Require user confirmation before live trading."""
    print("="*70)
    print("⚠️  LIVE TRADING MODE - REAL MONEY ⚠️")
    print("="*70)
    print("\nThis will execute REAL trades with REAL money!")
    print("Have you:")
    print("  1. ✅ Run backtest successfully?")
    print("  2. ✅ Run paper trading for 24+ hours?")
    print("  3. ✅ Reviewed all trades and performance?")
    print("  4. ✅ Set correct API keys?")
    print("  5. ✅ Verified capital amount?")
    
    print("\n" + "="*70)
    response = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
    
    if response != "CONFIRM LIVE TRADING":
        print("\n❌ Live trading cancelled")
        print("Run paper trading first: python paper_trade_v3.py")
        sys.exit(0)
    
    # Second confirmation
    capital = input("\nEnter starting capital amount (e.g., 6090): $")
    try:
        capital = float(capital)
        if capital <= 0:
            raise ValueError()
    except:
        print("\n❌ Invalid capital amount")
        sys.exit(1)
    
    confirm = input(f"\nConfirm: Start live trading with ${capital:,.0f}? (yes/no): ")
    if confirm.lower() != 'yes':
        print("\n❌ Live trading cancelled")
        sys.exit(0)
    
    return capital

def main():
    """Run live trading."""
    import logging
    
    # Safety check first
    capital = safety_check()
    
    # Check API keys
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("\n❌ ANTHROPIC_API_KEY not set!")
        sys.exit(1)
    
    if not os.getenv('ALPACA_API_KEY'):
        print("\n❌ ALPACA_API_KEY not set!")
        print("Set it in .env file or:")
        print("  export ALPACA_API_KEY='your-key'")
        sys.exit(1)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('logs/live_trading.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("🚀 LIVE TRADING STARTED")
    logger.info("="*70)
    logger.info(f"   Capital: ${capital:,.0f}")
    logger.info(f"   Start time: {datetime.now()}")
    logger.info(f"   Mode: REAL TRADES, REAL MONEY")
    logger.info("="*70)
    
    # Import bot
    from ultimate_bot_v3_llm import LLMTradingBot
    
    # Initialize with real capital
    bot = LLMTradingBot(capital=capital)
    
    # Add extra safety wrapper
    original_execute = bot.execute_trade
    trades_today = 0
    max_trades_per_day = 20  # Safety limit
    
    def safe_execute_trade(symbol, decision):
        """Safety-wrapped execution."""
        nonlocal trades_today
        
        if trades_today >= max_trades_per_day:
            logger.warning(f"⚠️ Max trades per day reached ({max_trades_per_day})")
            return None
        
        logger.info(f"\n💰 LIVE TRADE #{trades_today + 1}:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Signal: {decision['signal']}")
        logger.info(f"   Size: ${decision['size']:.2f}")
        logger.info(f"   Price: ${decision['price']:.2f}")
        logger.info(f"   ⚠️  EXECUTING REAL TRADE...")
        
        # Execute
        result = original_execute(symbol, decision)
        
        if result:
            trades_today += 1
            logger.info(f"   ✅ Trade executed (Position ID: {result})")
        else:
            logger.error(f"   ❌ Trade failed")
        
        return result
    
    bot.execute_trade = safe_execute_trade
    
    # Run bot
    logger.info("\n🟢 Bot is now trading LIVE")
    logger.info("   Press Ctrl+C to stop gracefully\n")
    
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Live trading stopped by user")
        bot.end_of_day_learning()
        logger.info("✅ Shutdown complete")
    except Exception as e:
        logger.error(f"\n\n❌ FATAL ERROR: {e}", exc_info=True)
        logger.info("⚠️ EMERGENCY SHUTDOWN - Check all positions!")
        raise

if __name__ == '__main__':
    main()
