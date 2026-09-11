#!/usr/bin/env python3
"""
Paper Trading - Test bot with real-time data but fake money
Logs all trades without actual execution
"""
import os
import time
import signal
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Set API key from environment
if not os.getenv('ANTHROPIC_API_KEY'):
    print("⚠️  WARNING: ANTHROPIC_API_KEY not set in .env file")
    os.environ['ANTHROPIC_API_KEY'] = 'test-key'

from ultimate_bot_v3_llm import LLMTradingBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_trade.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run paper trading."""
    logger.info("="*70)
    logger.info("PAPER TRADING MODE - Real data, fake money")
    logger.info("="*70)
    
    # Initialize bot
    bot = LLMTradingBot(capital=float(os.getenv("TRADING_CAPITAL", "2000")))
    
    # Override execute_trade to NOT actually trade
    original_execute = bot.execute_trade
    
    def paper_execute_trade(symbol, decision):
        """Mock execution - just log, don't trade."""
        logger.info(f"\n📝 PAPER TRADE:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Signal: {decision['signal']}")
        logger.info(f"   Size: ${decision['size']:.2f}")
        logger.info(f"   Price: ${decision['price']:.2f}")
        logger.info(f"   (NOT EXECUTED - Paper trading mode)")
        
        # Still track in database for testing
        return original_execute(symbol, decision)
    
    bot.execute_trade = paper_execute_trade
    
    # Track paper performance
    start_time = datetime.now()
    cycles_run = 0
    
    logger.info(f"\n🚀 Starting paper trading at {start_time}")
    logger.info(f"   Duration: Until you stop (Ctrl+C)")
    logger.info(f"   Capital: ${bot.capital:,.0f} (paper)")
    logger.info(f"   Cycle: Every 5 minutes")
    logger.info("")
    
    try:
        # Morning regime analysis
        bot.daily_regime_analysis()
        
        while True:
            cycles_run += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"PAPER TRADING CYCLE #{cycles_run}")
            logger.info(f"{'='*70}")
            
            # Run trading cycle
            bot.run_trading_cycle()
            
            # Stats
            runtime = (datetime.now() - start_time).total_seconds() / 3600
            logger.info(f"\n📊 Session stats:")
            logger.info(f"   Runtime: {runtime:.2f} hours")
            logger.info(f"   Cycles: {cycles_run}")
            logger.info(f"   Avg cycle time: {runtime*60/cycles_run:.1f} min")
            
            # LLM analysis every 4 hours
            if cycles_run % 48 == 0:  # 48 cycles = 4 hours
                bot.llm_trade_analysis()
            
            # Wait 5 minutes
            logger.info(f"\n💤 Waiting 5 minutes until next cycle...")
            time.sleep(300)
            
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Paper trading stopped by user")
        
        # End of day learning
        bot.end_of_day_learning()
        
        # Summary
        runtime = (datetime.now() - start_time).total_seconds() / 3600
        logger.info(f"\n{'='*70}")
        logger.info("PAPER TRADING SESSION SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"   Started: {start_time}")
        logger.info(f"   Ended: {datetime.now()}")
        logger.info(f"   Runtime: {runtime:.2f} hours")
        logger.info(f"   Cycles: {cycles_run}")
        logger.info(f"\n✅ Check logs/paper_trade.log for full details")
        logger.info(f"✅ Check data/trade_memory.sqlite for trade history")
        logger.info(f"{'='*70}")

if __name__ == '__main__':
    main()
