#!/usr/bin/env python3
"""
Paper Trading - Test bot with real-time data but fake money
Logs all trades without actual execution
"""
import os
import fcntl
import time
import signal
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Set API key from environment
if not os.getenv('ANTHROPIC_API_KEY'):
    print("⚠️  WARNING: ANTHROPIC_API_KEY not set in .env file")
    os.environ['ANTHROPIC_API_KEY'] = 'test-key'

from ultimate_bot_v3_llm import LLMTradingBot
from core.paper_maintenance import PaperMaintenance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_trade.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

_PID_LOCKS = {}


def _claim_pid(path="pids/bot_paper.pid"):
    pid_path = Path(path)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(pid_path.resolve())
    if key in _PID_LOCKS:
        raise RuntimeError(f"Paper bot already running with PID {os.getpid()}")
    handle = pid_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        existing = handle.read().strip() or "unknown"
        handle.close()
        raise RuntimeError(f"Paper bot already running with PID {existing}")
    handle.seek(0)
    raw = handle.read().strip()
    if raw:
        try:
            existing = int(raw)
            if existing != os.getpid():
                try:
                    os.kill(existing, 0)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    raise RuntimeError(
                        f"Paper bot already running with PID {existing}")
                else:
                    raise RuntimeError(
                        f"Paper bot already running with PID {existing}")
        except (ValueError, ProcessLookupError):
            pass
        except Exception:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            raise
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    os.fsync(handle.fileno())
    _PID_LOCKS[key] = handle
    return pid_path


def _release_pid(pid_path):
    key = str(Path(pid_path).resolve())
    handle = _PID_LOCKS.pop(key, None)
    try:
        if int(pid_path.read_text().strip()) == os.getpid():
            pid_path.unlink()
    except (OSError, ValueError):
        pass
    if handle is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _handle_shutdown(signum, _frame):
    logger.info("Shutdown signal %s received", signum)
    raise KeyboardInterrupt

def main():
    """Run paper trading."""
    logger.info("="*70)
    logger.info("PAPER TRADING MODE - Real data, fake money")
    logger.info("="*70)
    
    # Pin the broker before construction; this entry point cannot route live orders.
    if os.environ.get("TRADING_MODE", "").upper() == "LIVE":
        raise RuntimeError("Paper runner refuses TRADING_MODE=LIVE")
    os.environ["TRADING_MODE"] = "PAPER"
    pid_path = _claim_pid()
    try:
        from core.broker import PaperBroker
        bot = LLMTradingBot(capital=float(os.getenv("TRADING_CAPITAL", "2000")))
        if not isinstance(bot.broker, PaperBroker):
            raise RuntimeError("Paper runner requires PaperBroker")
        bot._initialize_core_accounting()
        bot._startup_reconciliation()
        maintenance = PaperMaintenance()
    except Exception:
        _release_pid(pid_path)
        raise

    # Track paper performance
    start_time = datetime.now()
    cycles_run = 0
    
    logger.info(f"\n🚀 Starting paper trading at {start_time}")
    logger.info(f"   Duration: Until you stop (Ctrl+C)")
    logger.info(f"   Capital: ${bot.capital:,.0f} (paper)")
    logger.info(f"   Cycle: Every 5 minutes")
    logger.info("")

    signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        # Morning regime analysis
        bot.daily_regime_analysis()
        
        while True:
            maintenance.tick(bot)
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
        try:
            bot.end_of_day_learning()
        except Exception as exc:
            logger.warning(f"End-of-day learning failed during shutdown: {exc}")
        
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
    finally:
        _release_pid(pid_path)

if __name__ == '__main__':
    main()
