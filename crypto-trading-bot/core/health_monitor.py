"""Health monitoring and graceful shutdown."""
import signal
import sys
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

class HealthMonitor:
    """Monitors bot health and handles graceful shutdown."""
    
    def __init__(self, bot_instance, max_cycle_minutes: int = 15):
        self.bot = bot_instance
        self.last_heartbeat = datetime.now()
        self.max_cycle_time = timedelta(minutes=max_cycle_minutes)
        self.shutdown_requested = False
    
    def setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        logger.info("✅ Signal handlers registered (SIGTERM, SIGINT)")
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self.graceful_shutdown()
        sys.exit(0)
    
    def update_heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = datetime.now()
    
    def check_heartbeat(self) -> bool:
        """Check if heartbeat is current.
        
        Returns:
            True if healthy, False if stalled
        """
        elapsed = datetime.now() - self.last_heartbeat
        if elapsed > self.max_cycle_time:
            logger.critical(f"🔴 Heartbeat stalled for {elapsed.total_seconds()/60:.1f} minutes!")
            return False
        return True
    
    def graceful_shutdown(self) -> None:
        """Perform graceful shutdown tasks."""
        logger.info("🛑 Graceful shutdown initiated")
        
        # Close any open positions (if implemented)
        if hasattr(self.bot, 'positions'):
            open_count = getattr(self.bot.positions, 'count_open', lambda: 0)()
            if open_count > 0:
                logger.warning(f"⚠️ {open_count} positions still open during shutdown!")
        
        # Log final metrics
        if hasattr(self.bot, 'risk_manager'):
            metrics = self.bot.risk_manager.get_risk_metrics()
            logger.info(f"Final daily P&L: ${metrics['daily_pnl']:+.2f}")
            logger.info(f"Final positions: {metrics['active_positions']}")
        
        logger.info("✅ Shutdown complete")
