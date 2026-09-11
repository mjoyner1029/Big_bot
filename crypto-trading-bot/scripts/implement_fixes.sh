#!/bin/bash
# Quick-start script to implement all critical fixes
# Usage: ./scripts/implement_fixes.sh [phase]

set -e  # Exit on error

REPO_ROOT="/Users/asad/Code/Big_bot/crypto-trading-bot"
KRONOS_ROOT="/Users/asad/Code/Big_bot/Kronos"

cd "$REPO_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Big Bot Critical Fixes Implementation${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""

# Phase selection
PHASE=${1:-all}

if [[ "$PHASE" == "all" || "$PHASE" == "1" ]]; then
    echo -e "${YELLOW}[Phase 1] Creating Risk Management Components${NC}"
    echo ""
    
    # 1. Create core/risk_manager.py
    echo "Creating core/risk_manager.py..."
    cat > core/risk_manager.py << 'EOF'
"""Central risk management for the entire bot."""
import logging
from datetime import datetime, date
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class RiskManager:
    """Enforces portfolio-level risk limits."""
    
    def __init__(self, capital: float, config: dict = None):
        self.capital = capital
        config = config or {}
        
        # Limits
        self.max_daily_loss_pct = config.get('max_daily_loss_pct', 0.02)  # 2%
        self.max_positions = config.get('max_positions', 5)
        self.max_position_size_pct = config.get('max_position_size_pct', 0.10)  # 10%
        self.max_consecutive_losses = config.get('max_consecutive_losses', 3)
        
        # State
        self.daily_pnl = 0.0
        self.active_positions = 0
        self.consecutive_losses = 0
        self.last_reset_date = date.today()
        self.circuit_breaker_triggered = False
        
        logger.info(f"✅ RiskManager initialized: Max daily loss {self.max_daily_loss_pct:.0%}, Max positions {self.max_positions}")
    
    def check_can_trade(self, position_size: float) -> Tuple[bool, Optional[str]]:
        """Check if trade is allowed. Returns (allowed, reason)."""
        
        # Reset daily P&L at start of new day
        if date.today() != self.last_reset_date:
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.circuit_breaker_triggered = False
            self.last_reset_date = date.today()
            logger.info("📅 Daily P&L and limits reset")
        
        # Check circuit breaker
        if self.circuit_breaker_triggered:
            return False, "Circuit breaker triggered"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.circuit_breaker_triggered = True
            logger.critical(f"🛑 Circuit breaker: {self.consecutive_losses} consecutive losses")
            return False, f"{self.consecutive_losses} consecutive losses"
        
        # Check daily loss limit
        max_loss = self.capital * self.max_daily_loss_pct
        if self.daily_pnl <= -max_loss:
            self.circuit_breaker_triggered = True
            logger.critical(f"🛑 Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False, f"Daily loss ${self.daily_pnl:.2f} exceeds limit ${max_loss:.2f}"
        
        # Check position count
        if self.active_positions >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"
        
        # Check position size
        max_size = self.capital * self.max_position_size_pct
        if position_size > max_size:
            return False, f"Position ${position_size:.2f} exceeds ${max_size:.2f} limit"
        
        return True, None
    
    def on_position_open(self) -> None:
        """Call when opening a new position."""
        self.active_positions += 1
        logger.info(f"📈 Position opened (total: {self.active_positions})")
    
    def on_position_close(self, pnl: float) -> None:
        """Call when closing a position."""
        self.active_positions = max(0, self.active_positions - 1)
        self.daily_pnl += pnl
        
        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        logger.info(f"📉 Position closed: ${pnl:+.2f}, Daily P&L: ${self.daily_pnl:+.2f}, Consecutive losses: {self.consecutive_losses}")
    
    def get_risk_metrics(self) -> dict:
        """Get current risk status."""
        max_loss = self.capital * self.max_daily_loss_pct
        return {
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': max_loss,
            'daily_loss_used_pct': abs(self.daily_pnl / max_loss) if max_loss > 0 else 0,
            'active_positions': self.active_positions,
            'max_positions': self.max_positions,
            'consecutive_losses': self.consecutive_losses,
            'circuit_breaker_triggered': self.circuit_breaker_triggered
        }
EOF
    echo -e "${GREEN}✅ Created core/risk_manager.py${NC}"
    
    # 2. Create core/transaction_costs.py
    echo "Creating core/transaction_costs.py..."
    cat > core/transaction_costs.py << 'EOF'
"""Calculate real transaction costs."""

class TransactionCostModel:
    """Models exchange fees, slippage, funding."""
    
    EXCHANGE_FEES = {
        'alpaca': 0.0,      # Free for crypto
        'binance': 0.001,   # 0.1%
        'coinbase': 0.006,  # 0.6%
    }
    
    def __init__(self, exchange: str = 'alpaca', slippage_pct: float = 0.0005):
        """
        Args:
            exchange: Exchange name for fee lookup
            slippage_pct: Expected slippage (default 0.05%)
        """
        self.exchange = exchange
        self.fee_pct = self.EXCHANGE_FEES.get(exchange, 0.001)
        self.slippage_pct = slippage_pct
    
    def calculate_cost(self, position_size: float, price: float) -> float:
        """Calculate total round-trip cost in dollars.
        
        Args:
            position_size: Position size in dollars
            price: Entry price
            
        Returns:
            Total cost in dollars (fees + slippage for buy and sell)
        """
        # Round trip (buy + sell)
        total_cost_pct = 2 * (self.fee_pct + self.slippage_pct)
        return total_cost_pct * position_size
    
    def get_min_profitable_move(self, entry_price: float) -> float:
        """Get minimum price movement needed to break even.
        
        Args:
            entry_price: Entry price
            
        Returns:
            Minimum profitable move as percentage (e.g., 0.005 = 0.5%)
        """
        # Need to overcome round-trip costs
        return 2 * (self.fee_pct + self.slippage_pct)
    
    def adjust_profit_target(self, target_price: float, entry_price: float, 
                            position_size: float) -> float:
        """Adjust profit target to account for costs.
        
        Args:
            target_price: Gross profit target
            entry_price: Entry price
            position_size: Position size in dollars
            
        Returns:
            Adjusted target price that accounts for costs
        """
        gross_return = (target_price - entry_price) / entry_price
        cost = self.calculate_cost(position_size, entry_price)
        cost_pct = cost / position_size
        
        net_return = gross_return - cost_pct
        adjusted_target = entry_price * (1 + net_return)
        
        return adjusted_target
    
    def calculate_net_pnl(self, entry_price: float, exit_price: float, 
                         position_size: float) -> float:
        """Calculate net P&L after all costs.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price  
            position_size: Position size in dollars
            
        Returns:
            Net P&L in dollars
        """
        # Gross P&L
        gross_pnl = (exit_price - entry_price) / entry_price * position_size
        
        # Costs
        costs = self.calculate_cost(position_size, entry_price)
        
        # Net P&L
        return gross_pnl - costs
EOF
    echo -e "${GREEN}✅ Created core/transaction_costs.py${NC}"
    
    # 3. Create core/health_monitor.py
    echo "Creating core/health_monitor.py..."
    cat > core/health_monitor.py << 'EOF'
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
EOF
    echo -e "${GREEN}✅ Created core/health_monitor.py${NC}"
    
    echo ""
    echo -e "${GREEN}✅ Phase 1 Complete: Risk Management Components Created${NC}"
    echo ""
fi

if [[ "$PHASE" == "all" || "$PHASE" == "2" ]]; then
    echo -e "${YELLOW}[Phase 2] Creating Position Tracking${NC}"
    echo ""
    
    # 4. Create core/position_manager.py
    echo "Creating core/position_manager.py..."
    cat > core/position_manager.py << 'EOF'
"""Position tracking and management."""
import sqlite3
import logging
from datetime import datetime
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class PositionManager:
    """Tracks open and closed positions."""
    
    def __init__(self, db_path: str = 'data/trade_memory.sqlite'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize positions table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_price REAL,
                    exit_time TIMESTAMP,
                    pnl REAL,
                    status TEXT DEFAULT 'OPEN'
                )
            """)
            conn.commit()
        logger.info(f"✅ PositionManager initialized: {self.db_path}")
    
    def open_position(self, symbol: str, signal: str, size: float, entry_price: float) -> int:
        """Open a new position.
        
        Returns:
            Position ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO positions (symbol, signal, size, entry_price, entry_time, status)
                VALUES (?, ?, ?, ?, ?, 'OPEN')
            """, (symbol, signal, size, entry_price, datetime.now()))
            conn.commit()
            position_id = cursor.lastrowid
        
        logger.info(f"📈 Opened position #{position_id}: {signal} {symbol} @ ${entry_price:.2f}, size ${size:.2f}")
        return position_id
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """Close the most recent open position for symbol.
        
        Returns:
            P&L in dollars, or None if no position found
        """
        with sqlite3.connect(self.db_path) as conn:
            # Find open position
            row = conn.execute("""
                SELECT id, signal, size, entry_price
                FROM positions
                WHERE symbol = ? AND status = 'OPEN'
                ORDER BY entry_time DESC
                LIMIT 1
            """, (symbol,)).fetchone()
            
            if not row:
                logger.warning(f"⚠️ No open position found for {symbol}")
                return None
            
            position_id, signal, size, entry_price = row
            
            # Calculate P&L
            if signal == 'BUY':
                pnl = (exit_price - entry_price) / entry_price * size
            else:  # SELL
                pnl = (entry_price - exit_price) / entry_price * size
            
            # Update position
            conn.execute("""
                UPDATE positions
                SET exit_price = ?, exit_time = ?, pnl = ?, status = 'CLOSED'
                WHERE id = ?
            """, (exit_price, datetime.now(), pnl, position_id))
            conn.commit()
        
        logger.info(f"📉 Closed position #{position_id}: {symbol} @ ${exit_price:.2f}, P&L ${pnl:+.2f}")
        return pnl
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM positions WHERE status = 'OPEN'
                ORDER BY entry_time DESC
            """).fetchall()
        
        return [dict(row) for row in rows]
    
    def count_open(self) -> int:
        """Count open positions."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("""
                SELECT COUNT(*) FROM positions WHERE status = 'OPEN'
            """).fetchone()[0]
        return count
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get most recent open position for symbol."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT * FROM positions 
                WHERE symbol = ? AND status = 'OPEN'
                ORDER BY entry_time DESC
                LIMIT 1
            """, (symbol,)).fetchone()
        
        return dict(row) if row else None
EOF
    echo -e "${GREEN}✅ Created core/position_manager.py${NC}"
    
    echo ""
    echo -e "${GREEN}✅ Phase 2 Complete: Position Tracking Created${NC}"
    echo ""
fi

if [[ "$PHASE" == "all" || "$PHASE" == "3" ]]; then
    echo -e "${YELLOW}[Phase 3] Setting Up Kronos Integration${NC}"
    echo ""
    
    # 5. Install Kronos dependencies
    echo "Installing Kronos dependencies..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        cd "$KRONOS_ROOT"
        pip install -q -r requirements.txt
        cd "$REPO_ROOT"
        echo -e "${GREEN}✅ Kronos dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠️ No .venv found, skipping pip install${NC}"
        echo "   Run manually: cd $KRONOS_ROOT && pip install -r requirements.txt"
    fi
    
    echo ""
    echo -e "${GREEN}✅ Phase 3 Complete: Kronos Setup${NC}"
    echo ""
fi

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Implementation Status${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "Created files:"
echo "  - core/risk_manager.py"
echo "  - core/transaction_costs.py"
echo "  - core/health_monitor.py"
echo "  - core/position_manager.py"
echo ""
echo "Next steps:"
echo "  1. Review IMPLEMENTATION_PLAN.md for full details"
echo "  2. Test components: python -m pytest tests/"
echo "  3. Create Kronos wrapper (Phase 3)"
echo "  4. Integrate into bot.py (Phase 4)"
echo ""
echo -e "${YELLOW}📝 Full plan: /Users/asad/Code/Big_bot/crypto-trading-bot/IMPLEMENTATION_PLAN.md${NC}"
echo ""
