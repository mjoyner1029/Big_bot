"""Profit Pile — Sacred accounting for realized profits.

Tracks realized profits separately from working capital. The profit pile is
sacred: it only grows from profitable trades, never shrinks, and the bot is
forbidden from using it for trading.

Inspired by the Meteora LP bot's profit pile that gave clear visibility into
lifetime earnings vs capital at risk.

Features:
  • Separate tracking of realized profits vs working capital
  • Configurable profit split (e.g., 60% reinvest, 40% pile)
  • Lifetime P&L tracking independent of current positions
  • Protection: bot can never trade with profit pile money

Usage:
    from trading.profit_pile import get_profit_pile, record_profit
    
    # After closing a profitable trade:
    if pnl > 0:
        split = record_profit(pnl, reinvest_pct=60)
        # split["piled"] goes to sacred pile
        # split["reinvested"] goes back to trading capital
    
    # Check profit pile status:
    pile = get_profit_pile()
    print(f"Lifetime earnings: ${pile.total_piled:.2f}")
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os

from config.config import CONFIG
from config.state_manager import StateManager


class ProfitPile:
    """Manager for the sacred profit pile."""
    
    def __init__(self):
        self.state_manager = StateManager("profit_pile.json")
        self.data = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """Load profit pile state."""
        state = self.state_manager.load()
        
        if state is None:
            return {
                "total_piled": 0.0,
                "total_reinvested": 0.0,
                "lifetime_profits": 0.0,
                "lifetime_losses": 0.0,
                "profit_events": [],  # Last 100 profit events
                "created_at": datetime.now().isoformat(),
            }
        
        return state
    
    def _save(self) -> None:
        """Save profit pile state."""
        self.state_manager.save(self.data)
    
    def record_profit(
        self,
        profit_usd: float,
        reinvest_pct: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Record a profit and split between pile and reinvestment.
        
        Args:
            profit_usd: Net profit from trade (can be negative for losses)
            reinvest_pct: % to reinvest (rest goes to pile). If None, uses config.
            metadata: Optional metadata about the trade
        
        Returns:
            Dict with "piled" and "reinvested" amounts
        """
        if reinvest_pct is None:
            reinvest_pct = CONFIG.get("profit_reinvest_pct", 60)
        
        # Split the profit
        if profit_usd > 0:
            reinvested = (profit_usd * reinvest_pct) / 100
            piled = profit_usd - reinvested
            
            self.data["total_piled"] += piled
            self.data["total_reinvested"] += reinvested
            self.data["lifetime_profits"] += profit_usd
            
            logging.info(
                f"[ProfitPile] Recorded profit: ${profit_usd:.2f} → "
                f"${piled:.2f} piled, ${reinvested:.2f} reinvested"
            )
        else:
            # Loss — doesn't affect pile, but track it
            reinvested = 0
            piled = 0
            self.data["lifetime_losses"] += abs(profit_usd)
            
            logging.info(
                f"[ProfitPile] Recorded loss: ${profit_usd:.2f}"
            )
        
        # Record the event
        event = {
            "timestamp": datetime.now().isoformat(),
            "profit_usd": profit_usd,
            "piled": piled,
            "reinvested": reinvested,
            "reinvest_pct": reinvest_pct,
            "metadata": metadata or {},
        }
        
        self.data["profit_events"].append(event)
        
        # Keep only last 100 events
        if len(self.data["profit_events"]) > 100:
            self.data["profit_events"] = self.data["profit_events"][-100:]
        
        self._save()
        
        return {
            "piled": piled,
            "reinvested": reinvested,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current profit pile status.
        
        Returns:
            Dict with pile metrics
        """
        net_lifetime = self.data["lifetime_profits"] - self.data["lifetime_losses"]
        
        return {
            "total_piled": self.data["total_piled"],
            "total_reinvested": self.data["total_reinvested"],
            "lifetime_profits": self.data["lifetime_profits"],
            "lifetime_losses": self.data["lifetime_losses"],
            "net_lifetime": net_lifetime,
            "profit_events_count": len(self.data["profit_events"]),
            "created_at": self.data.get("created_at"),
        }
    
    def get_recent_events(self, count: int = 10) -> list:
        """Get recent profit events."""
        return self.data["profit_events"][-count:]
    
    def reset(self, reason: str = "manual reset") -> None:
        """Reset the profit pile (use with extreme caution!).
        
        This should only be used when starting a new trading session or
        after withdrawing profits from the account.
        """
        logging.warning(f"[ProfitPile] RESET triggered: {reason}")
        
        # Archive old pile before reset
        archive_event = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "archived_pile": self.data["total_piled"],
            "archived_reinvested": self.data["total_reinvested"],
            "archived_lifetime_profits": self.data["lifetime_profits"],
            "archived_lifetime_losses": self.data["lifetime_losses"],
        }
        
        # Save archive
        archive_manager = StateManager("profit_pile_archive.json")
        archive_state = archive_manager.load() or {"archives": []}
        archive_state["archives"].append(archive_event)
        archive_manager.save(archive_state)
        
        # Reset pile
        self.data = {
            "total_piled": 0.0,
            "total_reinvested": 0.0,
            "lifetime_profits": 0.0,
            "lifetime_losses": 0.0,
            "profit_events": [],
            "created_at": datetime.now().isoformat(),
            "reset_reason": reason,
            "reset_at": datetime.now().isoformat(),
        }
        self._save()
    
    def adjust_pile(self, amount: float, reason: str) -> None:
        """Manually adjust pile amount (e.g., after external deposit/withdrawal).
        
        Args:
            amount: Amount to add (positive) or subtract (negative)
            reason: Explanation for the adjustment
        """
        logging.warning(
            f"[ProfitPile] Manual adjustment: {amount:+.2f} ({reason})"
        )
        
        self.data["total_piled"] += amount
        
        # Record as event
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "manual_adjustment",
            "amount": amount,
            "reason": reason,
        }
        self.data["profit_events"].append(event)
        
        self._save()


# ── Global instance ───────────────────────────────────────────────

_profit_pile: Optional[ProfitPile] = None


def get_profit_pile() -> ProfitPile:
    """Get the global profit pile instance."""
    global _profit_pile
    if _profit_pile is None:
        _profit_pile = ProfitPile()
    return _profit_pile


# ── Convenience functions ─────────────────────────────────────────

def record_profit(
    profit_usd: float,
    reinvest_pct: Optional[float] = None,
    symbol: Optional[str] = None,
    trade_id: Optional[str] = None,
) -> Dict[str, float]:
    """Record a profit and return the split.
    
    Args:
        profit_usd: Net profit/loss from trade
        reinvest_pct: % to reinvest (rest to pile)
        symbol: Trading symbol (for metadata)
        trade_id: Trade identifier (for metadata)
    
    Returns:
        Dict with "piled" and "reinvested" amounts
    """
    metadata = {}
    if symbol:
        metadata["symbol"] = symbol
    if trade_id:
        metadata["trade_id"] = trade_id
    
    pile = get_profit_pile()
    return pile.record_profit(profit_usd, reinvest_pct, metadata)


def get_pile_status() -> Dict[str, Any]:
    """Get current profit pile status."""
    pile = get_profit_pile()
    return pile.get_status()


def get_total_piled() -> float:
    """Get total amount in profit pile."""
    pile = get_profit_pile()
    return pile.data["total_piled"]


def get_total_reinvested() -> float:
    """Get total amount reinvested."""
    pile = get_profit_pile()
    return pile.data["total_reinvested"]


def get_net_lifetime_pnl() -> float:
    """Get net lifetime P&L (profits - losses)."""
    pile = get_profit_pile()
    return pile.data["lifetime_profits"] - pile.data["lifetime_losses"]


# ── Integration helpers ───────────────────────────────────────────

def calculate_available_capital(base_capital: float) -> float:
    """Calculate available capital for trading.
    
    Working capital = base capital + reinvested profits
    (Profit pile is never used for trading)
    
    Args:
        base_capital: Starting capital or current cash
    
    Returns:
        Total available capital for trading
    """
    pile = get_profit_pile()
    return base_capital + pile.data["total_reinvested"]


def should_withdraw_profits() -> tuple[bool, float]:
    """Check if it's time to withdraw accumulated profits.
    
    Returns:
        (should_withdraw, amount)
    """
    pile = get_profit_pile()
    total_piled = pile.data["total_piled"]
    
    # Get withdrawal threshold from config
    withdrawal_threshold = CONFIG.get("profit_withdrawal_threshold", 1000)
    
    if total_piled >= withdrawal_threshold:
        return True, total_piled
    
    return False, 0


def format_profit_summary() -> str:
    """Format a nice summary of profit pile status."""
    status = get_pile_status()
    
    return f"""
╔════════════════════════════════════════════════╗
║          PROFIT PILE STATUS                    ║
╠════════════════════════════════════════════════╣
║ Sacred Pile:        ${status['total_piled']:>12,.2f}  ║
║ Reinvested:         ${status['total_reinvested']:>12,.2f}  ║
║ ────────────────────────────────────────────── ║
║ Lifetime Profits:   ${status['lifetime_profits']:>12,.2f}  ║
║ Lifetime Losses:    ${status['lifetime_losses']:>12,.2f}  ║
║ Net Lifetime P&L:   ${status['net_lifetime']:>12,.2f}  ║
║ ────────────────────────────────────────────── ║
║ Profit Events:      {status['profit_events_count']:>16}  ║
╚════════════════════════════════════════════════╝
    """.strip()
