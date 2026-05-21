"""Position Health Monitor — Automatic position auditing and management.

Continuously monitors open positions and closes underwater positions automatically.
Inspired by the Meteora LP bot's IL watchdog that prevented holding losers.

Features:
  • Loss Watchdog: Closes positions where loss > gain for N consecutive checks
  • Stale Position Detection: Closes positions past their optimal hold time
  • Profit Taking: Automatically takes profit at configurable thresholds
  • Fee/Commission Tracking: Accounts for trading costs in P&L

Usage:
    from trading.position_health import audit_positions, should_close_position
    
    # In main loop:
    positions_to_close = audit_positions(portfolio)
    for position in positions_to_close:
        close_position(position)
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import json
import os

from config.config import CONFIG


# ── Position tracking state ───────────────────────────────────────

class PositionHealthTracker:
    """Track health metrics for each open position."""
    
    def __init__(self):
        self.state_file = os.path.join(
            CONFIG.get("state_dir", "state"),
            "position_health.json"
        )
        self.data: Dict[str, Dict[str, Any]] = {}
        self._load()
    
    def _load(self) -> None:
        """Load state from disk."""
        if not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, "r") as f:
                self.data = json.load(f)
            logging.info(
                f"[PosHealth] Loaded {len(self.data)} position health records"
            )
        except Exception as e:
            logging.warning(f"[PosHealth] Failed to load state: {e}")
    
    def _save(self) -> None:
        """Save state to disk."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logging.error(f"[PosHealth] Failed to save state: {e}")
    
    def get(self, position_id: str) -> Dict[str, Any]:
        """Get health record for a position."""
        if position_id not in self.data:
            self.data[position_id] = {
                "position_id": position_id,
                "first_seen": datetime.now().isoformat(),
                "loss_strikes": 0,
                "stale_strikes": 0,
                "last_check": None,
                "best_pnl_usd": 0,
                "best_pnl_pct": 0,
                "checks_count": 0,
            }
            self._save()
        
        return self.data[position_id]
    
    def update(self, position_id: str, updates: Dict[str, Any]) -> None:
        """Update health record."""
        record = self.get(position_id)
        record.update(updates)
        record["last_check"] = datetime.now().isoformat()
        self.data[position_id] = record
        self._save()
    
    def remove(self, position_id: str) -> None:
        """Remove position from tracking."""
        if position_id in self.data:
            del self.data[position_id]
            self._save()
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all tracked positions."""
        return list(self.data.values())


# Global tracker instance
_tracker = PositionHealthTracker()


# ── Position health checks ────────────────────────────────────────

def check_position_health(position: Dict[str, Any]) -> Dict[str, Any]:
    """Check health of a single position and return verdict.
    
    Args:
        position: Dict with keys:
            - id: str (unique identifier)
            - symbol: str
            - side: "buy" or "sell"
            - entry_price: float
            - current_price: float
            - quantity: float
            - opened_at: str (ISO timestamp)
            - fees_paid: float (optional)
    
    Returns:
        Dict with:
            - should_close: bool
            - reason: str
            - pnl_usd: float
            - pnl_pct: float
            - loss_strikes: int
            - stale_strikes: int
            - hold_hours: float
    """
    position_id = position["id"]
    health = _tracker.get(position_id)
    
    # Calculate P&L
    entry_price = position["entry_price"]
    current_price = position["current_price"]
    quantity = position["quantity"]
    side = position["side"]
    fees_paid = position.get("fees_paid", 0)
    
    # Calculate raw P&L
    if side == "buy":
        raw_pnl_usd = (current_price - entry_price) * quantity
    else:  # sell/short
        raw_pnl_usd = (entry_price - current_price) * quantity
    
    # Subtract fees
    net_pnl_usd = raw_pnl_usd - fees_paid
    position_value = entry_price * quantity
    pnl_pct = (net_pnl_usd / position_value * 100) if position_value > 0 else 0
    
    # Track best P&L
    if net_pnl_usd > health["best_pnl_usd"]:
        health["best_pnl_usd"] = net_pnl_usd
        health["best_pnl_pct"] = pnl_pct
    
    # Calculate hold time (handle both timezone-aware and naive datetimes)
    opened_at = datetime.fromisoformat(position["opened_at"])
    
    # Make both datetimes timezone-aware (UTC)
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=timezone.utc)
    
    now = datetime.now(timezone.utc)
    hold_hours = (now - opened_at).total_seconds() / 3600
    
    # Increment check count
    health["checks_count"] += 1
    
    # ── Loss Watchdog ─────────────────────────────────────────────
    # If position is underwater, increment strike counter
    # If underwater for N consecutive checks, close it
    
    loss_threshold_pct = CONFIG.get("loss_watchdog_threshold_pct", -2.0)
    max_loss_strikes = CONFIG.get("loss_watchdog_strikes", 3)
    
    is_losing = pnl_pct < loss_threshold_pct
    
    if is_losing:
        health["loss_strikes"] += 1
        logging.debug(
            f"[PosHealth] {position['symbol']} loss strike {health['loss_strikes']}"
            f"/{max_loss_strikes} (P&L: {pnl_pct:.2f}%)"
        )
    else:
        # Reset if position recovers
        if health["loss_strikes"] > 0:
            logging.debug(
                f"[PosHealth] {position['symbol']} loss strikes reset "
                f"(P&L: {pnl_pct:.2f}%)"
            )
        health["loss_strikes"] = 0
    
    # Check if we should close due to losses
    if health["loss_strikes"] >= max_loss_strikes:
        _tracker.update(position_id, health)
        return {
            "should_close": True,
            "reason": (
                f"Loss watchdog: underwater for {max_loss_strikes} checks "
                f"(P&L: {net_pnl_usd:.2f} USD / {pnl_pct:.2f}%)"
            ),
            "pnl_usd": net_pnl_usd,
            "pnl_pct": pnl_pct,
            "loss_strikes": health["loss_strikes"],
            "stale_strikes": health["stale_strikes"],
            "hold_hours": hold_hours,
        }
    
    # ── Stale Position Detection ──────────────────────────────────
    # Close positions that have been open too long without profit
    
    max_hold_hours = CONFIG.get("max_position_hold_hours", 48)
    stale_pnl_threshold = CONFIG.get("stale_position_pnl_threshold", 0.5)
    max_stale_strikes = CONFIG.get("stale_position_strikes", 4)
    
    is_stale = (
        hold_hours > max_hold_hours and
        pnl_pct < stale_pnl_threshold
    )
    
    if is_stale:
        health["stale_strikes"] += 1
        logging.debug(
            f"[PosHealth] {position['symbol']} stale strike "
            f"{health['stale_strikes']}/{max_stale_strikes} "
            f"(hold: {hold_hours:.1f}h, P&L: {pnl_pct:.2f}%)"
        )
    else:
        health["stale_strikes"] = 0
    
    if health["stale_strikes"] >= max_stale_strikes:
        _tracker.update(position_id, health)
        return {
            "should_close": True,
            "reason": (
                f"Stale position: held {hold_hours:.1f}h with minimal profit "
                f"(P&L: {net_pnl_usd:.2f} USD / {pnl_pct:.2f}%)"
            ),
            "pnl_usd": net_pnl_usd,
            "pnl_pct": pnl_pct,
            "loss_strikes": health["loss_strikes"],
            "stale_strikes": health["stale_strikes"],
            "hold_hours": hold_hours,
        }
    
    # ── Trailing Stop Loss ────────────────────────────────────────
    # If position has been profitable but is now giving back gains
    
    trailing_stop_enabled = CONFIG.get("trailing_stop_enabled", True)
    trailing_stop_pct = CONFIG.get("trailing_stop_pct", 50)  # Give back 50% of best gain
    
    if trailing_stop_enabled and health["best_pnl_pct"] > 2.0:
        # Position was profitable at some point
        drawdown_from_best = health["best_pnl_pct"] - pnl_pct
        drawdown_pct_of_best = (
            (drawdown_from_best / health["best_pnl_pct"]) * 100
            if health["best_pnl_pct"] > 0 else 0
        )
        
        if drawdown_pct_of_best >= trailing_stop_pct:
            _tracker.update(position_id, health)
            return {
                "should_close": True,
                "reason": (
                    f"Trailing stop: gave back {drawdown_pct_of_best:.1f}% of "
                    f"best gain (best: {health['best_pnl_pct']:.2f}%, "
                    f"current: {pnl_pct:.2f}%)"
                ),
                "pnl_usd": net_pnl_usd,
                "pnl_pct": pnl_pct,
                "loss_strikes": health["loss_strikes"],
                "stale_strikes": health["stale_strikes"],
                "hold_hours": hold_hours,
            }
    
    # ── Take Profit ───────────────────────────────────────────────
    # Automatic profit taking at configured threshold
    
    take_profit_enabled = CONFIG.get("auto_take_profit_enabled", False)
    take_profit_pct = CONFIG.get("take_profit_threshold_pct", 10.0)
    
    if take_profit_enabled and pnl_pct >= take_profit_pct:
        _tracker.update(position_id, health)
        return {
            "should_close": True,
            "reason": (
                f"Take profit: target {take_profit_pct}% reached "
                f"(P&L: {net_pnl_usd:.2f} USD / {pnl_pct:.2f}%)"
            ),
            "pnl_usd": net_pnl_usd,
            "pnl_pct": pnl_pct,
            "loss_strikes": health["loss_strikes"],
            "stale_strikes": health["stale_strikes"],
            "hold_hours": hold_hours,
        }
    
    # Position is healthy — update and continue
    _tracker.update(position_id, health)
    
    return {
        "should_close": False,
        "reason": "position healthy",
        "pnl_usd": net_pnl_usd,
        "pnl_pct": pnl_pct,
        "loss_strikes": health["loss_strikes"],
        "stale_strikes": health["stale_strikes"],
        "hold_hours": hold_hours,
    }


def audit_positions(positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Audit all open positions and return list of positions to close.
    
    Args:
        positions: List of position dicts (see check_position_health for schema)
    
    Returns:
        List of positions that should be closed, each with added "close_reason"
    """
    if not CONFIG.get("position_health_monitor_enabled", True):
        return []
    
    to_close = []
    
    for pos in positions:
        try:
            verdict = check_position_health(pos)
            
            if verdict["should_close"]:
                pos["close_reason"] = verdict["reason"]
                pos["health_verdict"] = verdict
                to_close.append(pos)
                
                logging.warning(
                    f"[PosHealth] 🚨 Position flagged for close: "
                    f"{pos['symbol']} - {verdict['reason']}"
                )
                try:
                    from alerts.sms_notifier import send_alert
                    send_alert(
                        f"POSITION FORCE-CLOSED: {pos['symbol']} - {verdict['reason'][:80]}"
                    )
                except Exception:
                    pass
        except Exception as e:
            logging.error(
                f"[PosHealth] Error checking position {pos.get('id')}: {e}"
            )
    
    return to_close


def cleanup_closed_position(position_id: str) -> None:
    """Remove tracking data for a closed position."""
    _tracker.remove(position_id)
    logging.debug(f"[PosHealth] Cleaned up tracking for {position_id}")


def get_position_health_summary() -> Dict[str, Any]:
    """Get summary of all tracked positions (for monitoring/dashboard)."""
    all_positions = _tracker.get_all()
    
    if not all_positions:
        return {
            "total_positions": 0,
            "positions_with_strikes": 0,
            "avg_hold_hours": 0,
            "positions": [],
        }
    
    positions_with_strikes = sum(
        1 for p in all_positions
        if p.get("loss_strikes", 0) > 0 or p.get("stale_strikes", 0) > 0
    )
    
    total_checks = sum(p.get("checks_count", 0) for p in all_positions)
    
    return {
        "total_positions": len(all_positions),
        "positions_with_strikes": positions_with_strikes,
        "total_checks": total_checks,
        "positions": all_positions,
    }


def reset_all_strikes() -> None:
    """Reset all strike counters (emergency use only)."""
    logging.warning("[PosHealth] Resetting all strike counters")
    
    for position_id in _tracker.data:
        _tracker.data[position_id]["loss_strikes"] = 0
        _tracker.data[position_id]["stale_strikes"] = 0
    
    _tracker._save()
