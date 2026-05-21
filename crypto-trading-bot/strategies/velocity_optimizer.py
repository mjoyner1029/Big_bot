"""Velocity Optimizer — manages trading frequency and execution speed.

Inspired by Claude's autonomous 48hr experiment: 5,200+ trades, +1,322% return.
Monitors trading velocity, optimizes decision speed, and prevents stagnation.

Key features:
  • Real-time trades/hour tracking
  • Adaptive confidence thresholds based on velocity
  • Dead-time detection (prevents the "OpenClaw syndrome" of too few trades)
  • Transaction cost efficiency monitoring
"""
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from config.config import CONFIG, get_mode_config


class VelocityOptimizer:
    """Tracks and optimizes trading frequency to maximize opportunities
    while maintaining risk discipline.
    
    The Claude agent succeeded with ~108 trades/hour. This optimizer helps
    identify when we're trading too slowly and missing opportunities.
    """
    
    def __init__(self, target_velocity_per_hour: Optional[float] = None):
        """
        Args:
            target_velocity_per_hour: Target trades/hour. If None, auto-set by trading mode.
        """
        self.target_velocity = target_velocity_per_hour or self._get_target_velocity()
        
        # Track recent trade timestamps (rolling window)
        self.trade_timestamps: deque = deque(maxlen=1000)
        self.signal_timestamps: deque = deque(maxlen=1000)  # All signals generated
        self.rejected_timestamps: deque = deque(maxlen=1000)  # Rejected by discipline
        
        # Performance metrics
        self.total_trades = 0
        self.total_signals = 0
        self.start_time = time.time()
        
        # Stagnation detection
        self.last_trade_time: Optional[float] = None
        self.stagnation_warnings = 0
        
        logging.info(
            f"[Velocity] Optimizer initialized. Target: {self.target_velocity:.1f} trades/hour "
            f"(mode: {CONFIG.get('trading_mode', 'balanced')})"
        )
    
    def _get_target_velocity(self) -> float:
        """Auto-calculate target velocity from trading mode and max daily trades."""
        max_daily = get_mode_config("max_trades_per_day")
        # Target 60% of max to leave headroom for prime opportunities
        return (max_daily * 0.6) / 24.0
    
    def record_signal(self, executed: bool = False) -> None:
        """Record a trading signal generation."""
        now = time.time()
        self.signal_timestamps.append(now)
        self.total_signals += 1
        
        if executed:
            self.trade_timestamps.append(now)
            self.total_trades += 1
            self.last_trade_time = now
        else:
            self.rejected_timestamps.append(now)
    
    def get_current_velocity(self, window_hours: float = 1.0) -> float:
        """Calculate current trades per hour over the specified window."""
        if not self.trade_timestamps:
            return 0.0
        
        cutoff = time.time() - (window_hours * 3600)
        recent_trades = sum(1 for ts in self.trade_timestamps if ts > cutoff)
        return recent_trades / window_hours
    
    def get_signal_velocity(self, window_hours: float = 1.0) -> float:
        """Calculate signals generated per hour."""
        if not self.signal_timestamps:
            return 0.0
        
        cutoff = time.time() - (window_hours * 3600)
        recent_signals = sum(1 for ts in self.signal_timestamps if ts > cutoff)
        return recent_signals / window_hours
    
    def get_execution_rate(self) -> float:
        """Return the % of signals that resulted in executed trades."""
        if self.total_signals == 0:
            return 0.0
        return (self.total_trades / self.total_signals) * 100
    
    def get_time_since_last_trade(self) -> Optional[float]:
        """Return seconds since last trade, or None if no trades yet."""
        if self.last_trade_time is None:
            return None
        return time.time() - self.last_trade_time
    
    def check_stagnation(self, alert_threshold_minutes: float = 30) -> Tuple[bool, str]:
        """Detect if trading has stalled (OpenClaw syndrome).
        
        The losing OpenClaw agent only made 200 trades and stagnated.
        This check helps prevent that pattern.
        
        Returns:
            (is_stagnant, message)
        """
        mode = CONFIG.get("trading_mode", "balanced")
        
        # High-frequency mode: more aggressive stagnation detection
        if mode == "claude_hf":
            alert_threshold_minutes = 10
        
        time_since = self.get_time_since_last_trade()
        if time_since is None:
            # No trades yet — only warn if system has been running a while
            runtime_minutes = (time.time() - self.start_time) / 60
            if runtime_minutes > alert_threshold_minutes:
                return True, f"No trades executed in {runtime_minutes:.1f} minutes"
            return False, ""
        
        if time_since > (alert_threshold_minutes * 60):
            self.stagnation_warnings += 1
            return True, f"Stagnant: {time_since/60:.1f} min since last trade"
        
        return False, ""
    
    def should_reduce_confidence_threshold(self) -> Tuple[bool, float]:
        """Suggest reducing confidence threshold if velocity too low.
        
        Returns:
            (should_reduce, suggested_adjustment)
            e.g., (True, -0.03) means "reduce threshold by 0.03"
        """
        current_vel = self.get_current_velocity(window_hours=0.5)  # Last 30 min
        exec_rate = self.get_execution_rate()
        
        # If we're below 50% of target velocity and execution rate is low
        if current_vel < (self.target_velocity * 0.5) and exec_rate < 30:
            # Suggest modest reduction
            adjustment = -0.02 if exec_rate < 15 else -0.01
            return True, adjustment
        
        # If we're executing too aggressively (>80% of signals), tighten
        if exec_rate > 80 and current_vel > self.target_velocity:
            return True, 0.01
        
        return False, 0.0
    
    def get_velocity_metrics(self) -> Dict[str, any]:
        """Get comprehensive velocity and efficiency metrics."""
        runtime_hours = (time.time() - self.start_time) / 3600
        current_vel = self.get_current_velocity(1.0)
        signal_vel = self.get_signal_velocity(1.0)
        
        return {
            "current_velocity_per_hour": round(current_vel, 2),
            "target_velocity_per_hour": round(self.target_velocity, 2),
            "signal_velocity_per_hour": round(signal_vel, 2),
            "execution_rate_pct": round(self.get_execution_rate(), 1),
            "total_trades": self.total_trades,
            "total_signals": self.total_signals,
            "avg_trades_per_hour": round(self.total_trades / runtime_hours, 2) if runtime_hours > 0 else 0,
            "projected_daily_trades": round((self.total_trades / runtime_hours) * 24, 0) if runtime_hours > 0 else 0,
            "time_since_last_trade_sec": self.get_time_since_last_trade(),
            "stagnation_warnings": self.stagnation_warnings,
            "velocity_target_achievement_pct": round((current_vel / self.target_velocity) * 100, 1) if self.target_velocity > 0 else 0,
        }
    
    def log_status(self) -> None:
        """Log current velocity status for monitoring."""
        metrics = self.get_velocity_metrics()
        stagnant, msg = self.check_stagnation()
        
        status = "WARNING  STAGNANT" if stagnant else "Active"
        
        logging.info(
            f"[Velocity] {status} | "
            f"Trades: {metrics['total_trades']} "
            f"({metrics['current_velocity_per_hour']:.1f}/hr, "
            f"target: {metrics['target_velocity_per_hour']:.1f}/hr) | "
            f"Execution: {metrics['execution_rate_pct']}% | "
            f"Projected daily: {metrics['projected_daily_trades']:.0f}"
        )
        
        if stagnant:
            logging.warning(f"[Velocity] {msg}")
        
        # Recommend adjustment if needed
        should_adjust, adjustment = self.should_reduce_confidence_threshold()
        if should_adjust:
            direction = "decrease" if adjustment < 0 else "increase"
            logging.info(
                f"[Velocity] Recommendation: {direction} confidence threshold by "
                f"{abs(adjustment):.3f} to optimize velocity"
            )


# Singleton instance
_velocity_optimizer: Optional[VelocityOptimizer] = None


def get_velocity_optimizer() -> VelocityOptimizer:
    """Get or create the global velocity optimizer instance."""
    global _velocity_optimizer
    if _velocity_optimizer is None:
        _velocity_optimizer = VelocityOptimizer()
    return _velocity_optimizer


def reset_velocity_optimizer() -> None:
    """Reset the velocity optimizer (useful for testing or mode changes)."""
    global _velocity_optimizer
    _velocity_optimizer = None
