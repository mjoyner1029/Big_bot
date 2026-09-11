"""
Portfolio-level safety limits and circuit breakers.

Hard risk rules are enforced here. No external caller (including LLM output)
can override these limits. All rule checks are deterministic Python code.
"""
import logging
from datetime import datetime, date
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


class SafetyManager:
    """Enforces hard trading safety limits.

    All methods are intentionally simple and side-effect free on external
    systems — they update internal state only. The bot is responsible for
    calling these methods at the right points in the trade lifecycle.

    Limits that can NEVER be overridden by Claude or any config at runtime:
        - max_daily_loss_pct
        - max_consecutive_losses
        - circuit_breaker (once triggered, requires restart)
    """

    def __init__(self, capital: float, config: Dict = None):
        self.capital = capital
        config = config or {}

        # ── Hard limits (read-only after init) ────────────────────────────────
        self.max_daily_loss_pct    = config.get('max_daily_loss_pct', 0.02)
        self.max_positions         = config.get('max_positions', 5)
        self.max_position_size_pct = config.get('max_position_size_pct', 0.10)
        self.max_consecutive_losses= config.get('max_consecutive_losses', 3)
        self.max_drawdown_pct      = config.get('max_drawdown_pct', 0.05)
        self.max_daily_trades      = config.get('max_daily_trades', 50)

        # ── Mutable state ─────────────────────────────────────────────────────
        self.daily_pnl              = 0.0
        self.daily_trade_count      = 0
        self.active_positions       = 0
        self.consecutive_losses     = 0
        self.consecutive_wins       = 0
        self.peak_equity            = capital
        self.current_equity         = capital
        self.last_reset_date        = date.today()
        self.circuit_breaker_triggered = False
        self.halt_reason: Optional[str] = None

        logger.info(
            f"SafetyManager ready: max_loss={self.max_daily_loss_pct:.0%} "
            f"max_pos={self.max_positions} circuit_breaker={self.max_consecutive_losses} losses"
        )

    # ── Daily reset ───────────────────────────────────────────────────────────

    def _maybe_reset_daily(self) -> None:
        if date.today() != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self.consecutive_losses = 0
            self.consecutive_wins = 0
            self.circuit_breaker_triggered = False
            self.halt_reason = None
            self.last_reset_date = date.today()
            logger.info("SafetyManager: daily state reset")

    # ── Gate check ────────────────────────────────────────────────────────────

    def check_can_trade(self, position_size: float = 0.0) -> Tuple[bool, Optional[str]]:
        """Return (allowed, reason). Reason is None when allowed."""
        self._maybe_reset_daily()

        if self.circuit_breaker_triggered:
            return False, f"Circuit breaker: {self.halt_reason}"

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.circuit_breaker_triggered = True
            self.halt_reason = f"{self.consecutive_losses} consecutive losses"
            logger.critical(f"CIRCUIT BREAKER: {self.halt_reason}")
            return False, self.halt_reason

        max_loss = self.capital * self.max_daily_loss_pct
        if self.daily_pnl <= -max_loss:
            self.circuit_breaker_triggered = True
            self.halt_reason = f"Daily loss ${self.daily_pnl:.2f} >= limit ${max_loss:.2f}"
            logger.critical(f"CIRCUIT BREAKER: {self.halt_reason}")
            return False, self.halt_reason

        drawdown_pct = (self.peak_equity - self.current_equity) / self.peak_equity
        if drawdown_pct >= self.max_drawdown_pct:
            self.circuit_breaker_triggered = True
            self.halt_reason = f"Max drawdown {drawdown_pct:.1%} reached"
            logger.critical(f"CIRCUIT BREAKER: {self.halt_reason}")
            return False, self.halt_reason

        if self.active_positions >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"

        if self.daily_trade_count >= self.max_daily_trades:
            return False, f"Max daily trades ({self.max_daily_trades}) reached"

        if position_size > 0:
            max_size = self.capital * self.max_position_size_pct
            if position_size > max_size:
                return False, f"Position ${position_size:.2f} > max ${max_size:.2f}"

        return True, None

    # ── Event hooks ───────────────────────────────────────────────────────────

    def on_position_open(self) -> None:
        """Call immediately after a position is opened."""
        self.active_positions += 1
        self.daily_trade_count += 1
        logger.debug(f"SafetyManager: position opened ({self.active_positions}/{self.max_positions})")

    def on_realized_pnl(self, pnl: float) -> None:
        """
        Record realized P&L from any close event (full or partial).

        Updates daily_pnl, current_equity, and peak_equity.
        Does NOT touch active_positions or consecutive counters —
        call on_position_closed() or on_partial_close() for that.
        """
        self.daily_pnl     += pnl
        self.current_equity += pnl
        self.peak_equity    = max(self.peak_equity, self.current_equity)
        logger.debug(
            f"SafetyManager: realized pnl={pnl:+.2f} daily={self.daily_pnl:+.2f} "
            f"equity={self.current_equity:,.2f}"
        )

    def on_partial_close(self, pnl: float) -> None:
        """
        Call when a partial close fills at the broker.

        Records the realized P&L but does NOT decrement active_positions —
        the position is still open (just smaller).
        Does NOT update consecutive win/loss counters.
        """
        self.on_realized_pnl(pnl)
        logger.debug(f"SafetyManager: partial close pnl={pnl:+.2f} (position still open)")

    def on_position_closed(self, pnl: float) -> None:
        """
        Call when a full position close fills at the broker.

        Records realized P&L, decrements active_positions, and updates
        consecutive win/loss counters.
        """
        self.active_positions = max(0, self.active_positions - 1)
        self.on_realized_pnl(pnl)

        if pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins   = 0
        else:
            self.consecutive_wins   += 1
            self.consecutive_losses  = 0

        logger.debug(
            f"SafetyManager: position closed pnl={pnl:+.2f} "
            f"active={self.active_positions} consec_losses={self.consecutive_losses}"
        )

    # ── Legacy aliases (kept for backward compatibility) ─────────────────────

    def on_position_close(self, pnl: float) -> None:
        """Alias for on_position_closed(). Prefer the explicit method."""
        self.on_position_closed(pnl)

    def on_win(self, pnl: float) -> None:
        """Convenience alias — call on a profitable full close."""
        if pnl < 0:
            logger.warning(f"on_win called with negative pnl={pnl:.2f}; use on_loss instead")
        self.on_position_closed(abs(pnl))

    def on_loss(self, pnl: float) -> None:
        """Convenience alias — call on a losing full close."""
        self.on_position_closed(-abs(pnl))

    # ── Observability ─────────────────────────────────────────────────────────

    def get_safety_metrics(self) -> Dict:
        """Snapshot of current safety state for logging / Claude consumption."""
        max_loss = self.capital * self.max_daily_loss_pct
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        return {
            'daily_pnl':               round(self.daily_pnl, 2),
            'daily_pnl_pct':           round(self.daily_pnl / self.capital, 4),
            'daily_loss_limit':        round(max_loss, 2),
            'daily_loss_used_pct':     round(-self.daily_pnl / max_loss, 4) if self.daily_pnl < 0 else 0.0,
            'daily_trade_count':       self.daily_trade_count,
            'active_positions':        self.active_positions,
            'max_positions':           self.max_positions,
            'consecutive_losses':      self.consecutive_losses,
            'consecutive_wins':        self.consecutive_wins,
            'circuit_breaker':         self.circuit_breaker_triggered,
            'halt_reason':             self.halt_reason,
            'current_equity':          round(self.current_equity, 2),
            'peak_equity':             round(self.peak_equity, 2),
            'drawdown_pct':            round(drawdown, 4),
            'max_drawdown_pct':        self.max_drawdown_pct,
        }
