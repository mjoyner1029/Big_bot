"""Discipline layer — the single gate every signal must pass before execution.

This module orchestrates all professional risk discipline checks:
  1. Daily drawdown kill switch
  2. Market session / hours enforcement
  3. Correlation & cluster exposure limits
  4. Loss-streak cool-down (reduce size or pause)
  5. Max trades per day cap
  6. Volume / liquidity sanity check
  7. Overnight risk management (flatten before close)
  8. Regime-aware parameter adjustments

Usage:
    from strategies.discipline import DisciplineGate
    gate = DisciplineGate(portfolio)
    ok, reason, adjustments = gate.check(signal, market_data, current_prices)
"""
import logging
from collections import defaultdict
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.config import CONFIG, is_crypto, get_mode_config
from strategies.session_manager import (
    is_trading_allowed,
    session_confidence_multiplier,
    should_flatten_overnight,
    get_current_session,
)
from strategies.correlation import (
    can_add_to_cluster,
    get_cluster,
    find_highly_correlated,
)
from strategies.regime import detect_regime
from strategies.velocity_optimizer import get_velocity_optimizer


class DisciplineGate:
    """Central risk-discipline gate.  Every signal passes through here
    before execution.  The gate can:
      • BLOCK a trade entirely (returns allowed=False)
      • ADJUST parameters (reduce size, tighten stops)
      • LOG the reason for transparency
    """

    def __init__(self, portfolio):
        self.portfolio = portfolio
        # Daily tracking — reset at midnight UTC
        self._today: Optional[date] = None
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._daily_peak_equity: float = 0.0
        self._consecutive_losses: int = 0
        self._loss_streak_today: int = 0
        self._recent_results: List[str] = []  # last N trade results
        
        # Circuit breaker state
        self._emergency_halt: bool = False
        self._halt_reason: str = ""
        self._pause_until: float = 0.0  # Unix timestamp
        self._last_rapid_loss_check: float = 0.0
        self._rapid_loss_timestamps: List[float] = []
        
        # Velocity tracking (inspired by Claude's high-frequency success)
        self.velocity_optimizer = get_velocity_optimizer()
        self._last_velocity_check: float = 0.0

    # ── Daily state management ────────────────────────────────────

    def _ensure_day(self, now: Optional[datetime] = None) -> None:
        """Reset daily counters if the date has changed."""
        today = (now or datetime.now(timezone.utc)).date()
        if self._today != today:
            self._today = today
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._loss_streak_today = 0
            equity = self.portfolio.total_equity(current_prices={})
            self._daily_peak_equity = equity if equity > 0 else CONFIG["capital"]
            logging.info(
                f"[Discipline] New trading day: {today}  "
                f"Starting equity=${self._daily_peak_equity:.2f}"
            )

    def record_trade_result(self, pnl: float) -> None:
        """Called after each trade closes to update discipline counters."""
        import time
        
        self._daily_pnl += pnl
        self._recent_results.append("win" if pnl > 0 else "loss")
        self._recent_results = self._recent_results[-20:]  # keep last 20

        if pnl <= 0:
            self._consecutive_losses += 1
            self._loss_streak_today += 1
            
            # Track timestamp for rapid loss detection
            self._rapid_loss_timestamps.append(time.time())
            # Keep only last 10 timestamps
            self._rapid_loss_timestamps = self._rapid_loss_timestamps[-10:]
        else:
            self._consecutive_losses = 0
            # Clear rapid loss timestamps on a win
            self._rapid_loss_timestamps = []

    def record_trade_opened(self) -> None:
        """Called when a trade is opened to increment daily counter."""
        self._daily_trades += 1

    # ── Main gate ─────────────────────────────────────────────────

    def check(
        self,
        signal: Dict[str, Any],
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str, Dict[str, float]]:
        """Run all discipline checks on a proposed trade signal.

        Args:
            signal:         Trade signal from strategy engine.
            market_data:    {symbol: DataFrame} for regime/correlation.
            current_prices: {symbol: float} for position valuation.
        Returns:
            (allowed, reason, adjustments)
            where adjustments is a dict of parameter modifications:
              - 'size_mult': multiply position size by this
              - 'tp_mult': multiply TP distance by this
              - 'sl_mult': multiply SL distance by this
              - 'confidence_adj': add this to confidence before threshold check
        """
        import time
        self._ensure_day()

        symbol = signal["symbol"]
        side = signal["side"]
        confidence = signal.get("confidence", 0.5)
        current_prices = current_prices or {}
        market_data = market_data or {}

        adjustments: Dict[str, float] = {
            "size_mult": 1.0,
            "tp_mult": 1.0,
            "sl_mult": 1.0,
            "confidence_adj": 0.0,
        }

        # ── 0. CIRCUIT BREAKERS (CRITICAL) ────────────────────────
        # Check for emergency halt
        if self._emergency_halt:
            return False, f"EMERGENCY HALT: {self._halt_reason}", adjustments
        
        # Check if we're in a pause period
        now = time.time()
        if now < self._pause_until:
            remaining = int(self._pause_until - now)
            return False, f"Trading paused for {remaining}s after rapid losses", adjustments
        
        # Check total loss circuit breaker
        ok, reason = self._check_circuit_breaker(current_prices)
        if not ok:
            return False, reason, adjustments
        
        # Check rapid loss protection
        ok, reason = self._check_rapid_losses()
        if not ok:
            return False, reason, adjustments

        # ── 1. DAILY DRAWDOWN KILL SWITCH ─────────────────────────
        ok, reason = self._check_daily_drawdown(current_prices)
        if not ok:
            return False, reason, adjustments

        # ── 2. MARKET SESSION / HOURS ─────────────────────────────
        ok, reason = self._check_session(symbol)
        if not ok:
            return False, reason, adjustments

        # Apply session confidence multiplier
        session_mult = session_confidence_multiplier(symbol)
        if session_mult < 1.0:
            adjustments["size_mult"] *= session_mult

        # ── 3. CORRELATION / CLUSTER EXPOSURE ─────────────────────
        ok, reason = self._check_correlation(signal, current_prices)
        if not ok:
            return False, reason, adjustments

        # ── 4. CONSECUTIVE LOSS COOL-DOWN ─────────────────────────
        ok, reason, loss_adj = self._check_loss_streak()
        if not ok:
            return False, reason, adjustments
        adjustments["size_mult"] *= loss_adj

        # ── 5. MAX TRADES PER DAY (mode-aware) ────────────────────
        ok, reason = self._check_daily_trade_limit()
        if not ok:
            return False, reason, adjustments
        
        # ── 5b. VELOCITY OPTIMIZATION ─────────────────────────────
        # In high-frequency modes, adjust confidence based on velocity
        vel_adjust = self._check_velocity_and_adjust()
        adjustments["confidence_adj"] += vel_adjust

        # ── 6. VOLUME / LIQUIDITY ─────────────────────────────────
        df = market_data.get(symbol)
        ok, reason = self._check_volume(symbol, df)
        if not ok:
            return False, reason, adjustments

        # ── 7. REGIME ADJUSTMENTS ─────────────────────────────────
        if df is not None:
            regime, regime_params = detect_regime(df)
            adjustments["tp_mult"] *= regime_params.get("tp_mult", 1.0)
            adjustments["sl_mult"] *= regime_params.get("sl_mult", 1.0)
            adjustments["size_mult"] *= regime_params.get("size_mult", 1.0)
            adjustments["confidence_adj"] += regime_params.get("confidence_adj", 0.0)

            # Regime minimum confidence gate
            min_conf = regime_params.get("min_confidence", 0.55)
            effective_conf = confidence + adjustments["confidence_adj"]
            if effective_conf < min_conf:
                reason = (
                    f"regime '{regime}' requires min confidence "
                    f"{min_conf:.2f}, got {effective_conf:.3f}"
                )
                logging.info(f"[Discipline] BLOCKED {symbol}: {reason}")
                return False, reason, adjustments

            signal["_regime"] = regime

        # ── 8. HIGH CORRELATION WARNING ───────────────────────────
        correlated = find_highly_correlated(
            symbol,
            self.portfolio.open_positions,
            market_data,
            threshold=CONFIG.get("correlation_warning_threshold", 0.75),
        )
        if correlated:
            # Reduce size proportionally to number of correlated positions
            corr_penalty = max(0.4, 1.0 - 0.2 * len(correlated))
            adjustments["size_mult"] *= corr_penalty
            logging.info(
                f"[Discipline] {symbol} correlated with {correlated} "
                f"→ size reduced to {corr_penalty*100:.0f}%"
            )

        logging.info(
            f"[Discipline] APPROVED {symbol} {side.upper()}  "
            f"size_mult={adjustments['size_mult']:.2f}  "
            f"tp_mult={adjustments['tp_mult']:.2f}  "
            f"sl_mult={adjustments['sl_mult']:.2f}  "
            f"conf_adj={adjustments['confidence_adj']:+.3f}"
        )

        return True, "approved", adjustments

    # ── Individual checks ─────────────────────────────────────────

    def _check_daily_drawdown(
        self,
        current_prices: Dict[str, float],
    ) -> Tuple[bool, str]:
        """Kill switch: stop all trading if daily drawdown exceeds limit."""
        max_dd_pct = CONFIG.get("max_daily_drawdown_pct", 0.03)  # 3% default

        equity = self.portfolio.total_equity(current_prices)
        if equity <= 0 or self._daily_peak_equity <= 0:
            return True, "ok"

        # Track intra-day peak
        if equity > self._daily_peak_equity:
            self._daily_peak_equity = equity

        daily_dd = (self._daily_peak_equity - equity) / self._daily_peak_equity

        if daily_dd >= max_dd_pct:
            reason = (
                f"DAILY DRAWDOWN KILL SWITCH: down {daily_dd*100:.2f}% "
                f"(limit={max_dd_pct*100:.1f}%)  "
                f"peak=${self._daily_peak_equity:.2f}  current=${equity:.2f}"
            )
            logging.warning(f"[Discipline] {reason}")
            return False, reason

        # Warn at 50% of limit
        if daily_dd >= max_dd_pct * 0.5:
            logging.warning(
                f"[Discipline] Daily drawdown WARNING: {daily_dd*100:.2f}% "
                f"(limit={max_dd_pct*100:.1f}%)"
            )

        return True, "ok"

    def _check_circuit_breaker(
        self,
        current_prices: Dict[str, float],
    ) -> Tuple[bool, str]:
        """CIRCUIT BREAKER: Halt trading if total loss exceeds threshold.
        
        This protects against catastrophic losses from flash crashes,
        bugs, or extreme market conditions.
        """
        max_total_loss_pct = CONFIG.get("max_total_loss_pct", 0.20)  # 20% default
        
        # Get starting capital (first equity value recorded)
        starting_capital = getattr(self.portfolio, 'starting_capital', CONFIG["capital"])
        if starting_capital <= 0:
            starting_capital = CONFIG["capital"]
        
        current_equity = self.portfolio.total_equity(current_prices)
        if current_equity <= 0:
            current_equity = self.portfolio.cash
        
        # Calculate total loss from start
        total_loss_pct = (starting_capital - current_equity) / starting_capital
        
        if total_loss_pct >= max_total_loss_pct:
            self._emergency_halt = True
            self._halt_reason = (
                f"Total loss {total_loss_pct:.1%} >= {max_total_loss_pct:.1%}. "
                f"Starting: ${starting_capital:.2f}, Current: ${current_equity:.2f}"
            )
            logging.critical(
                f"[CIRCUIT BREAKER] EMERGENCY HALT - {self._halt_reason}"
            )
            try:
                from alerts.sms_notifier import send_alert
                send_alert(f"CIRCUIT BREAKER FIRED: {self._halt_reason}")
            except Exception:
                pass
            return False, f"CIRCUIT BREAKER: {self._halt_reason}"
        
        # Warn at 75% of threshold
        if total_loss_pct >= max_total_loss_pct * 0.75:
            logging.warning(
                f"[CIRCUIT BREAKER] WARNING: Total loss {total_loss_pct:.1%} "
                f"approaching {max_total_loss_pct:.1%} limit"
            )
        
        return True, "ok"
    
    def _check_rapid_losses(self) -> Tuple[bool, str]:
        """Rapid loss protection: Pause trading after multiple quick losses.
        
        This catches trading bugs, bad market conditions, or strategy failures
        before they cause significant damage.
        """
        import time
        
        threshold = CONFIG.get("rapid_loss_threshold", 5)  # 5 consecutive losses
        window_sec = CONFIG.get("rapid_loss_window_sec", 600)  # 10 minutes
        pause_sec = CONFIG.get("rapid_loss_pause_sec", 3600)  # 1 hour pause
        
        # Check if we have enough loss history
        if len(self._recent_results) < threshold:
            return True, "ok"
        
        # Get last N results
        last_n = self._recent_results[-threshold:]
        
        # Check if all are losses
        if all(r == "loss" for r in last_n):
            now = time.time()
            
            # Check if these losses happened within the time window
            if self._rapid_loss_timestamps:
                oldest_loss = self._rapid_loss_timestamps[0] if len(self._rapid_loss_timestamps) >= threshold else 0
                time_span = now - oldest_loss
                
                if time_span < window_sec:
                    # Rapid losses detected - pause trading
                    self._pause_until = now + pause_sec
                    reason = (
                        f"RAPID LOSS PROTECTION: {threshold} consecutive losses "
                        f"in {time_span/60:.1f} minutes. Pausing for {pause_sec/60:.0f} min."
                    )
                    logging.critical(f"[Discipline] {reason}")
                    try:
                        from alerts.sms_notifier import send_alert
                        send_alert(
                            f"RAPID LOSS PROTECTION: {threshold} consecutive losses "
                            f"in {time_span/60:.1f}min. Trading paused {pause_sec//60}min."
                        )
                    except Exception:
                        pass

                    # Clear rapid loss timestamps
                    self._rapid_loss_timestamps = []
                    
                    return False, reason
        
        return True, "ok"

    def _check_session(self, symbol: str) -> Tuple[bool, str]:
        """Enforce market hours for stocks."""
        if CONFIG.get("backtest_mode", False):
            return True, "backtest_mode"

        allowed, reason = is_trading_allowed(symbol)
        if not allowed:
            logging.info(f"[Discipline] BLOCKED {symbol}: {reason}")
            return False, f"session: {reason}"
        return True, "ok"

    def _check_correlation(
        self,
        signal: Dict[str, Any],
        current_prices: Dict[str, float],
    ) -> Tuple[bool, str]:
        """Enforce cluster and asset-class exposure limits."""
        symbol = signal["symbol"]
        side = signal["side"]
        entry = signal.get("entry_price", 0)
        equity = self.portfolio.total_equity(current_prices)

        if equity <= 0 or entry <= 0:
            return True, "ok"

        # Estimate position size as % of equity
        risk_pct = CONFIG.get("risk_per_trade_pct", 0.02)
        position_pct = min(risk_pct * 3, CONFIG.get("max_position_pct", 0.25))

        return can_add_to_cluster(
            symbol, side, position_pct,
            self.portfolio.open_positions,
            current_prices, equity,
        )

    def _check_loss_streak(self) -> Tuple[bool, str, float]:
        """Adaptive behavior after consecutive losses.

        Returns:
            (allowed, reason, size_multiplier)
        """
        max_streak = CONFIG.get("max_consecutive_losses_pause", 4)
        streak_reduce_at = CONFIG.get("loss_streak_reduce_at", 2)

        # Hard pause after N consecutive losses
        if self._consecutive_losses >= max_streak:
            reason = (
                f"LOSS STREAK PAUSE: {self._consecutive_losses} consecutive losses "
                f"(limit={max_streak}). Wait for a winning trade or new session."
            )
            logging.warning(f"[Discipline] {reason}")
            return False, reason, 1.0

        # Reduce size after streak_reduce_at losses
        if self._consecutive_losses >= streak_reduce_at:
            # Reduce by 25% per loss beyond the threshold
            reductions = self._consecutive_losses - streak_reduce_at + 1
            size_mult = max(0.25, 1.0 - 0.25 * reductions)
            logging.info(
                f"[Discipline] Loss streak ({self._consecutive_losses}): "
                f"reducing size to {size_mult*100:.0f}%"
            )
            return True, "loss_streak_size_reduction", size_mult

        return True, "ok", 1.0

    def _check_daily_trade_limit(self) -> Tuple[bool, str]:
        """Cap the number of trades per day to prevent overtrading (mode-aware)."""
        max_trades = get_mode_config("max_trades_per_day")

        if self._daily_trades >= max_trades:
            reason = (
                f"MAX DAILY TRADES: {self._daily_trades} trades today "
                f"(limit={max_trades})"
            )
            logging.warning(f"[Discipline] {reason}")
            return False, reason

        return True, "ok"
    
    def _check_velocity_and_adjust(self) -> float:
        """Check trading velocity and suggest confidence adjustment.
        
        Inspired by Claude's high-frequency success: if we're trading too slowly,
        slightly reduce threshold to capture more opportunities.
        
        Returns:
            Confidence adjustment (negative = easier to trade, positive = harder)
        """
        import time
        now = time.time()
        
        # Only check velocity every 5 minutes
        if now - self._last_velocity_check < 300:
            return 0.0
        
        self._last_velocity_check = now
        
        # Check if we should adjust
        should_adjust, adjustment = self.velocity_optimizer.should_reduce_confidence_threshold()
        
        if should_adjust:
            mode = CONFIG.get("trading_mode", "balanced")
            # Only auto-adjust in aggressive and high-frequency modes
            if mode in ["aggressive", "claude_hf"]:
                logging.info(
                    f"[Discipline] Velocity optimizer suggests confidence adjustment: "
                    f"{adjustment:+.3f}"
                )
                return adjustment
        
        return 0.0

    def _check_volume(
        self,
        symbol: str,
        df: Optional[pd.DataFrame],
    ) -> Tuple[bool, str]:
        """Check that recent volume is sufficient for safe execution.

        Rejects trades on symbols where volume has dried up (below 20th
        percentile of their own recent history), which signals thin
        liquidity and potential slippage.
        """
        if df is None or "Volume" not in df.columns or len(df) < 20:
            return True, "ok"  # no data to judge — don't block

        vol = df["Volume"].tail(20)
        current_vol = vol.iloc[-1]
        median_vol = vol.median()

        if median_vol <= 0:
            return True, "ok"

        vol_ratio = current_vol / median_vol

        min_vol_ratio = CONFIG.get("min_volume_ratio", 0.20)
        if vol_ratio < min_vol_ratio:
            reason = (
                f"LOW VOLUME: {symbol} current volume is {vol_ratio:.1%} of "
                f"20-bar median (min={min_vol_ratio:.0%})"
            )
            logging.info(f"[Discipline] BLOCKED: {reason}")
            return False, reason

        return True, "ok"

    # ── Overnight risk management ─────────────────────────────────

    def get_positions_to_flatten(self) -> List[Dict]:
        """Return stock positions that should be closed before market close.

        Called by the main loop near end of day.  Professional day traders
        don't hold stock positions overnight — gap risk is unmanaged.
        """
        if CONFIG.get("backtest_mode", False):
            return []

        flatten = []
        for pos in self.portfolio.open_positions:
            if should_flatten_overnight(pos["symbol"]):
                flatten.append(pos)
                logging.info(
                    f"[Discipline] Flagging {pos['symbol']} for overnight flatten "
                    f"(entered @ ${pos['entry_price']:.2f})"
                )
        return flatten

    # ── Stats for logging / dashboard ─────────────────────────────

    def get_status(self, current_prices: Optional[Dict[str, float]] = None) -> Dict:
        """Return a snapshot of the discipline gate's state."""
        self._ensure_day()
        prices = current_prices or {}
        equity = self.portfolio.total_equity(prices)
        dd = 0.0
        if self._daily_peak_equity > 0:
            dd = (self._daily_peak_equity - equity) / self._daily_peak_equity

        return {
            "date": str(self._today),
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_trades": self._daily_trades,
            "consecutive_losses": self._consecutive_losses,
            "daily_drawdown_pct": round(dd * 100, 2),
            "daily_peak_equity": round(self._daily_peak_equity, 2),
            "current_equity": round(equity, 2),
            "kill_switch_active": dd >= CONFIG.get("max_daily_drawdown_pct", 0.03),
        }

    def reset_loss_streak(self) -> None:
        """Manually reset the consecutive loss counter (e.g. new day)."""
        self._consecutive_losses = 0
        self._loss_streak_today = 0
        logging.info("[Discipline] Loss streak manually reset")
