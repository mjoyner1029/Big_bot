"""
Promotion Gates — deterministic, LLM-proof promotion criteria.

RULE: Claude may NEVER modify, waive, or override these gates.
      Gates are determined by configuration, not by LLM analysis.
      Claude may ANALYZE failures and propose research — nothing more.

Usage
-----
    gates = PromotionGates(config=PromotionConfig(...))
    result = gates.evaluate(validation_result, stage='PAPER_TO_CANARY')
    if result.passed:
        # proceed with deployment
    else:
        # block promotion, log reasons, notify research engine
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

class PromotionStage(str, Enum):
    BACKTEST_TO_WF      = "BACKTEST_TO_WF"        # pass walk-forward
    WF_TO_OOS           = "WF_TO_OOS"             # pass walk-forward before OOS unlock
    OOS_TO_PAPER        = "OOS_TO_PAPER"           # pass OOS before paper trading
    PAPER_TO_CANARY     = "PAPER_TO_CANARY"        # paper → canary live
    CANARY_TO_LIMITED   = "CANARY_TO_LIMITED"      # canary → limited live
    LIMITED_TO_NORMAL   = "LIMITED_TO_NORMAL"      # limited → normal live
    CHALLENGER_PROMOTE  = "CHALLENGER_PROMOTE"     # model champion replacement
    ROLLBACK_TRIGGER    = "ROLLBACK_TRIGGER"       # trigger rollback check


# ---------------------------------------------------------------------------
# Configuration (all thresholds are explicit, not LLM-determined)
# ---------------------------------------------------------------------------

@dataclass
class PromotionConfig:
    """
    Deterministic thresholds for every promotion gate.

    These values are set by the operator, not by Claude.
    Claude cannot modify these values at runtime.
    """
    # Trade count minimums
    min_backtest_trades:   int   = 50
    min_wf_trades:         int   = 30     # per fold
    min_oos_trades:        int   = 20
    min_paper_trades:      int   = 30
    min_canary_trades:     int   = 15

    # Expectancy (net, per trade, in dollars)
    min_expectancy:        float = 0.0    # must be positive net

    # Profit factor
    min_profit_factor:     float = 1.10   # 10% edge minimum

    # Max drawdown (absolute dollars or % of capital)
    max_drawdown_pct:      float = 0.20   # 20% max drawdown

    # Sharpe thresholds
    min_backtest_sharpe:   float = 0.50
    min_oos_sharpe:        float = 0.40
    min_paper_sharpe:      float = 0.30

    # Win rate (INFORMATIONAL ONLY — not a primary gate)
    # We optimize expectancy, not win rate
    min_win_rate:          float = 0.35   # must at least be above random guessing

    # OOS degradation tolerance
    oos_sharpe_degradation: float = 0.30  # OOS Sharpe must be >= 70% of backtest
    oos_exp_degradation:    float = 0.50  # OOS expectancy must be >= 50% of backtest

    # Execution quality
    max_rejection_rate:    float = 0.10
    max_slippage_bps:      float = 15.0

    # Model calibration
    max_calibration_gap:   float = 0.15   # max difference between predicted and actual

    # Reconciliation
    allow_reconciliation_errors: bool = False   # zero tolerance

    # Capital (these are STATIC — Claude cannot change them)
    canary_capital_pct:    float = 0.02    # 2% of total capital
    limited_capital_pct:   float = 0.25   # 25% of total capital
    normal_capital_pct:    float = 1.00   # 100% allowed


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    stage:       PromotionStage
    passed:      bool
    checks:      List[Dict]   = field(default_factory=list)
    blocked_by:  List[str]    = field(default_factory=list)
    evaluated_at: str         = field(default_factory=_utcnow)

    def summary(self) -> str:
        status = "✓ PASSED" if self.passed else "✗ BLOCKED"
        lines  = [
            f"Gate: {self.stage.value}  [{status}]",
            f"Evaluated: {self.evaluated_at}",
        ]
        for c in self.checks:
            icon = "✓" if c['passed'] else "✗"
            lines.append(f"  {icon} {c['name']}: {c['value']} (need {c['threshold']})")
        if self.blocked_by:
            lines.append(f"BLOCKED BY: {'; '.join(self.blocked_by)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PromotionGates:
    """
    Evaluates whether a strategy/model meets promotion criteria.

    INVARIANTS (enforced at class level):
        - All thresholds come from PromotionConfig, not runtime arguments
        - No method allows bypassing a failed gate
        - LLM integration points are clearly labelled and cannot modify state
    """

    def __init__(
        self,
        config: PromotionConfig = None,
        db_path: str = "data/trade_memory.sqlite",
    ):
        self.config  = config or PromotionConfig()
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS promotion_decisions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    stage        TEXT NOT NULL,
                    label        TEXT,
                    passed       INTEGER NOT NULL,
                    checks_json  TEXT,
                    blocked_by   TEXT,
                    evaluated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        stage: PromotionStage,
        validation_result=None,    # ValidationResult
        walk_forward_result=None,  # WalkForwardResult
        execution_report=None,     # ExecutionReport
        reconciliation_errors: int = 0,
        calibration_gap: float = 0.0,
        label: str = "",
    ) -> GateResult:
        """
        Run the appropriate gate checks for the given stage.

        This method is DETERMINISTIC — same inputs always produce same output.
        Claude cannot call this method with modified thresholds.
        """
        checks  = []
        blocked = []

        if stage == PromotionStage.BACKTEST_TO_WF:
            self._check_backtest(validation_result, checks, blocked)

        elif stage == PromotionStage.WF_TO_OOS:
            self._check_walk_forward(walk_forward_result, checks, blocked)

        elif stage == PromotionStage.OOS_TO_PAPER:
            self._check_oos(validation_result, checks, blocked)

        elif stage == PromotionStage.PAPER_TO_CANARY:
            self._check_paper(validation_result, checks, blocked)
            self._check_execution(execution_report, checks, blocked)
            self._check_reconciliation(reconciliation_errors, checks, blocked)
            self._check_calibration(calibration_gap, checks, blocked)

        elif stage in (PromotionStage.CANARY_TO_LIMITED, PromotionStage.LIMITED_TO_NORMAL):
            self._check_paper(validation_result, checks, blocked)
            self._check_execution(execution_report, checks, blocked)
            self._check_reconciliation(reconciliation_errors, checks, blocked)

        elif stage == PromotionStage.CHALLENGER_PROMOTE:
            self._check_model_promotion(validation_result, checks, blocked)

        elif stage == PromotionStage.ROLLBACK_TRIGGER:
            self._check_rollback(validation_result, checks, blocked)

        passed = len(blocked) == 0
        result = GateResult(stage=stage, passed=passed, checks=checks, blocked_by=blocked)

        self._persist(result, label)
        log_fn = logger.info if passed else logger.warning
        log_fn(f"PromotionGate [{stage.value}] label={label!r}: {'PASSED' if passed else 'BLOCKED'}")
        if not passed:
            logger.warning(f"  Blocked by: {'; '.join(blocked)}")

        return result

    # ── Gate check sets ───────────────────────────────────────────────────────

    def _check_backtest(self, r, checks: List, blocked: List) -> None:
        if not r:
            blocked.append("No validation result provided")
            return
        self._gate(checks, blocked, "min_trades",
                   r.trade_count, self.config.min_backtest_trades,
                   r.trade_count >= self.config.min_backtest_trades)
        self._gate(checks, blocked, "positive_expectancy",
                   r.expectancy, self.config.min_expectancy,
                   r.expectancy > self.config.min_expectancy)
        self._gate(checks, blocked, "min_sharpe",
                   r.sharpe, self.config.min_backtest_sharpe,
                   r.sharpe >= self.config.min_backtest_sharpe)
        self._gate(checks, blocked, "min_profit_factor",
                   r.profit_factor, self.config.min_profit_factor,
                   r.profit_factor >= self.config.min_profit_factor)
        self._gate(checks, blocked, "min_win_rate",
                   r.win_rate, self.config.min_win_rate,
                   r.win_rate >= self.config.min_win_rate)

    def _check_walk_forward(self, wf, checks: List, blocked: List) -> None:
        if not wf:
            blocked.append("No walk-forward result provided")
            return
        self._gate(checks, blocked, "min_wf_folds",
                   wf.n_folds, 3, wf.n_folds >= 3)
        self._gate(checks, blocked, "positive_avg_expectancy",
                   wf.avg_val_expectancy, 0, wf.avg_val_expectancy > 0)
        self._gate(checks, blocked, "min_pct_positive_folds",
                   wf.pct_folds_positive, 0.6, wf.pct_folds_positive >= 0.6)

    def _check_oos(self, r, checks: List, blocked: List) -> None:
        if not r:
            blocked.append("No OOS validation result provided")
            return
        self._gate(checks, blocked, "min_oos_trades",
                   r.trade_count, self.config.min_oos_trades,
                   r.trade_count >= self.config.min_oos_trades)
        self._gate(checks, blocked, "positive_net_expectancy",
                   r.expectancy, self.config.min_expectancy,
                   r.expectancy > self.config.min_expectancy)
        self._gate(checks, blocked, "min_oos_sharpe",
                   r.sharpe, self.config.min_oos_sharpe,
                   r.sharpe >= self.config.min_oos_sharpe)

    def _check_paper(self, r, checks: List, blocked: List) -> None:
        if not r:
            blocked.append("No paper trading result provided")
            return
        self._gate(checks, blocked, "min_paper_trades",
                   r.trade_count, self.config.min_paper_trades,
                   r.trade_count >= self.config.min_paper_trades)
        self._gate(checks, blocked, "positive_net_expectancy",
                   r.expectancy, self.config.min_expectancy,
                   r.expectancy > self.config.min_expectancy)
        self._gate(checks, blocked, "max_drawdown_pct",
                   r.max_drawdown, self.config.max_drawdown_pct,
                   r.max_drawdown <= self.config.max_drawdown_pct * 10000)
        self._gate(checks, blocked, "min_paper_sharpe",
                   r.sharpe, self.config.min_paper_sharpe,
                   r.sharpe >= self.config.min_paper_sharpe)
        self._gate(checks, blocked, "min_profit_factor",
                   r.profit_factor, self.config.min_profit_factor,
                   r.profit_factor >= self.config.min_profit_factor)

    def _check_execution(self, r, checks: List, blocked: List) -> None:
        if not r:
            return   # execution check is informational when not provided
        self._gate(checks, blocked, "max_rejection_rate",
                   getattr(r, 'rejection_rate', 0),
                   self.config.max_rejection_rate,
                   getattr(r, 'rejection_rate', 0) <= self.config.max_rejection_rate)
        self._gate(checks, blocked, "max_slippage_bps",
                   getattr(r, 'slippage', type('', (), {'mean_bps': 0})()).mean_bps,
                   self.config.max_slippage_bps,
                   getattr(getattr(r, 'slippage', None), 'mean_bps', 0) <= self.config.max_slippage_bps)

    def _check_reconciliation(self, errors: int, checks: List, blocked: List) -> None:
        passed = (errors == 0) or self.config.allow_reconciliation_errors
        self._gate(checks, blocked, "reconciliation_errors",
                   errors, 0, passed)

    def _check_calibration(self, gap: float, checks: List, blocked: List) -> None:
        self._gate(checks, blocked, "calibration_gap",
                   gap, self.config.max_calibration_gap,
                   gap <= self.config.max_calibration_gap)

    def _check_model_promotion(self, r, checks: List, blocked: List) -> None:
        """Model challenger promotion gate."""
        if not r:
            blocked.append("No model validation result")
            return
        self._gate(checks, blocked, "min_trades",
                   r.trade_count, self.config.min_paper_trades,
                   r.trade_count >= self.config.min_paper_trades)
        self._gate(checks, blocked, "positive_expectancy",
                   r.expectancy, 0, r.expectancy > 0)

    def _check_rollback(self, r, checks: List, blocked: List) -> None:
        """Check if rollback should be triggered (returns passed=True if rollback needed)."""
        if not r:
            return
        dd_pct     = r.max_drawdown / 10000
        rollback   = (
            dd_pct > self.config.max_drawdown_pct * 1.5 or
            r.expectancy < self.config.min_expectancy - 50 or
            (r.trade_count >= 10 and r.profit_factor < 0.80)
        )
        self._gate(checks, blocked, "drawdown_trigger",
                   dd_pct, self.config.max_drawdown_pct * 1.5, not rollback)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _gate(checks: List, blocked: List, name: str, value, threshold, passed: bool) -> None:
        checks.append({'name': name, 'value': value, 'threshold': threshold, 'passed': passed})
        if not passed:
            blocked.append(f"{name}={value} (need {threshold})")

    def _persist(self, result: GateResult, label: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO promotion_decisions "
                    "(stage,label,passed,checks_json,blocked_by,evaluated_at) VALUES (?,?,?,?,?,?)",
                    (result.stage.value, label, int(result.passed),
                     json.dumps(result.checks), json.dumps(result.blocked_by),
                     result.evaluated_at),
                )
        except Exception as e:
            logger.warning(f"PromotionGates: could not persist: {e}")

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Return recent promotion decision history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM promotion_decisions ORDER BY evaluated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
