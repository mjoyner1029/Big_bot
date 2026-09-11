"""
Shadow Mode — live market data, real decisions, NO order submission.

TRADING MODES
─────────────
BACKTEST   Historical replay
PAPER      Simulated live (orders simulated locally)
SHADOW     Real infrastructure, real decisions, NO orders
LIVE       Real capital

Shadow mode allows the production decision pipeline to run
against live market data without any financial exposure.

All shadow decisions, hypothetical fills, and outcomes are
permanently recorded for later analysis.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


class TradingMode(str, Enum):
    BACKTEST = "BACKTEST"
    PAPER    = "PAPER"
    SHADOW   = "SHADOW"
    LIVE     = "LIVE"


@dataclass
class ShadowDecision:
    """A decision made in shadow mode — recorded but NOT executed."""
    decision_id:      str
    symbol:           str
    action:           str       # ENTER | EXIT | NO_TRADE | SKIP
    strategy:         str
    direction:        str
    size:             float
    signal_price:     float
    reason:           str
    meta_prediction:  str
    meta_confidence:  float
    opportunity_score: float
    regime:           str
    decided_at:       str = field(default_factory=_utcnow)
    hypothetical_fill: Optional[float] = None
    hypothetical_pnl:  Optional[float] = None
    outcome_recorded:  bool = False


_CREATE_SHADOW = """
CREATE TABLE IF NOT EXISTS shadow_decisions (
    decision_id       TEXT PRIMARY KEY,
    symbol            TEXT NOT NULL,
    action            TEXT NOT NULL,
    strategy          TEXT,
    direction         TEXT,
    size              REAL DEFAULT 0,
    signal_price      REAL DEFAULT 0,
    reason            TEXT,
    meta_prediction   TEXT,
    meta_confidence   REAL DEFAULT 0,
    opportunity_score REAL DEFAULT 0,
    regime            TEXT,
    decided_at        TEXT NOT NULL,
    hypothetical_fill REAL,
    hypothetical_pnl  REAL,
    outcome_recorded  INTEGER DEFAULT 0
)
"""


class ShadowModeTracker:
    """
    Records all trading decisions made in SHADOW mode.

    Shadow mode: real infrastructure + real decisions + NO order submission.

    Use this to:
        - Validate the decision pipeline before committing capital
        - Accumulate forward-test performance data without risk
        - Compare shadow results vs paper/live results
    """

    def __init__(self, db_path: str = "data/trade_memory.sqlite"):
        self.db_path = db_path
        self._active: bool = False
        self._mode: TradingMode = TradingMode.PAPER
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_SHADOW)
            conn.commit()

    # ── Mode control ─────────────────────────────────────────────────────────

    def set_mode(self, mode: TradingMode) -> None:
        """Set the trading mode. Only SHADOW mode activates recording."""
        self._mode = mode
        self._active = (mode == TradingMode.SHADOW)
        logger.info(f"ShadowMode: mode set to {mode.value}")

    @property
    def is_shadow(self) -> bool:
        return self._active

    @property
    def mode(self) -> TradingMode:
        return self._mode

    def should_submit_order(self) -> bool:
        """Returns True only in PAPER or LIVE mode — never in SHADOW."""
        return self._mode in (TradingMode.PAPER, TradingMode.LIVE)

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_decision(
        self,
        symbol:            str,
        action:            str,
        strategy:          str = '',
        direction:         str = 'LONG',
        size:              float = 0.0,
        signal_price:      float = 0.0,
        reason:            str = '',
        meta_prediction:   str = 'UNKNOWN',
        meta_confidence:   float = 0.0,
        opportunity_score: float = 0.0,
        regime:            str = '',
    ) -> Optional[str]:
        """
        Record a trading decision. Returns decision_id or None if not in SHADOW mode.

        In PAPER/LIVE mode, this is a no-op (orders go through broker instead).
        """
        if not self._active:
            return None

        decision_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO shadow_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (decision_id, symbol, action, strategy, direction, size, signal_price,
                 reason, meta_prediction, meta_confidence, opportunity_score, regime,
                 _utcnow(), None, None, 0),
            )
            conn.commit()

        logger.debug(f"ShadowMode: recorded {action} {symbol} @ {signal_price:.4f} [{decision_id[:8]}]")
        return decision_id

    def record_outcome(
        self,
        decision_id:      str,
        hypothetical_fill: float,
        hypothetical_pnl:  float,
    ) -> None:
        """Record the hypothetical outcome of a shadow decision."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE shadow_decisions SET hypothetical_fill=?,hypothetical_pnl=?,outcome_recorded=1 "
                "WHERE decision_id=?",
                (hypothetical_fill, hypothetical_pnl, decision_id),
            )
            conn.commit()

    # ── Analysis ──────────────────────────────────────────────────────────────

    def get_decisions(self, days: int = 7, action: str = None) -> List[Dict]:
        cutoff = (datetime.now(timezone.utc) -
                  __import__('datetime').timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if action:
                rows = conn.execute(
                    "SELECT * FROM shadow_decisions WHERE decided_at>=? AND action=? ORDER BY decided_at DESC",
                    (cutoff, action),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM shadow_decisions WHERE decided_at>=? ORDER BY decided_at DESC",
                    (cutoff,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_performance_summary(self, days: int = 30) -> Dict:
        """Summarise shadow P&L and decision quality."""
        decisions = self.get_decisions(days=days, action='ENTER')
        with_outcomes = [d for d in decisions if d.get('outcome_recorded')]

        pnls = [d['hypothetical_pnl'] for d in with_outcomes if d.get('hypothetical_pnl') is not None]
        if not pnls:
            return {'decisions': len(decisions), 'pnl_recorded': 0, 'total_pnl': 0}

        return {
            'decisions':      len(decisions),
            'pnl_recorded':   len(pnls),
            'total_pnl':      sum(pnls),
            'win_rate':       sum(1 for p in pnls if p > 0) / len(pnls),
            'expectancy':     sum(pnls) / len(pnls),
            'avg_confidence': sum(d.get('meta_confidence', 0) for d in with_outcomes) / len(with_outcomes),
        }

    def compare_with_paper(self, paper_result: Dict) -> Dict:
        """Compare shadow performance with live paper performance."""
        shadow = self.get_performance_summary()
        return {
            'shadow_expectancy': shadow.get('expectancy', 0),
            'paper_expectancy':  paper_result.get('expectancy', 0),
            'shadow_win_rate':   shadow.get('win_rate', 0),
            'paper_win_rate':    paper_result.get('win_rate', 0),
            'decisions_vs_trades': (shadow.get('decisions', 0), paper_result.get('total_trades', 0)),
        }
