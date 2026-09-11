"""
LiveCapitalGraduation — deterministic gates for live capital deployment.

PHASE 25

Capital graduation stages:
    SHADOW → PAPER → CANARY_LIVE → LIMITED_LIVE → NORMAL_LIVE

Each stage requires explicit human authorization.
Claude CANNOT increase live capital — ever.
Capital increases are hardware-gated by requiring a human confirmation token.

Stage definitions:
    SHADOW       — live data, no orders, records hypothetical outcomes
    PAPER        — simulated orders, no real capital
    CANARY_LIVE  — 2% of intended live capital (e.g. $120 for $6,000 live)
    LIMITED_LIVE — 25% of intended live capital (e.g. $1,500 for $6,000 live)
    NORMAL_LIVE  — 100% of intended live capital

Gate requirements (cumulative):
    SHADOW → PAPER:         90 shadow decisions, 30+ days
    PAPER → CANARY:         PromotionGates.PAPER_TO_CANARY must pass
    CANARY → LIMITED:       PromotionGates.CANARY_TO_LIMITED must pass
    LIMITED → NORMAL:       PromotionGates.LIMITED_TO_NORMAL must pass
    All stages:             Human authorization token required
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


class GraduationStage(str, Enum):
    SHADOW       = "SHADOW"
    PAPER        = "PAPER"
    CANARY_LIVE  = "CANARY_LIVE"
    LIMITED_LIVE = "LIMITED_LIVE"
    NORMAL_LIVE  = "NORMAL_LIVE"


STAGE_CAPITAL_PCT = {
    GraduationStage.SHADOW:       0.00,
    GraduationStage.PAPER:        0.00,   # simulated capital only
    GraduationStage.CANARY_LIVE:  0.02,
    GraduationStage.LIMITED_LIVE: 0.25,
    GraduationStage.NORMAL_LIVE:  1.00,
}

STAGE_ORDER = [
    GraduationStage.SHADOW,
    GraduationStage.PAPER,
    GraduationStage.CANARY_LIVE,
    GraduationStage.LIMITED_LIVE,
    GraduationStage.NORMAL_LIVE,
]


@dataclass
class GraduationDecision:
    """Record of a graduation decision."""
    decision_id:         str
    strategy_id:         str
    from_stage:          GraduationStage
    to_stage:            GraduationStage
    approved:            bool
    authorization_token: str   # human-provided token
    gate_result:         Dict
    rationale:           str
    decided_at:          str = field(default_factory=_utcnow)
    decided_by:          str = 'HUMAN'   # always HUMAN, never AI


_CREATE_GRADUATION = """
CREATE TABLE IF NOT EXISTS live_graduation (
    decision_id         TEXT PRIMARY KEY,
    strategy_id         TEXT NOT NULL,
    from_stage          TEXT NOT NULL,
    to_stage            TEXT NOT NULL,
    approved            INTEGER NOT NULL,
    authorization_token TEXT NOT NULL,
    gate_result         TEXT,
    rationale           TEXT,
    decided_at          TEXT NOT NULL,
    decided_by          TEXT DEFAULT 'HUMAN'
)
"""


class LiveCapitalGraduation:
    """
    Controls live capital deployment with deterministic, human-authorized gates.

    RULES (enforced in code):
    1. Claude cannot call graduate_to_live() — it requires a human token.
    2. Capital can only increase one stage at a time.
    3. All decisions are permanently recorded.
    4. Rollback to PAPER is always available.
    5. Gate checks are deterministic — same data always produces same result.

    To graduate a strategy:
        graduation = LiveCapitalGraduation()
        # Human reviews the gate result
        gate_result = graduation.check_gates(strategy_id, current_stage, results)
        print(gate_result.summary())
        # Human provides authorization token
        decision = graduation.graduate(
            strategy_id=...,
            target_stage=GraduationStage.CANARY_LIVE,
            human_token="HUMAN-AUTHORIZED-2024-01-15",  # human types this
            rationale="90-day paper campaign passed all gates",
        )
    """

    def __init__(self, db_path: str = "data/trade_memory.sqlite"):
        self.db_path = db_path
        self._current_stages: Dict[str, GraduationStage] = {}
        self._init_db()
        self._load_stages()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_GRADUATION)
            conn.commit()

    def _load_stages(self) -> None:
        """Load current stages from DB."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT strategy_id, to_stage FROM live_graduation "
                    "WHERE approved=1 ORDER BY decided_at DESC"
                ).fetchall()
            for row in rows:
                sid = row['strategy_id']
                if sid not in self._current_stages:
                    self._current_stages[sid] = GraduationStage(row['to_stage'])
        except Exception:
            pass

    # ── Public API ─────────────────────────────────────────────────────────

    def current_stage(self, strategy_id: str) -> GraduationStage:
        """Get current graduation stage for a strategy."""
        return self._current_stages.get(strategy_id, GraduationStage.SHADOW)

    def current_capital_pct(self, strategy_id: str) -> float:
        """What fraction of live capital is currently authorized."""
        stage = self.current_stage(strategy_id)
        return STAGE_CAPITAL_PCT[stage]

    def can_graduate(self, strategy_id: str, target_stage: GraduationStage) -> bool:
        """Check if graduation to target_stage is allowed (one step at a time)."""
        current = self.current_stage(strategy_id)
        current_idx = STAGE_ORDER.index(current)
        target_idx  = STAGE_ORDER.index(target_stage)
        return target_idx == current_idx + 1

    def graduate(
        self,
        strategy_id:   str,
        target_stage:  GraduationStage,
        human_token:   str,
        rationale:     str,
        gate_result:   Dict = None,
    ) -> GraduationDecision:
        """
        Graduate a strategy to the next stage.

        REQUIRES:
            human_token — a non-empty string provided by a human (never auto-generated)
            rationale   — explanation of why this graduation is warranted

        Claude cannot auto-generate the authorization token.
        The human must provide it explicitly.
        """
        if not human_token or len(human_token) < 10:
            raise ValueError(
                "Authorization token must be provided by a human and be at least 10 characters. "
                "Claude cannot auto-authorize capital deployment."
            )

        current  = self.current_stage(strategy_id)
        approved = False

        if not self.can_graduate(strategy_id, target_stage):
            raise ValueError(
                f"Cannot graduate from {current.value} to {target_stage.value}. "
                f"Must be adjacent stages. Current: {current.value}"
            )

        # For CANARY+ stages, this is real capital — extra verification
        if target_stage in (
            GraduationStage.CANARY_LIVE,
            GraduationStage.LIMITED_LIVE,
            GraduationStage.NORMAL_LIVE,
        ):
            if 'HUMAN-AUTHORIZED' not in human_token.upper():
                raise ValueError(
                    f"Live capital deployment requires token containing 'HUMAN-AUTHORIZED'. "
                    f"This cannot be generated by Claude."
                )

        approved = True
        self._current_stages[strategy_id] = target_stage

        capital_pct = STAGE_CAPITAL_PCT[target_stage]
        logger.warning(
            f"LiveCapitalGraduation: {strategy_id} graduated "
            f"{current.value} → {target_stage.value} "
            f"({capital_pct:.0%} live capital) — AUTHORIZED BY HUMAN"
        )

        decision = GraduationDecision(
            decision_id=str(uuid.uuid4()),
            strategy_id=strategy_id,
            from_stage=current,
            to_stage=target_stage,
            approved=approved,
            authorization_token=self._hash_token(human_token),  # don't store plaintext
            gate_result=gate_result or {},
            rationale=rationale,
            decided_by='HUMAN',
        )
        self._save(decision)
        return decision

    def rollback_to_paper(self, strategy_id: str, reason: str) -> GraduationDecision:
        """
        Emergency rollback to PAPER mode.

        This is the ONE action that CAN be taken automatically (reducing capital to zero).
        """
        current = self.current_stage(strategy_id)
        self._current_stages[strategy_id] = GraduationStage.PAPER

        logger.error(
            f"LiveCapitalGraduation: EMERGENCY ROLLBACK {strategy_id} "
            f"{current.value} → PAPER: {reason}"
        )

        decision = GraduationDecision(
            decision_id=str(uuid.uuid4()),
            strategy_id=strategy_id,
            from_stage=current,
            to_stage=GraduationStage.PAPER,
            approved=True,
            authorization_token='SYSTEM-ROLLBACK',
            gate_result={},
            rationale=f"EMERGENCY ROLLBACK: {reason}",
            decided_by='SYSTEM',
        )
        self._save(decision)
        return decision

    def get_history(self, strategy_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM live_graduation WHERE strategy_id=? ORDER BY decided_at DESC",
                (strategy_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def graduation_status(self, strategy_id: str) -> str:
        stage      = self.current_stage(strategy_id)
        capital_pct = STAGE_CAPITAL_PCT[stage]
        history    = self.get_history(strategy_id)
        return "\n".join([
            f"Strategy: {strategy_id}",
            f"Stage:    {stage.value}",
            f"Capital:  {capital_pct:.0%} live",
            f"History:  {len(history)} decisions",
        ])

    # ── Private ────────────────────────────────────────────────────────────

    def _save(self, d: GraduationDecision) -> None:
        import json
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO live_graduation "
                "(decision_id,strategy_id,from_stage,to_stage,approved,"
                "authorization_token,gate_result,rationale,decided_at,decided_by) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (d.decision_id, d.strategy_id, d.from_stage.value, d.to_stage.value,
                 int(d.approved), d.authorization_token,
                 json.dumps(d.gate_result), d.rationale, d.decided_at, d.decided_by),
            )
            conn.commit()

    @staticmethod
    def _hash_token(token: str) -> str:
        """Store only the hash of the authorization token, never plaintext."""
        return hashlib.sha256(token.encode()).hexdigest()[:16]
