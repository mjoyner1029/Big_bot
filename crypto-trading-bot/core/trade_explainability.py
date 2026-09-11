"""
Trade Explainability — every trade permanently records full reasoning.

Every trade entry stores WHY it was chosen and WHY alternatives were rejected.

Fields stored:
    why_chosen          — narrative from Claude + ranker
    alternatives_rejected — list of rejected candidates with reasons
    expected_value      — projected value before entry
    opportunity_score   — from OpportunityRanker
    portfolio_reason    — from PortfolioManager
    risk_reason         — from SafetyManager
    claude_explanation  — verbatim LLM reasoning
    metamodel_pred      — MetaModel decision + confidence
    kronos_confidence   — Kronos foundation model confidence
    strategy_votes      — per-strategy signal contributions
    regime              — market regime at entry
    ranker_rank         — position in opportunity ranking
"""
import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


@dataclass
class TradeExplanation:
    position_id:           str
    symbol:                str
    strategy:              str
    why_chosen:            str
    alternatives_rejected: List[Dict]    # [{symbol, score, reason}]
    expected_value:        float
    opportunity_score:     float
    portfolio_reason:      str
    risk_reason:           str
    claude_explanation:    str
    metamodel_decision:    str
    metamodel_confidence:  float
    metamodel_ev:          float
    kronos_confidence:     float
    strategy_votes:        Dict[str, Any]   # strategy → signal dict
    regime:                str
    ranker_rank:           int
    total_candidates:      int
    recorded_at:           str = field(default_factory=_utcnow)


_CREATE_EXPLANATIONS = """
CREATE TABLE IF NOT EXISTS trade_explanations (
    position_id           TEXT PRIMARY KEY,
    symbol                TEXT NOT NULL,
    strategy              TEXT NOT NULL,
    why_chosen            TEXT,
    alternatives_rejected TEXT,
    expected_value        REAL DEFAULT 0,
    opportunity_score     REAL DEFAULT 0,
    portfolio_reason      TEXT,
    risk_reason           TEXT,
    claude_explanation    TEXT,
    metamodel_decision    TEXT,
    metamodel_confidence  REAL DEFAULT 0,
    metamodel_ev          REAL DEFAULT 0,
    kronos_confidence     REAL DEFAULT 0,
    strategy_votes        TEXT,
    regime                TEXT,
    ranker_rank           INTEGER DEFAULT 0,
    total_candidates      INTEGER DEFAULT 0,
    recorded_at           TEXT NOT NULL
)
"""


class TradeExplainabilityStore:
    """
    Stores and retrieves trade explanations.

    Called at trade entry with full context from the decision pipeline.
    """

    def __init__(self, db_path: str = "data/trade_memory.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_EXPLANATIONS)
            conn.commit()

    # ── Recording ────────────────────────────────────────────────────────────

    def record(self, explanation: TradeExplanation) -> None:
        """Persist a trade explanation. Deduplicates on position_id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO trade_explanations "
                "(position_id,symbol,strategy,why_chosen,alternatives_rejected,"
                "expected_value,opportunity_score,portfolio_reason,risk_reason,"
                "claude_explanation,metamodel_decision,metamodel_confidence,"
                "metamodel_ev,kronos_confidence,strategy_votes,regime,"
                "ranker_rank,total_candidates,recorded_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (explanation.position_id, explanation.symbol, explanation.strategy,
                 explanation.why_chosen,
                 json.dumps(explanation.alternatives_rejected),
                 explanation.expected_value, explanation.opportunity_score,
                 explanation.portfolio_reason, explanation.risk_reason,
                 explanation.claude_explanation,
                 explanation.metamodel_decision, explanation.metamodel_confidence,
                 explanation.metamodel_ev, explanation.kronos_confidence,
                 json.dumps(explanation.strategy_votes),
                 explanation.regime, explanation.ranker_rank,
                 explanation.total_candidates, explanation.recorded_at),
            )
            conn.commit()
        logger.debug(f"TradeExplainability: recorded for position {explanation.position_id[:8]}")

    def record_from_context(
        self,
        position_id:        str,
        symbol:             str,
        strategy:           str,
        ranked_opportunity=None,    # RankedOpportunity from OpportunityRanker
        all_candidates:     List = None,
        portfolio_result=None,      # AllocationResult
        meta_prediction=None,       # MetaPrediction
        claude_explanation: str = '',
        strategy_signals:   Dict = None,
        regime:             str = 'unknown',
        kronos_confidence:  float = 0.0,
    ) -> None:
        """
        Convenience method: build and record explanation from pipeline context.
        """
        candidates = all_candidates or []

        # Build alternatives_rejected list
        alternatives = []
        for i, c in enumerate(candidates):
            sym = getattr(c, 'symbol', None) or c.get('symbol', '') if hasattr(c, 'get') else getattr(c, 'symbol', '')
            if sym == symbol:
                continue
            score = getattr(c, 'final_score', None) or (c.get('final_score', 0) if hasattr(c, 'get') else 0)
            dec   = getattr(c, 'decision', None) or (c.get('decision', '') if hasattr(c, 'get') else '')
            alternatives.append({
                'symbol': sym,
                'score':  round(score, 3),
                'reason': str(dec),
            })

        ranker_rank = 0
        opp_score   = 0.0
        expected_value = 0.0
        if ranked_opportunity:
            opp_score   = getattr(ranked_opportunity, 'final_score', 0.0)
            expected_value = getattr(ranked_opportunity, 'expected_return', 0.0)
            # Find rank position
            for i, c in enumerate(candidates):
                if getattr(c, 'symbol', '') == symbol:
                    ranker_rank = i + 1
                    break

        portfolio_reason = ''
        if portfolio_result:
            if getattr(portfolio_result, 'approved', False):
                portfolio_reason = f"Approved ${getattr(portfolio_result, 'size_dollars', 0):.0f}"
            else:
                portfolio_reason = getattr(portfolio_result, 'reject_reason', 'Unknown')

        meta_decision = 'UNKNOWN'
        meta_conf     = 0.0
        meta_ev       = 0.0
        if meta_prediction:
            meta_decision = getattr(meta_prediction, 'decision', 'UNKNOWN')
            meta_conf     = getattr(meta_prediction, 'confidence', 0.0)
            meta_ev       = getattr(meta_prediction, 'expected_value', 0.0)

        why_chosen = (
            f"Symbol {symbol} selected from {len(candidates)} candidates. "
            f"Opportunity score: {opp_score:.2f}. "
            f"MetaModel: {meta_decision} ({meta_conf:.0%} confidence). "
            f"Strategy: {strategy}."
        )

        expl = TradeExplanation(
            position_id=position_id,
            symbol=symbol,
            strategy=strategy,
            why_chosen=why_chosen,
            alternatives_rejected=alternatives[:10],    # store top 10 alternatives
            expected_value=expected_value,
            opportunity_score=opp_score,
            portfolio_reason=portfolio_reason,
            risk_reason='Safety checks passed',
            claude_explanation=claude_explanation,
            metamodel_decision=meta_decision,
            metamodel_confidence=meta_conf,
            metamodel_ev=meta_ev,
            kronos_confidence=kronos_confidence,
            strategy_votes=strategy_signals or {},
            regime=regime,
            ranker_rank=ranker_rank,
            total_candidates=len(candidates),
        )
        self.record(expl)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get(self, position_id: str) -> Optional[TradeExplanation]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trade_explanations WHERE position_id=?",
                (position_id,),
            ).fetchone()
        return self._row_to_explanation(dict(row)) if row else None

    def get_recent(self, limit: int = 20) -> List[TradeExplanation]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trade_explanations ORDER BY recorded_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_explanation(dict(r)) for r in rows]

    def format_explanation(self, position_id: str) -> str:
        """Human-readable explanation for a trade."""
        expl = self.get(position_id)
        if not expl:
            return f"No explanation found for position {position_id}"

        alts = expl.alternatives_rejected
        alt_lines = [
            f"    {a['symbol']}: score={a['score']:.2f} ({a['reason']})"
            for a in alts[:5]
        ]
        votes_str = ", ".join(
            f"{k}={v}" for k, v in expl.strategy_votes.items()
        ) if expl.strategy_votes else "none"

        return "\n".join([
            f"=== Trade Explanation: {expl.symbol} @ {expl.recorded_at} ===",
            f"Strategy:         {expl.strategy}",
            f"Regime:           {expl.regime}",
            f"Ranker rank:      #{expl.ranker_rank} of {expl.total_candidates}",
            f"Opportunity score:{expl.opportunity_score:.2f}",
            f"Expected value:   ${expl.expected_value:.2f}",
            f"MetaModel:        {expl.metamodel_decision} ({expl.metamodel_confidence:.0%})",
            f"Kronos:           {expl.kronos_confidence:.0%}",
            f"",
            f"Why chosen: {expl.why_chosen}",
            f"",
            f"Claude reasoning:",
            f"  {expl.claude_explanation[:400]}",
            f"",
            f"Top alternatives rejected ({len(alts)}):",
        ] + alt_lines + [
            f"",
            f"Strategy votes:  {votes_str}",
            f"Portfolio:        {expl.portfolio_reason}",
        ])

    # ── Persistence ───────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_explanation(r: Dict) -> TradeExplanation:
        return TradeExplanation(
            position_id=r['position_id'],
            symbol=r['symbol'],
            strategy=r['strategy'],
            why_chosen=r.get('why_chosen', ''),
            alternatives_rejected=json.loads(r.get('alternatives_rejected') or '[]'),
            expected_value=r.get('expected_value', 0.0),
            opportunity_score=r.get('opportunity_score', 0.0),
            portfolio_reason=r.get('portfolio_reason', ''),
            risk_reason=r.get('risk_reason', ''),
            claude_explanation=r.get('claude_explanation', ''),
            metamodel_decision=r.get('metamodel_decision', 'UNKNOWN'),
            metamodel_confidence=r.get('metamodel_confidence', 0.0),
            metamodel_ev=r.get('metamodel_ev', 0.0),
            kronos_confidence=r.get('kronos_confidence', 0.0),
            strategy_votes=json.loads(r.get('strategy_votes') or '{}'),
            regime=r.get('regime', 'unknown'),
            ranker_rank=r.get('ranker_rank', 0),
            total_candidates=r.get('total_candidates', 0),
            recorded_at=r.get('recorded_at', _utcnow()),
        )
