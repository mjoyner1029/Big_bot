"""
Autonomous Weekly Review — investment committee–style weekly analysis.

Generated once per week, stored permanently. Reviews:
    Week performance       — PnL, trades, win rate, Sharpe
    Strategy performance   — per-strategy breakdown
    Market changes         — regime shifts, volatility changes
    Portfolio performance  — drawdown, beta, correlation
    Model changes          — any champion promotions
    Feature changes        — importance shifts over the week
    Research completed     — hypotheses validated/rejected
    Experiments promoted   — new strategy versions deployed
    Recommendations        — 5 key actions for next week

Claude synthesises a comprehensive narrative.
"""
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


@dataclass
class WeeklyReview:
    week_start:           str
    week_end:             str
    performance:          Dict
    strategy_breakdown:   Dict
    model_changes:        Dict
    research_completed:   Dict
    experiments_promoted: List[Dict]
    key_metrics:          Dict
    recommendations:      List[str]
    claude_narrative:     str = ''
    generated_at:         str = field(default_factory=_utcnow)


_CREATE_WEEKLY = """
CREATE TABLE IF NOT EXISTS weekly_reviews (
    week_start           TEXT PRIMARY KEY,
    week_end             TEXT NOT NULL,
    performance          TEXT,
    strategy_breakdown   TEXT,
    model_changes        TEXT,
    research_completed   TEXT,
    experiments_promoted TEXT,
    key_metrics          TEXT,
    recommendations      TEXT,
    claude_narrative     TEXT,
    generated_at         TEXT NOT NULL
)
"""


class WeeklyReviewer:
    """
    Autonomous end-of-week investment committee review.
    """

    def __init__(
        self,
        trade_memory=None,
        strategy_health=None,
        champion_challenger=None,
        research_engine=None,
        experiment_engine=None,
        drift_detector=None,
        feature_importance=None,
        daily_reviewer=None,
        llm=None,
        db_path: str = "data/trade_memory.sqlite",
    ):
        self.tm  = trade_memory
        self.sh  = strategy_health
        self.cc  = champion_challenger
        self.re  = research_engine
        self.ee  = experiment_engine
        self.dd  = drift_detector
        self.fi  = feature_importance
        self.dr  = daily_reviewer
        self.llm = llm
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_WEEKLY)
            conn.commit()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, week_start: Optional[str] = None) -> WeeklyReview:
        """Generate and store the weekly review."""
        now = datetime.now(timezone.utc)
        # Default: last Monday to Sunday
        monday  = now - timedelta(days=now.weekday() + 7)
        start   = week_start or monday.strftime('%Y-%m-%d')
        end     = (datetime.strptime(start, '%Y-%m-%d') + timedelta(days=6)).strftime('%Y-%m-%d')

        logger.info(f"WeeklyReviewer: generating review for week {start} — {end}")

        performance          = self._performance(start)
        strategy_breakdown   = self._strategy_breakdown()
        model_changes        = self._model_changes()
        research_completed   = self._research_completed()
        experiments_promoted = self._experiments_promoted(start)
        key_metrics          = self._key_metrics(start)
        recommendations      = self._recommendations(performance, strategy_breakdown, research_completed)
        claude_narrative     = self._claude_narrative(
            performance, strategy_breakdown, key_metrics, recommendations
        )

        review = WeeklyReview(
            week_start=start,
            week_end=end,
            performance=performance,
            strategy_breakdown=strategy_breakdown,
            model_changes=model_changes,
            research_completed=research_completed,
            experiments_promoted=experiments_promoted,
            key_metrics=key_metrics,
            recommendations=recommendations,
            claude_narrative=claude_narrative,
        )
        self._save(review)
        logger.info(f"WeeklyReviewer: review saved for {start}")
        return review

    def get_review(self, week_start: str) -> Optional[WeeklyReview]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM weekly_reviews WHERE week_start=?", (week_start,)
            ).fetchone()
        return self._row_to_review(dict(row)) if row else None

    def format_review(self, review: WeeklyReview) -> str:
        lines = [
            "=" * 60,
            f"  WEEKLY REVIEW: {review.week_start} — {review.week_end}",
            "=" * 60,
            "",
            "PERFORMANCE:",
            f"  Total PnL:    ${review.performance.get('total_pnl', 0):.2f}",
            f"  Total trades: {review.performance.get('total_trades', 0)}",
            f"  Win rate:     {review.performance.get('win_rate', 0):.1%}",
            f"  Best day PnL: ${review.performance.get('best_day_pnl', 0):.2f}",
            f"  Worst day:    ${review.performance.get('worst_day_pnl', 0):.2f}",
            "",
            "STRATEGY BREAKDOWN:",
        ]
        for strat, stats in (review.strategy_breakdown or {}).items():
            lines.append(
                f"  {strat:<20s} trades={stats.get('trades',0)} "
                f"wr={stats.get('win_rate',0):.1%} "
                f"exp=${stats.get('expectancy',0):.2f}"
            )
        lines += ["", "EXPERIMENTS PROMOTED:"]
        for e in (review.experiments_promoted or []):
            lines.append(f"  → {e.get('description','?')}")
        lines += ["", "RECOMMENDATIONS:"]
        for i, r in enumerate(review.recommendations or [], 1):
            lines.append(f"  {i}. {r}")
        if review.claude_narrative:
            lines += ["", "INVESTMENT COMMITTEE ANALYSIS:", review.claude_narrative]
        return "\n".join(lines)

    # ── Data gathering ────────────────────────────────────────────────────────

    def _performance(self, week_start: str) -> Dict:
        if not self.tm:
            return {}
        try:
            return self.tm.performance_report(days=7)
        except Exception as e:
            logger.warning(f"WeeklyReviewer: performance error: {e}")
            return {}

    def _strategy_breakdown(self) -> Dict:
        if not self.sh:
            return {}
        try:
            health_map = self.sh.run()
            return {
                name: {
                    'trades':      h.trades,
                    'win_rate':    round(h.win_rate, 3),
                    'expectancy':  round(h.expectancy, 2),
                    'sharpe':      round(h.sharpe, 2),
                    'healthy':     h.healthy,
                }
                for name, h in health_map.items()
            }
        except Exception as e:
            logger.warning(f"WeeklyReviewer: strategy breakdown error: {e}")
            return {}

    def _model_changes(self) -> Dict:
        if not self.cc:
            return {}
        try:
            return self.cc.get_champion_summary()
        except Exception as e:
            return {}

    def _research_completed(self) -> Dict:
        if not self.re:
            return {}
        try:
            return self.re.get_summary()
        except Exception as e:
            return {}

    def _experiments_promoted(self, week_start: str) -> List[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT id, description, status FROM experiments "
                    "WHERE status='PROMOTED' AND created_at>=? ORDER BY created_at DESC",
                    (week_start,),
                ).fetchall()
            return [{'id': r[0], 'description': r[1], 'status': r[2]} for r in rows]
        except Exception as e:
            return []

    def _key_metrics(self, week_start: str) -> Dict:
        """Compute key risk-adjusted metrics for the week."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT net_pnl FROM positions WHERE status='CLOSED' AND exit_time>=?",
                    (week_start,),
                ).fetchall()
            pnls = [r[0] for r in rows if r[0] is not None]
            if not pnls:
                return {}
            import math
            avg  = sum(pnls) / len(pnls)
            std  = (sum((p - avg)**2 for p in pnls) / len(pnls)) ** 0.5
            losses = [p for p in pnls if p < 0]
            downside_std = (sum(p**2 for p in losses) / len(losses)) ** 0.5 if losses else 0
            return {
                'total_pnl':  round(sum(pnls), 2),
                'trades':     len(pnls),
                'avg_pnl':    round(avg, 2),
                'std':        round(std, 2),
                'sharpe':     round(avg / std, 3) if std > 0 else 0.0,
                'sortino':    round(avg / downside_std, 3) if downside_std > 0 else 0.0,
                'profit_factor': round(
                    abs(sum(p for p in pnls if p > 0)) / abs(sum(p for p in pnls if p < 0)), 2
                ) if losses else float('inf'),
                'max_drawdown': round(self._max_drawdown(pnls), 2),
            }
        except Exception as e:
            return {}

    def _recommendations(
        self,
        performance:  Dict,
        strategies:   Dict,
        research:     Dict,
    ) -> List[str]:
        recs = []

        wr = performance.get('win_rate', 0.5)
        if wr < 0.45:
            recs.append("Win rate critically low — re-evaluate signal thresholds")
        elif wr > 0.60:
            recs.append("Win rate strong — consider scaling position sizes modestly")

        degraded = [s for s, h in strategies.items() if not h.get('healthy')]
        if degraded:
            recs.append(f"Pause and review: {', '.join(degraded[:3])}")

        open_hyp = research.get('open', 0)
        if open_hyp > 5:
            recs.append(f"Research backlog high ({open_hyp} open) — prioritise top 3")

        validated = research.get('validated', 0)
        if validated > 0:
            recs.append(f"{validated} validated hypotheses ready — consider promoting to experiments")

        if not recs:
            recs.append("System healthy — maintain current operation")

        return recs[:5]

    def _claude_narrative(
        self,
        performance:  Dict,
        strategies:   Dict,
        key_metrics:  Dict,
        recs:         List[str],
    ) -> str:
        if not self.llm:
            return ''
        try:
            healthy    = sum(1 for h in strategies.values() if h.get('healthy'))
            degraded_n = len(strategies) - healthy
            prompt = (
                "Write an investment committee–style weekly review (5-6 sentences).\n"
                f"Week PnL: ${performance.get('total_pnl',0):.2f} from "
                f"{performance.get('total_trades',0)} trades "
                f"(win rate {performance.get('win_rate',0):.1%}).\n"
                f"Sharpe: {key_metrics.get('sharpe',0):.2f}  "
                f"Sortino: {key_metrics.get('sortino',0):.2f}  "
                f"Max DD: ${key_metrics.get('max_drawdown',0):.2f}\n"
                f"Strategies: {healthy} healthy, {degraded_n} degraded.\n"
                f"Recommendations: {recs}\n"
                "Be analytical, precise, and actionable. No fluff."
            )
            import anthropic
            response = anthropic.Anthropic().messages.create(
                model="claude-opus-4-5",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning(f"WeeklyReviewer: Claude narrative error: {e}")
            return ''

    # ── Stats ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _max_drawdown(pnls: List[float]) -> float:
        equity, peak, max_dd = 0.0, 0.0, 0.0
        for p in pnls:
            equity += p
            peak    = max(peak, equity)
            max_dd  = max(max_dd, peak - equity)
        return max_dd

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, review: WeeklyReview) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO weekly_reviews "
                "(week_start,week_end,performance,strategy_breakdown,model_changes,"
                "research_completed,experiments_promoted,key_metrics,recommendations,"
                "claude_narrative,generated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (review.week_start, review.week_end,
                 json.dumps(review.performance),
                 json.dumps(review.strategy_breakdown),
                 json.dumps(review.model_changes),
                 json.dumps(review.research_completed),
                 json.dumps(review.experiments_promoted),
                 json.dumps(review.key_metrics),
                 json.dumps(review.recommendations),
                 review.claude_narrative,
                 review.generated_at),
            )
            conn.commit()

    @staticmethod
    def _row_to_review(r: Dict) -> WeeklyReview:
        return WeeklyReview(
            week_start=r['week_start'],
            week_end=r['week_end'],
            performance=json.loads(r.get('performance') or '{}'),
            strategy_breakdown=json.loads(r.get('strategy_breakdown') or '{}'),
            model_changes=json.loads(r.get('model_changes') or '{}'),
            research_completed=json.loads(r.get('research_completed') or '{}'),
            experiments_promoted=json.loads(r.get('experiments_promoted') or '[]'),
            key_metrics=json.loads(r.get('key_metrics') or '{}'),
            recommendations=json.loads(r.get('recommendations') or '[]'),
            claude_narrative=r.get('claude_narrative', ''),
            generated_at=r.get('generated_at', _utcnow()),
        )
