"""
Autonomous Daily Review — end-of-day analysis report.

Generated once per day, stored permanently. Reviews:
    PnL summary            — best/worst trades, daily PnL
    Strategy health        — degraded strategies, alerts
    Market regime          — which regime dominated today
    Execution quality      — slippage, fees, latency
    Research output        — new hypotheses, queued experiments
    Drift report           — any drift detected today
    Feature importance     — top contributors today
    Recommendations        — 3 key actions for tomorrow

Claude synthesises the analysis into a narrative summary.
"""
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


@dataclass
class DailyReview:
    date:                str
    pnl_summary:         Dict
    strategy_health:     Dict
    execution_quality:   Dict
    research_output:     Dict
    drift_report:        Dict
    regime_summary:      Dict
    feature_importance:  Dict
    recommendations:     List[str]
    claude_narrative:    str = ''
    generated_at:        str = field(default_factory=_utcnow)


_CREATE_DAILY = """
CREATE TABLE IF NOT EXISTS daily_reviews (
    date           TEXT PRIMARY KEY,
    pnl_summary    TEXT,
    strategy_health TEXT,
    execution_quality TEXT,
    research_output TEXT,
    drift_report   TEXT,
    regime_summary TEXT,
    feature_importance TEXT,
    recommendations TEXT,
    claude_narrative TEXT,
    generated_at   TEXT NOT NULL
)
"""


class DailyReviewer:
    """
    Autonomous end-of-day review generator.

    Call run() at end of trading day. Saves permanently to SQLite.
    """

    def __init__(
        self,
        trade_memory=None,
        strategy_health=None,
        execution_quality=None,
        research_engine=None,
        drift_detector=None,
        feature_importance=None,
        position_manager=None,
        llm=None,
        db_path: str = "data/trade_memory.sqlite",
    ):
        self.tm  = trade_memory
        self.sh  = strategy_health
        self.eq  = execution_quality
        self.re  = research_engine
        self.dd  = drift_detector
        self.fi  = feature_importance
        self.pm  = position_manager
        self.llm = llm
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_DAILY)
            conn.commit()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, date: Optional[str] = None) -> DailyReview:
        """
        Generate the daily review. Idempotent — safe to run multiple times.
        """
        today = date or datetime.now(timezone.utc).strftime('%Y-%m-%d')
        logger.info(f"DailyReviewer: generating review for {today}")

        pnl_summary       = self._pnl_summary()
        strategy_health   = self._strategy_health_summary()
        exec_quality      = self._execution_summary()
        research_output   = self._research_summary()
        drift_report      = self._drift_summary()
        regime_summary    = self._regime_summary()
        fi_summary        = self._feature_importance_summary()
        recommendations   = self._recommendations(
            pnl_summary, strategy_health, drift_report
        )
        claude_narrative  = self._claude_narrative(
            pnl_summary, strategy_health, exec_quality, recommendations
        )

        review = DailyReview(
            date=today,
            pnl_summary=pnl_summary,
            strategy_health=strategy_health,
            execution_quality=exec_quality,
            research_output=research_output,
            drift_report=drift_report,
            regime_summary=regime_summary,
            feature_importance=fi_summary,
            recommendations=recommendations,
            claude_narrative=claude_narrative,
        )

        self._save(review)
        logger.info(f"DailyReviewer: review saved for {today}")
        return review

    def get_review(self, date: str) -> Optional[DailyReview]:
        """Load a stored daily review."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM daily_reviews WHERE date=?", (date,)
            ).fetchone()
        return self._row_to_review(dict(row)) if row else None

    def format_review(self, review: DailyReview) -> str:
        """Human-readable review text."""
        lines = [
            f"{'='*60}",
            f"  DAILY REVIEW: {review.date}",
            f"{'='*60}",
            "",
            "PNL SUMMARY:",
            f"  Total PnL:    ${review.pnl_summary.get('total_pnl', 0):.2f}",
            f"  Trades:       {review.pnl_summary.get('total_trades', 0)}",
            f"  Win rate:     {review.pnl_summary.get('win_rate', 0):.1%}",
            f"  Best trade:   ${review.pnl_summary.get('best_trade', 0):.2f}",
            f"  Worst trade:  ${review.pnl_summary.get('worst_trade', 0):.2f}",
            "",
            "STRATEGY HEALTH:",
        ]
        for k, v in review.strategy_health.items():
            lines.append(f"  {k}: {v}")

        lines += ["", "RECOMMENDATIONS:"]
        for i, r in enumerate(review.recommendations, 1):
            lines.append(f"  {i}. {r}")

        if review.claude_narrative:
            lines += ["", "ANALYSIS:", review.claude_narrative]

        return "\n".join(lines)

    # ── Data gathering ────────────────────────────────────────────────────────

    def _pnl_summary(self) -> Dict:
        if not self.tm:
            return {}
        try:
            return self.tm.performance_report(days=1)
        except Exception as e:
            logger.warning(f"DailyReviewer: pnl error: {e}")
            return {}

    def _strategy_health_summary(self) -> Dict:
        if not self.sh:
            return {}
        try:
            return self.sh.summary()
        except Exception as e:
            logger.warning(f"DailyReviewer: strategy_health error: {e}")
            return {}

    def _execution_summary(self) -> Dict:
        if not self.eq:
            return {}
        try:
            r = self.eq.run(period_days=1)
            return {
                'fills': r.fills_analyzed,
                'mean_slippage_bps': r.slippage.mean_bps,
                'avg_fee_pct': r.avg_fee_pct,
                'alerts': r.alerts,
            }
        except Exception as e:
            logger.warning(f"DailyReviewer: execution error: {e}")
            return {}

    def _research_summary(self) -> Dict:
        if not self.re:
            return {}
        try:
            return self.re.get_summary()
        except Exception as e:
            logger.warning(f"DailyReviewer: research error: {e}")
            return {}

    def _drift_summary(self) -> Dict:
        if not self.dd:
            return {}
        try:
            report = self.dd.get_report()
            if not report:
                return {}
            return {
                'drift_detected':    report.drift_detected,
                'allocation_factor': report.allocation_factor,
                'retrain_required':  report.retrain_required,
                'summary':           report.summary[:200],
            }
        except Exception as e:
            logger.warning(f"DailyReviewer: drift error: {e}")
            return {}

    def _regime_summary(self) -> Dict:
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT market_regime, COUNT(*) as cnt FROM positions "
                    "WHERE exit_time>=? AND market_regime IS NOT NULL "
                    "GROUP BY market_regime ORDER BY cnt DESC",
                    (cutoff,),
                ).fetchall()
            return {'regimes': {r[0]: r[1] for r in rows}}
        except Exception as e:
            return {}

    def _feature_importance_summary(self) -> Dict:
        if not self.fi:
            return {}
        try:
            report = self.fi.get_last_report()
            if not report:
                return {}
            gi = report.get('global_importance', {})
            top5 = sorted(gi.items(), key=lambda kv: kv[1], reverse=True)[:5]
            return {'top_features': top5, 'calibration_gap': report.get('calibration', {}).get('max_gap', 0)}
        except Exception as e:
            return {}

    def _recommendations(
        self,
        pnl:      Dict,
        health:   Dict,
        drift:    Dict,
    ) -> List[str]:
        recs = []

        # PnL based
        wr = pnl.get('win_rate', 0.5)
        if wr < 0.4:
            recs.append("Win rate below 40% — review signal thresholds and opportunity ranking")

        # Health based
        degraded = health.get('degraded', [])
        if degraded:
            recs.append(f"Investigate degraded strategies: {', '.join(degraded[:3])}")

        # Drift based
        if drift.get('retrain_required'):
            recs.append("Drift detected — trigger model retraining immediately")
        elif drift.get('drift_detected'):
            recs.append("Mild drift detected — monitor closely, reduce allocation by 25%")

        if not recs:
            recs.append("System operating normally — continue monitoring")

        return recs[:5]

    def _claude_narrative(
        self,
        pnl:      Dict,
        health:   Dict,
        exec_q:   Dict,
        recs:     List[str],
    ) -> str:
        if not self.llm:
            return ''
        try:
            prompt = (
                f"Write a brief investment committee–style daily review (3-4 sentences).\n"
                f"PnL: ${pnl.get('total_pnl',0):.2f} from {pnl.get('total_trades',0)} trades "
                f"(win rate {pnl.get('win_rate',0):.1%}).\n"
                f"Degraded strategies: {health.get('degraded',[])}.\n"
                f"Execution alerts: {exec_q.get('alerts',[])}.\n"
                f"Recommendations: {recs}.\n"
                "Be concise. Focus on what matters."
            )
            import anthropic
            response = anthropic.Anthropic().messages.create(
                model="claude-opus-4-5",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning(f"DailyReviewer: Claude narrative error: {e}")
            return ''

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, review: DailyReview) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO daily_reviews "
                "(date,pnl_summary,strategy_health,execution_quality,research_output,"
                "drift_report,regime_summary,feature_importance,recommendations,"
                "claude_narrative,generated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (review.date,
                 json.dumps(review.pnl_summary),
                 json.dumps(review.strategy_health),
                 json.dumps(review.execution_quality),
                 json.dumps(review.research_output),
                 json.dumps(review.drift_report),
                 json.dumps(review.regime_summary),
                 json.dumps(review.feature_importance),
                 json.dumps(review.recommendations),
                 review.claude_narrative,
                 review.generated_at),
            )
            conn.commit()

    @staticmethod
    def _row_to_review(r: Dict) -> DailyReview:
        return DailyReview(
            date=r['date'],
            pnl_summary=json.loads(r.get('pnl_summary') or '{}'),
            strategy_health=json.loads(r.get('strategy_health') or '{}'),
            execution_quality=json.loads(r.get('execution_quality') or '{}'),
            research_output=json.loads(r.get('research_output') or '{}'),
            drift_report=json.loads(r.get('drift_report') or '{}'),
            regime_summary=json.loads(r.get('regime_summary') or '{}'),
            feature_importance=json.loads(r.get('feature_importance') or '{}'),
            recommendations=json.loads(r.get('recommendations') or '[]'),
            claude_narrative=r.get('claude_narrative', ''),
            generated_at=r.get('generated_at', _utcnow()),
        )
