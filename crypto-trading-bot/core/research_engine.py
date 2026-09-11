"""
Autonomous Research Engine — continuously asks "What can improve profitability?"

ARCHITECTURE RULE:
    Claude proposes hypotheses. The Research Engine tracks them.
    Nothing reaches production automatically.
    Every hypothesis must pass the full Experiment Worker pipeline.

The engine analyzes:
    • recent winners/losers (from TradeMemory)
    • market regime changes (from RegimeDetector)
    • feature importance shifts (from FeatureImportanceEngine)
    • strategy degradation signals (from StrategyHealthMonitor)
    • MetaModel accuracy and calibration
    • execution quality issues
    • opportunity distribution over time

Output: structured Hypothesis objects → ExperimentEngine queue
"""
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


class HypothesisStatus(str, Enum):
    OPEN       = "OPEN"        # generated, not yet acted on
    QUEUED     = "QUEUED"      # submitted to ExperimentWorker
    TESTING    = "TESTING"     # experiment in progress
    VALIDATED  = "VALIDATED"   # experiment confirmed the hypothesis
    REJECTED   = "REJECTED"    # hypothesis disproved
    PROMOTED   = "PROMOTED"    # changes live in production


class HypothesisCategory(str, Enum):
    STRATEGY_PARAM    = "STRATEGY_PARAM"      # strategy parameter tuning
    FEATURE_WEIGHT    = "FEATURE_WEIGHT"      # reweight input features
    RISK_PARAM        = "RISK_PARAM"          # risk management (via experiment, not directly)
    REGIME_FILTER     = "REGIME_FILTER"       # skip/favor specific regimes
    SIGNAL_THRESHOLD  = "SIGNAL_THRESHOLD"   # confidence/vote threshold changes
    EXECUTION         = "EXECUTION"           # entry/exit timing
    UNIVERSE          = "UNIVERSE"            # add/remove assets from universe
    META_MODEL        = "META_MODEL"          # MetaModel architecture / training


@dataclass
class Hypothesis:
    id:           str
    title:        str
    description:  str
    category:     HypothesisCategory
    evidence:     List[str]              # data points that triggered this
    proposed_by:  str                   # 'research_engine' | 'claude' | 'operator'
    status:       HypothesisStatus = HypothesisStatus.OPEN
    priority:     int = 5               # 1=highest, 10=lowest
    experiment_id: Optional[str] = None
    created_at:   str = field(default_factory=_utcnow)
    updated_at:   str = field(default_factory=_utcnow)
    outcome:      Optional[str] = None


_CREATE_HYPOTHESES = """
CREATE TABLE IF NOT EXISTS research_hypotheses (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    description   TEXT,
    category      TEXT,
    evidence      TEXT,
    proposed_by   TEXT DEFAULT 'research_engine',
    status        TEXT DEFAULT 'OPEN',
    priority      INTEGER DEFAULT 5,
    experiment_id TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT,
    outcome       TEXT
)
"""


class ResearchEngine:
    """
    Continuously generates improvement hypotheses from live performance data.

    Run `analyze()` periodically (e.g., hourly). Each call inspects all data
    sources and emits new hypotheses when anomalies are detected.
    """

    # Degradation thresholds that trigger hypothesis generation
    STRATEGY_MIN_WIN_RATE    = 0.45    # below this → hypothesis about strategy
    STRATEGY_MIN_EXPECTANCY  = -5.0    # below this → hypothesis about strategy
    STRATEGY_MIN_TRADES      = 10      # minimum sample before evaluating
    METAMODEL_MIN_ACCURACY   = 0.52    # below this → retrain hypothesis
    SLIPPAGE_MAX_PCT         = 0.003   # above 0.3% avg → execution hypothesis
    MAX_REGIME_DRAWDOWN_PCT  = 0.03    # over 3% drawdown in a regime → filter

    def __init__(self, db_path: str = "data/trade_memory.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_HYPOTHESES)
            conn.commit()

    # ── Main analysis loop ────────────────────────────────────────────────────

    def analyze(
        self,
        trade_memory=None,
        strategy_health=None,
        execution_quality=None,
        drift_detector=None,
        meta_model=None,
        llm=None,
    ) -> List[Hypothesis]:
        """
        Run all analysis passes and emit new hypotheses.

        Returns list of newly created hypotheses this cycle.
        """
        new_hypotheses: List[Hypothesis] = []

        if trade_memory:
            new_hypotheses += self._analyze_strategy_performance(trade_memory)
            new_hypotheses += self._analyze_regime_performance(trade_memory)
            new_hypotheses += self._analyze_execution_quality(trade_memory)

        if strategy_health:
            new_hypotheses += self._analyze_strategy_degradation(strategy_health)

        if drift_detector:
            new_hypotheses += self._analyze_drift(drift_detector)

        if meta_model and hasattr(meta_model, '_is_trained') and meta_model._is_trained:
            new_hypotheses += self._analyze_metamodel(meta_model)

        # Deduplicate: skip hypotheses with identical titles that are already OPEN/QUEUED
        existing_open = {h['title'] for h in self._load_open_hypotheses()}
        filtered = [h for h in new_hypotheses if h.title not in existing_open]

        for h in filtered:
            self._save(h)
            logger.info(f"ResearchEngine: new hypothesis [{h.category.value}] '{h.title}'")

        # Optionally ask Claude to synthesize additional hypotheses
        if llm and filtered:
            claude_hyps = self._ask_claude_for_hypotheses(llm, filtered)
            for h in claude_hyps:
                if h.title not in existing_open:
                    self._save(h)
                    filtered.append(h)

        return filtered

    # ── Analysis passes ────────────────────────────────────────────────────────

    def _analyze_strategy_performance(self, trade_memory) -> List[Hypothesis]:
        """Generate hypotheses from per-strategy win rate and expectancy."""
        hyps = []
        try:
            report = trade_memory.performance_report(days=30)
            for strategy, stats in report.get('by_strategy', {}).items():
                trades = stats.get('trades', 0)
                if trades < self.STRATEGY_MIN_TRADES:
                    continue

                wr  = stats.get('win_rate', 0.5)
                exp = stats.get('avg_pnl', 0.0)

                if wr < self.STRATEGY_MIN_WIN_RATE:
                    hyps.append(Hypothesis(
                        id=str(uuid.uuid4()),
                        title=f"{strategy} win rate below threshold ({wr:.0%})",
                        description=(
                            f"Strategy '{strategy}' has win rate {wr:.1%} over {trades} trades "
                            f"(threshold {self.STRATEGY_MIN_WIN_RATE:.0%}). "
                            f"Consider: regime filter, tighter entry criteria, or retirement."
                        ),
                        category=HypothesisCategory.STRATEGY_PARAM,
                        evidence=[f"win_rate={wr:.3f}", f"trades={trades}"],
                        proposed_by='research_engine',
                        priority=3,
                    ))

                if exp < self.STRATEGY_MIN_EXPECTANCY:
                    hyps.append(Hypothesis(
                        id=str(uuid.uuid4()),
                        title=f"{strategy} negative expectancy (${exp:.2f}/trade)",
                        description=(
                            f"Strategy '{strategy}' has average PnL ${exp:.2f} per trade. "
                            f"Investigate: stop losses too tight, exits too early, "
                            f"or strategy has lost its edge."
                        ),
                        category=HypothesisCategory.STRATEGY_PARAM,
                        evidence=[f"avg_pnl={exp:.2f}", f"trades={trades}"],
                        proposed_by='research_engine',
                        priority=2,
                    ))
        except Exception as e:
            logger.warning(f"ResearchEngine: strategy analysis error: {e}")
        return hyps

    def _analyze_regime_performance(self, trade_memory) -> List[Hypothesis]:
        """Detect regime-specific underperformance."""
        hyps = []
        try:
            history = trade_memory.get_history(days=30)
            by_regime: Dict[str, List[float]] = {}
            for trade in history:
                regime = trade.get('market_regime', 'unknown') or 'unknown'
                pnl    = trade.get('net_pnl', 0.0) or 0.0
                by_regime.setdefault(regime, []).append(pnl)

            for regime, pnls in by_regime.items():
                if len(pnls) < 5:
                    continue
                total = sum(pnls)
                avg   = total / len(pnls)
                if total < -self.capital_at_risk(pnls) * self.MAX_REGIME_DRAWDOWN_PCT:
                    hyps.append(Hypothesis(
                        id=str(uuid.uuid4()),
                        title=f"Underperformance in '{regime}' regime",
                        description=(
                            f"Total PnL in '{regime}' regime: ${total:.2f} "
                            f"({len(pnls)} trades, avg ${avg:.2f}). "
                            f"Consider adding a regime filter to skip this environment."
                        ),
                        category=HypothesisCategory.REGIME_FILTER,
                        evidence=[f"regime={regime}", f"total_pnl={total:.2f}", f"trades={len(pnls)}"],
                        proposed_by='research_engine',
                        priority=4,
                    ))
        except Exception as e:
            logger.warning(f"ResearchEngine: regime analysis error: {e}")
        return hyps

    def _analyze_execution_quality(self, trade_memory) -> List[Hypothesis]:
        """Detect systematically poor execution (high slippage, bad fills)."""
        hyps = []
        try:
            history = trade_memory.get_history(days=14)
            slippages = []
            for t in history:
                entry = t.get('entry_fill_price', 0)
                entry_ref = t.get('entry_price', entry)
                if entry_ref and entry_ref > 0:
                    slip = abs(entry - entry_ref) / entry_ref
                    slippages.append(slip)

            if len(slippages) >= 10:
                avg_slip = sum(slippages) / len(slippages)
                if avg_slip > self.SLIPPAGE_MAX_PCT:
                    hyps.append(Hypothesis(
                        id=str(uuid.uuid4()),
                        title=f"High average entry slippage ({avg_slip:.2%})",
                        description=(
                            f"Average entry slippage is {avg_slip:.2%} "
                            f"(threshold {self.SLIPPAGE_MAX_PCT:.2%}) over {len(slippages)} trades. "
                            f"Consider limit orders or reducing trade size during low liquidity."
                        ),
                        category=HypothesisCategory.EXECUTION,
                        evidence=[f"avg_slippage={avg_slip:.4f}", f"sample={len(slippages)}"],
                        proposed_by='research_engine',
                        priority=4,
                    ))
        except Exception as e:
            logger.warning(f"ResearchEngine: execution analysis error: {e}")
        return hyps

    def _analyze_strategy_degradation(self, strategy_health) -> List[Hypothesis]:
        """Ask StrategyHealthMonitor for already-degraded strategies."""
        hyps = []
        try:
            alerts = strategy_health.get_alerts()
            for alert in alerts:
                hyps.append(Hypothesis(
                    id=str(uuid.uuid4()),
                    title=f"Strategy health alert: {alert['strategy']} — {alert['metric']}",
                    description=alert.get('description', ''),
                    category=HypothesisCategory.STRATEGY_PARAM,
                    evidence=[f"{k}={v}" for k, v in alert.items() if k != 'description'],
                    proposed_by='research_engine',
                    priority=2,
                ))
        except Exception as e:
            logger.warning(f"ResearchEngine: strategy health analysis error: {e}")
        return hyps

    def _analyze_drift(self, drift_detector) -> List[Hypothesis]:
        """Create hypotheses when drift is detected."""
        hyps = []
        try:
            report = drift_detector.get_report()
            if report.get('drift_detected'):
                for drift in report.get('drifts', []):
                    hyps.append(Hypothesis(
                        id=str(uuid.uuid4()),
                        title=f"Drift detected: {drift['type']} in {drift['feature']}",
                        description=(
                            f"{drift['type']} drift in '{drift['feature']}': "
                            f"{drift.get('description', '')}. "
                            f"Action: retrain MetaModel, reduce confidence, reduce allocation."
                        ),
                        category=HypothesisCategory.META_MODEL,
                        evidence=[f"drift_type={drift['type']}", f"feature={drift['feature']}"],
                        proposed_by='research_engine',
                        priority=1,
                    ))
        except Exception as e:
            logger.warning(f"ResearchEngine: drift analysis error: {e}")
        return hyps

    def _analyze_metamodel(self, meta_model) -> List[Hypothesis]:
        """Check if MetaModel accuracy is degrading."""
        hyps = []
        try:
            if hasattr(meta_model, '_recent_accuracy'):
                acc = meta_model._recent_accuracy
                if acc < self.METAMODEL_MIN_ACCURACY:
                    hyps.append(Hypothesis(
                        id=str(uuid.uuid4()),
                        title=f"MetaModel accuracy below threshold ({acc:.1%})",
                        description=(
                            f"MetaModel recent accuracy {acc:.1%} < "
                            f"minimum {self.METAMODEL_MIN_ACCURACY:.0%}. "
                            f"Schedule retraining with fresh feature store data."
                        ),
                        category=HypothesisCategory.META_MODEL,
                        evidence=[f"accuracy={acc:.4f}"],
                        proposed_by='research_engine',
                        priority=2,
                    ))
        except Exception as e:
            logger.warning(f"ResearchEngine: metamodel analysis error: {e}")
        return hyps

    def _ask_claude_for_hypotheses(self, llm, recent_hyps: List['Hypothesis']) -> List['Hypothesis']:
        """Ask Claude to synthesize additional research questions."""
        hyps = []
        try:
            context = "\n".join([f"- {h.title}: {h.description}" for h in recent_hyps[:5]])
            prompt = (
                f"You are a quantitative researcher reviewing these recent performance issues:\n\n"
                f"{context}\n\n"
                f"Generate 1-2 additional research hypotheses that might explain the root causes. "
                f"Respond with one hypothesis per line in format:\n"
                f"HYPOTHESIS: [title] | [brief description]"
            )
            response = llm.quick_analysis(prompt)
            if response:
                for line in response.strip().split('\n'):
                    if line.startswith('HYPOTHESIS:'):
                        parts = line[len('HYPOTHESIS:'):].split('|')
                        if len(parts) == 2:
                            hyps.append(Hypothesis(
                                id=str(uuid.uuid4()),
                                title=parts[0].strip(),
                                description=parts[1].strip(),
                                category=HypothesisCategory.STRATEGY_PARAM,
                                evidence=['claude_synthesis'],
                                proposed_by='claude',
                                priority=5,
                            ))
        except Exception as e:
            logger.warning(f"ResearchEngine: Claude synthesis error: {e}")
        return hyps

    @staticmethod
    def capital_at_risk(pnls: List[float]) -> float:
        """Rough capital estimate from trade sizes (fallback)."""
        return max(abs(p) for p in pnls) * 10 if pnls else 10_000

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def get_open_hypotheses(self, limit: int = 50) -> List[Hypothesis]:
        rows = self._load_open_hypotheses(limit)
        return [self._row_to_hyp(r) for r in rows]

    def update_status(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
        experiment_id: str = None,
        outcome: str = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE research_hypotheses SET status=?, experiment_id=COALESCE(?,experiment_id),"
                " outcome=COALESCE(?,outcome), updated_at=? WHERE id=?",
                (status.value, experiment_id, outcome, _utcnow(), hypothesis_id),
            )
            conn.commit()

    def get_summary(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM research_hypotheses GROUP BY status"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def _save(self, h: Hypothesis) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_hypotheses "
                "(id,title,description,category,evidence,proposed_by,status,priority,"
                "experiment_id,created_at,updated_at,outcome) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (h.id, h.title, h.description, h.category.value,
                 json.dumps(h.evidence), h.proposed_by, h.status.value,
                 h.priority, h.experiment_id, h.created_at, h.updated_at, h.outcome),
            )
            conn.commit()

    def _load_open_hypotheses(self, limit: int = 100) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM research_hypotheses WHERE status IN ('OPEN','QUEUED') "
                "ORDER BY priority ASC, created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _row_to_hyp(row: Dict) -> Hypothesis:
        return Hypothesis(
            id=row['id'],
            title=row['title'],
            description=row['description'] or '',
            category=HypothesisCategory(row['category']),
            evidence=json.loads(row.get('evidence') or '[]'),
            proposed_by=row.get('proposed_by', 'research_engine'),
            status=HypothesisStatus(row['status']),
            priority=row.get('priority', 5),
            experiment_id=row.get('experiment_id'),
            created_at=row.get('created_at', _utcnow()),
            updated_at=row.get('updated_at', _utcnow()),
            outcome=row.get('outcome'),
        )
