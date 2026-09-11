"""
Experiment Worker — asynchronous pipeline that takes hypotheses through the
full validation gauntlet before any change can reach production.

Pipeline:
    Hypothesis
    → Experiment Queued
    → Historical Backtest
    → Walk-Forward Validation
    → Out-of-Sample Validation
    → Monte Carlo Simulation
    → Paper Trading
    → Canary Deployment
    → Promotion / Rejection

Each stage produces structured metrics. Failures at any stage reject the
experiment — they never proceed further.

The Worker runs as a background thread and processes one experiment at a time
to avoid overwhelming the system.
"""
import json
import logging
import os
import queue
import random
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


class PipelineStage(str, Enum):
    QUEUED      = "QUEUED"
    BACKTESTING = "BACKTESTING"
    WALK_FORWARD = "WALK_FORWARD"
    OUT_OF_SAMPLE = "OUT_OF_SAMPLE"
    MONTE_CARLO   = "MONTE_CARLO"
    PAPER_TEST    = "PAPER_TEST"
    PAPER_RUNNING = "PAPER_RUNNING"          # parked: collecting attributed paper trades
    CANARY        = "CANARY"
    CANARY_PENDING_APPROVAL = "CANARY_PENDING_APPROVAL"  # needs explicit human approval
    CANARY_RUNNING = "CANARY_RUNNING"        # parked: collecting attributed canary trades
    PROMOTED      = "PROMOTED"
    REJECTED      = "REJECTED"


@dataclass
class PipelineResult:
    """Metrics produced by a single pipeline stage."""
    stage:        PipelineStage
    passed:       bool
    metrics:      Dict[str, Any] = field(default_factory=dict)
    reject_reason: Optional[str] = None
    completed_at:  str = field(default_factory=_utcnow)
    pending:      bool = False   # stage is waiting for real observations — not pass, not fail


@dataclass
class WorkItem:
    """An item in the experiment queue."""
    experiment_id:   str
    hypothesis_id:   Optional[str]
    params:          Dict[str, Any]
    description:     str
    submitted_at:    str = field(default_factory=_utcnow)
    priority:        int = 5     # 1=highest


# ── Validation gates (must ALL pass before promotion) ─────────────────────────
GATE_BACKTEST_WIN_RATE   = 0.52
GATE_BACKTEST_EXPECTANCY = 0.0
GATE_BACKTEST_TRADES     = 20
GATE_BACKTEST_SHARPE     = 0.5
GATE_OOS_WIN_RATE        = 0.50
GATE_MC_POSITIVE_RUNS    = 0.60   # 60% of Monte Carlo runs must be profitable
GATE_PAPER_TRADES        = 30     # minimum live paper trades before promotion
GATE_CANARY_TRADES       = 15     # minimum canary trades before full promotion

# Default experiment-worker configuration (overridable per instance/strategy)
DEFAULT_WORKER_CONFIG: Dict[str, Any] = {
    # backtest / statistics
    "min_backtest_trades": GATE_BACKTEST_TRADES,
    "min_expectancy": GATE_BACKTEST_EXPECTANCY,
    "fdr_alpha": 0.10,
    "min_parameter_robustness": 0.40,
    "max_profit_concentration_top5": 0.85,
    "min_positive_fold_fraction": 0.5,
    # paper gate
    "min_paper_trades": GATE_PAPER_TRADES,
    "min_paper_days": 10,
    "max_paper_drawdown": 250.0,           # $ on standard test sizing
    # canary gate
    "auto_live_promotion": False,          # canary NEVER auto-starts
    "canary_allocation_pct": 0.03,
    "max_canary_notional": 500.0,
    "min_canary_trades": GATE_CANARY_TRADES,
    "min_canary_days": 7,
    "max_canary_drawdown": 100.0,
    "max_canary_slippage_deviation": 0.002,
    # data
    "symbols": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "period": "90d",
    "interval": "1h",
}

# Keys in item.params that are experiment metadata, not strategy parameters
_RESERVED_PARAM_KEYS = frozenset({
    "strategy", "strategy_id", "baseline", "symbols", "period", "interval",
    "multi_variable", "rare_event", "hypothesis_family",
    "min_paper_trades", "min_paper_days", "min_canary_trades", "min_canary_days",
})

_CREATE_HYPOTHESES = """
CREATE TABLE IF NOT EXISTS tested_hypotheses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT,
    experiment_id TEXT,
    family        TEXT NOT NULL,
    p_value       REAL NOT NULL,
    metric        REAL,
    tested_at     TEXT NOT NULL
)
"""

_CREATE_CANARY_APPROVALS = """
CREATE TABLE IF NOT EXISTS canary_approvals (
    experiment_id TEXT PRIMARY KEY,
    approved_by   TEXT,
    approved_at   TEXT
)
"""


_CREATE_PIPELINE = """
CREATE TABLE IF NOT EXISTS experiment_pipeline (
    id             TEXT PRIMARY KEY,
    experiment_id  TEXT NOT NULL,
    hypothesis_id  TEXT,
    description    TEXT,
    params         TEXT,
    stage          TEXT DEFAULT 'QUEUED',
    results        TEXT,
    submitted_at   TEXT,
    updated_at     TEXT,
    promoted_at    TEXT,
    reject_reason  TEXT
)
"""


class ExperimentWorker:
    """
    Background worker that processes experiments through the full pipeline.

    Thread-safe. Only processes one experiment at a time to avoid
    resource contention with the live trading bot.
    """

    def __init__(
        self,
        db_path: str = "data/trade_memory.sqlite",
        experiment_engine=None,
        research_engine=None,
        data_provider: Optional[Callable[[List[str], str, str], Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.db_path           = db_path
        self.experiment_engine = experiment_engine
        self.research_engine   = research_engine
        self.config            = {**DEFAULT_WORKER_CONFIG, **(config or {})}
        self.data_provider     = data_provider or self._default_data_provider
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._running          = False
        self._thread: Optional[threading.Thread] = None
        self._current_item: Optional[WorkItem] = None
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_PIPELINE)
            conn.execute(_CREATE_HYPOTHESES)
            conn.execute(_CREATE_CANARY_APPROVALS)
            conn.commit()

    @staticmethod
    def _default_data_provider(symbols: List[str], period: str, interval: str) -> Dict[str, Any]:
        """Fetch real historical OHLCV per symbol. Returns {} on failure (fail-fast)."""
        try:
            from data.fetcher import fetch_latest_market_data
        except ImportError as e:
            logger.error(f"ExperimentWorker: data fetcher unavailable: {e}")
            return {}
        out = {}
        for sym in symbols:
            try:
                df = fetch_latest_market_data(sym, period=period, interval=interval)
                if df is not None and len(df) >= 100:
                    df = df.copy()
                    df.columns = [str(c).lower() for c in df.columns]
                    out[sym] = df
            except Exception as e:
                logger.warning(f"ExperimentWorker: fetch failed for {sym}: {e}")
        return out

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(
        self,
        params: Dict[str, Any],
        description: str,
        hypothesis_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        priority: int = 5,
    ) -> str:
        """
        Submit a new experiment to the queue.

        Returns the pipeline record ID.
        """
        if self.experiment_engine:
            try:
                exp = self.experiment_engine.propose(
                    description=description,
                    params=params,
                    proposed_by='experiment_worker',
                )
                exp_id = exp.id
            except Exception as e:
                logger.warning(f"ExperimentWorker: could not register with engine: {e}")
                exp_id = experiment_id or str(uuid.uuid4())
        else:
            exp_id = experiment_id or str(uuid.uuid4())

        item = WorkItem(
            experiment_id=exp_id,
            hypothesis_id=hypothesis_id,
            params=params,
            description=description,
            priority=priority,
        )
        self._save_pipeline(item)
        # Priority queue: (priority, timestamp, item) for deterministic ordering
        self._queue.put((priority, item.submitted_at, item))
        logger.info(f"ExperimentWorker: queued '{description}' (priority={priority})")
        return exp_id

    def start(self) -> None:
        """Start the background processing thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run_loop, daemon=True, name="ExperimentWorker")
        self._thread.start()
        logger.info("ExperimentWorker: started background thread")

    def stop(self) -> None:
        """Gracefully stop the worker after the current experiment completes."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("ExperimentWorker: stopped")

    def queue_size(self) -> int:
        return self._queue.qsize()

    def is_busy(self) -> bool:
        return self._current_item is not None

    def get_pipeline_status(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM experiment_pipeline ORDER BY submitted_at DESC LIMIT 50"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Worker loop ───────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        while self._running:
            try:
                priority, ts, item = self._queue.get(timeout=10)
                self._current_item = item
                self._process(item)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"ExperimentWorker: uncaught error: {e}", exc_info=True)
            finally:
                self._current_item = None

    def _process(self, item: WorkItem, start_stage: Optional[PipelineStage] = None) -> None:
        """Run an item through all pipeline stages.

        Stages may return pending=True (paper / canary collecting real
        observations): the experiment is PARKED at that stage — it neither
        passes nor fails until evidence arrives. Resume via check_pending().
        """
        logger.info(f"ExperimentWorker: processing '{item.description}' [{item.experiment_id[:8]}]")
        results: List[PipelineResult] = []
        self._stage_context: Dict[str, Any] = {}

        stages = [
            (PipelineStage.BACKTESTING,    self._run_backtest),
            (PipelineStage.WALK_FORWARD,   self._run_walk_forward),
            (PipelineStage.OUT_OF_SAMPLE,  self._run_oos),
            (PipelineStage.MONTE_CARLO,    self._run_monte_carlo),
            (PipelineStage.PAPER_TEST,     self._run_paper_test),
            (PipelineStage.CANARY,         self._run_canary),
        ]
        start_idx = 0
        if start_stage is not None:
            for i, (stage, _) in enumerate(stages):
                if stage == start_stage:
                    start_idx = i
                    break

        for stage, fn in stages[start_idx:]:
            self._update_stage(item.experiment_id, stage)
            result = fn(item)
            results.append(result)

            if result.pending:
                parked = (PipelineStage.PAPER_RUNNING if stage == PipelineStage.PAPER_TEST
                          else PipelineStage.CANARY_PENDING_APPROVAL
                          if result.metrics.get("awaiting_approval")
                          else PipelineStage.CANARY_RUNNING)
                self._update_stage(item.experiment_id, parked,
                                   results={stage.value: result.metrics})
                logger.info(
                    f"ExperimentWorker: ⏸ PARKED {item.experiment_id[:8]} at {parked.value}: "
                    f"{result.metrics.get('note', 'awaiting observations')}"
                )
                return

            if not result.passed:
                self._reject(item, result)
                return

            self._update_stage(item.experiment_id, stage,
                               results={stage.value: result.metrics})
            # Brief pause between stages so the live bot isn't starved
            time.sleep(0.1)

        self._promote(item, results)

    # ── Pipeline stage implementations ────────────────────────────────────────

    def _extract_strategy_config(self, item: WorkItem) -> Tuple[Optional[str], Dict, Dict, bool]:
        """Split item.params into (strategy_id, baseline, challenger_overrides, multi_variable)."""
        strategy_id = item.params.get("strategy") or item.params.get("strategy_id")
        baseline = dict(item.params.get("baseline") or {})
        challenger_overrides = {
            k: v for k, v in item.params.items() if k not in _RESERVED_PARAM_KEYS
        }
        multi = bool(item.params.get("multi_variable", False))
        return strategy_id, baseline, challenger_overrides, multi

    def _load_data(self, item: WorkItem) -> Dict[str, Any]:
        symbols = item.params.get("symbols") or self.config["symbols"]
        period = item.params.get("period") or self.config["period"]
        interval = item.params.get("interval") or self.config["interval"]
        return self.data_provider(symbols, period, interval)

    def _run_backtest(self, item: WorkItem) -> PipelineResult:
        """REAL strategy-specific backtest: run baseline (control) and
        challenger configurations of the ACTUAL strategy over the same
        historical market data with identical execution assumptions.

        Never validates by rescaling unrelated closed trades.
        """
        from core.holdout_manager import three_way_split
        from core.strategy_experiment_runner import StrategyExperimentRunner
        from core.validation_stats import parameter_robustness_score

        try:
            strategy_id, baseline, challenger_overrides, multi = self._extract_strategy_config(item)
            if not strategy_id:
                return PipelineResult(
                    PipelineStage.BACKTESTING, False,
                    reject_reason="Experiment must name a target strategy "
                                  "(params['strategy']); proxy validation from "
                                  "unrelated historical trades is not permitted",
                )
            if not challenger_overrides:
                return PipelineResult(
                    PipelineStage.BACKTESTING, False,
                    reject_reason="Experiment proposes no parameter changes",
                )

            data = self._load_data(item)
            if not data:
                return PipelineResult(
                    PipelineStage.BACKTESTING, False,
                    reject_reason="No historical market data available — "
                                  "cannot validate without a real backtest",
                )

            # Reserve the final 20% as untouched holdout; backtest uses discovery only
            n_bars = min(len(df) for df in data.values())
            split = three_way_split(n_bars)
            discovery_data = {s: df.iloc[: split.validation[1]] for s, df in data.items()}

            runner = StrategyExperimentRunner()
            challenger_params = {**baseline, **challenger_overrides}
            comparison = runner.run_experiment(
                strategy_id=strategy_id,
                baseline_params=baseline,
                challenger_params=challenger_params,
                data=discovery_data,
                allow_multi_variable=multi,
            )

            cm = comparison.challenger_metrics
            cfg = self.config
            checks: List[str] = []
            if cm.trades < cfg["min_backtest_trades"]:
                checks.append(f"challenger trades={cm.trades} < {cfg['min_backtest_trades']}")
            if cm.expectancy <= cfg["min_expectancy"]:
                checks.append(f"net expectancy={cm.expectancy:.4f} not positive")
            if cm.concentration.get("top_5_trades_pct", 0) > cfg["max_profit_concentration_top5"] \
                    and not item.params.get("rare_event"):
                checks.append(
                    f"profit concentration top5={cm.concentration['top_5_trades_pct']:.0%} "
                    f"> {cfg['max_profit_concentration_top5']:.0%} (mark rare_event to opt in)"
                )

            # Multiple-hypothesis correction: record this test in its family and
            # require Benjamini-Hochberg FDR survival — p<0.05 alone never qualifies.
            family = item.params.get("hypothesis_family") or (
                f"{strategy_id}:{'+'.join(sorted(comparison.changed_parameters))}"
            )
            fdr_ok = self._record_and_check_fdr(
                item, family, comparison.p_value_challenger_positive, cm.expectancy)
            if not fdr_ok:
                checks.append(
                    f"failed FDR correction (family='{family}', "
                    f"p={comparison.p_value_challenger_positive:.4f}, "
                    f"alpha={cfg['fdr_alpha']})"
                )

            # Parameter robustness: numeric single-parameter changes must sit on
            # a plateau, not a cliff.
            robustness = None
            changed = comparison.changed_parameters
            if len(changed) == 1:
                pname, (old_v, new_v) = next(iter(changed.items()))
                if isinstance(new_v, (int, float)) and not isinstance(new_v, bool):
                    neighbors = [new_v * m for m in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)]
                    sweep = runner.run_parameter_sweep(
                        strategy_id, challenger_params, pname, neighbors, discovery_data)
                    robustness = parameter_robustness_score(sweep)
                    if robustness < cfg["min_parameter_robustness"]:
                        checks.append(
                            f"parameter_robustness_score={robustness:.2f} < "
                            f"{cfg['min_parameter_robustness']} (narrow peak)"
                        )

            self._stage_context = {
                "comparison": comparison,
                "data": data,
                "split": split,
                "strategy_id": strategy_id,
                "challenger_params": challenger_params,
                "baseline_params": baseline,
            }

            metrics = {
                **comparison.to_dict(),
                "parameter_robustness_score": robustness,
                "hypothesis_family": family,
            }
            passed = not checks
            logger.info(
                f"ExperimentWorker backtest [{item.experiment_id[:8]}]: passed={passed} "
                f"challenger exp={cm.expectancy:.4f} trades={cm.trades} "
                f"delta_exp={comparison.delta_metrics['expectancy']:.4f}"
            )
            return PipelineResult(PipelineStage.BACKTESTING, passed, metrics,
                                  None if passed else "; ".join(checks))

        except ValueError as e:
            return PipelineResult(PipelineStage.BACKTESTING, False, reject_reason=str(e))
        except Exception as e:
            logger.error(f"ExperimentWorker backtest error: {e}", exc_info=True)
            return PipelineResult(PipelineStage.BACKTESTING, False, reject_reason=str(e))

    def _record_and_check_fdr(self, item: WorkItem, family: str,
                              p_value: float, metric: float) -> bool:
        from core.validation_stats import benjamini_hochberg
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO tested_hypotheses "
                "(hypothesis_id, experiment_id, family, p_value, metric, tested_at) "
                "VALUES (?,?,?,?,?,?)",
                (item.hypothesis_id, item.experiment_id, family, p_value, metric, _utcnow()),
            )
            conn.commit()
            rows = conn.execute(
                "SELECT experiment_id, p_value FROM tested_hypotheses WHERE family=?",
                (family,),
            ).fetchall()
        passed = benjamini_hochberg([r[1] for r in rows], alpha=self.config["fdr_alpha"])
        for (exp_id, _), ok in zip(rows, passed):
            if exp_id == item.experiment_id:
                return ok
        return False

    def _run_walk_forward(self, item: WorkItem) -> PipelineResult:
        """Walk-forward: run the challenger strategy over sequential unseen
        time windows of REAL market data (never shuffled folds of trade P&L)."""
        try:
            ctx = self._stage_context
            if not ctx:
                return PipelineResult(PipelineStage.WALK_FORWARD, False,
                                      reject_reason="missing backtest context")
            from core.strategy_experiment_runner import StrategyExperimentRunner
            runner = StrategyExperimentRunner()
            split = ctx["split"]
            wf_data = {s: df.iloc[: split.validation[1]] for s, df in ctx["data"].items()}
            wf = runner.run_walk_forward(ctx["strategy_id"], ctx["challenger_params"], wf_data)

            cfg = self.config
            passed = (
                wf["n_folds_with_trades"] >= 2
                and wf["positive_fold_fraction"] >= cfg["min_positive_fold_fraction"]
                and wf["mean_fold_expectancy"] > 0
            )
            reject = None if passed else (
                f"walk-forward: folds_with_trades={wf['n_folds_with_trades']} "
                f"positive_fraction={wf['positive_fold_fraction']:.0%} "
                f"mean_expectancy={wf['mean_fold_expectancy']:.4f}"
            )
            return PipelineResult(PipelineStage.WALK_FORWARD, passed, wf, reject)
        except Exception as e:
            return PipelineResult(PipelineStage.WALK_FORWARD, False, reject_reason=str(e))

    def _run_oos(self, item: WorkItem) -> PipelineResult:
        """Final holdout: the untouched last 20% of data, evaluated ONCE via
        the HoldoutManager after parameters are frozen."""
        try:
            ctx = self._stage_context
            if not ctx:
                return PipelineResult(PipelineStage.OUT_OF_SAMPLE, False,
                                      reject_reason="missing backtest context")
            from core.holdout_manager import HoldoutManager
            from core.strategy_experiment_runner import (
                BacktestMetrics, StrategyBacktester, StrategyExperimentRunner,
            )

            split = ctx["split"]
            data = ctx["data"]
            h_start, h_end = split.holdout
            sample_df = next(iter(data.values()))
            hs = str(sample_df.index[h_start]) if h_start < len(sample_df) else str(h_start)
            he = str(sample_df.index[-1])
            version = StrategyExperimentRunner._strategy_code_hash(ctx["strategy_id"])
            version_key = f"{version}:{item.experiment_id[:8]}"

            hm = HoldoutManager(self.db_path)
            hm.register(ctx["strategy_id"], hs, he, version_key)
            ok, reason = hm.can_access(ctx["strategy_id"], hs, he)
            if not ok:
                return PipelineResult(PipelineStage.OUT_OF_SAMPLE, False,
                                      reject_reason=f"holdout access denied: {reason}")

            backtester = StrategyBacktester()
            trades = backtester.run(ctx["strategy_id"], ctx["challenger_params"], data,
                                    index_range=(h_start, h_end - 1))
            m = BacktestMetrics.from_trades(trades)
            metrics = {**m.to_dict(), "holdout_start": hs, "holdout_end": he}
            hm.record_access(ctx["strategy_id"], hs, he, version_key, metrics)

            if m.trades < 5:
                return PipelineResult(
                    PipelineStage.OUT_OF_SAMPLE, False, metrics,
                    f"holdout produced only {m.trades} trades (need ≥5 for evidence)",
                )
            passed = m.expectancy > 0 and m.win_rate >= 0.35
            self._stage_context["oos_trades"] = trades
            return PipelineResult(
                PipelineStage.OUT_OF_SAMPLE, passed, metrics,
                None if passed else
                f"holdout expectancy={m.expectancy:.4f} win_rate={m.win_rate:.1%}",
            )
        except Exception as e:
            return PipelineResult(PipelineStage.OUT_OF_SAMPLE, False, reject_reason=str(e))

    def _run_monte_carlo(self, item: WorkItem) -> PipelineResult:
        """Monte Carlo: bootstrap THIS experiment's challenger trades (not
        unrelated closed positions) to estimate robustness of the P&L path."""
        try:
            ctx = self._stage_context
            comparison = ctx.get("comparison") if ctx else None
            if comparison is None:
                return PipelineResult(PipelineStage.MONTE_CARLO, False,
                                      reject_reason="missing challenger trades")
            pnls = [t.net_pnl for t in comparison.challenger_trades]
            pnls += [t.net_pnl for t in ctx.get("oos_trades", [])]
            if len(pnls) < 10:
                return PipelineResult(PipelineStage.MONTE_CARLO, False,
                                      reject_reason=f"only {len(pnls)} challenger trades for MC")

            rng = random.Random(42)
            n_sims, positive = 1000, 0
            for _ in range(n_sims):
                sample = rng.choices(pnls, k=len(pnls))
                if sum(sample) > 0:
                    positive += 1

            positive_rate = positive / n_sims
            passed = positive_rate >= GATE_MC_POSITIVE_RUNS
            return PipelineResult(
                PipelineStage.MONTE_CARLO, passed,
                {'positive_rate': positive_rate, 'simulations': n_sims,
                 'n_trades': len(pnls)},
                None if passed else f"MC positive={positive_rate:.0%} < {GATE_MC_POSITIVE_RUNS:.0%}",
            )
        except Exception as e:
            return PipelineResult(PipelineStage.MONTE_CARLO, False, reject_reason=str(e))

    def _run_paper_test(self, item: WorkItem) -> PipelineResult:
        """REAL paper gate: the experiment stays in PAPER_RUNNING until enough
        trades ATTRIBUTED TO THIS EXPERIMENT have closed in paper mode.

        Requirements (configurable): min_paper_trades, min_paper_days,
        positive expectancy, bounded drawdown. Never auto-passes.
        """
        from core.experiment_observations import ExperimentObservationTracker

        cfg = self.config
        tracker = ExperimentObservationTracker(self.db_path)
        summary = tracker.summary(item.experiment_id, phase="paper")

        needed_trades = int(item.params.get("min_paper_trades", cfg["min_paper_trades"]))
        needed_days = float(item.params.get("min_paper_days", cfg["min_paper_days"]))

        if summary["trades"] < needed_trades or summary["days"] < needed_days:
            return PipelineResult(
                PipelineStage.PAPER_TEST, passed=False, pending=True,
                metrics={
                    "note": "collecting attributed paper observations",
                    "trades": summary["trades"], "required_trades": needed_trades,
                    "days": round(summary["days"], 2), "required_days": needed_days,
                    **{k: summary[k] for k in ("net_return", "expectancy", "win_rate")},
                },
            )

        expected = self._stored_backtest_expectancy(item.experiment_id)
        checks: List[str] = []
        if summary["expectancy"] <= 0:
            checks.append(f"paper expectancy={summary['expectancy']:.4f} not positive")
        if summary["max_drawdown"] > cfg["max_paper_drawdown"]:
            checks.append(f"paper drawdown={summary['max_drawdown']:.2f} > "
                          f"{cfg['max_paper_drawdown']}")
        if expected is not None and expected > 0 and summary["expectancy"] < 0.25 * expected:
            checks.append(
                f"paper expectancy {summary['expectancy']:.4f} < 25% of backtest "
                f"expectation {expected:.4f} (edge not confirmed live)"
            )

        metrics = {**summary, "backtest_expectancy": expected}
        passed = not checks
        return PipelineResult(PipelineStage.PAPER_TEST, passed, metrics,
                              None if passed else "; ".join(checks))

    def _run_canary(self, item: WorkItem) -> PipelineResult:
        """REAL canary gate: limited real-money deployment requires EXPLICIT
        approval (auto_live_promotion=False by default) and then enough
        attributed canary observations. Never auto-passes.
        """
        from core.experiment_observations import ExperimentObservationTracker

        cfg = self.config
        if not cfg["auto_live_promotion"] and not self._canary_approved(item.experiment_id):
            return PipelineResult(
                PipelineStage.CANARY, passed=False, pending=True,
                metrics={
                    "note": "awaiting explicit live approval (auto_live_promotion=False)",
                    "awaiting_approval": True,
                    "canary_allocation_pct": cfg["canary_allocation_pct"],
                    "max_canary_notional": cfg["max_canary_notional"],
                },
            )

        tracker = ExperimentObservationTracker(self.db_path)
        summary = tracker.summary(item.experiment_id, phase="canary")
        needed_trades = int(item.params.get("min_canary_trades", cfg["min_canary_trades"]))
        needed_days = float(item.params.get("min_canary_days", cfg["min_canary_days"]))

        if summary["trades"] < needed_trades or summary["days"] < needed_days:
            return PipelineResult(
                PipelineStage.CANARY, passed=False, pending=True,
                metrics={
                    "note": "collecting attributed canary observations",
                    "trades": summary["trades"], "required_trades": needed_trades,
                    "days": round(summary["days"], 2), "required_days": needed_days,
                    "canary_allocation_pct": cfg["canary_allocation_pct"],
                    "max_canary_notional": cfg["max_canary_notional"],
                },
            )

        checks: List[str] = []
        if summary["expectancy"] <= 0:
            checks.append(f"canary expectancy={summary['expectancy']:.4f} not positive")
        if summary["max_drawdown"] > cfg["max_canary_drawdown"]:
            checks.append(f"canary drawdown={summary['max_drawdown']:.2f} > "
                          f"{cfg['max_canary_drawdown']}")
        if (summary["avg_slippage_pct"] is not None
                and summary["avg_slippage_pct"] > cfg["max_canary_slippage_deviation"]):
            checks.append(
                f"canary slippage={summary['avg_slippage_pct']:.4f} > "
                f"{cfg['max_canary_slippage_deviation']}"
            )

        passed = not checks
        return PipelineResult(PipelineStage.CANARY, passed, dict(summary),
                              None if passed else "; ".join(checks))

    # ── Paper/canary lifecycle helpers ───────────────────────────────────────

    def approve_canary(self, experiment_id: str, approved_by: str = "operator") -> None:
        """Explicit human approval required before any canary capital."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO canary_approvals (experiment_id, approved_by, approved_at) "
                "VALUES (?,?,?)",
                (experiment_id, approved_by, _utcnow()),
            )
            conn.commit()
        logger.info(f"ExperimentWorker: canary APPROVED for {experiment_id[:8]} by {approved_by}")

    def _canary_approved(self, experiment_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM canary_approvals WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
        return row is not None

    def check_pending(self) -> List[str]:
        """Re-evaluate experiments parked in PAPER_RUNNING / CANARY_* states.

        Call periodically (e.g. daily). Returns experiment ids whose state
        advanced (promoted or rejected).
        """
        advanced: List[str] = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM experiment_pipeline WHERE stage IN "
                "('PAPER_RUNNING','CANARY_PENDING_APPROVAL','CANARY_RUNNING')"
            ).fetchall()
        for row in rows:
            item = WorkItem(
                experiment_id=row["experiment_id"],
                hypothesis_id=row["hypothesis_id"],
                params=json.loads(row["params"] or "{}"),
                description=row["description"] or "",
            )
            resume = (PipelineStage.PAPER_TEST if row["stage"] == "PAPER_RUNNING"
                      else PipelineStage.CANARY)
            before = row["stage"]
            self._stage_context = {}
            self._process(item, start_stage=resume)
            after = self._current_stage(item.experiment_id)
            if after not in (before, None):
                advanced.append(item.experiment_id)
        return advanced

    def _current_stage(self, experiment_id: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT stage FROM experiment_pipeline WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
        return row[0] if row else None

    def _stored_backtest_expectancy(self, experiment_id: str) -> Optional[float]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT results FROM experiment_pipeline WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
        if not row or not row[0]:
            return None
        try:
            results = json.loads(row[0])
            return results.get("BACKTESTING", {}).get("challenger_metrics", {}).get("expectancy")
        except (json.JSONDecodeError, AttributeError):
            return None

    # ── Outcome ───────────────────────────────────────────────────────────────

    def _promote(self, item: WorkItem, results: List[PipelineResult]) -> None:
        all_metrics = {r.stage.value: r.metrics for r in results}
        self._update_stage(item.experiment_id, PipelineStage.PROMOTED,
                           results=all_metrics)
        self._register_promoted_alpha(item, all_metrics)
        if self.research_engine and item.hypothesis_id:
            from core.research_engine import HypothesisStatus
            self.research_engine.update_status(
                item.hypothesis_id, HypothesisStatus.VALIDATED,
                experiment_id=item.experiment_id, outcome='promoted',
            )
        logger.info(f"ExperimentWorker: ✅ PROMOTED {item.experiment_id[:8]} '{item.description}'")

    def _register_promoted_alpha(self, item: WorkItem, metrics: Dict) -> None:
        """Record the validated edge in the Alpha Library (LIVE_ELIGIBLE)."""
        try:
            from core.alpha_library import AlphaLibrary, AlphaState
            strategy_id = item.params.get("strategy") or item.params.get("strategy_id")
            if not strategy_id:
                return
            lib = AlphaLibrary(self.db_path)
            alpha_id = f"{strategy_id}:{item.experiment_id[:8]}"
            backtest = metrics.get("BACKTESTING", {})
            challenger = backtest.get("challenger_metrics", {}) if isinstance(backtest, dict) else {}
            lib.register(
                alpha_id, name=item.description or alpha_id,
                strategy_id=strategy_id,
                params={k: v for k, v in item.params.items()},
                experiment_id=item.experiment_id,
                sample_size=challenger.get("trades"),
                net_expectancy=challenger.get("expectancy"),
                win_rate=challenger.get("win_rate"),
                sharpe=challenger.get("sharpe"),
                max_drawdown=challenger.get("max_drawdown"),
                oos_metrics=metrics.get("OUT_OF_SAMPLE"),
                walk_forward_metrics=metrics.get("WALK_FORWARD"),
                paper_metrics=metrics.get("PAPER_TEST"),
                canary_metrics=metrics.get("CANARY"),
                parameter_robustness_score=backtest.get("parameter_robustness_score")
                if isinstance(backtest, dict) else None,
            )
            for state in (AlphaState.VALIDATING, AlphaState.PAPER,
                          AlphaState.LIVE_ELIGIBLE):
                lib.transition(alpha_id, state, reason="experiment pipeline promotion")
        except Exception as e:
            logger.warning(f"ExperimentWorker: alpha library registration failed: {e}")

    def _reject(self, item: WorkItem, result: PipelineResult) -> None:
        self._update_stage(item.experiment_id, PipelineStage.REJECTED,
                           reject_reason=result.reject_reason)
        if self.research_engine and item.hypothesis_id:
            from core.research_engine import HypothesisStatus
            self.research_engine.update_status(
                item.hypothesis_id, HypothesisStatus.REJECTED,
                experiment_id=item.experiment_id,
                outcome=f"rejected at {result.stage.value}: {result.reject_reason}",
            )
        logger.warning(
            f"ExperimentWorker: ❌ REJECTED {item.experiment_id[:8]} "
            f"at {result.stage.value}: {result.reject_reason}"
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_pipeline(self, item: WorkItem) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO experiment_pipeline "
                "(id,experiment_id,hypothesis_id,description,params,stage,submitted_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), item.experiment_id, item.hypothesis_id,
                 item.description, json.dumps(item.params),
                 PipelineStage.QUEUED.value, item.submitted_at, _utcnow()),
            )
            conn.commit()

    def _update_stage(
        self,
        experiment_id: str,
        stage: PipelineStage,
        results: Dict = None,
        reject_reason: str = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            merged_json = None
            if results:
                row = conn.execute(
                    "SELECT results FROM experiment_pipeline WHERE experiment_id=?",
                    (experiment_id,),
                ).fetchone()
                existing = {}
                if row and row[0]:
                    try:
                        existing = json.loads(row[0])
                    except json.JSONDecodeError:
                        existing = {}
                existing.update(results)
                merged_json = json.dumps(existing, default=str)
            conn.execute(
                "UPDATE experiment_pipeline SET stage=?, results=COALESCE(?,results), "
                "reject_reason=COALESCE(?,reject_reason), updated_at=?, "
                "promoted_at=CASE WHEN ?='PROMOTED' THEN ? ELSE promoted_at END "
                "WHERE experiment_id=?",
                (stage.value,
                 merged_json,
                 reject_reason,
                 _utcnow(),
                 stage.value, _utcnow(),
                 experiment_id),
            )
            conn.commit()

    @staticmethod
    def _sharpe(pnls: List[float]) -> float:
        if len(pnls) < 2:
            return 0.0
        avg = sum(pnls) / len(pnls)
        var = sum((p - avg)**2 for p in pnls) / len(pnls)
        std = var ** 0.5
        return avg / std if std > 0 else 0.0
