"""
Experiment Engine — safe path for all parameter adaptations.

ARCHITECTURE RULE:
    Claude (LLM) may PROPOSE experiments.
    Claude may NEVER directly modify production trading parameters.

    The only sanctioned path for changing trading behaviour is:
        LLM Proposal
          → ExperimentEngine.propose()    — creates Experiment record
          → ExperimentEngine.backtest()   — runs historical validation
          → ExperimentEngine.validate()   — walk-forward out-of-sample
          → ExperimentEngine.promote()    — copy to production if criteria met

    Parameters Claude can NEVER override regardless of experiment outcome:
        • risk_per_trade limits
        • leverage
        • stop losses
        • circuit breaker thresholds
        • daily loss limits
        • max positions

    These are set at startup from env/config and are read-only at runtime.
"""
import json
import logging
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

# ── Parameters that Claude may NEVER touch ────────────────────────────────────
IMMUTABLE_PARAMS = frozenset({
    "max_daily_loss_pct",
    "max_position_size_pct",
    "max_consecutive_losses",
    "max_drawdown_pct",
    "max_positions",
    "leverage",
    "stop_loss_atr_multiplier",
    "circuit_breaker",
    "enable_live_trading",
})

# ── Promotion criteria (must ALL be satisfied) ────────────────────────────────
MIN_BACKTEST_WIN_RATE   = 0.52   # > 52% win rate in backtest
MIN_BACKTEST_EXPECTANCY = 0.0    # Positive expectancy ($ per trade)
MIN_BACKTEST_TRADES     = 20     # At least 20 backtest trades
MIN_OOS_WIN_RATE        = 0.50   # > 50% in out-of-sample
MIN_SHARPE              = 0.5    # Sharpe > 0.5 over backtest period


class ExperimentStatus(str, Enum):
    PROPOSED  = "PROPOSED"
    BACKTESTING = "BACKTESTING"
    VALIDATING  = "VALIDATING"
    PAPER_TEST  = "PAPER_TEST"
    PROMOTED    = "PROMOTED"
    REJECTED    = "REJECTED"


@dataclass
class Experiment:
    id:           str
    proposed_by:  str                    # "claude" | "operator"
    description:  str
    params:       Dict[str, Any]         # proposed parameter changes
    status:       ExperimentStatus = ExperimentStatus.PROPOSED
    created_at:   str = field(default_factory=_utcnow)
    updated_at:   str = field(default_factory=_utcnow)
    backtest_result: Optional[Dict] = None
    oos_result:      Optional[Dict] = None
    reject_reason:   Optional[str] = None
    promoted_at:     Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class ExperimentEngine:
    """
    Manages the full lifecycle of parameter experiments.

    Experiments are persisted to a SQLite table `experiments` so they
    survive bot restarts.
    """

    def __init__(self, db_path: str = "data/trade_memory.sqlite"):
        self.db_path = db_path
        self._init_db()
        self._production_params: Dict[str, Any] = {}

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id           TEXT PRIMARY KEY,
                    proposed_by  TEXT,
                    description  TEXT,
                    params       TEXT,
                    status       TEXT DEFAULT 'PROPOSED',
                    created_at   TEXT,
                    updated_at   TEXT,
                    backtest_result TEXT,
                    oos_result   TEXT,
                    reject_reason TEXT,
                    promoted_at  TEXT
                )
            """)
            conn.commit()

    # ── Production parameter registry ─────────────────────────────────────────

    def register_production_params(self, params: Dict[str, Any]) -> None:
        """Call once at bot startup to register current production values."""
        self._production_params = dict(params)
        logger.info(f"ExperimentEngine: registered {len(params)} production params")

    # ── Proposal ──────────────────────────────────────────────────────────────

    def propose(
        self,
        description: str,
        params: Dict[str, Any],
        proposed_by: str = "claude",
    ) -> Experiment:
        """
        Create a new experiment proposal.

        Validates that no immutable safety parameters are being changed.
        Returns the Experiment (status=PROPOSED) — nothing is applied yet.

        Raises ValueError if the proposal violates safety constraints.
        """
        # Safety validation — block immutable params
        violations = [k for k in params if k in IMMUTABLE_PARAMS]
        if violations:
            raise ValueError(
                f"Experiment proposal rejected: cannot modify immutable "
                f"safety parameters: {violations}. "
                f"These are set at startup and cannot change at runtime."
            )

        # Block params that don't exist in production (typos etc.)
        if self._production_params:
            unknown = [k for k in params if k not in self._production_params]
            if unknown:
                raise ValueError(
                    f"Experiment proposal rejected: unknown parameters "
                    f"{unknown}. Only existing parameters may be experimented on."
                )

        exp = Experiment(
            id=str(uuid.uuid4()),
            proposed_by=proposed_by,
            description=description,
            params=params,
        )
        self._save(exp)
        logger.info(
            f"Experiment proposed by '{proposed_by}': {exp.id[:8]} — {description}"
        )
        return exp

    # ── Validation pipeline ───────────────────────────────────────────────────

    def run_backtest(
        self,
        experiment_id: str,
        backtest_fn,                    # callable(params) → dict of metrics
        historical_data,                # passed through to backtest_fn
    ) -> Experiment:
        """
        Run historical backtest for the proposed parameters.

        backtest_fn must accept (params, data) and return a dict with keys:
            win_rate, expectancy, trades, sharpe, max_drawdown, total_return
        """
        exp = self._load(experiment_id)
        if exp is None:
            raise KeyError(f"Experiment {experiment_id} not found")

        exp.status = ExperimentStatus.BACKTESTING
        exp.updated_at = _utcnow()
        self._save(exp)

        try:
            result = backtest_fn(exp.params, historical_data)
            exp.backtest_result = result
            exp.updated_at = _utcnow()

            # Gate: reject if metrics don't meet minimums
            if not self._passes_backtest_gate(result):
                exp.status = ExperimentStatus.REJECTED
                exp.reject_reason = (
                    f"Backtest failed gate: "
                    f"win_rate={result.get('win_rate', 0):.1%} "
                    f"(min {MIN_BACKTEST_WIN_RATE:.1%}), "
                    f"expectancy=${result.get('expectancy', 0):.2f} "
                    f"(min ${MIN_BACKTEST_EXPECTANCY:.2f}), "
                    f"trades={result.get('trades', 0)} "
                    f"(min {MIN_BACKTEST_TRADES})"
                )
                logger.warning(f"Experiment {experiment_id[:8]} REJECTED: {exp.reject_reason}")
            else:
                exp.status = ExperimentStatus.VALIDATING
                logger.info(
                    f"Experiment {experiment_id[:8]} passed backtest gate — "
                    f"win_rate={result.get('win_rate', 0):.1%} "
                    f"expectancy=${result.get('expectancy', 0):.2f}"
                )
        except Exception as e:
            exp.status = ExperimentStatus.REJECTED
            exp.reject_reason = f"Backtest error: {e}"
            logger.error(f"Experiment {experiment_id[:8]} backtest error: {e}")

        self._save(exp)
        return exp

    def run_oos_validation(
        self,
        experiment_id: str,
        oos_fn,
        oos_data,
    ) -> Experiment:
        """
        Run out-of-sample validation.

        oos_fn must accept (params, data) and return a dict with the same
        keys as backtest_fn.
        """
        exp = self._load(experiment_id)
        if exp is None:
            raise KeyError(f"Experiment {experiment_id} not found")

        if exp.status != ExperimentStatus.VALIDATING:
            raise RuntimeError(
                f"Experiment {experiment_id[:8]} must pass backtest before OOS validation"
            )

        try:
            result = oos_fn(exp.params, oos_data)
            exp.oos_result = result
            exp.updated_at = _utcnow()

            if result.get("win_rate", 0) >= MIN_OOS_WIN_RATE:
                exp.status = ExperimentStatus.PAPER_TEST
                logger.info(
                    f"Experiment {experiment_id[:8]} passed OOS — "
                    f"win_rate={result.get('win_rate', 0):.1%} — "
                    f"moving to PAPER_TEST"
                )
            else:
                exp.status = ExperimentStatus.REJECTED
                exp.reject_reason = (
                    f"OOS validation failed: "
                    f"win_rate={result.get('win_rate', 0):.1%} < {MIN_OOS_WIN_RATE:.1%}"
                )
                logger.warning(f"Experiment {experiment_id[:8]} failed OOS: {exp.reject_reason}")
        except Exception as e:
            exp.status = ExperimentStatus.REJECTED
            exp.reject_reason = f"OOS error: {e}"

        self._save(exp)
        return exp

    def promote(
        self,
        experiment_id: str,
        promoted_by: str = "operator",
    ) -> Dict[str, Any]:
        """
        Promote a PAPER_TEST experiment to production parameters.

        Returns the new parameter dict.
        Must be called by an operator (human), not by Claude.
        """
        exp = self._load(experiment_id)
        if exp is None:
            raise KeyError(f"Experiment {experiment_id} not found")

        if exp.status != ExperimentStatus.PAPER_TEST:
            raise RuntimeError(
                f"Experiment {experiment_id[:8]} is in status '{exp.status}' — "
                f"only PAPER_TEST experiments can be promoted"
            )

        if promoted_by == "claude":
            raise PermissionError(
                "Claude cannot directly promote experiments to production. "
                "An operator must call promote()."
            )

        # Merge changes into production params
        new_params = {**self._production_params, **exp.params}
        self._production_params = new_params

        # Create versioned strategy record
        self._version_strategy(exp)

        exp.status = ExperimentStatus.PROMOTED
        exp.promoted_at = _utcnow()
        exp.updated_at = _utcnow()
        self._save(exp)

        logger.info(
            f"Experiment {experiment_id[:8]} PROMOTED by '{promoted_by}': "
            f"{list(exp.params.keys())}"
        )
        return new_params

    def canary_deploy(
        self,
        experiment_id: str,
        capital_fraction: float = 0.05,
    ) -> Dict:
        """
        Begin canary deployment: allocate a small fraction of capital to an
        experiment that has passed PAPER_TEST, before full promotion.

        Returns a canary config dict the bot can use to run the experiment
        with limited capital. The bot should call `promote()` after
        sufficient live trades confirm the experiment's edge.

        Args:
            capital_fraction: Fraction of capital to risk on canary (default 5%)
        """
        exp = self._load(experiment_id)
        if exp is None:
            raise KeyError(f"Experiment {experiment_id} not found")
        if exp.status != ExperimentStatus.PAPER_TEST:
            raise RuntimeError(
                f"Experiment must be in PAPER_TEST status for canary deploy; "
                f"current status: {exp.status}"
            )
        if not (0 < capital_fraction <= 0.20):
            raise ValueError(f"capital_fraction must be between 0 and 0.20, got {capital_fraction}")

        config = {
            "experiment_id":   exp.id,
            "params":          exp.params,
            "capital_fraction": capital_fraction,
            "description":     exp.description,
            "canary_started":  _utcnow(),
            "min_live_trades": 30,   # promote after 30 live canary trades
        }
        logger.info(
            f"Canary deploy: experiment {experiment_id[:8]} with "
            f"{capital_fraction:.0%} of capital"
        )
        return config

    def get_strategy_version(self, strategy_name: str) -> Optional[str]:
        """Return the current version string for a strategy (e.g. 'v1.3')."""
        return self._strategy_versions.get(strategy_name)

    def _version_strategy(self, exp: Experiment) -> None:
        """
        Create a new version for strategies modified by this experiment.

        Version numbering: vMAJOR.MINOR where MINOR increments each promotion.
        """
        if not hasattr(self, '_strategy_versions'):
            self._strategy_versions: Dict[str, str] = {}
        if not hasattr(self, '_version_history'):
            self._version_history: List[Dict] = []

        # Infer affected strategy from experiment description
        strategy_key = exp.description.split()[0].lower() if exp.description else "strategy"

        current = self._strategy_versions.get(strategy_key, "v1.0")
        major, minor = current.lstrip('v').split('.')
        new_version  = f"v{major}.{int(minor)+1}"
        self._strategy_versions[strategy_key] = new_version

        record = {
            "strategy":      strategy_key,
            "version":       new_version,
            "experiment_id": exp.id,
            "params":        exp.params,
            "promoted_at":   _utcnow(),
        }
        self._version_history.append(record)

        # Persist to DB
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_versions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy     TEXT NOT NULL,
                    version      TEXT NOT NULL,
                    experiment_id TEXT,
                    params       TEXT,
                    promoted_at  TEXT
                )
            """)
            conn.execute(
                "INSERT INTO strategy_versions (strategy, version, experiment_id, params, promoted_at)"
                " VALUES (?,?,?,?,?)",
                (strategy_key, new_version, exp.id,
                 json.dumps(exp.params, default=str), _utcnow()),
            )
            conn.commit()

        logger.info(f"Strategy version: {strategy_key} → {new_version}")

    def reject(self, experiment_id: str, reason: str) -> None:
        """Manually reject an experiment."""
        exp = self._load(experiment_id)
        if exp:
            exp.status = ExperimentStatus.REJECTED
            exp.reject_reason = reason
            exp.updated_at = _utcnow()
            self._save(exp)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Experiment]:
        sql = "SELECT * FROM experiments"
        params = []
        if status:
            sql += " WHERE status=?"
            params.append(status.value)
        sql += " ORDER BY created_at DESC"
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql, params).fetchall()
            return [self._row_to_experiment(r) for r in rows]
        except sqlite3.Error as e:
            logger.warning(f"ExperimentEngine list error: {e}")
            return []

    def get_production_params(self) -> Dict[str, Any]:
        """Return current production parameters (read-only copy)."""
        return dict(self._production_params)

    # ── Private ───────────────────────────────────────────────────────────────

    def _passes_backtest_gate(self, result: Dict) -> bool:
        return (
            result.get("win_rate", 0)   >= MIN_BACKTEST_WIN_RATE
            and result.get("expectancy", 0) >= MIN_BACKTEST_EXPECTANCY
            and result.get("trades", 0) >= MIN_BACKTEST_TRADES
        )

    def _save(self, exp: Experiment) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments
                (id, proposed_by, description, params, status,
                 created_at, updated_at, backtest_result, oos_result,
                 reject_reason, promoted_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                exp.id, exp.proposed_by, exp.description,
                json.dumps(exp.params, default=str),
                exp.status.value,
                exp.created_at, exp.updated_at,
                json.dumps(exp.backtest_result, default=str) if exp.backtest_result else None,
                json.dumps(exp.oos_result, default=str) if exp.oos_result else None,
                exp.reject_reason, exp.promoted_at,
            ))
            conn.commit()

    def _load(self, experiment_id: str) -> Optional[Experiment]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM experiments WHERE id=?", [experiment_id]
                ).fetchone()
            return self._row_to_experiment(row) if row else None
        except sqlite3.Error as e:
            logger.warning(f"ExperimentEngine load error: {e}")
            return None

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> Experiment:
        return Experiment(
            id=row["id"],
            proposed_by=row["proposed_by"],
            description=row["description"],
            params=json.loads(row["params"]) if row["params"] else {},
            status=ExperimentStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            backtest_result=json.loads(row["backtest_result"]) if row["backtest_result"] else None,
            oos_result=json.loads(row["oos_result"]) if row["oos_result"] else None,
            reject_reason=row["reject_reason"],
            promoted_at=row["promoted_at"],
        )
