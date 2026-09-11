"""
Validation Report — standardised report for any candidate.

PHASE 22

Generates a complete validation report including:
    PERFORMANCE           RISK               BENCHMARKS
    STATISTICAL TESTS     REGIME PERFORMANCE EXECUTION COSTS
    CALIBRATION           MONTE CARLO        STRESS SCENARIOS
    OOS PERFORMANCE       PAPER PERFORMANCE  PROMOTION STATUS

Reports are persisted as JSON and human-readable text.
Every report gets a unique report_id for traceability.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from validation.engine import ValidationEngine, ValidationResult
from validation.benchmark_engine import BenchmarkEngine, BenchmarkReport
from validation.monte_carlo import MonteCarloEngine, MonteCarloResult
from validation.promotion_gates import PromotionGates, PromotionStage, PromotionConfig

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

_CREATE_REPORTS = """
CREATE TABLE IF NOT EXISTS validation_reports (
    report_id       TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    report_type     TEXT NOT NULL,
    stage           TEXT,
    generated_at    TEXT NOT NULL,
    report_json     TEXT NOT NULL,
    promotion_passed INTEGER DEFAULT 0,
    notes           TEXT DEFAULT ''
)
"""


@dataclass
class ValidationReport:
    """Complete validation report for a candidate strategy/model/experiment."""
    report_id:          str
    label:              str
    report_type:        str       # 'strategy' | 'model' | 'experiment' | 'campaign'
    generated_at:       str

    # Core sections (all optional — populated as data is available)
    performance:        Optional[ValidationResult] = None
    oos_performance:    Optional[ValidationResult] = None
    paper_performance:  Optional[ValidationResult] = None
    benchmark:          Optional[BenchmarkReport] = None
    monte_carlo:        Optional[MonteCarloResult] = None
    walk_forward:       Optional[Any] = None     # WalkForwardResult
    promotion_gate:     Optional[Any] = None     # GateResult
    statistical_tests:  Dict = field(default_factory=dict)
    regime_performance: Dict = field(default_factory=dict)
    execution_summary:  Dict = field(default_factory=dict)
    calibration:        Dict = field(default_factory=dict)
    notes:              str = ''

    def promotion_passed(self) -> bool:
        if self.promotion_gate is None:
            return False
        return getattr(self.promotion_gate, 'passed', False)

    def to_dict(self) -> Dict:
        d = {
            'report_id':    self.report_id,
            'label':        self.label,
            'report_type':  self.report_type,
            'generated_at': self.generated_at,
            'notes':        self.notes,
            'promotion_passed': self.promotion_passed(),
        }
        if self.performance:
            d['performance'] = self.performance.to_dict()
        if self.oos_performance:
            d['oos_performance'] = self.oos_performance.to_dict()
        if self.paper_performance:
            d['paper_performance'] = self.paper_performance.to_dict()
        if self.benchmark:
            d['benchmark'] = {'label': self.benchmark.label,
                              'comparisons': [vars(c) for c in self.benchmark.comparisons]}
        if self.monte_carlo:
            d['monte_carlo'] = {
                'n_simulations': self.monte_carlo.n_simulations,
                'median_return': self.monte_carlo.median_return,
                'p5_return':     self.monte_carlo.p5_return,
                'p1_return':     self.monte_carlo.p1_return,
                'prob_loss':     self.monte_carlo.prob_loss,
                'prob_ruin':     self.monte_carlo.prob_ruin,
                'median_sharpe': self.monte_carlo.median_sharpe,
            }
        d['statistical_tests']  = self.statistical_tests
        d['regime_performance'] = self.regime_performance
        d['execution_summary']  = self.execution_summary
        d['calibration']        = self.calibration
        return d

    def text_report(self) -> str:
        """Complete human-readable text report."""
        sections = [
            self._header(),
            self._performance_section(),
            self._risk_section(),
            self._oos_section(),
            self._paper_section(),
            self._benchmark_section(),
            self._mc_section(),
            self._gate_section(),
            self._footer(),
        ]
        return "\n\n".join(s for s in sections if s)

    # ── Section renderers ─────────────────────────────────────────────────────

    def _header(self) -> str:
        status = "✓ PROMOTION PASSED" if self.promotion_passed() else "✗ PROMOTION BLOCKED"
        return "\n".join([
            "=" * 70,
            f"  VALIDATION REPORT: {self.label}",
            f"  Type: {self.report_type.upper()}   Generated: {self.generated_at}",
            f"  Status: {status}",
            "=" * 70,
        ])

    def _performance_section(self) -> str:
        if not self.performance:
            return ""
        r = self.performance
        return "\n".join([
            "BACKTEST PERFORMANCE (NET AFTER COSTS)",
            "─" * 40,
            f"  Trades:           {r.trade_count:>10}",
            f"  Net PnL:          ${r.total_pnl_net:>10,.2f}",
            f"  Total return:     {r.total_return_pct:>10.2f}%",
            f"  Ann. return:      {r.ann_return_pct:>10.2f}%",
            f"  Fees paid:        ${r.total_fees:>10,.2f}",
            f"  Slippage:         ${r.total_slippage:>10,.2f}",
        ])

    def _risk_section(self) -> str:
        if not self.performance:
            return ""
        r = self.performance
        return "\n".join([
            "RISK METRICS",
            "─" * 40,
            f"  Win rate:         {r.win_rate:>10.1%}",
            f"  Expectancy:       ${r.expectancy:>10.4f}",
            f"  Profit factor:    {r.profit_factor:>10.3f}",
            f"  Sharpe:           {r.sharpe:>10.3f}",
            f"  Sortino:          {r.sortino:>10.3f}",
            f"  Calmar:           {r.calmar:>10.3f}",
            f"  Max drawdown:     ${r.max_drawdown:>10,.2f}",
            f"  CVaR 5%:          ${r.cvar_5pct:>10,.2f}",
        ])

    def _oos_section(self) -> str:
        if not self.oos_performance:
            return ""
        r = self.oos_performance
        return "\n".join([
            "OUT-OF-SAMPLE PERFORMANCE",
            "─" * 40,
            f"  Trades:           {r.trade_count:>10}",
            f"  Net PnL:          ${r.total_pnl_net:>10,.2f}",
            f"  Expectancy:       ${r.expectancy:>10.4f}",
            f"  Sharpe:           {r.sharpe:>10.3f}",
        ])

    def _paper_section(self) -> str:
        if not self.paper_performance:
            return ""
        r = self.paper_performance
        return "\n".join([
            "PAPER TRADING PERFORMANCE",
            "─" * 40,
            f"  Trades:           {r.trade_count:>10}",
            f"  Net PnL:          ${r.total_pnl_net:>10,.2f}",
            f"  Expectancy:       ${r.expectancy:>10.4f}",
            f"  Sharpe:           {r.sharpe:>10.3f}",
        ])

    def _benchmark_section(self) -> str:
        if not self.benchmark:
            return ""
        return self.benchmark.summary()

    def _mc_section(self) -> str:
        if not self.monte_carlo:
            return ""
        mc = self.monte_carlo
        return "\n".join([
            "MONTE CARLO STRESS TEST",
            "─" * 40,
            f"  Simulations:      {mc.n_simulations:>10,}",
            f"  Median return:    {mc.median_return:>10.2f}%",
            f"  5th pct return:   {mc.p5_return:>10.2f}%",
            f"  P(loss):          {mc.prob_loss:>10.1%}",
            f"  P(ruin):          {mc.prob_ruin:>10.1%}",
        ])

    def _gate_section(self) -> str:
        if not self.promotion_gate:
            return ""
        return self.promotion_gate.summary()

    def _footer(self) -> str:
        return "=" * 70


class ValidationReportGenerator:
    """Generates and persists validation reports."""

    def __init__(
        self,
        capital: float = 10_000.0,
        db_path: str = "data/trade_memory.sqlite",
        promotion_config: PromotionConfig = None,
    ):
        self.capital    = capital
        self.db_path    = db_path
        self._ve        = ValidationEngine(capital=capital)
        self._bm        = BenchmarkEngine()
        self._mc        = MonteCarloEngine(capital=capital)
        self._gates     = PromotionGates(config=promotion_config, db_path=db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_REPORTS)
            conn.commit()

    def generate(
        self,
        label:             str,
        report_type:       str,
        trades:            List = None,        # backtest / train trades
        oos_trades:        List = None,
        paper_trades:      List = None,
        walk_forward:      Any = None,
        promotion_stage:   PromotionStage = None,
        execution_report:  Any = None,
        calibration_gap:   float = 0.0,
        notes:             str = '',
    ) -> ValidationReport:
        """Generate a complete validation report."""
        from validation.engine import Trade

        report_id = str(uuid.uuid4())
        report = ValidationReport(
            report_id=report_id,
            label=label,
            report_type=report_type,
            generated_at=_utcnow(),
            notes=notes,
        )

        # Core performance
        if trades:
            result = self._ve.evaluate(trades, label=label)
            report.performance = result

            # Benchmark comparison
            if result.trade_count > 0:
                report.benchmark = self._bm.compare(result)

            # Monte Carlo (only for sufficient trades)
            if result.trade_count >= 20:
                report.monte_carlo = self._mc.run(trades, label=label)

        # OOS
        if oos_trades:
            report.oos_performance = self._ve.evaluate(oos_trades, label=f"{label} OOS")

        # Paper
        if paper_trades:
            report.paper_performance = self._ve.evaluate(paper_trades, label=f"{label} Paper")

        # Walk-forward
        report.walk_forward = walk_forward

        # Promotion gate
        if promotion_stage:
            report.promotion_gate = self._gates.evaluate(
                stage=promotion_stage,
                validation_result=report.performance,
                walk_forward_result=walk_forward,
                execution_report=execution_report,
                calibration_gap=calibration_gap,
                label=label,
            )

        # Persist
        self._save(report)
        logger.info(f"ValidationReport: generated {report_id[:8]} for {label}")
        return report

    def get(self, report_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM validation_reports WHERE report_id=?", (report_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d['data'] = json.loads(d.pop('report_json', '{}'))
        return d

    def list_reports(self, label: str = None, limit: int = 20) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if label:
                rows = conn.execute(
                    "SELECT report_id,label,report_type,generated_at,promotion_passed "
                    "FROM validation_reports WHERE label=? ORDER BY generated_at DESC LIMIT ?",
                    (label, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT report_id,label,report_type,generated_at,promotion_passed "
                    "FROM validation_reports ORDER BY generated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def _save(self, report: ValidationReport) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO validation_reports "
                    "(report_id,label,report_type,stage,generated_at,report_json,promotion_passed,notes)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (report.report_id, report.label, report.report_type,
                     getattr(report.promotion_gate, 'stage', {__class__: None}.get(__class__) or '') if report.promotion_gate else '',
                     report.generated_at, json.dumps(report.to_dict()),
                     int(report.promotion_passed()), report.notes),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"ValidationReport: persist error: {e}")
