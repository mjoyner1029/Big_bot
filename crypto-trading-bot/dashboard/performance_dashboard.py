"""
Performance Dashboard — CLI text dashboards for all system dimensions.

Sections:
    Portfolio       — equity, open positions, PnL
    Strategies      — per-strategy health, expectancy, Sharpe
    Experiments     — active experiments, pipeline status
    Models          — Champion/Challenger status, AUC, training history
    Feature Importance — top features by SHAP / native importance
    Market Regimes  — regime distribution, which regimes are profitable
    Execution Quality — slippage, fees, latency, rejection rate
    Research Queue  — open hypotheses, queued experiments
    Drift           — last drift report summary

Each section is a method returning a formatted string.
The run() method prints all sections or a subset.
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

_BAR = "=" * 60
_SEP = "-" * 60


class PerformanceDashboard:
    """
    Aggregates data from all system components and renders CLI dashboards.

    All components are optional — missing components produce placeholder text.
    """

    def __init__(
        self,
        position_manager=None,
        strategy_health: "StrategyHealthMonitor" = None,
        experiment_engine=None,
        champion_challenger=None,
        feature_importance=None,
        execution_quality=None,
        research_engine=None,
        drift_detector=None,
        trade_memory=None,
        capital: float = 0.0,
    ):
        self.pm    = position_manager
        self.sh    = strategy_health
        self.ee    = experiment_engine
        self.cc    = champion_challenger
        self.fi    = feature_importance
        self.eq    = execution_quality
        self.re    = research_engine
        self.dd    = drift_detector
        self.tm    = trade_memory
        self.capital = capital

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, sections: Optional[List[str]] = None) -> str:
        """
        Render the dashboard.

        Args:
            sections: list of section names to include, or None for all.
        """
        all_sections = [
            'header', 'portfolio', 'strategies', 'models',
            'experiments', 'execution', 'research', 'drift',
        ]
        to_render = sections or all_sections
        parts = []
        for section in to_render:
            method = getattr(self, f'_section_{section}', None)
            if method:
                try:
                    parts.append(method())
                except Exception as e:
                    parts.append(f"[{section}: error — {e}]")
        return "\n\n".join(parts)

    # ── Sections ──────────────────────────────────────────────────────────────

    def _section_header(self) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return "\n".join([
            _BAR,
            "  AUTONOMOUS TRADING SYSTEM — PERFORMANCE DASHBOARD",
            f"  {now}",
            _BAR,
        ])

    def _section_portfolio(self) -> str:
        lines = ["PORTFOLIO", _SEP]
        if self.tm:
            try:
                report = self.tm.performance_report(days=7)
                lines += [
                    f"  7-day PnL:        ${report.get('total_pnl', 0):.2f}",
                    f"  7-day trades:     {report.get('total_trades', 0)}",
                    f"  7-day win rate:   {report.get('win_rate', 0):.1%}",
                    f"  Capital:          ${self.capital:,.0f}",
                ]
            except Exception as e:
                lines.append(f"  [error: {e}]")
        else:
            lines.append("  [trade_memory not configured]")

        if self.pm:
            try:
                positions = self.pm.get_open_positions()
                lines += [
                    f"  Open positions:   {len(positions)}",
                ]
                for p in positions[:5]:
                    pnl = p.get('unrealized_pnl', p.get('net_pnl', 0)) or 0
                    lines.append(
                        f"    {p.get('symbol','?'):12s} "
                        f"{p.get('strategy','?'):15s} "
                        f"PnL=${pnl:.2f}"
                    )
            except Exception as e:
                lines.append(f"  [positions error: {e}]")
        return "\n".join(lines)

    def _section_strategies(self) -> str:
        lines = ["STRATEGY HEALTH", _SEP]
        if self.sh:
            try:
                summary = self.sh.summary()
                lines.append(f"  Healthy:  {len(summary.get('healthy', []))}")
                lines.append(f"  Degraded: {len(summary.get('degraded', []))}")
                lines.append(f"  Paused:   {len(summary.get('paused', []))}")
                if summary.get('alerts'):
                    lines.append("  Alerts:")
                    for a in summary['alerts'][:3]:
                        lines.append(f"    ⚠ [{a.get('severity','')}] {a.get('description','')}")
                health_map = self.sh.run()
                if health_map:
                    lines.append("")
                    lines.append(f"  {'Strategy':<20s} {'Trades':>6} {'WR':>6} {'Exp':>8} {'Sharpe':>7}")
                    for name, h in sorted(health_map.items())[:10]:
                        status = "✓" if h.healthy else "✗"
                        lines.append(
                            f"  {status} {name:<18s} {h.trades:>6} {h.win_rate:>5.1%} "
                            f"${h.expectancy:>7.2f} {h.sharpe:>6.2f}"
                        )
            except Exception as e:
                lines.append(f"  [error: {e}]")
        else:
            lines.append("  [strategy_health not configured]")
        return "\n".join(lines)

    def _section_models(self) -> str:
        lines = ["MODEL STATUS (CHAMPION/CHALLENGER)", _SEP]
        if self.cc:
            try:
                summary = self.cc.get_champion_summary()
                champ = summary.get('champion')
                chal  = summary.get('challenger')

                if champ:
                    lines += [
                        f"  Champion:   v{champ['version_id']} "
                        f"(trained {champ.get('training_rows',0)} rows, "
                        f"AUC={champ.get('auc',0):.3f})",
                        f"  Promoted:   {champ.get('promoted_at','unknown')}",
                    ]
                else:
                    lines.append("  Champion:   [none]")

                if chal:
                    lines += [
                        f"  Challenger: v{chal['version_id']} "
                        f"({chal.get('training_rows',0)} rows, "
                        f"AUC={chal.get('auc',0):.3f})",
                    ]
                else:
                    lines.append("  Challenger: [none]")

                lines.append(f"  Total versions: {summary.get('total_versions', 0)}")
            except Exception as e:
                lines.append(f"  [error: {e}]")

        if self.fi:
            try:
                report = self.fi.get_last_report()
                if report:
                    gi = report.get('global_importance', {})
                    top = sorted(gi.items(), key=lambda kv: kv[1], reverse=True)[:5]
                    lines.append("\n  Feature Importance (Top 5):")
                    for feat, imp in top:
                        bar = "█" * int(imp * 10)
                        lines.append(f"    {feat:<25s} {bar:<10s} {imp:.3f}")
            except Exception as e:
                lines.append(f"  [feature importance error: {e}]")

        return "\n".join(lines)

    def _section_experiments(self) -> str:
        lines = ["EXPERIMENTS & RESEARCH", _SEP]
        if self.ee:
            try:
                # Get recent experiments from DB
                import sqlite3
                with sqlite3.connect("data/trade_memory.sqlite") as conn:
                    rows = conn.execute(
                        "SELECT id, description, status FROM experiments "
                        "ORDER BY created_at DESC LIMIT 5"
                    ).fetchall()
                if rows:
                    lines.append("  Recent experiments:")
                    for r in rows:
                        lines.append(f"    [{r[2]:<12}] {r[1][:50]}")
                else:
                    lines.append("  No experiments recorded")
            except Exception as e:
                lines.append(f"  [error: {e}]")
        if self.re:
            try:
                summary = self.re.get_summary()
                lines += [
                    f"  Hypotheses: {summary.get('total', 0)} total, "
                    f"{summary.get('open', 0)} open, "
                    f"{summary.get('validated', 0)} validated",
                ]
            except Exception as e:
                lines.append(f"  [research error: {e}]")
        return "\n".join(lines)

    def _section_execution(self) -> str:
        lines = ["EXECUTION QUALITY", _SEP]
        if self.eq:
            try:
                summary = self.eq.summarize(period_days=7)
                for line in summary.split("\n")[1:]:  # skip header
                    lines.append(f"  {line}")
            except Exception as e:
                lines.append(f"  [error: {e}]")
        else:
            lines.append("  [execution_quality not configured]")
        return "\n".join(lines)

    def _section_research(self) -> str:
        lines = ["RESEARCH QUEUE", _SEP]
        if self.re:
            try:
                hypotheses = self.re.get_open_hypotheses()
                lines.append(f"  Open hypotheses: {len(hypotheses)}")
                for h in hypotheses[:5]:
                    lines.append(f"    [{h.get('priority','?'):<4}] {h.get('title','?')[:50]}")
            except Exception as e:
                lines.append(f"  [error: {e}]")
        else:
            lines.append("  [research_engine not configured]")
        return "\n".join(lines)

    def _section_drift(self) -> str:
        lines = ["DRIFT DETECTION", _SEP]
        if self.dd:
            try:
                report = self.dd.get_report()
                if report:
                    lines += [
                        f"  Drift detected: {report.drift_detected}",
                        f"  Allocation factor: {report.allocation_factor:.0%}",
                        f"  Retrain required: {report.retrain_required}",
                        f"  Summary: {report.summary[:100]}",
                    ]
                else:
                    lines.append("  [no drift report yet — run drift_detector.run()]")
            except Exception as e:
                lines.append(f"  [error: {e}]")
        else:
            lines.append("  [drift_detector not configured]")
        return "\n".join(lines)
