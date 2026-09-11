"""
ValidationEngine — centralised, standardised strategy/model evaluation.

RULE: NET performance after realistic costs is authoritative.
      Never report gross returns. All metrics use net PnL after fees+slippage.

Every candidate (strategy, model, experiment, portfolio) is evaluated with the
exact same metric set so results are directly comparable.

Metrics produced
----------------
Total return          Annualised return     Win rate
Expectancy            Expectancy (R)        Profit factor
Sharpe                Sortino               Calmar
Max drawdown          Avg drawdown          Recovery time (bars)
Avg winner            Avg loser             Payoff ratio
Exposure %            Turnover              Trade count
Fees paid             Slippage paid         Net PnL
Tail loss (CVaR 5%)   Tail loss (CVaR 1%)

Usage
-----
    from validation.engine import ValidationEngine, Trade

    trades = [Trade(entry_time, exit_time, pnl_net, size, fees, slippage), ...]
    engine = ValidationEngine(risk_free_rate=0.05, periods_per_year=252)
    result = engine.evaluate(trades, label="MyStrategy v1.2")
    print(result.summary())
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A single completed trade (entry → exit)."""
    entry_time:   datetime
    exit_time:    datetime
    pnl_net:      float       # net PnL after ALL costs
    pnl_gross:    float       # gross PnL before costs
    size:         float       # dollar size at entry
    fees:         float       # total fees paid
    slippage:     float       # estimated slippage cost
    symbol:       str = ''
    strategy:     str = ''
    direction:    str = 'LONG'   # LONG | SHORT
    regime:       str = ''
    session_id:   str = ''
    config_hash:  str = ''

    @property
    def holding_hours(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 3600

    @property
    def r_multiple(self) -> float:
        """PnL expressed in units of initial risk (size * 2% assumed)."""
        risk = self.size * 0.02
        return self.pnl_net / risk if risk != 0 else 0.0


@dataclass
class ValidationResult:
    """Full validation metric set for one evaluation run."""
    label:            str
    trade_count:      int

    # Returns
    total_pnl_net:    float
    total_pnl_gross:  float
    total_fees:       float
    total_slippage:   float
    total_return_pct: float    # net, relative to starting capital

    # Annualised
    ann_return_pct:   float
    ann_volatility:   float

    # Win/loss
    win_rate:         float
    avg_winner:       float
    avg_loser:        float
    payoff_ratio:     float
    profit_factor:    float

    # Expectancy
    expectancy:       float    # avg net PnL per trade
    expectancy_r:     float    # avg R-multiple per trade

    # Risk-adjusted
    sharpe:           float
    sortino:          float
    calmar:           float

    # Drawdown
    max_drawdown:     float
    avg_drawdown:     float
    recovery_bars:    int      # trades needed to recover from max DD

    # Tail risk
    cvar_5pct:        float    # 5th-percentile trade PnL (tail loss)
    cvar_1pct:        float

    # Activity
    exposure_pct:     float    # fraction of time in market
    turnover:         float    # average trades per period
    avg_hold_hours:   float

    # Period
    eval_start:       Optional[datetime] = None
    eval_end:         Optional[datetime] = None
    eval_days:        float = 0.0

    # Status flags
    sufficient_trades: bool = True    # False if < MIN_TRADES
    notes:             List[str] = field(default_factory=list)

    MIN_TRADES = 20

    def summary(self, width: int = 60) -> str:
        sep = "─" * width
        lines = [
            "═" * width,
            f"  VALIDATION RESULT: {self.label}",
            "═" * width,
            f"  Trades:        {self.trade_count:>10}",
            f"  Period:        {self.eval_days:>10.1f} days",
            sep,
            "  RETURNS",
            f"  Net PnL:       ${self.total_pnl_net:>10,.2f}",
            f"  Gross PnL:     ${self.total_pnl_gross:>10,.2f}",
            f"  Fees:          ${self.total_fees:>10,.2f}",
            f"  Slippage:      ${self.total_slippage:>10,.2f}",
            f"  Total return:  {self.total_return_pct:>10.2f}%",
            f"  Ann. return:   {self.ann_return_pct:>10.2f}%",
            sep,
            "  EDGE",
            f"  Win rate:      {self.win_rate:>10.1%}",
            f"  Expectancy:    ${self.expectancy:>10,.4f}",
            f"  Expectancy R:  {self.expectancy_r:>10.4f}",
            f"  Profit factor: {self.profit_factor:>10.3f}",
            f"  Payoff ratio:  {self.payoff_ratio:>10.3f}",
            f"  Avg winner:    ${self.avg_winner:>10,.2f}",
            f"  Avg loser:     ${self.avg_loser:>10,.2f}",
            sep,
            "  RISK-ADJUSTED",
            f"  Sharpe:        {self.sharpe:>10.3f}",
            f"  Sortino:       {self.sortino:>10.3f}",
            f"  Calmar:        {self.calmar:>10.3f}",
            sep,
            "  DRAWDOWN",
            f"  Max DD:        ${self.max_drawdown:>10,.2f}",
            f"  Avg DD:        ${self.avg_drawdown:>10,.2f}",
            f"  Recovery bars: {self.recovery_bars:>10}",
            sep,
            "  TAIL RISK",
            f"  CVaR 5%:       ${self.cvar_5pct:>10,.2f}",
            f"  CVaR 1%:       ${self.cvar_1pct:>10,.2f}",
            sep,
            "  ACTIVITY",
            f"  Exposure:      {self.exposure_pct:>10.1%}",
            f"  Avg hold (h):  {self.avg_hold_hours:>10.1f}",
            "═" * width,
        ]
        if not self.sufficient_trades:
            lines.append(f"  ⚠ Insufficient trades ({self.trade_count} < {self.MIN_TRADES})")
        for note in self.notes:
            lines.append(f"  ℹ {note}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['eval_start'] = self.eval_start.isoformat() if self.eval_start else None
        d['eval_end']   = self.eval_end.isoformat() if self.eval_end else None
        return d


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ValidationEngine:
    """
    Evaluates a list of Trade objects and produces a ValidationResult.

    Parameters
    ----------
    capital : float
        Starting capital for return calculations.
    risk_free_rate : float
        Annual risk-free rate (e.g. 0.05 = 5%).
    periods_per_year : int
        Trading periods per year for annualisation (use 252 for daily,
        365 for crypto, or trade count / years for trade-frequency-based).
    """

    MIN_TRADES = 20

    def __init__(
        self,
        capital: float = 10_000.0,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
    ):
        self.capital          = capital
        self.risk_free_rate   = risk_free_rate
        self.periods_per_year = periods_per_year

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        trades: Sequence[Trade],
        label: str = "Strategy",
        start_capital: float = None,
    ) -> ValidationResult:
        """
        Evaluate a sequence of completed trades.

        IMPORTANT: trades must be sorted chronologically (entry_time ASC).
        """
        capital = start_capital or self.capital
        trades  = sorted(trades, key=lambda t: t.entry_time)
        pnls    = [t.pnl_net for t in trades]
        n       = len(trades)

        if n == 0:
            return self._empty_result(label)

        # ── Period ────────────────────────────────────────────────────────────
        eval_start = trades[0].entry_time
        eval_end   = trades[-1].exit_time
        eval_days  = max((eval_end - eval_start).total_seconds() / 86400, 1.0)

        # ── PnL ───────────────────────────────────────────────────────────────
        total_net     = sum(pnls)
        total_gross   = sum(t.pnl_gross for t in trades)
        total_fees    = sum(t.fees for t in trades)
        total_slip    = sum(t.slippage for t in trades)
        total_ret_pct = total_net / capital * 100

        # ── Annualised return ─────────────────────────────────────────────────
        years        = eval_days / 365.0
        ann_ret_pct  = ((1 + total_net / capital) ** (1 / max(years, 0.01)) - 1) * 100
        ann_ret_pct  = max(ann_ret_pct, -100.0)

        # ── Volatility (annualised std of per-trade PnL) ─────────────────────
        ann_vol = 0.0
        if n >= 2:
            std     = statistics.stdev(pnls)
            ann_vol = std * math.sqrt(self.periods_per_year)

        # ── Win/loss ──────────────────────────────────────────────────────────
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr      = len(wins) / n
        avg_w   = sum(wins)   / len(wins)   if wins   else 0.0
        avg_l   = sum(losses) / len(losses) if losses else 0.0
        payoff  = abs(avg_w / avg_l) if avg_l != 0 else float('inf')
        pf      = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

        # ── Expectancy ────────────────────────────────────────────────────────
        expectancy   = total_net / n
        expectancy_r = sum(t.r_multiple for t in trades) / n

        # ── Sharpe ───────────────────────────────────────────────────────────
        rf_per_trade = self.risk_free_rate / self.periods_per_year
        excess       = [p / capital - rf_per_trade for p in pnls]
        sharpe       = 0.0
        if n >= 2:
            ex_mean = sum(excess) / n
            ex_std  = statistics.stdev(excess)
            sharpe  = (ex_mean / ex_std * math.sqrt(self.periods_per_year)) if ex_std > 0 else 0.0

        # ── Sortino ──────────────────────────────────────────────────────────
        sortino     = 0.0
        neg_returns = [p / capital - rf_per_trade for p in pnls if p < 0]
        if neg_returns:
            downside_var = sum(r ** 2 for r in neg_returns) / len(neg_returns)
            downside_std = math.sqrt(downside_var) * math.sqrt(self.periods_per_year)
            ann_ret_dec  = ann_ret_pct / 100
            sortino      = ann_ret_dec / downside_std if downside_std > 0 else 0.0

        # ── Drawdown ─────────────────────────────────────────────────────────
        max_dd, avg_dd, recovery = self._drawdown_stats(pnls)

        # ── Calmar ───────────────────────────────────────────────────────────
        calmar = (ann_ret_pct / 100) / (abs(max_dd) / capital) if max_dd != 0 else 0.0

        # ── Tail risk ────────────────────────────────────────────────────────
        cvar_5  = self._cvar(pnls, 0.05)
        cvar_1  = self._cvar(pnls, 0.01)

        # ── Activity ─────────────────────────────────────────────────────────
        total_hold_hours = sum(t.holding_hours for t in trades)
        exposure_pct     = (total_hold_hours / (eval_days * 24)) if eval_days > 0 else 0.0
        avg_hold_hours   = total_hold_hours / n if n > 0 else 0.0
        turnover         = n / max(eval_days / 30, 1.0)   # trades per month

        notes = []
        if n < self.MIN_TRADES:
            notes.append(f"Only {n} trades — results may not be statistically meaningful")
        if total_fees > abs(total_net) * 0.5:
            notes.append(f"Fees (${total_fees:.0f}) exceed 50% of gross PnL — cost drag is severe")

        return ValidationResult(
            label=label,
            trade_count=n,
            total_pnl_net=total_net,
            total_pnl_gross=total_gross,
            total_fees=total_fees,
            total_slippage=total_slip,
            total_return_pct=total_ret_pct,
            ann_return_pct=ann_ret_pct,
            ann_volatility=ann_vol,
            win_rate=wr,
            avg_winner=avg_w,
            avg_loser=avg_l,
            payoff_ratio=payoff,
            profit_factor=pf,
            expectancy=expectancy,
            expectancy_r=expectancy_r,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            recovery_bars=recovery,
            cvar_5pct=cvar_5,
            cvar_1pct=cvar_1,
            exposure_pct=min(exposure_pct, 1.0),
            turnover=turnover,
            avg_hold_hours=avg_hold_hours,
            eval_start=eval_start,
            eval_end=eval_end,
            eval_days=eval_days,
            sufficient_trades=(n >= self.MIN_TRADES),
            notes=notes,
        )

    def evaluate_from_dicts(
        self,
        rows: List[Dict],
        label: str = "Strategy",
        start_capital: float = None,
    ) -> ValidationResult:
        """
        Convenience: evaluate from a list of position dicts (from PositionManager).

        Required keys: exit_time, entry_time, net_pnl, pnl, size, fees, slippage (optional).
        """
        trades = []
        for r in rows:
            if r.get('status') != 'CLOSED':
                continue
            try:
                entry_t = _parse_dt(r.get('entry_time', ''))
                exit_t  = _parse_dt(r.get('exit_time', ''))
                if not entry_t or not exit_t:
                    continue
                pnl_net   = float(r.get('net_pnl') or r.get('pnl') or 0.0)
                pnl_gross = float(r.get('pnl') or pnl_net)
                size      = float(r.get('size') or 100.0)
                fees      = float(r.get('fees') or 0.0)
                slippage  = float(r.get('slippage') or 0.0)
                trades.append(Trade(
                    entry_time=entry_t, exit_time=exit_t,
                    pnl_net=pnl_net, pnl_gross=pnl_gross,
                    size=size, fees=fees, slippage=slippage,
                    symbol=r.get('symbol', ''),
                    strategy=r.get('strategy', ''),
                    direction=r.get('direction', 'LONG'),
                    regime=r.get('regime') or r.get('market_regime', ''),
                    session_id=r.get('session_id', ''),
                    config_hash=r.get('config_hash', ''),
                ))
            except Exception:
                continue
        return self.evaluate(trades, label=label, start_capital=start_capital)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _empty_result(self, label: str) -> ValidationResult:
        return ValidationResult(
            label=label, trade_count=0,
            total_pnl_net=0, total_pnl_gross=0, total_fees=0, total_slippage=0,
            total_return_pct=0, ann_return_pct=0, ann_volatility=0,
            win_rate=0, avg_winner=0, avg_loser=0, payoff_ratio=0, profit_factor=0,
            expectancy=0, expectancy_r=0,
            sharpe=0, sortino=0, calmar=0,
            max_drawdown=0, avg_drawdown=0, recovery_bars=0,
            cvar_5pct=0, cvar_1pct=0,
            exposure_pct=0, turnover=0, avg_hold_hours=0,
            sufficient_trades=False,
            notes=["No trades to evaluate"],
        )

    @staticmethod
    def _drawdown_stats(pnls: List[float]):
        """Return (max_dd, avg_dd, recovery_bars_from_max_dd)."""
        equity, peak = 0.0, 0.0
        drawdowns    = []
        in_dd        = False
        dd_start     = 0
        max_dd       = 0.0
        max_dd_idx   = 0

        for i, p in enumerate(pnls):
            equity += p
            if equity > peak:
                peak   = equity
                in_dd  = False
            dd = peak - equity
            drawdowns.append(dd)
            if dd > max_dd:
                max_dd     = dd
                max_dd_idx = i

        # Recovery bars from max drawdown point
        recovery = 0
        running  = 0.0
        peak_at_dd = sum(pnls[:max_dd_idx + 1]) + max_dd
        for p in pnls[max_dd_idx + 1:]:
            running += p
            recovery += 1
            if running >= max_dd:
                break
        else:
            recovery = -1   # Never recovered within sample

        avg_dd = sum(d for d in drawdowns if d > 0) / max(sum(1 for d in drawdowns if d > 0), 1)
        return max_dd, avg_dd, recovery

    @staticmethod
    def _cvar(pnls: List[float], alpha: float) -> float:
        """Conditional Value-at-Risk at given alpha (e.g. 0.05 = 5%)."""
        if not pnls:
            return 0.0
        sorted_pnls = sorted(pnls)
        cutoff_idx  = max(int(len(sorted_pnls) * alpha), 1)
        tail        = sorted_pnls[:cutoff_idx]
        return sum(tail) / len(tail) if tail else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(s) -> Optional[datetime]:
    if isinstance(s, datetime):
        if s.tzinfo is None:
            return s.replace(tzinfo=timezone.utc)
        return s
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None
