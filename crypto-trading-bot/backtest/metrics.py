"""Backtest performance metrics.

Computes a full suite of statistics from the trade list and equity curve
produced by the backtester.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from config.config import CONFIG


def _infer_bars_per_year(equity_curve: pd.Series) -> float:
    """
    Infer the number of bars in a year from the equity curve's index.

    Falls back to the configured interval if the index is not datetime-based.
    """
    if hasattr(equity_curve.index, 'freq') and equity_curve.index.freq is not None:
        freq = equity_curve.index.freq
        td = pd.Timedelta(freq)
        return pd.Timedelta(days=365) / td

    # Try to infer from consecutive timestamps
    if isinstance(equity_curve.index, pd.DatetimeIndex) and len(equity_curve) >= 2:
        deltas = equity_curve.index.to_series().diff().dropna()
        median_delta = deltas.median()
        if median_delta > pd.Timedelta(0):
            return pd.Timedelta(days=365) / median_delta

    # Fall back based on the configured interval
    interval = CONFIG.get("interval", "1h")
    interval_map = {
        "1m": 365 * 24 * 60,
        "5m": 365 * 24 * 12,
        "15m": 365 * 24 * 4,
        "30m": 365 * 24 * 2,
        "1h": 365 * 24,
        "4h": 365 * 6,
        "1d": 365,
        "1wk": 52,
        "1mo": 12,
    }
    return interval_map.get(interval, 365 * 24)


def compute_backtest_metrics(
    trades: List[Dict[str, Any]],
    equity_curve: pd.Series,
    initial_capital: float = 500,
    risk_free_rate: Optional[float] = None,
    bars_per_year: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive performance metrics.

    Args:
        trades:          List of trade dicts (must have 'pnl' key).
        equity_curve:    pd.Series of equity values, one per bar.
        initial_capital: Starting cash.
        risk_free_rate:  Annual risk-free rate for Sharpe calculation.
                         Defaults to CONFIG value or 0.04.
        bars_per_year:   Number of bars in a year (for annualisation).
                         Auto-detected from data if not provided.
    Returns:
        Dict of metric name -> value.
    """
    if not trades:
        return _empty_metrics(initial_capital)

    # Resolve annualisation parameters from real data
    if risk_free_rate is None:
        risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
    if bars_per_year is None:
        bars_per_year = _infer_bars_per_year(equity_curve)

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    final_equity = equity_curve.iloc[-1] if len(equity_curve) else initial_capital
    total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100

    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")

    # -- Drawdown -----
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_pct = abs(drawdown.min()) * 100 if len(drawdown) else 0

    # -- Sharpe ratio (annualised) -----
    returns = equity_curve.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        rf_per_bar = (1 + risk_free_rate) ** (1 / bars_per_year) - 1
        excess = returns - rf_per_bar
        sharpe_ratio = (excess.mean() / excess.std()) * np.sqrt(bars_per_year)
    else:
        sharpe_ratio = 0.0

    # -- Sortino ratio -----
    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino_ratio = (returns.mean() / downside.std()) * np.sqrt(bars_per_year)
    else:
        sortino_ratio = 0.0

    # -- Expectancy -----
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "expectancy": round(expectancy, 2),
        "final_equity": round(final_equity, 2),
        "bars_per_year_used": round(bars_per_year, 1),
        "risk_free_rate_used": risk_free_rate,
    }


def _empty_metrics(initial_capital: float = 500) -> Dict[str, Any]:
    """Return clearly-labelled empty metrics when no trades occurred."""
    return {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": None,
        "total_pnl": 0.0,
        "total_return_pct": 0.0,
        "avg_win": None,
        "avg_loss": None,
        "profit_factor": None,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "expectancy": None,
        "final_equity": initial_capital,
        "bars_per_year_used": None,
        "risk_free_rate_used": None,
    }
