"""StrategyExperimentRunner — real control/challenger backtesting.

Every proposed parameter change is validated by ACTUALLY RUNNING the modified
strategy against historical market data, never by rescaling unrelated closed
trades.

Baseline and challenger are evaluated on the SAME historical data with the
SAME execution-cost assumptions, isolating the effect of the changed
parameters (causal comparison).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from core.signal_flipper import Signal, SignalType
from core.strategy_registry import StrategyRegistry
from core.transaction_costs import TransactionCostModel
from core.validation_stats import (
    max_drawdown,
    permutation_test_mean,
    profit_concentration,
    profit_factor,
    sharpe_ratio,
    sign_test_p_value,
    sortino_ratio,
    uncertainty_summary,
    walk_forward_splits,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulatedTrade:
    symbol: str
    direction: str          # "long" | "short"
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    cost: float
    net_pnl: float
    exit_reason: str
    strategy: str = ""
    notional: float = 0.0   # actual entry notional of THIS trade

    @property
    def net_return(self) -> float:
        """Fractional net return normalized by this trade's own notional."""
        return self.net_pnl / self.notional if self.notional > 0 else 0.0


@dataclass
class BacktestMetrics:
    trades: int = 0
    net_pnl: float = 0.0
    expectancy: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_costs: float = 0.0
    gross_pnl: float = 0.0
    concentration: Dict[str, float] = field(default_factory=dict)
    uncertainty: Dict[str, float] = field(default_factory=dict)
    # ── Normalized return statistics (FRACTIONAL returns: 0.01 = +1%) ────────
    # Computed from per-trade net_pnl_i / entry_notional_i — never from an
    # assumed global position size.
    mean_net_return: float = 0.0
    median_net_return: float = 0.0
    gross_expectancy_return: float = 0.0
    net_expectancy_return: float = 0.0
    return_std: float = 0.0
    standard_error_return: float = 0.0
    confidence_lower_return: float = 0.0
    confidence_upper_return: float = 0.0
    effective_sample_size_returns: float = 0.0

    @classmethod
    def from_trades(cls, trades: Sequence[SimulatedTrade]) -> "BacktestMetrics":
        if not trades:
            return cls()
        pnls = [t.net_pnl for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        returns = [t.net_return for t in trades]
        gross_returns = [t.gross_pnl / t.notional if t.notional > 0 else 0.0
                         for t in trades]
        ret_stats = uncertainty_summary(returns)
        return cls(
            trades=len(trades),
            net_pnl=sum(pnls),
            expectancy=sum(pnls) / len(pnls),
            win_rate=wins / len(pnls),
            sharpe=sharpe_ratio(pnls),
            sortino=sortino_ratio(pnls),
            profit_factor=profit_factor(pnls),
            max_drawdown=max_drawdown(pnls),
            total_costs=sum(t.cost for t in trades),
            gross_pnl=sum(t.gross_pnl for t in trades),
            concentration=profit_concentration(pnls),
            uncertainty=uncertainty_summary(pnls),
            mean_net_return=ret_stats["mean"],
            median_net_return=ret_stats["median"],
            gross_expectancy_return=sum(gross_returns) / len(gross_returns),
            net_expectancy_return=ret_stats["mean"],
            return_std=(sum((r - ret_stats["mean"]) ** 2 for r in returns)
                        / max(len(returns) - 1, 1)) ** 0.5,
            standard_error_return=ret_stats["std_error"],
            confidence_lower_return=ret_stats["ci_lower"],
            confidence_upper_return=ret_stats["ci_upper"],
            effective_sample_size_returns=ret_stats["effective_sample_size"],
        )

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        pf = d.get("profit_factor")
        if pf == float("inf"):
            d["profit_factor"] = 1e9
        so = d.get("sortino")
        if so == float("inf"):
            d["sortino"] = 1e9
        return d


@dataclass
class ExperimentComparison:
    """Full output of a control-vs-challenger experiment."""
    strategy_id: str
    baseline_params: Dict[str, Any]
    challenger_params: Dict[str, Any]
    changed_parameters: Dict[str, Tuple[Any, Any]]
    baseline_metrics: BacktestMetrics
    challenger_metrics: BacktestMetrics
    delta_metrics: Dict[str, float]
    p_value_difference: float
    p_value_challenger_positive: float
    baseline_trades: List[SimulatedTrade]
    challenger_trades: List[SimulatedTrade]
    reproducibility_hash: str
    validation_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "baseline_params": self.baseline_params,
            "challenger_params": self.challenger_params,
            "changed_parameters": {k: list(v) for k, v in self.changed_parameters.items()},
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "challenger_metrics": self.challenger_metrics.to_dict(),
            "delta_metrics": self.delta_metrics,
            "p_value_difference": self.p_value_difference,
            "p_value_challenger_positive": self.p_value_challenger_positive,
            "n_baseline_trades": len(self.baseline_trades),
            "n_challenger_trades": len(self.challenger_trades),
            "reproducibility_hash": self.reproducibility_hash,
            "validation_metadata": self.validation_metadata,
        }


class StrategyBacktester:
    """Bar-by-bar simulator that runs ONE strategy config on ONE dataset.

    Deterministic: no network access, no randomness. Signals are generated
    from data available up to and including the decision bar; fills occur at
    the next bar's open (no look-ahead).
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        position_size_usd: float = 1000.0,
        max_hold_bars: int = 96,
        warmup_bars: int = 50,
    ) -> None:
        self.cost_model = cost_model or TransactionCostModel()
        self.position_size_usd = position_size_usd
        self.max_hold_bars = max_hold_bars
        self.warmup_bars = warmup_bars

    def run(
        self,
        strategy_id: str,
        params: Dict[str, Any],
        data: Dict[str, pd.DataFrame],
        index_range: Optional[Tuple[int, int]] = None,
    ) -> List[SimulatedTrade]:
        strategy_cls = StrategyRegistry.get(strategy_id)
        if strategy_cls is None:
            raise ValueError(f"Unknown strategy '{strategy_id}'")
        trades: List[SimulatedTrade] = []
        for symbol, df in data.items():
            strategy = strategy_cls(config=dict(params))
            trades.extend(
                self._run_symbol(strategy, symbol, df, index_range)
            )
        trades.sort(key=lambda t: t.entry_index)
        return trades

    def _run_symbol(
        self,
        strategy,
        symbol: str,
        df: pd.DataFrame,
        index_range: Optional[Tuple[int, int]],
    ) -> List[SimulatedTrade]:
        trades: List[SimulatedTrade] = []
        n = len(df)
        start = max(self.warmup_bars, index_range[0] if index_range else 0)
        end = min(n - 1, index_range[1] if index_range else n - 1)
        open_trade: Optional[Dict[str, Any]] = None

        closes = pd.to_numeric(df["close"], errors="coerce")
        highs = pd.to_numeric(df["high"], errors="coerce")
        lows = pd.to_numeric(df["low"], errors="coerce")
        opens = pd.to_numeric(df["open"], errors="coerce") if "open" in df.columns else closes

        for i in range(start, end):
            if open_trade is not None:
                exit_price, reason = self._check_exit(open_trade, i, highs, lows, closes)
                if exit_price is not None:
                    trades.append(self._close(open_trade, i, exit_price, reason, symbol,
                                              strategy.name))
                    open_trade = None
                continue

            window = df.iloc[: i + 1]
            try:
                sig: Signal = strategy.generate_signal(symbol, {"df": window})
            except Exception as e:
                logger.debug(f"Backtester: strategy error at bar {i}: {e}")
                continue
            if not sig.is_actionable or sig.entry is None:
                continue

            fill_price = float(opens.iloc[i + 1]) if i + 1 < n else float(closes.iloc[i])
            if not math.isfinite(fill_price) or fill_price <= 0:
                continue
            direction = "long" if sig.signal == SignalType.BUY else "short"
            stop = sig.stop_loss
            target = sig.targets[0] if sig.targets else None
            open_trade = {
                "direction": direction,
                "entry_index": i + 1,
                "entry_price": fill_price,
                "stop": stop,
                "target": target,
                "qty": self.position_size_usd / fill_price,
            }

        if open_trade is not None:
            final_price = float(closes.iloc[end])
            trades.append(self._close(open_trade, end, final_price, "end_of_data",
                                      symbol, strategy.name))
        return trades

    def _check_exit(self, trade, i, highs, lows, closes) -> Tuple[Optional[float], str]:
        hi, lo = float(highs.iloc[i]), float(lows.iloc[i])
        stop, target = trade["stop"], trade["target"]
        if trade["direction"] == "long":
            if stop is not None and lo <= stop:
                return stop, "stop_loss"
            if target is not None and hi >= target:
                return target, "take_profit"
        else:
            if stop is not None and hi >= stop:
                return stop, "stop_loss"
            if target is not None and lo <= target:
                return target, "take_profit"
        if i - trade["entry_index"] >= self.max_hold_bars:
            return float(closes.iloc[i]), "time_stop"
        return None, ""

    def _close(self, trade, exit_index, exit_price, reason, symbol, strategy_name) -> SimulatedTrade:
        qty = trade["qty"]
        notional = qty * trade["entry_price"]
        if trade["direction"] == "long":
            gross = (exit_price - trade["entry_price"]) * qty
        else:
            gross = (trade["entry_price"] - exit_price) * qty
        cost = self.cost_model.calculate_cost(notional, trade["entry_price"])
        return SimulatedTrade(
            symbol=symbol,
            direction=trade["direction"],
            entry_index=trade["entry_index"],
            exit_index=exit_index,
            entry_price=trade["entry_price"],
            exit_price=exit_price,
            gross_pnl=gross,
            cost=cost,
            net_pnl=gross - cost,
            exit_reason=reason,
            strategy=strategy_name,
            notional=notional,
        )


class StrategyExperimentRunner:
    """Run a causal control-vs-challenger experiment for one strategy.

    Both configurations are executed on identical historical data with
    identical execution assumptions. Only the changed parameters differ.
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        position_size_usd: float = 1000.0,
        seed: int = 42,
    ) -> None:
        self.cost_model = cost_model or TransactionCostModel()
        self.position_size_usd = position_size_usd
        self.seed = seed

    def run_experiment(
        self,
        strategy_id: str,
        baseline_params: Dict[str, Any],
        challenger_params: Dict[str, Any],
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        universe_version: str = "v1",
        dataset_version: str = "v1",
        allow_multi_variable: bool = False,
    ) -> ExperimentComparison:
        changed = {
            k: (baseline_params.get(k), challenger_params[k])
            for k in challenger_params
            if baseline_params.get(k) != challenger_params[k]
        }
        removed = {
            k: (baseline_params[k], None)
            for k in baseline_params
            if k not in challenger_params
        }
        changed.update(removed)
        if len(changed) > 1 and not allow_multi_variable:
            raise ValueError(
                f"Experiment changes {len(changed)} parameters {sorted(changed)}; "
                "single-variable experiments required unless allow_multi_variable=True"
            )

        backtester = StrategyBacktester(
            cost_model=self.cost_model,
            position_size_usd=self.position_size_usd,
        )
        baseline_trades = backtester.run(strategy_id, baseline_params, data)
        challenger_trades = backtester.run(strategy_id, challenger_params, data)

        baseline_metrics = BacktestMetrics.from_trades(baseline_trades)
        challenger_metrics = BacktestMetrics.from_trades(challenger_trades)

        delta = {
            "expectancy": challenger_metrics.expectancy - baseline_metrics.expectancy,
            "net_pnl": challenger_metrics.net_pnl - baseline_metrics.net_pnl,
            "win_rate": challenger_metrics.win_rate - baseline_metrics.win_rate,
            "sharpe": challenger_metrics.sharpe - baseline_metrics.sharpe,
            "max_drawdown": challenger_metrics.max_drawdown - baseline_metrics.max_drawdown,
            "trades": challenger_metrics.trades - baseline_metrics.trades,
        }

        p_diff = permutation_test_mean(
            [t.net_pnl for t in challenger_trades],
            [t.net_pnl for t in baseline_trades],
            seed=self.seed,
        )
        p_positive = sign_test_p_value([t.net_pnl for t in challenger_trades])

        repro_hash = self.reproducibility_hash(
            strategy_id, baseline_params, challenger_params, data,
            dataset_version, universe_version,
        )

        metadata = {
            "start_date": start_date or self._first_ts(data),
            "end_date": end_date or self._last_ts(data),
            "universe": sorted(data.keys()),
            "universe_version": universe_version,
            "dataset_version": dataset_version,
            "execution_assumptions": {
                "exchange": self.cost_model.exchange,
                "fee_pct": self.cost_model.fee_pct,
                "slippage_pct": self.cost_model.slippage_pct,
                "position_size_usd": self.position_size_usd,
                "fill": "next_bar_open",
            },
            "seed": self.seed,
        }

        return ExperimentComparison(
            strategy_id=strategy_id,
            baseline_params=baseline_params,
            challenger_params=challenger_params,
            changed_parameters=changed,
            baseline_metrics=baseline_metrics,
            challenger_metrics=challenger_metrics,
            delta_metrics=delta,
            p_value_difference=p_diff,
            p_value_challenger_positive=p_positive,
            baseline_trades=baseline_trades,
            challenger_trades=challenger_trades,
            reproducibility_hash=repro_hash,
            validation_metadata=metadata,
        )

    def run_walk_forward(
        self,
        strategy_id: str,
        params: Dict[str, Any],
        data: Dict[str, pd.DataFrame],
        n_folds: int = 4,
    ) -> Dict[str, Any]:
        """Walk-forward evaluation: run the strategy over sequential
        out-of-sample windows (train windows are implicit — parameters are
        fixed, so each fold is a genuine unseen segment)."""
        n = min(len(df) for df in data.values())
        splits = walk_forward_splits(n, n_folds=n_folds, min_train=max(60, n // (n_folds + 1)))
        backtester = StrategyBacktester(cost_model=self.cost_model,
                                        position_size_usd=self.position_size_usd)
        folds = []
        for train_rng, test_rng in splits:
            trades = backtester.run(strategy_id, params, data,
                                    index_range=(test_rng.start, test_rng.stop - 1))
            m = BacktestMetrics.from_trades(trades)
            folds.append({"start": test_rng.start, "stop": test_rng.stop,
                          "trades": m.trades, "expectancy": m.expectancy,
                          "net_pnl": m.net_pnl, "win_rate": m.win_rate})
        expectancies = [f["expectancy"] for f in folds if f["trades"] > 0]
        positive_folds = sum(1 for e in expectancies if e > 0)
        return {
            "folds": folds,
            "n_folds_with_trades": len(expectancies),
            "positive_fold_fraction": (positive_folds / len(expectancies)) if expectancies else 0.0,
            "mean_fold_expectancy": (sum(expectancies) / len(expectancies)) if expectancies else 0.0,
        }

    def run_parameter_sweep(
        self,
        strategy_id: str,
        base_params: Dict[str, Any],
        param_name: str,
        values: Sequence[float],
        data: Dict[str, pd.DataFrame],
    ) -> Dict[float, float]:
        """Run the strategy at neighboring parameter values; return
        value -> net expectancy (for robustness scoring)."""
        backtester = StrategyBacktester(cost_model=self.cost_model,
                                        position_size_usd=self.position_size_usd)
        out: Dict[float, float] = {}
        for v in values:
            p = dict(base_params)
            p[param_name] = v
            trades = backtester.run(strategy_id, p, data)
            m = BacktestMetrics.from_trades(trades)
            out[v] = m.expectancy if m.trades > 0 else 0.0
        return out

    # ── Reproducibility ───────────────────────────────────────────────────────

    def reproducibility_hash(
        self,
        strategy_id: str,
        baseline_params: Dict[str, Any],
        challenger_params: Dict[str, Any],
        data: Dict[str, pd.DataFrame],
        dataset_version: str,
        universe_version: str,
    ) -> str:
        data_sig = {
            sym: (len(df), str(df.index[0]) if len(df) else "", str(df.index[-1]) if len(df) else "",
                  round(float(pd.to_numeric(df["close"], errors="coerce").sum()), 4))
            for sym, df in sorted(data.items())
        }
        payload = json.dumps({
            "strategy": strategy_id,
            "strategy_code_version": self._strategy_code_hash(strategy_id),
            "baseline": baseline_params,
            "challenger": challenger_params,
            "data": data_sig,
            "dataset_version": dataset_version,
            "universe_version": universe_version,
            "costs": {"exchange": self.cost_model.exchange,
                      "fee": self.cost_model.fee_pct,
                      "slippage": self.cost_model.slippage_pct},
            "position_size_usd": self.position_size_usd,
            "seed": self.seed,
        }, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _strategy_code_hash(strategy_id: str) -> str:
        import inspect
        cls = StrategyRegistry.get(strategy_id)
        if cls is None:
            return "unknown"
        try:
            src = inspect.getsource(cls)
            return hashlib.sha256(src.encode()).hexdigest()[:16]
        except (OSError, TypeError):
            return "unavailable"

    @staticmethod
    def _first_ts(data: Dict[str, pd.DataFrame]) -> str:
        for df in data.values():
            if len(df):
                return str(df.index[0])
        return ""

    @staticmethod
    def _last_ts(data: Dict[str, pd.DataFrame]) -> str:
        for df in data.values():
            if len(df):
                return str(df.index[-1])
        return ""
