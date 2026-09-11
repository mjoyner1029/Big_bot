"""Adversarial Synthetic Alpha Lab.

Deterministic synthetic datasets with KNOWN properties are pushed through the
validation stack to prove the research engine can tell real alpha from fake:

  A  true momentum edge          -> PASS
  B  random walk / no edge       -> REJECT
  C  look-ahead leakage          -> leakage prevented / detected
  D  single-outlier profits      -> REJECT
  E  cost-sensitive edge         -> REJECT (net negative after costs)
  F  in-sample overfit           -> REJECT at holdout
  G  regime-specific edge        -> per-fold evidence exposed for restriction
  I  parameter cliff             -> REJECT / heavy penalty
  J  broad robust plateau        -> PASS
  K  high win rate, neg expectancy -> REJECT
  L  low win rate, pos expectancy  -> can PASS
"""
import math
import uuid

import numpy as np
import pandas as pd
import pytest

from core.signal_flipper import Signal, SignalType
from core.strategy_base import StrategyBase
from core.strategy_registry import StrategyRegistry
from core.strategy_experiment_runner import (
    BacktestMetrics,
    StrategyBacktester,
    StrategyExperimentRunner,
)
from core.transaction_costs import TransactionCostModel
from core.validation_stats import (
    benjamini_hochberg,
    parameter_robustness_score,
    profit_concentration,
    purged_time_series_splits,
    uncertainty_summary,
)

SEED = 1337


# ── Synthetic data generators (deterministic) ────────────────────────────────


def _to_ohlcv(prices: np.ndarray, volume: float = 1e6) -> pd.DataFrame:
    prices = np.maximum(prices, 1e-3)
    close = prices
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    spread = np.abs(close - open_)
    high = np.maximum(open_, close) + spread * 0.25 + close * 0.0005
    low = np.minimum(open_, close) - spread * 0.25 - close * 0.0005
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.full(len(prices), volume)},
        index=idx,
    )


def make_momentum_series(n: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Price with genuine positive autocorrelation (real momentum edge)."""
    rng = np.random.default_rng(seed)
    rets = np.zeros(n)
    for i in range(1, n):
        rets[i] = 0.5 * rets[i - 1] + rng.normal(0.0008, 0.004)
    return _to_ohlcv(100 * np.exp(np.cumsum(rets)))


def make_random_walk(n: int = 2000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.004, n)
    return _to_ohlcv(100 * np.exp(np.cumsum(rets)))


def make_overfit_series(n: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Strong momentum in first 80%, pure noise in the final 20% (holdout)."""
    split = int(n * 0.8)
    head = make_momentum_series(split, seed)
    rng = np.random.default_rng(seed + 1)
    noise = rng.normal(0.0, 0.004, n - split)
    tail_prices = float(head["close"].iloc[-1]) * np.exp(np.cumsum(noise))
    tail = _to_ohlcv(tail_prices)
    tail.index = pd.date_range(head.index[-1] + pd.Timedelta(hours=1),
                               periods=len(tail), freq="1h")
    return pd.concat([head, tail])


# ── Synthetic strategies ──────────────────────────────────────────────────────


@StrategyRegistry.register("synthetic_momentum")
class SyntheticMomentumStrategy(StrategyBase):
    """Simple SMA-momentum entry; used to detect the planted momentum edge."""

    name = "synthetic_momentum"

    def __init__(self, config=None):
        super().__init__(config)
        self.fast = int(self._cfg("fast", 5))
        self.slow = int(self._cfg("slow", 15))
        self.stop_pct = float(self._cfg("stop_pct", 0.025))
        self.target_pct = float(self._cfg("target_pct", 0.02))

    def generate_signal(self, symbol, data):
        df = data.get("df")
        if df is None or len(df) < self.slow + 2:
            return self._no_trade(symbol, reason="insufficient data")
        tail = df.iloc[-(self.slow + 2):]  # only the tail is needed for the cross
        close = pd.to_numeric(tail["close"], errors="coerce")
        fast = close.rolling(self.fast).mean()
        slow = close.rolling(self.slow).mean()
        price = float(close.iloc[-1])
        if fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]:
            return Signal(
                symbol=symbol, signal=SignalType.BUY, confidence=60.0,
                entry=price, stop_loss=price * (1 - self.stop_pct),
                targets=[price * (1 + self.target_pct)],
                strategy_name=self.name, reason="sma cross up",
            )
        return self._no_trade(symbol, reason="no cross")


@StrategyRegistry.register("synthetic_lookahead")
class SyntheticLookaheadStrategy(StrategyBase):
    """Intentionally leaky: only signals when the CURRENT bar closed up and
    claims that bar's open as entry. A naive same-bar-fill backtester would
    credit it with impossible profits; a correct simulator must not."""

    name = "synthetic_lookahead"

    def generate_signal(self, symbol, data):
        df = data.get("df")
        if df is None or len(df) < 3:
            return self._no_trade(symbol, reason="insufficient data")
        last_open = float(df["open"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if last_close > last_open * 1.002:
            return Signal(
                symbol=symbol, signal=SignalType.BUY, confidence=99.0,
                entry=last_open,  # pretends it could have bought at this bar's open
                stop_loss=last_open * 0.99,
                targets=[last_close],  # "knows" the future close
                strategy_name=self.name, reason="leak",
            )
        return self._no_trade(symbol, reason="no leak bar")


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def runner():
    return StrategyExperimentRunner(
        cost_model=TransactionCostModel("binance"), position_size_usd=1000.0)


@pytest.fixture
def worker(tmp_path):
    from core.experiment_worker import ExperimentWorker

    def _make(data: dict):
        return ExperimentWorker(
            db_path=str(tmp_path / f"lab_{uuid.uuid4().hex}.sqlite"),
            data_provider=lambda symbols, period, interval: data,
        )
    return _make


def _run_pipeline(worker_factory, data, params, description="lab experiment"):
    from core.experiment_worker import WorkItem
    w = worker_factory(data)
    item = WorkItem(experiment_id=str(uuid.uuid4()), hypothesis_id=None,
                    params=params, description=description)
    w._save_pipeline(item)
    w._process(item)
    return w, item, w._current_stage(item.experiment_id)


# ── CASE A: true momentum edge → PASS quantitative stages ────────────────────


class TestCaseATrueMomentum:
    def test_momentum_edge_reaches_paper_stage(self, worker):
        data = {"SYN-A": make_momentum_series()}
        _, _, stage = _run_pipeline(
            worker, data,
            {"strategy": "synthetic_momentum", "fast": 4,
             "baseline": {"fast": 5}},
        )
        # A real edge must survive all quantitative gates and park at paper
        # (never auto-promote without observations)
        assert stage in ("PAPER_RUNNING",), f"expected PAPER_RUNNING, got {stage}"


# ── CASE B: random walk → REJECT ─────────────────────────────────────────────


class TestCaseBRandomWalk:
    def test_random_walk_rejected(self, worker):
        data = {"SYN-B": make_random_walk()}
        _, _, stage = _run_pipeline(
            worker, data,
            {"strategy": "synthetic_momentum", "fast": 4,
             "baseline": {"fast": 5}},
        )
        assert stage == "REJECTED"


# ── CASE C: look-ahead leakage ───────────────────────────────────────────────


class TestCaseCLookahead:
    def test_simulator_denies_lookahead_profits(self, runner):
        """The leaky strategy 'knows' each up-bar in advance. A correct
        simulator fills at the NEXT bar's open, so the leak conveys no
        systematic advantage on a random walk."""
        data = {"SYN-C": make_random_walk()}
        backtester = StrategyBacktester(cost_model=TransactionCostModel("binance"))
        trades = backtester.run("synthetic_lookahead", {}, data)
        m = BacktestMetrics.from_trades(trades)
        # With leakage honored, expectancy would be strongly positive with
        # ~100% win rate. Correct simulation keeps it at noise level.
        assert m.trades == 0 or m.win_rate < 0.9
        assert m.expectancy < 1.0  # $ per $1000 position: noise, not free money

    def test_purged_splits_prevent_label_leakage(self):
        """No training index's label window may overlap the test block."""
        n, horizon, embargo = 200, 5, 5
        splits = purged_time_series_splits(n, n_folds=5, embargo=embargo,
                                           label_horizon=horizon)
        assert splits, "expected splits"
        for train_idx, test_idx in splits:
            t_start, t_end = min(test_idx), max(test_idx)
            for i in train_idx:
                # label window [i, i+horizon] must not reach into test block
                assert not (i <= t_end and i + horizon >= t_start), (
                    f"leaky train index {i} for test block [{t_start},{t_end}]"
                )

    def test_embargo_removes_post_test_samples(self):
        splits = purged_time_series_splits(100, n_folds=4, embargo=10)
        train_idx, test_idx = splits[0]
        t_end = max(test_idx)
        for i in train_idx:
            assert not (t_end < i <= t_end + 10), f"embargo violated at {i}"


# ── CASE D: single outlier → REJECT ──────────────────────────────────────────


class TestCaseDSingleOutlier:
    def test_concentration_metric_flags_outlier(self):
        pnls = [-1.0] * 60 + [500.0]  # all profit in one trade
        conc = profit_concentration(pnls)
        assert conc["top_1_trade_pct"] == 1.0
        assert conc["concentration_score"] > 0.85

    def test_diversified_profits_not_flagged(self):
        pnls = [2.0, -1.0] * 50
        conc = profit_concentration(pnls)
        assert conc["concentration_score"] < 0.85


# ── CASE E: cost-sensitive edge → REJECT ─────────────────────────────────────


class TestCaseECostSensitive:
    def test_gross_positive_net_negative_rejected(self):
        """Edge earning less than round-trip costs must show negative NET
        expectancy once realistic costs are applied."""
        cost_model = TransactionCostModel("coinbase")  # 0.6% fees
        cost = cost_model.calculate_cost(1000.0, 100.0)
        assert cost > 0
        gross_per_trade = cost * 0.5  # edge is worth half its costs
        pnls = [gross_per_trade - cost] * 40
        summary = uncertainty_summary(pnls)
        assert summary["mean"] < 0
        assert summary["p_net_positive"] == 0.0


# ── CASE F: in-sample overfit → REJECT at holdout ────────────────────────────


class TestCaseFOverfit:
    def test_overfit_edge_fails_holdout(self, worker):
        data = {"SYN-F": make_overfit_series()}
        w, item, stage = _run_pipeline(
            worker, data,
            {"strategy": "synthetic_momentum", "fast": 4,
             "baseline": {"fast": 5}},
        )
        # Must NOT reach paper; the untouched final segment kills it,
        # or an earlier quantitative gate already rejected it.
        assert stage == "REJECTED"


# ── CASE G: regime-specific edge → evidence exposed per fold ─────────────────


class TestCaseGRegimeSpecific:
    def test_walk_forward_exposes_per_fold_results(self, runner):
        data = {"SYN-G": make_momentum_series()}
        wf = runner.run_walk_forward("synthetic_momentum", {"fast": 4}, data)
        assert "folds" in wf and isinstance(wf["folds"], list)
        for fold in wf["folds"]:
            assert {"start", "stop", "trades", "expectancy"} <= set(fold)


# ── CASE I / J: parameter cliff vs plateau ───────────────────────────────────


class TestCaseIJParameterRobustness:
    def test_cliff_scores_low(self):
        cliff = {24: -5.0, 25: -4.0, 26: -6.0, 27: 50.0, 28: -5.0, 29: -3.0, 30: -6.0}
        assert parameter_robustness_score(cliff) < 0.4

    def test_plateau_scores_high(self):
        plateau = {24: 40.0, 25: 44.0, 26: 47.0, 27: 50.0, 28: 48.0, 29: 45.0, 30: 41.0}
        assert parameter_robustness_score(plateau) > 0.7


# ── CASE K / L: win rate vs expectancy ───────────────────────────────────────


class TestCaseKLExpectancyOverWinRate:
    def test_high_win_rate_negative_expectancy_rejected(self):
        # 90% win rate, but losses dwarf wins
        pnls = ([1.0] * 90) + ([-20.0] * 10)
        summary = uncertainty_summary(pnls)
        assert summary["p_net_positive"] > 0.85  # win rate high...
        assert summary["mean"] < 0               # ...but expectancy negative
        # Promotion gates key on expectancy, not win rate:
        from core.experiment_worker import DEFAULT_WORKER_CONFIG
        assert summary["mean"] <= DEFAULT_WORKER_CONFIG["min_expectancy"]

    def test_low_win_rate_positive_expectancy_viable(self):
        # 40% win rate with 3:1 payoff
        pnls = ([30.0] * 40) + ([-10.0] * 60)
        summary = uncertainty_summary(pnls)
        assert summary["p_net_positive"] < 0.5
        assert summary["mean"] > 0
        assert summary["ci_lower"] > 0  # even lower bound positive


# ── Multiple-testing correction ───────────────────────────────────────────────


class TestMultipleTestingCorrection:
    def test_bh_rejects_chance_level_discoveries_in_large_family(self):
        # Under the null, p-values are uniform: a family of 100 tests yields a
        # few p<0.05 by pure chance. Naive p<0.05 would accept them; BH must not.
        p_values = [(i + 1) / 100 for i in range(100)]  # 0.01, 0.02, ..., 1.0
        passed = benjamini_hochberg(p_values, alpha=0.05)
        assert not any(passed)

    def test_bh_keeps_strong_discoveries(self):
        p_values = [0.0001, 0.6, 0.7, 0.8, 0.9]
        passed = benjamini_hochberg(p_values, alpha=0.05)
        assert passed[0] and not any(passed[1:])


# ── Paper / canary gates (never auto-pass) ────────────────────────────────────


class TestPaperAndCanaryGates:
    def _park_at_paper(self, worker_factory):
        data = {"SYN-P": make_momentum_series()}
        w, item, stage = _run_pipeline(
            worker_factory, data,
            {"strategy": "synthetic_momentum", "fast": 4,
             "baseline": {"fast": 5},
             "min_paper_trades": 5, "min_paper_days": 1,
             "min_canary_trades": 3, "min_canary_days": 1},
        )
        assert stage == "PAPER_RUNNING"
        return w, item

    def test_paper_does_not_pass_without_observations(self, worker):
        w, item = self._park_at_paper(worker)
        # Re-checking without observations must keep it parked
        w.check_pending()
        assert w._current_stage(item.experiment_id) == "PAPER_RUNNING"

    def test_paper_passes_only_with_attributed_observations(self, worker):
        from core.experiment_observations import ExperimentObservationTracker
        w, item = self._park_at_paper(worker)
        tracker = ExperimentObservationTracker(w.db_path)

        # Trades from OTHER experiments must not count
        for i in range(10):
            tracker.record_trade("some-other-experiment", "paper", "SYN-P", 5.0)
        w.check_pending()
        assert w._current_stage(item.experiment_id) == "PAPER_RUNNING"

        # Attributed profitable observations spanning enough days
        for i in range(6):
            tracker.record_trade(
                item.experiment_id, "paper", "SYN-P", 4.0,
                recorded_at=f"2026-01-0{i + 1}T00:00:00+00:00",
            )
        w.check_pending()
        stage = w._current_stage(item.experiment_id)
        # Paper passed -> moved on to canary which requires explicit approval
        assert stage == "CANARY_PENDING_APPROVAL"

    def test_canary_requires_explicit_approval_and_observations(self, worker):
        from core.experiment_observations import ExperimentObservationTracker
        w, item = self._park_at_paper(worker)
        tracker = ExperimentObservationTracker(w.db_path)
        for i in range(6):
            tracker.record_trade(
                item.experiment_id, "paper", "SYN-P", 4.0,
                recorded_at=f"2026-01-0{i + 1}T00:00:00+00:00",
            )
        w.check_pending()
        assert w._current_stage(item.experiment_id) == "CANARY_PENDING_APPROVAL"

        # Approval alone is not enough — needs attributed canary trades
        w.approve_canary(item.experiment_id, approved_by="test")
        w.check_pending()
        assert w._current_stage(item.experiment_id) == "CANARY_RUNNING"

        for i in range(4):
            tracker.record_trade(
                item.experiment_id, "canary", "SYN-P", 2.0,
                recorded_at=f"2026-02-0{i + 1}T00:00:00+00:00",
            )
        w.check_pending()
        assert w._current_stage(item.experiment_id) == "PROMOTED"

    def test_losing_paper_experiment_rejected(self, worker):
        from core.experiment_observations import ExperimentObservationTracker
        w, item = self._park_at_paper(worker)
        tracker = ExperimentObservationTracker(w.db_path)
        for i in range(6):
            tracker.record_trade(
                item.experiment_id, "paper", "SYN-P", -4.0,
                recorded_at=f"2026-01-0{i + 1}T00:00:00+00:00",
            )
        w.check_pending()
        assert w._current_stage(item.experiment_id) == "REJECTED"


# ── Experiment causality requirements ────────────────────────────────────────


class TestCausalExperimentRequirements:
    def test_multi_variable_change_rejected_unless_marked(self, runner):
        data = {"SYN-M": make_momentum_series(600)}
        with pytest.raises(ValueError, match="single-variable"):
            runner.run_experiment(
                "synthetic_momentum",
                baseline_params={},
                challenger_params={"fast": 4, "slow": 20},
                data=data,
            )
        # explicitly marked multi-variable is allowed
        comparison = runner.run_experiment(
            "synthetic_momentum", baseline_params={},
            challenger_params={"fast": 4, "slow": 20},
            data=data, allow_multi_variable=True,
        )
        assert set(comparison.changed_parameters) == {"fast", "slow"}

    def test_reproducibility_hash_stable_and_sensitive(self, runner):
        data = {"SYN-R": make_momentum_series(600)}
        c1 = runner.run_experiment("synthetic_momentum", {}, {"fast": 4}, data)
        c2 = runner.run_experiment("synthetic_momentum", {}, {"fast": 4}, data)
        c3 = runner.run_experiment("synthetic_momentum", {}, {"fast": 6}, data)
        assert c1.reproducibility_hash == c2.reproducibility_hash
        assert c1.reproducibility_hash != c3.reproducibility_hash

    def test_baseline_and_challenger_share_data_and_costs(self, runner):
        data = {"SYN-S": make_momentum_series(600)}
        comparison = runner.run_experiment("synthetic_momentum", {}, {"fast": 4}, data)
        meta = comparison.validation_metadata
        assert meta["universe"] == ["SYN-S"]
        assert meta["execution_assumptions"]["fill"] == "next_bar_open"
        # deterministic: same experiment reproduces identical trade counts
        again = runner.run_experiment("synthetic_momentum", {}, {"fast": 4}, data)
        assert len(again.baseline_trades) == len(comparison.baseline_trades)
        assert len(again.challenger_trades) == len(comparison.challenger_trades)
