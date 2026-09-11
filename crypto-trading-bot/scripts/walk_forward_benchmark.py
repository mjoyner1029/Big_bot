#!/usr/bin/env python3
"""Walk-forward benchmark runner: Bot vs S&P (SPY).

One-command script that:
1) Evaluates all registered strategies individually plus an ensemble candidate.
2) Uses rolling walk-forward folds (train -> test).
3) Selects the best candidate on each train window.
4) Benchmarks out-of-sample performance against SPY buy-and-hold.
5) Prints promotion pass/fail gates and writes JSON+Markdown reports.

Example:
    .venv/bin/python scripts/walk_forward_benchmark.py

Fast custom run:
    .venv/bin/python scripts/walk_forward_benchmark.py \
      --symbols BTC-USD,ETH-USD,NVDA,SPY \
      --period 1y --interval 1h --folds 3 --capital 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.backtester import run_backtest
from config.config import CONFIG, get_all_symbols
from core.signal_flipper import SignalType
from core.strategy_registry import StrategyRegistry
from data.fetcher import fetch_latest_market_data
from strategies import *  # noqa: F401,F403 - trigger strategy imports/registration
from strategies.thresholds import get_trade_thresholds


@dataclass
class FoldResult:
    fold_idx: int
    train_range: str
    test_range: str
    selected_candidate: str
    train_return_pct: float
    test_return_pct: float
    test_profit_factor: float
    test_max_drawdown_pct: float
    test_sharpe_ratio: float
    spy_return_pct: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward Bot vs S&P benchmark")
    # ── Mode ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--mode",
        default="benchmark",
        choices=["benchmark", "grid-search"],
        help=(
            "benchmark = standard walk-forward (default); "
            "grid-search = tune confidence/risk/positions then auto-run benchmark with best params"
        ),
    )
    # ── Shared benchmark params ──────────────────────────────────────────────
    p.add_argument("--symbols", default=",".join(get_all_symbols()[:8]), help="Comma-separated symbols")
    p.add_argument("--period", default="2y", help="Fetch lookback period (e.g. 1y, 2y)")
    p.add_argument("--interval", default="1h", help="Bar interval (e.g. 1h, 1d)")
    p.add_argument("--folds", type=int, default=4, help="Number of walk-forward folds")
    p.add_argument("--train-frac", type=float, default=0.65, help="Initial train fraction")
    p.add_argument("--capital", type=float, default=10000.0, help="Initial capital per symbol")
    p.add_argument("--warmup", type=int, default=60, help="Warmup bars for backtester")
    p.add_argument("--out-json", default="reports/walk_forward_bot_vs_spy.json", help="Output JSON path")
    p.add_argument("--out-md", default="reports/walk_forward_bot_vs_spy.md", help="Output markdown path")
    # ── Grid-search tuning params ────────────────────────────────────────────
    p.add_argument(
        "--gs-confidence-thresholds",
        default="0.45,0.55,0.65,0.75",
        dest="gs_confidence_thresholds",
        help="Comma-separated confidence thresholds to sweep (grid-search mode)",
    )
    p.add_argument(
        "--gs-risk-pcts",
        default="0.005,0.01,0.015,0.02",
        dest="gs_risk_pcts",
        help="Comma-separated risk-per-trade fractions to sweep (grid-search mode)",
    )
    p.add_argument(
        "--gs-max-positions",
        default="3,5,8",
        dest="gs_max_positions",
        help="Comma-separated max-open-positions values to sweep (grid-search mode)",
    )
    p.add_argument(
        "--gs-eval-frac",
        type=float,
        default=0.30,
        dest="gs_eval_frac",
        help="Fraction of data tail used for quick grid-search scoring, 0.10-0.50 (grid-search mode)",
    )
    return p.parse_args()


def _is_crypto(symbol: str) -> bool:
    return "-" in symbol or "/" in symbol


def _normalize_confidence(c: float) -> float:
    c = float(c)
    if c > 1.0:
        c /= 100.0
    return max(0.0, min(1.0, c))


def _build_data_bundle(df_slice: pd.DataFrame) -> Dict[str, Any]:
    last = df_slice.iloc[-1]
    prev = df_slice.iloc[-2] if len(df_slice) > 1 else last

    # Approximate previous-day levels when true session data is absent.
    prev_window = df_slice.tail(min(len(df_slice), 24))
    prev_day = {
        "high": float(prev_window["high"].max()) if "high" in prev_window else float(prev.get("high", 0) or 0),
        "low": float(prev_window["low"].min()) if "low" in prev_window else float(prev.get("low", 0) or 0),
        "open": float(prev_window["open"].iloc[0]) if "open" in prev_window else float(prev.get("open", 0) or 0),
        "close": float(prev_window["close"].iloc[-1]) if "close" in prev_window else float(prev.get("close", 0) or 0),
    }

    atr = float(last.get("atr", abs(float(last.get("high", 0)) - float(last.get("low", 0))) or 0))

    current = {
        "open": float(last.get("open", last.get("Open", 0)) or 0),
        "high": float(last.get("high", last.get("High", 0)) or 0),
        "low": float(last.get("low", last.get("Low", 0)) or 0),
        "close": float(last.get("close", last.get("Close", 0)) or 0),
    }

    return {
        "current_candle": current,
        "previous_day_ohlc": prev_day,
        "df": df_slice,
        "atr": atr,
    }


def _signal_to_trade_dict(sig: Any, symbol: str, close_px: float) -> Optional[Dict[str, Any]]:
    side = sig.signal.value if hasattr(sig, "signal") else "NO_TRADE"
    if side not in {SignalType.BUY.value, SignalType.SELL.value}:
        return None

    entry = float(sig.entry if sig.entry is not None else close_px)
    conf = _normalize_confidence(float(sig.confidence if hasattr(sig, "confidence") else 0.5))

    sl = float(sig.stop_loss) if getattr(sig, "stop_loss", None) is not None else 0.0
    tp = float(sig.targets[0]) if getattr(sig, "targets", None) else 0.0
    if sl <= 0 or tp <= 0:
        th = get_trade_thresholds(
            current_price=entry,
            confidence=conf,
            side="buy" if side == SignalType.BUY.value else "sell",
            asset_type="crypto" if _is_crypto(symbol) else "stock",
        )
        if sl <= 0:
            sl = float(th.get("stop_loss_price", 0) or 0)
        if tp <= 0:
            tp = float(th.get("take_profit_price", 0) or 0)

    return {
        "symbol": symbol,
        "side": "buy" if side == SignalType.BUY.value else "sell",
        "asset_type": "crypto" if _is_crypto(symbol) else "stock",
        "entry_price": entry,
        "confidence": conf,
        "stop_loss_price": sl,
        "take_profit_price": tp,
        "trailing_stop_pct": float(getattr(sig, "metadata", {}).get("trailing_stop_pct", 0) or 0),
        "strategy_name": getattr(sig, "strategy_name", "unknown"),
    }


def _single_strategy_fn(strategy_name: str) -> Callable[[pd.DataFrame, str], Optional[Dict[str, Any]]]:
    klass = StrategyRegistry.get(strategy_name)
    if klass is None:
        raise KeyError(f"Strategy not found: {strategy_name}")
    strategy = klass()

    def fn(df_slice: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        bundle = _build_data_bundle(df_slice)
        sig = strategy.generate_signal(symbol, bundle)
        close_px = float(df_slice["Close"].iloc[-1])
        return _signal_to_trade_dict(sig, symbol, close_px)

    return fn


def _ensemble_strategy_fn(strategy_names: List[str]) -> Callable[[pd.DataFrame, str], Optional[Dict[str, Any]]]:
    klasses = [StrategyRegistry.get(n) for n in strategy_names]
    klasses = [k for k in klasses if k is not None]
    strategies = [k() for k in klasses]

    def fn(df_slice: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        bundle = _build_data_bundle(df_slice)
        close_px = float(df_slice["Close"].iloc[-1])

        votes = {"BUY": 0.0, "SELL": 0.0}
        best_trade: Optional[Dict[str, Any]] = None
        best_weight = -1.0

        for strat in strategies:
            try:
                sig = strat.generate_signal(symbol, bundle)
            except Exception:
                continue
            trade = _signal_to_trade_dict(sig, symbol, close_px)
            if trade is None:
                continue
            side = "BUY" if trade["side"] == "buy" else "SELL"
            w = _normalize_confidence(trade["confidence"]) * 100.0
            votes[side] += w
            if w > best_weight:
                best_weight = w
                best_trade = trade

        if best_trade is None:
            return None

        winner = "BUY" if votes["BUY"] >= votes["SELL"] else "SELL"
        if (winner == "BUY" and best_trade["side"] != "buy") or (winner == "SELL" and best_trade["side"] != "sell"):
            best_trade["side"] = "buy" if winner == "BUY" else "sell"
            # Recompute thresholds for the switched side.
            th = get_trade_thresholds(
                current_price=best_trade["entry_price"],
                confidence=best_trade["confidence"],
                side=best_trade["side"],
                asset_type=best_trade["asset_type"],
            )
            best_trade["stop_loss_price"] = float(th.get("stop_loss_price", best_trade["stop_loss_price"]))
            best_trade["take_profit_price"] = float(th.get("take_profit_price", best_trade["take_profit_price"]))

        best_trade["strategy_name"] = "ensemble_all"
        return best_trade

    return fn


def _build_folds(n: int, folds: int, train_frac: float) -> List[Tuple[int, int, int, int]]:
    """Return list of (train_start, train_end, test_start, test_end) indices."""
    if n < 200:
        return []
    folds = max(2, int(folds))
    train_end0 = int(n * train_frac)
    test_len = max(30, (n - train_end0) // folds)

    out = []
    train_start = 0
    for i in range(folds):
        train_end = train_end0 + (i * test_len)
        test_start = train_end
        test_end = min(n, test_start + test_len)
        if test_end - test_start < 20 or train_end - train_start < 120:
            break
        out.append((train_start, train_end, test_start, test_end))
    return out


def _aggregate_metric(per_symbol: List[Dict[str, Any]]) -> Dict[str, float]:
    if not per_symbol:
        return {
            "total_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "total_trades": 0.0,
            "win_rate": 0.0,
        }

    def avg(name: str) -> float:
        vals = [float(x.get(name, 0) or 0) for x in per_symbol]
        return float(sum(vals) / max(len(vals), 1))

    return {
        "total_return_pct": avg("total_return_pct"),
        "profit_factor": avg("profit_factor"),
        "max_drawdown_pct": avg("max_drawdown_pct"),
        "sharpe_ratio": avg("sharpe_ratio"),
        "total_trades": avg("total_trades"),
        "win_rate": avg("win_rate"),
    }


def _buy_hold_return_pct(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    start = float(df["Close"].iloc[0])
    end = float(df["Close"].iloc[-1])
    if start <= 0:
        return 0.0
    return ((end - start) / start) * 100.0


def _score_candidate(metrics: Dict[str, float]) -> float:
    # Favor return and Sharpe, penalize drawdown.
    return (
        float(metrics.get("total_return_pct", 0.0))
        + (float(metrics.get("sharpe_ratio", 0.0)) * 5.0)
        - (float(metrics.get("max_drawdown_pct", 0.0)) * 0.7)
    )


# ── Grid-search helpers ───────────────────────────────────────────────────────

def _with_confidence_filter(
    fn: Callable[[pd.DataFrame, str], Optional[Dict[str, Any]]],
    min_confidence: float,
) -> Callable[[pd.DataFrame, str], Optional[Dict[str, Any]]]:
    """Wraps a strategy function to gate out signals below *min_confidence*."""
    def filtered(df_slice: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        trade = fn(df_slice, symbol)
        if trade is None:
            return None
        conf = _normalize_confidence(trade.get("confidence", 0.0))
        if conf < min_confidence:
            return None
        return trade
    return filtered


@dataclass
class GridSearchResult:
    confidence_threshold: float
    risk_per_trade_pct: float
    max_positions: int
    score: float
    return_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    best_candidate: str


def _grid_eval_params(
    candidate_fns: Dict[str, Callable[[pd.DataFrame, str], Optional[Dict[str, Any]]]],
    symbol_data: Dict[str, pd.DataFrame],
    eval_start: int,
    eval_end: int,
    capital: float,
    warmup: int,
    confidence_threshold: float,
    risk_per_trade_pct: float,
    max_positions: int,
) -> GridSearchResult:
    """Evaluate a single grid-search parameter combination.

    *max_positions* limits the number of symbols evaluated concurrently,
    approximating the real-world constraint of holding at most N open positions
    across the full universe at any time.
    """
    # Limit active symbol set to top *max_positions* by data length (liquidity proxy).
    sorted_syms = sorted(symbol_data.keys(), key=lambda s: len(symbol_data[s]), reverse=True)
    active_syms = {s: symbol_data[s] for s in sorted_syms[:max_positions]}

    # Patch CONFIG for this evaluation, restore unconditionally afterwards.
    orig_risk = CONFIG.get("risk_per_trade_pct", 0.01)
    CONFIG["risk_per_trade_pct"] = risk_per_trade_pct

    best_score = -1e9
    best_m: Optional[Dict[str, float]] = None
    best_name = "none"

    try:
        for cname, base_fn in candidate_fns.items():
            filtered_fn = _with_confidence_filter(base_fn, confidence_threshold)
            m = _run_candidate(cname, filtered_fn, active_syms, eval_start, eval_end, capital, warmup)
            sc = _score_candidate(m)
            if sc > best_score:
                best_score = sc
                best_m = m
                best_name = cname
    finally:
        CONFIG["risk_per_trade_pct"] = orig_risk

    m = best_m or {}
    return GridSearchResult(
        confidence_threshold=confidence_threshold,
        risk_per_trade_pct=risk_per_trade_pct,
        max_positions=max_positions,
        score=best_score,
        return_pct=float(m.get("total_return_pct", 0.0)),
        profit_factor=float(m.get("profit_factor", 0.0)),
        max_drawdown_pct=float(m.get("max_drawdown_pct", 0.0)),
        sharpe_ratio=float(m.get("sharpe_ratio", 0.0)),
        best_candidate=best_name,
    )


def _run_grid_search(
    args: argparse.Namespace,
    symbol_data: Dict[str, pd.DataFrame],
    candidates: Dict[str, Callable[[pd.DataFrame, str], Optional[Dict[str, Any]]]],
) -> GridSearchResult:
    """Sweep all grid-search parameter combinations and return the best result.

    After this function returns the caller should inject the winning params
    into *args* before calling *_run_benchmark()*.
    """
    conf_thresholds = [float(x) for x in args.gs_confidence_thresholds.split(",") if x.strip()]
    risk_pcts = [float(x) for x in args.gs_risk_pcts.split(",") if x.strip()]
    max_positions_list = [int(x) for x in args.gs_max_positions.split(",") if x.strip()]

    # Clamp gs_eval_frac to a sane range.
    eval_frac = max(0.10, min(0.50, float(args.gs_eval_frac)))

    min_len = min(len(df) for df in symbol_data.values())
    eval_start = int(min_len * (1.0 - eval_frac))
    eval_end = min_len

    # Safety: need at least warmup + 30 bars.
    if eval_end - eval_start < args.warmup + 30:
        eval_start = 0
        eval_end = min_len // 2

    total = len(conf_thresholds) * len(risk_pcts) * len(max_positions_list)
    print(f"\n{'='*60}")
    print(f"  GRID SEARCH  ({total} combinations)")
    print(f"{'='*60}")
    print(f"  confidence thresholds : {conf_thresholds}")
    print(f"  risk-per-trade fracs  : {risk_pcts}")
    print(f"  max-open-positions    : {max_positions_list}")
    print(f"  eval window           : bars {eval_start}..{eval_end} ({eval_end - eval_start} bars)")
    print()

    results: List[GridSearchResult] = []
    done = 0
    for conf in conf_thresholds:
        for risk in risk_pcts:
            for mp in max_positions_list:
                gs = _grid_eval_params(
                    candidate_fns=candidates,
                    symbol_data=symbol_data,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    capital=args.capital,
                    warmup=args.warmup,
                    confidence_threshold=conf,
                    risk_per_trade_pct=risk,
                    max_positions=mp,
                )
                results.append(gs)
                done += 1
                print(
                    f"  [{done:3d}/{total}]  conf={conf:.2f}  risk={risk:.4f}  maxpos={mp:2d}"
                    f"  -> score={gs.score:+8.3f}  ret={gs.return_pct:+6.2f}%"
                    f"  pf={gs.profit_factor:.3f}  dd={gs.max_drawdown_pct:.2f}%"
                    f"  [{gs.best_candidate}]"
                )

    results.sort(key=lambda r: r.score, reverse=True)
    best = results[0]

    print()
    print(f"{'='*60}")
    print("  TOP 5 GRID RESULTS:")
    print(f"{'='*60}")
    for rank, r in enumerate(results[:5], start=1):
        print(
            f"  #{rank}  conf={r.confidence_threshold:.2f}  risk={r.risk_per_trade_pct:.4f}"
            f"  maxpos={r.max_positions:2d}  score={r.score:+8.3f}"
            f"  ret={r.return_pct:+6.2f}%  pf={r.profit_factor:.3f}"
            f"  [{r.best_candidate}]"
        )
    print()
    print(f"  >>> Best params selected:")
    print(f"        confidence_threshold = {best.confidence_threshold}")
    print(f"        risk_per_trade_pct   = {best.risk_per_trade_pct}")
    print(f"        max_open_positions   = {best.max_positions}")
    print(f"        score                = {best.score:.3f}")
    print(f"        best_candidate       = {best.best_candidate}")
    print(f"{'='*60}")
    print("  Injecting best params into walk-forward benchmark...\n")

    return best


def _run_candidate(
    candidate_name: str,
    strategy_fn: Callable[[pd.DataFrame, str], Optional[Dict[str, Any]]],
    symbol_data: Dict[str, pd.DataFrame],
    start_idx: int,
    end_idx: int,
    capital: float,
    warmup: int,
) -> Dict[str, float]:
    per_symbol = []
    for sym, full_df in symbol_data.items():
        seg = full_df.iloc[start_idx:end_idx].copy()
        if seg is None or seg.empty or len(seg) < warmup + 30:
            continue
        bt = run_backtest(
            strategy_fn=strategy_fn,
            historical_df=seg,
            symbol=sym,
            initial_capital=capital,
            warmup=warmup,
            use_llm=False,
        )
        per_symbol.append(bt["metrics"])

    out = _aggregate_metric(per_symbol)
    out["candidate"] = candidate_name
    return out


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _load_data_and_candidates(
    args: argparse.Namespace,
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, Callable[[pd.DataFrame, str], Optional[Dict[str, Any]]]],
    Optional[pd.DataFrame],
]:
    """Fetch market data and build the strategy candidate map.

    Returns ``(symbol_data, candidates, spy_df)``.
    Prints warnings and returns empty dicts on failure; callers should check.
    """
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("No symbols provided")
        return {}, {}, None

    StrategyRegistry.autodiscover()
    strategy_names = StrategyRegistry.list_names()
    if not strategy_names:
        print("No registered strategies found")
        return {}, {}, None

    candidates: Dict[str, Callable[[pd.DataFrame, str], Optional[Dict[str, Any]]]] = {}
    for name in strategy_names:
        candidates[name] = _single_strategy_fn(name)
    candidates["ensemble_all"] = _ensemble_strategy_fn(strategy_names)

    print(f"Loaded {len(strategy_names)} strategies")
    print("Candidates:", ", ".join(candidates.keys()))

    symbol_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_latest_market_data(ticker=sym, period=args.period, interval=args.interval)
        if df is None or df.empty:
            print(f"[WARN] No data for {sym}, skipping")
            continue
        df = df.copy()
        if "Close" not in df.columns:
            print(f"[WARN] Data for {sym} missing Close column, skipping")
            continue
        symbol_data[sym] = df

    spy_df = fetch_latest_market_data(ticker="SPY", period=args.period, interval=args.interval)
    if spy_df is None or spy_df.empty:
        print("[WARN] SPY data unavailable, benchmark fallback will be 0%")

    return symbol_data, candidates, spy_df


def _run_benchmark(
    args: argparse.Namespace,
    symbol_data: Dict[str, pd.DataFrame],
    candidates: Dict[str, Callable[[pd.DataFrame, str], Optional[Dict[str, Any]]]],
    spy_df: Optional[pd.DataFrame],
    *,
    confidence_threshold: Optional[float] = None,
    risk_per_trade_pct: Optional[float] = None,
    max_active_symbols: Optional[int] = None,
) -> int:
    """Run the walk-forward benchmark.

    When *confidence_threshold*, *risk_per_trade_pct*, or *max_active_symbols*
    are supplied (by the grid-search caller) they override the CONFIG defaults
    for the duration of this run.
    """
    if not symbol_data:
        print("No symbols with valid data")
        return 2

    # ── Apply grid-search injected overrides ─────────────────────────────
    orig_risk = CONFIG.get("risk_per_trade_pct", 0.01)
    if risk_per_trade_pct is not None:
        CONFIG["risk_per_trade_pct"] = risk_per_trade_pct

    # Limit active symbol set when max_active_symbols is specified.
    active_symbol_data = symbol_data
    if max_active_symbols is not None and max_active_symbols < len(symbol_data):
        sorted_syms = sorted(symbol_data.keys(), key=lambda s: len(symbol_data[s]), reverse=True)
        active_symbol_data = {s: symbol_data[s] for s in sorted_syms[:max_active_symbols]}

    # Wrap all candidate fns with the confidence filter if a threshold is set.
    active_candidates = candidates
    if confidence_threshold is not None:
        active_candidates = {
            name: _with_confidence_filter(fn, confidence_threshold)
            for name, fn in candidates.items()
        }

    try:
        return _benchmark_loop(args, active_symbol_data, active_candidates, spy_df)
    finally:
        CONFIG["risk_per_trade_pct"] = orig_risk


def _benchmark_loop(
    args: argparse.Namespace,
    symbol_data: Dict[str, pd.DataFrame],
    candidates: Dict[str, Callable[[pd.DataFrame, str], Optional[Dict[str, Any]]]],
    spy_df: Optional[pd.DataFrame],
) -> int:
    min_len = min(len(df) for df in symbol_data.values())
    folds = _build_folds(min_len, args.folds, args.train_frac)
    if not folds:
        print("Insufficient data for walk-forward folds")
        return 2

    fold_results: List[FoldResult] = []

    for i, (tr0, tr1, te0, te1) in enumerate(folds, start=1):
        train_scores = []
        for cname, fn in candidates.items():
            m = _run_candidate(cname, fn, symbol_data, tr0, tr1, args.capital, args.warmup)
            m["score"] = _score_candidate(m)
            train_scores.append(m)

        train_scores.sort(key=lambda x: x["score"], reverse=True)
        selected = train_scores[0]
        selected_name = str(selected["candidate"])
        selected_fn = candidates[selected_name]

        test_m = _run_candidate(selected_name, selected_fn, symbol_data, te0, te1, args.capital, args.warmup)

        spy_ret = 0.0
        if spy_df is not None and not spy_df.empty and len(spy_df) >= te1:
            spy_seg = spy_df.iloc[te0:te1].copy()
            spy_ret = _buy_hold_return_pct(spy_seg)

        any_df = next(iter(symbol_data.values()))
        train_range = f"{any_df.index[tr0]} -> {any_df.index[tr1-1]}"
        test_range = f"{any_df.index[te0]} -> {any_df.index[te1-1]}"

        fr = FoldResult(
            fold_idx=i,
            train_range=train_range,
            test_range=test_range,
            selected_candidate=selected_name,
            train_return_pct=float(selected.get("total_return_pct", 0.0)),
            test_return_pct=float(test_m.get("total_return_pct", 0.0)),
            test_profit_factor=float(test_m.get("profit_factor", 0.0)),
            test_max_drawdown_pct=float(test_m.get("max_drawdown_pct", 0.0)),
            test_sharpe_ratio=float(test_m.get("sharpe_ratio", 0.0)),
            spy_return_pct=float(spy_ret),
        )
        fold_results.append(fr)

        print(
            f"[Fold {i}] selected={fr.selected_candidate} "
            f"train={fr.train_return_pct:.2f}% test={fr.test_return_pct:.2f}% "
            f"SPY={fr.spy_return_pct:.2f}%"
        )

    # Aggregate out-of-sample performance.
    bot_return = sum(f.test_return_pct for f in fold_results) / len(fold_results)
    bot_pf = sum(f.test_profit_factor for f in fold_results) / len(fold_results)
    bot_dd = sum(f.test_max_drawdown_pct for f in fold_results) / len(fold_results)
    bot_sharpe = sum(f.test_sharpe_ratio for f in fold_results) / len(fold_results)
    spy_return = sum(f.spy_return_pct for f in fold_results) / len(fold_results)

    gates = {
        "return_ge_30pct": bot_return >= 30.0,
        "beat_spy": bot_return > spy_return,
        "profit_factor_ge_1_30": bot_pf >= 1.30,
        "max_drawdown_le_12pct": bot_dd <= 12.0,
        "sharpe_ge_1_20": bot_sharpe >= 1.20,
    }
    promote = all(gates.values())

    selected_counts: Dict[str, int] = {}
    for f in fold_results:
        selected_counts[f.selected_candidate] = selected_counts.get(f.selected_candidate, 0) + 1
    top_strategy = max(selected_counts.items(), key=lambda kv: kv[1])[0]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": list(symbol_data.keys()),
        "period": args.period,
        "interval": args.interval,
        "folds": len(fold_results),
        "bot_return_pct_oos": round(bot_return, 2),
        "spy_return_pct_oos": round(spy_return, 2),
        "bot_profit_factor_oos": round(bot_pf, 3),
        "bot_max_drawdown_pct_oos": round(bot_dd, 2),
        "bot_sharpe_oos": round(bot_sharpe, 2),
        "promotion_gates": gates,
        "promote_to_next_stage": promote,
        "most_selected_strategy": top_strategy,
        "selection_counts": selected_counts,
        "fold_results": [f.__dict__ for f in fold_results],
    }

    _ensure_dir(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        "# Walk-Forward Benchmark: Bot vs S&P",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Symbols: {', '.join(summary['symbols'])}",
        f"- Period / Interval: {args.period} / {args.interval}",
        f"- Folds: {summary['folds']}",
        "",
        "## Out-of-Sample Summary",
        "",
        f"- Bot Return: {summary['bot_return_pct_oos']:.2f}%",
        f"- S&P (SPY) Return: {summary['spy_return_pct_oos']:.2f}%",
        f"- Profit Factor: {summary['bot_profit_factor_oos']:.3f}",
        f"- Max Drawdown: {summary['bot_max_drawdown_pct_oos']:.2f}%",
        f"- Sharpe: {summary['bot_sharpe_oos']:.2f}",
        f"- Most Selected Strategy: {summary['most_selected_strategy']}",
        "",
        "## Promotion Gates",
        "",
    ]
    for gate, passed in gates.items():
        md_lines.append(f"- {'PASS' if passed else 'FAIL'}: {gate}")
    md_lines += ["", "## Promotion Decision", "", f"- {'PROMOTE' if promote else 'HOLD'}"]

    _ensure_dir(args.out_md)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n=== Bot vs S&P ===")
    print(f"Bot OOS Return: {bot_return:.2f}%")
    print(f"SPY OOS Return: {spy_return:.2f}%")
    print(f"Promotion: {'PROMOTE' if promote else 'HOLD'}")
    print(f"Best Practical Option (most selected): {top_strategy}")
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved MD:   {args.out_md}")
    return 0


def main() -> int:
    args = _parse_args()

    symbol_data, candidates, spy_df = _load_data_and_candidates(args)
    if not symbol_data or not candidates:
        return 2

    if args.mode == "grid-search":
        best = _run_grid_search(args, symbol_data, candidates)
        return _run_benchmark(
            args,
            symbol_data,
            candidates,
            spy_df,
            confidence_threshold=best.confidence_threshold,
            risk_per_trade_pct=best.risk_per_trade_pct,
            max_active_symbols=best.max_positions,
        )

    # Default: standard benchmark (no param injection).
    return _run_benchmark(args, symbol_data, candidates, spy_df)


if __name__ == "__main__":
    raise SystemExit(main())
