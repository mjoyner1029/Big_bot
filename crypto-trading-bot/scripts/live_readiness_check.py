#!/usr/bin/env python3
"""Strict live-money readiness validator.

This script performs objective GO / NO-GO checks for live trading readiness.
It does not place orders and does not enable live trading.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _add(results: List[CheckResult], name: str, passed: bool, detail: str) -> None:
    results.append(CheckResult(name=name, passed=passed, detail=detail))


def _find_first_existing(root: Path, rel_paths: List[str]) -> Optional[Path]:
    for rel in rel_paths:
        p = root / rel
        if p.exists():
            return p
    return None


def check_tests(root: Path) -> List[CheckResult]:
    results: List[CheckResult] = []
    pytest_results = root / "pytest-results.txt"

    _add(
        results,
        "pytest-results.txt exists",
        pytest_results.exists(),
        f"expected file: {pytest_results.relative_to(root)}",
    )

    if not pytest_results.exists():
        _add(results, "latest pytest all passing", False, "missing pytest-results.txt")
        return results

    text = _read_text(pytest_results)
    summary_match = re.search(
        r"(?P<passed>\d+)\s+passed(?:,\s*(?P<failed>\d+)\s+failed)?(?:,\s*(?P<errors>\d+)\s+error[s]?)?",
        text,
        flags=re.IGNORECASE,
    )
    if not summary_match:
        _add(results, "latest pytest all passing", False, "could not parse pytest summary line")
        return results

    passed_n = int(summary_match.group("passed") or 0)
    failed_n = int(summary_match.group("failed") or 0)
    errors_n = int(summary_match.group("errors") or 0)
    ok = passed_n > 0 and failed_n == 0 and errors_n == 0
    _add(
        results,
        "latest pytest all passing",
        ok,
        f"parsed summary: passed={passed_n}, failed={failed_n}, errors={errors_n}",
    )

    return results


def _parse_trade_rows(csv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    failures: List[str] = []
    rows: List[Dict[str, str]] = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as exc:
        failures.append(f"failed to read {csv_path.name}: {exc}")
    return rows, failures


def _parse_ts(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def check_paper_trading(root: Path, cfg: Dict) -> List[CheckResult]:
    results: List[CheckResult] = []
    configured_log = str(cfg.get("trade_log_path", "logs/trade_log.csv"))
    trade_log = root / configured_log

    _add(
        results,
        "paper trade log exists",
        trade_log.exists(),
        f"expected file: {trade_log.relative_to(root)}",
    )

    if not trade_log.exists():
        missing_msg = f"missing {trade_log.relative_to(root)}"
        _add(results, "30 days of paper logs", False, missing_msg)
        _add(results, "100 trades OR 30 days complete", False, missing_msg)
        _add(results, "expectancy after fees/slippage", False, missing_msg)
        _add(results, "max drawdown below configured threshold", False, missing_msg)
        _add(results, "paper metrics computed", False, missing_msg)
        return results

    rows, parse_failures = _parse_trade_rows(trade_log)
    if parse_failures:
        for f in parse_failures:
            _add(results, "paper trade log parse", False, f)
        return results

    if not rows:
        empty_msg = f"{trade_log.name} has no rows"
        _add(results, "30 days of paper logs", False, empty_msg)
        _add(results, "100 trades OR 30 days complete", False, empty_msg)
        _add(results, "expectancy after fees/slippage", False, "no trade rows")
        _add(results, "max drawdown below configured threshold", False, "no trade rows")
        _add(results, "paper metrics computed", False, "no trade rows")
        return results

    seed_tags = {"readiness_seed", "synthetic_seed", "seed"}
    real_rows = [
        r for r in rows
        if (r.get("strategy_name", "").strip().lower() not in seed_tags)
    ]
    _add(
        results,
        "paper log contains non-seeded rows",
        len(real_rows) > 0,
        f"real_rows={len(real_rows)}, total_rows={len(rows)}",
    )

    if not real_rows:
        _add(results, "30 days of paper logs", False, "no non-seeded trade rows")
        _add(results, "100 trades OR 30 days complete", False, "no non-seeded trade rows")
        _add(results, "expectancy after fees/slippage", False, "no non-seeded closed trades")
        _add(results, "max drawdown below configured threshold", False, "no non-seeded closed trades")
        _add(results, "paper metrics computed", False, "no non-seeded closed trades")
        return results

    timestamps = [
        _parse_ts(r.get("timestamp", ""))
        for r in real_rows
        if r.get("timestamp")
    ]
    timestamps = [t for t in timestamps if t is not None]
    if len(timestamps) >= 2:
        span_days = (max(timestamps) - min(timestamps)).days
    else:
        span_days = 0

    trade_count = len(real_rows)
    thirty_days_ok = span_days >= 30
    min_volume_ok = (trade_count >= 100) or thirty_days_ok

    if timestamps:
        last_trade_ts = max(timestamps)
        age_hours = (datetime.now(last_trade_ts.tzinfo) - last_trade_ts).total_seconds() / 3600.0
        recent_ok = age_hours <= 72.0
    else:
        age_hours = float("inf")
        recent_ok = False
    _add(
        results,
        "recent paper activity (<=72h)",
        recent_ok,
        f"last_trade_age_hours={age_hours:.1f}",
    )

    _add(
        results,
        "30 days of paper logs",
        thirty_days_ok,
        f"observed log span: {span_days} day(s)",
    )
    _add(
        results,
        "100 trades OR 30 days complete",
        min_volume_ok,
        f"trades={trade_count}, span_days={span_days}",
    )

    # Closed trades required for expectancy and PnL-derived metrics.
    # NOTE: slippage is already applied to fill prices in the CSV (paper_trader
    # adjusts entry price by slip_bps on open). Only subtract commission here
    # to avoid double-counting.
    closed = []
    fee_pct = float(cfg.get("exchange_fee_pct", 0.0) or 0.0)
    cost_rate = fee_pct  # slip already in fill price; don't double-count

    for r in real_rows:
        pnl_raw = (r.get("pnl") or "").strip()
        exit_raw = (r.get("exit_price") or "").strip()
        if not pnl_raw or not exit_raw:
            continue
        try:
            pnl = float(pnl_raw)
            entry = float(r.get("entry_price") or 0.0)
            exit_price = float(exit_raw)
            qty = float(r.get("qty") or 0.0)
        except Exception:
            continue

        est_cost = (entry * qty + exit_price * qty) * cost_rate
        net_pnl = pnl - est_cost
        closed.append(net_pnl)

    if not closed:
        _add(results, "expectancy after fees/slippage", False, "no closed trades with pnl/exit_price")
        _add(results, "max drawdown below configured threshold", False, "no closed trades for drawdown")
        _add(results, "paper metrics computed", False, "no closed trades for win rate/profit factor/sharpe/return")
        return results

    expectancy = sum(closed) / len(closed)
    _add(
        results,
        "expectancy after fees/slippage",
        expectancy > 0,
        f"expectancy={expectancy:.4f} (net per closed trade)",
    )

    start_capital = float(cfg.get("capital", 0.0) or 0.0)
    if start_capital <= 0:
        start_capital = 10000.0

    equity = start_capital
    peak = start_capital
    max_dd = 0.0
    for p in closed:
        equity += p
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

    dd_threshold = float(cfg.get("max_total_loss_pct", 0.20) or 0.20)
    dd_ok = max_dd <= dd_threshold
    _add(
        results,
        "max drawdown below configured threshold",
        dd_ok,
        f"max_drawdown={max_dd:.4%}, threshold={dd_threshold:.4%}",
    )

    wins = [p for p in closed if p > 0]
    losses = [p for p in closed if p <= 0]
    win_rate = (len(wins) / len(closed)) if closed else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    returns = [p / start_capital for p in closed]
    if len(returns) > 1:
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns)
        sharpe = (mean_r / std_r) * math.sqrt(len(returns)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    total_return = sum(closed) / start_capital
    metrics_detail = (
        f"win_rate={win_rate:.2%}, profit_factor={profit_factor:.4f}, "
        f"sharpe={sharpe:.4f}, total_return={total_return:.2%}"
    )
    _add(results, "paper metrics computed", True, metrics_detail)

    target_win_rate = float(cfg.get("target_win_rate", 0.60) or 0.60)
    target_avg_gain = float(cfg.get("target_average_gain_pct", 0.15) or 0.15)
    _add(
        results,
        "paper win rate meets target",
        win_rate >= target_win_rate,
        f"win_rate={win_rate:.2%}, target={target_win_rate:.2%}",
    )
    _add(
        results,
        "paper average gain meets target (CSV window)",
        total_return >= target_avg_gain,
        f"total_return={total_return:.2%}, target={target_avg_gain:.2%}",
    )

    # --- Portfolio-state total return (authoritative lifetime check) ----------
    import json as _json
    state_paths = [
        root / "state" / "paper_training_state.json",
        root / "state" / "intraday_evidence_state.json",
    ]
    portfolio_return_ok = False
    portfolio_return_detail = "no portfolio state file found"
    for sp in state_paths:
        if sp.exists():
            try:
                state = _json.loads(sp.read_text())
                port_cash = float(state.get("cash", 0))
                port_start = float(state.get("starting_capital", 0) or 0)
                closed_t = state.get("closed_trades", [])
                if port_start <= 0:
                    port_start = 10000.0
                port_pnl = sum(float(t.get("pnl", 0)) for t in closed_t)
                port_return = port_pnl / port_start
                portfolio_return_ok = port_return >= target_avg_gain
                portfolio_return_detail = (
                    f"portfolio_return={port_return:.2%}, trades={len(closed_t)}, "
                    f"target={target_avg_gain:.2%}, source={sp.name}"
                )
            except Exception as exc:
                portfolio_return_detail = f"failed to parse {sp.name}: {exc}"
            break
    _add(results, "portfolio lifetime return meets target", portfolio_return_ok, portfolio_return_detail)

    return results


def _check_keywords_in_path(path: Path, keywords: List[str]) -> bool:
    text = _read_text(path).lower()
    return all(k.lower() in text for k in keywords)


def check_broker_sandbox(root: Path) -> List[CheckResult]:
    results: List[CheckResult] = []
    candidates = [
        "reports/sandbox_certification.json",
        "reports/sandbox_certification.md",
        "docs/SANDBOX_CERTIFICATION.md",
        "state/sandbox_certification.json",
    ]

    artifact = _find_first_existing(root, candidates)
    _add(
        results,
        "sandbox certification results exist",
        artifact is not None,
        (
            f"found: {artifact.relative_to(root)}"
            if artifact is not None
            else f"missing all expected files: {', '.join(candidates)}"
        ),
    )

    lifecycle_checks = [
        ("sandbox place order tested", ["place order"]),
        ("sandbox cancel order tested", ["cancel order"]),
        ("sandbox partial fill tested", ["partial fill"]),
        ("sandbox close position tested", ["close position"]),
        ("sandbox rejected order handling tested", ["rejected order"]),
        ("sandbox insufficient balance handling tested", ["insufficient balance"]),
    ]

    if artifact is None:
        for name, keys in lifecycle_checks:
            _add(results, name, False, f"missing certification artifact: expected one of {', '.join(candidates)}")
        return results

    text = _read_text(artifact).lower()
    for name, keys in lifecycle_checks:
        ok = all(k in text for k in keys)
        _add(results, name, ok, f"source={artifact.relative_to(root)}; required keyword(s): {', '.join(keys)}")

    return results


def check_risk_controls(root: Path, cfg: Dict) -> List[CheckResult]:
    results: List[CheckResult] = []

    _add(
        results,
        "use_paper_trading defaults to true",
        bool(cfg.get("use_paper_trading", False)) is True,
        f"config use_paper_trading={cfg.get('use_paper_trading')}",
    )

    cfg_text = _read_text(root / "config" / "config.py")
    explicit_env_flag = any(
        k in cfg_text
        for k in ["ENABLE_LIVE_TRADING", "ALLOW_LIVE_TRADING", "LIVE_TRADING_ENABLED"]
    )
    _add(
        results,
        "live trading requires explicit environment flags",
        explicit_env_flag,
        "expected env-backed live flag key (e.g., ENABLE_LIVE_TRADING) in config/config.py",
    )

    _add(
        results,
        "max daily loss kill switch exists",
        "max_daily_drawdown_pct" in cfg,
        "required config key: max_daily_drawdown_pct",
    )
    _add(
        results,
        "max open positions check exists",
        "max_open_positions" in cfg,
        "required config key: max_open_positions",
    )
    _add(
        results,
        "max single position check exists",
        ("max_position_value" in cfg) or ("max_position_pct" in cfg),
        "required config key: max_position_value or max_position_pct",
    )

    # Check stop-loss enforcement across executor candidates
    executor_candidates = [
        root / "trading" / "trade_executor.py",
        root / "agent" / "trade_executor.py",
        root / "core" / "trade_executor.py",
    ]
    executor_text = ""
    for ep in executor_candidates:
        if ep.exists():
            executor_text = _read_text(ep).lower()
            break
    stop_loss_required = "stop_loss_price" in executor_text or "stop_loss" in executor_text
    _add(
        results,
        "stop-loss requirement exists",
        stop_loss_required,
        "expected stop-loss enforcement references in trade executor",
    )

    # Check emergency stop in entrypoints (not main.py — now paper_trade_v3 / live_test_v3)
    entrypoint_candidates = [
        root / "paper_trade_v3.py",
        root / "live_test_v3.py",
        root / "ultimate_bot_v3_llm.py",
    ]
    entrypoint_text = "\n".join(_read_text(p) for p in entrypoint_candidates if p.exists())
    _add(
        results,
        "emergency stop file is supported",
        "EMERGENCY_STOP" in entrypoint_text or "emergency_stop_file" in cfg,
        "expected emergency stop handling in entrypoints or config",
    )
    _add(
        results,
        "pause trading file is supported",
        "PAUSE_TRADING" in entrypoint_text or "pause_trading_file" in cfg,
        "expected pause trading handling in entrypoints or config",
    )

    return results


def check_backtesting(root: Path) -> List[CheckResult]:
    results: List[CheckResult] = []

    walk_forward_candidates = [
        "reports/walk_forward_results.json",
        "reports/walk_forward_results.md",
        "state/walk_forward_results.json",
    ]
    oos_candidates = [
        "reports/out_of_sample_results.json",
        "reports/out_of_sample_results.md",
        "state/out_of_sample_results.json",
    ]
    benchmark_candidates = [
        "reports/benchmark_comparison.json",
        "reports/benchmark_comparison.md",
        "state/benchmark_comparison.json",
    ]

    walk = _find_first_existing(root, walk_forward_candidates)
    oos = _find_first_existing(root, oos_candidates)
    bench = _find_first_existing(root, benchmark_candidates)

    _add(
        results,
        "walk-forward results exist",
        walk is not None,
        (f"found: {walk.relative_to(root)}" if walk else f"missing all expected files: {', '.join(walk_forward_candidates)}"),
    )
    _add(
        results,
        "out-of-sample results exist",
        oos is not None,
        (f"found: {oos.relative_to(root)}" if oos else f"missing all expected files: {', '.join(oos_candidates)}"),
    )

    fee_slip_ok = False
    fee_slip_detail = "missing evidence"
    for p in [walk, oos, bench]:
        if p is None:
            continue
        txt = _read_text(p).lower()
        if ("fee" in txt) and ("slippage" in txt):
            fee_slip_ok = True
            fee_slip_detail = f"found in {p.relative_to(root)}"
            break
    _add(results, "fees and slippage included", fee_slip_ok, fee_slip_detail)

    bench_ok = False
    bench_detail = "missing benchmark comparison artifact"
    if bench is not None:
        txt = _read_text(bench).lower()
        bench_ok = all(k in txt for k in ["spy", "qqq", "btc"])
        bench_detail = f"source={bench.relative_to(root)} requires SPY/QQQ/BTC references"
    _add(results, "benchmark comparison exists against SPY, QQQ, and BTC", bench_ok, bench_detail)

    return results


def check_operations(root: Path) -> List[CheckResult]:
    results: List[CheckResult] = []

    runbook_candidates = [
        "PRODUCTION_READINESS.md",
        "docs/PRODUCTION_RUNBOOK.md",
        "RUNBOOK.md",
    ]
    runbook = _find_first_existing(root, runbook_candidates)
    _add(
        results,
        "production runbook exists",
        runbook is not None,
        (f"found: {runbook.relative_to(root)}" if runbook else f"missing all expected files: {', '.join(runbook_candidates)}"),
    )

    # Search a small set of operations docs for required topics.
    doc_candidates = [
        root / "PRODUCTION_READINESS.md",
        root / "README.md",
        root / "INTEGRATION_GUIDE.md",
        root / "QUICK_REFERENCE.md",
    ]
    combined = "\n".join(_read_text(p).lower() for p in doc_candidates if p.exists())

    _add(
        results,
        "incident response plan exists",
        ("incident" in combined) or ("emergency procedures" in combined),
        "expected incident/emergency response section in operations docs",
    )
    _add(
        results,
        "rollback plan exists",
        "rollback" in combined,
        "expected rollback section in operations docs",
    )
    _add(
        results,
        "manual approval mode documented",
        ("manual approval" in combined) or ("manual mode" in combined),
        "expected manual approval mode section in operations docs",
    )
    _add(
        results,
        "monitoring/logging documented",
        ("monitor" in combined) and ("log" in combined),
        "expected monitoring + logging documentation in operations docs",
    )

    return results


def load_config_dict(root: Path) -> Dict:
    config_path = root / "config" / "config.py"
    namespace: Dict = {}
    try:
        code = config_path.read_text(encoding="utf-8", errors="ignore")
        exec(compile(code, str(config_path), "exec"), namespace, namespace)
        cfg = namespace.get("CONFIG", {})
        if isinstance(cfg, dict):
            return cfg
    except Exception as e:
        logging.warning(f"[Readiness] Failed to load config from {config_path}: {e}")
    return {}


def main() -> int:
    """
    IMPORTANT DISTINCTION:

        scripts/preflight.py        — startup readiness ("can the software start?")
        scripts/live_readiness_check.py — live-capital readiness ("has strategy earned real money?")

    A PAPER startup can PASS preflight while LIVE readiness remains NO-GO.
    This script does NOT activate or enable live trading.
    """
    parser = argparse.ArgumentParser(
        description="Live-capital readiness validator (separate from preflight)"
    )
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="Project root directory")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg = load_config_dict(root)

    print("=" * 72)
    print(" LIVE CAPITAL READINESS CHECK")
    print(" NOTE: This validates strategy evidence, not software startup.")
    print(" For startup readiness, run: python scripts/preflight.py")
    print("=" * 72)
    print()

    all_results: List[CheckResult] = []
    all_results.extend(check_tests(root))
    all_results.extend(check_paper_trading(root, cfg))
    all_results.extend(check_broker_sandbox(root))
    all_results.extend(check_risk_controls(root, cfg))
    all_results.extend(check_backtesting(root))
    all_results.extend(check_operations(root))

    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")

    failures = [r for r in all_results if not r.passed]
    print()
    print("-" * 72)
    if failures:
        print()
        print("  LIVE READINESS: NO-GO")
        print()
        print("  Missing requirements:")
        for f in failures:
            print(f"    ✗ {f.name}")
            print(f"      {f.detail}")
        print()
        print("  Complete a full PAPER trading campaign first.")
        print("  Run: python -m validation.cli campaign90 --name 'Baseline-90d'")
        print("-" * 72)
        return 1

    print()
    print("  LIVE READINESS: GO")
    print("  Strategy has met configured evidence requirements.")
    print("  STILL REQUIRES human authorization before deploying real capital.")
    print("  See: validation/live_graduation.py")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
