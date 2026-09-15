#!/usr/bin/env python3
"""
Preflight — single canonical startup readiness check.

Usage
─────
    python scripts/preflight.py             # default: check EXECUTION_MODE
    python scripts/preflight.py --mode PAPER
    python scripts/preflight.py --smoke     # add lightweight smoke pipeline test

Exit codes
──────────
    0   All checks passed — safe to start in selected mode
    1   One or more NO-GO failures — do NOT start the bot

Output ends with one of:
    ====================================
    PAPER TRADING READY
    ====================================

    ====================================
    SHADOW TRADING READY
    ====================================

    ====================================
    BACKTEST READY
    ====================================

    ====================================
    LIVE TRADING READY
    ====================================

    ====================================
    NO-GO
    ====================================
    Reason 1
    Reason 2
    ...
"""
from __future__ import annotations

import argparse
import importlib
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ── Project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

# ── Types ─────────────────────────────────────────────────────────────────────

class CheckResult:
    __slots__ = ("name", "status", "detail")

    def __init__(self, name: str, status: str, detail: str = ""):
        self.name   = name   # check label
        self.status = status # "PASS" | "WARN" | "FAIL"
        self.detail = detail

    @property
    def is_fail(self) -> bool:
        return self.status == "FAIL"

    @property
    def icon(self) -> str:
        return {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(self.status, "?")

    def __str__(self) -> str:
        base = f"  {self.icon} [{self.status:<4}] {self.name}"
        if self.detail:
            base += f"\n         {self.detail}"
        return base


def _pass(name: str, detail: str = "") -> CheckResult:
    return CheckResult(name, "PASS", detail)

def _warn(name: str, detail: str = "") -> CheckResult:
    return CheckResult(name, "WARN", detail)

def _fail(name: str, detail: str = "") -> CheckResult:
    return CheckResult(name, "FAIL", detail)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _env(key: str) -> str:
    return os.getenv(key, "")

def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() == "true"

def _scrub(value: str) -> str:
    """Show only first 4 chars of secret values."""
    if not value:
        return "(empty)"
    return value[:4] + "***" + f"({len(value)} chars)"


# ════════════════════════════════════════════════════════════════════════════
# CHECK GROUPS
# ════════════════════════════════════════════════════════════════════════════

def check_python_version() -> List[CheckResult]:
    results = []
    vi = sys.version_info
    if vi >= (3, 9):
        results.append(_pass("Python version", f"{vi.major}.{vi.minor}.{vi.micro}"))
    else:
        results.append(_fail("Python version",
            f"{vi.major}.{vi.minor}.{vi.micro} — requires ≥3.9"))
    return results


def check_execution_mode(requested_mode: Optional[str]) -> Tuple[List[CheckResult], str]:
    """Resolve and validate execution mode. Returns (results, resolved_mode_str)."""
    results = []
    from core.trading_mode import resolve_execution_mode, ExecutionMode

    raw = requested_mode or _env("EXECUTION_MODE") or _env("TRADING_MODE") or ""

    try:
        mode = resolve_execution_mode(raw if raw else "PAPER")
        results.append(_pass("Execution mode", f"{mode.value}"))
    except ValueError as e:
        results.append(_fail("Execution mode", str(e)))
        return results, "UNKNOWN"

    # Live guard
    if mode == ExecutionMode.LIVE:
        live_ok = _env_bool("ENABLE_LIVE_TRADING")
        if not live_ok:
            results.append(_fail("LIVE guard",
                "EXECUTION_MODE=LIVE but ENABLE_LIVE_TRADING is not 'true'. "
                "Both must be set to use live capital."))
        else:
            results.append(_warn("LIVE guard",
                "ENABLE_LIVE_TRADING=true — THIS IS REAL CAPITAL."))

    return results, mode.value


def check_dependencies() -> List[CheckResult]:
    """Verify critical imports are available."""
    results = []
    required = [
        ("dotenv",          "python-dotenv"),
        ("alpaca_trade_api", "alpaca-trade-api"),
        ("anthropic",       "anthropic"),
        ("pandas",          "pandas"),
        ("numpy",           "numpy"),
        ("requests",        "requests"),
        ("sqlite3",         "sqlite3 (stdlib)"),
    ]
    optional = [
        ("lightgbm",   "lightgbm"),
        ("sklearn",    "scikit-learn"),
        ("ta",         "ta (technical-analysis)"),
        ("yaml",       "pyyaml"),
    ]

    for module, pkg in required:
        try:
            importlib.import_module(module)
            results.append(_pass(f"Import {pkg}"))
        except ImportError:
            results.append(_fail(f"Import {pkg}",
                f"Missing. Run: pip install {pkg}"))

    for module, pkg in optional:
        try:
            importlib.import_module(module)
            results.append(_pass(f"Import {pkg} (optional)"))
        except ImportError:
            results.append(_warn(f"Import {pkg} (optional)",
                f"Not installed — some features will be disabled"))

    return results


def check_env_vars(mode: str) -> List[CheckResult]:
    """Check required environment variables for the given mode."""
    results = []

    # Always required
    mandatory = [
        ("ANTHROPIC_API_KEY", "Anthropic/Claude AI"),
        ("ALPACA_API_KEY",    "Alpaca broker key"),
        ("ALPACA_API_SECRET", "Alpaca broker secret"),
    ]
    for key, label in mandatory:
        val = _env(key)
        if not val:
            results.append(_fail(f"Env: {key}",
                f"{label} — not set. Add to .env"))
        else:
            results.append(_pass(f"Env: {key}", f"configured {_scrub(val)}"))

    # ALPACA_BASE_URL
    base_url = _env("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets"
    results.append(_pass("Env: ALPACA_BASE_URL", base_url))

    # EXECUTION_MODE
    val = _env("EXECUTION_MODE") or _env("TRADING_MODE")
    if val:
        results.append(_pass("Env: EXECUTION_MODE", val))
    else:
        results.append(_warn("Env: EXECUTION_MODE",
            "Not set — defaulting to PAPER. Set EXECUTION_MODE=PAPER in .env"))

    # ENABLE_LIVE_TRADING should explicitly exist
    elt = _env("ENABLE_LIVE_TRADING")
    if not elt:
        results.append(_warn("Env: ENABLE_LIVE_TRADING",
            "Not set — defaults to false (safe). Recommend adding ENABLE_LIVE_TRADING=false to .env"))
    elif elt.lower() == "false":
        results.append(_pass("Env: ENABLE_LIVE_TRADING", "false (safe)"))
    elif elt.lower() == "true":
        if mode == "LIVE":
            results.append(_warn("Env: ENABLE_LIVE_TRADING",
                "true — live capital ENABLED. Ensure this is intentional."))
        else:
            results.append(_warn("Env: ENABLE_LIVE_TRADING",
                f"true but EXECUTION_MODE={mode}. Live trading gate is set but mode prevents it."))
    else:
        results.append(_warn("Env: ENABLE_LIVE_TRADING",
            f"Unexpected value '{elt}' — should be 'true' or 'false'"))

    # TRADING_CAPITAL
    cap_str = _env("TRADING_CAPITAL")
    if cap_str:
        try:
            cap = float(cap_str)
            if cap <= 0:
                results.append(_fail("Env: TRADING_CAPITAL",
                    f"${cap:,.2f} — must be positive"))
            elif cap < 100:
                results.append(_warn("Env: TRADING_CAPITAL",
                    f"${cap:,.2f} — very small capital"))
            else:
                results.append(_pass("Env: TRADING_CAPITAL", f"${cap:,.2f}"))
        except ValueError:
            results.append(_fail("Env: TRADING_CAPITAL",
                f"'{cap_str}' is not a number"))
    else:
        results.append(_warn("Env: TRADING_CAPITAL",
            "Not set — will use default $2,000"))

    # Kronos (optional)
    kronos_path = _env("KRONOS_PATH") or str(PROJECT_ROOT.parent / "Kronos")
    if Path(kronos_path).exists():
        results.append(_pass("Env: KRONOS_PATH (optional)",
            f"directory exists: {kronos_path}"))
    else:
        results.append(_warn("Env: KRONOS_PATH (optional)",
            f"Not found at {kronos_path} — Kronos forecasting disabled"))

    return results


def check_broker_mode(mode: str) -> List[CheckResult]:
    """Verify broker URL matches the requested execution mode."""
    results = []
    from core.trading_mode import verify_broker_matches_mode, ExecutionMode

    base_url = _env("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets"

    try:
        em = ExecutionMode(mode)
        ok, reason = verify_broker_matches_mode(base_url, em)
        if ok:
            results.append(_pass("Broker/mode alignment", reason))
        else:
            # For PAPER/LIVE, this is a hard failure
            if em in (ExecutionMode.PAPER, ExecutionMode.LIVE):
                results.append(_fail("Broker/mode alignment", reason))
            else:
                results.append(_warn("Broker/mode alignment", reason))
    except Exception as e:
        results.append(_warn("Broker/mode alignment", str(e)))

    return results


def check_api_connectivity(mode: str) -> List[CheckResult]:
    """Test read-only API connectivity. Never places trades."""
    results = []

    # Alpaca connectivity (read-only account info)
    if mode in ("PAPER", "LIVE", "SHADOW"):
        try:
            import alpaca_trade_api as tradeapi
            key    = _env("ALPACA_API_KEY")
            secret = _env("ALPACA_API_SECRET")
            base   = _env("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets"

            if not key or not secret:
                results.append(_fail("Alpaca connectivity",
                    "ALPACA_API_KEY or ALPACA_API_SECRET not set"))
            else:
                t0  = time.time()
                api = tradeapi.REST(key, secret, base_url=base, api_version="v2")
                account = api.get_account()
                ms  = int((time.time() - t0) * 1000)
                status = account.status
                equity = getattr(account, 'equity', 'N/A')
                results.append(_pass("Alpaca connectivity",
                    f"account status={status}, equity={equity} ({ms}ms)"))
        except Exception as e:
            results.append(_fail("Alpaca connectivity",
                f"{type(e).__name__}: {str(e)[:120]}"))
    else:
        results.append(_pass("Alpaca connectivity",
            f"SKIPPED (mode={mode} does not use live broker)"))

    # Anthropic connectivity (client init only — no API call)
    try:
        import anthropic
        key = _env("ANTHROPIC_API_KEY")
        if not key:
            results.append(_fail("Anthropic client init",
                "ANTHROPIC_API_KEY not set"))
        else:
            _client = anthropic.Anthropic(api_key=key)
            results.append(_pass("Anthropic client init",
                f"client initialized (key: {_scrub(key)})"))
    except Exception as e:
        results.append(_fail("Anthropic client init",
            f"{type(e).__name__}: {str(e)[:120]}"))

    return results


def check_kronos(mode: str) -> List[CheckResult]:
    """Check Kronos availability."""
    results = []
    kronos_path = _env("KRONOS_PATH") or str(PROJECT_ROOT.parent / "Kronos")
    required    = _env_bool("KRONOS_REQUIRED")

    if Path(kronos_path).exists():
        # Try import
        try:
            kp = str(Path(kronos_path))
            if kp not in sys.path:
                sys.path.insert(0, kp)
            import importlib.util
            spec = importlib.util.find_spec("model")
            if spec:
                results.append(_pass("Kronos", f"AVAILABLE at {kronos_path}"))
            else:
                results.append(_warn("Kronos",
                    f"Path exists but model module not found — check Kronos installation"))
        except Exception as e:
            results.append(_warn("Kronos",
                f"Path exists but import failed: {e}"))
    else:
        if required:
            results.append(_fail("Kronos",
                f"KRONOS_REQUIRED=true but not found at {kronos_path}"))
        else:
            results.append(_warn("Kronos",
                f"DISABLED — not found at {kronos_path}. Set KRONOS_PATH to enable."))

    return results


def check_database(mode: str) -> List[CheckResult]:
    """Verify all required databases and tables initialize correctly."""
    results = []

    db_path = _env("TRADE_MEMORY_DB") or "data/trade_memory.sqlite"
    db_dir  = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    # Core trade memory DB
    try:
        conn = sqlite3.connect(db_path)
        tables_needed = [
            ("positions",          "CREATE TABLE IF NOT EXISTS positions (id TEXT PRIMARY KEY)"),
            ("orders",             "CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY)"),
            ("experiments",        "CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY)"),
            ("strategy_versions",  "CREATE TABLE IF NOT EXISTS strategy_versions (id TEXT PRIMARY KEY)"),
            ("model_versions",     "CREATE TABLE IF NOT EXISTS model_versions (id TEXT PRIMARY KEY)"),
            ("paper_campaigns",    "CREATE TABLE IF NOT EXISTS paper_campaigns (campaign_id TEXT PRIMARY KEY)"),
            ("session_configs",    "CREATE TABLE IF NOT EXISTS session_configs (session_id TEXT PRIMARY KEY)"),
            ("validation_reports", "CREATE TABLE IF NOT EXISTS validation_reports (report_id TEXT PRIMARY KEY)"),
        ]
        for tbl_name, ddl in tables_needed:
            conn.execute(ddl)
        conn.commit()

        # Count existing closed positions
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM positions WHERE status='CLOSED'"
            ).fetchone()
            closed = row[0] if row else 0
        except Exception:
            closed = 0

        conn.close()
        results.append(_pass("Database: trade_memory.sqlite",
            f"schema OK, {closed} closed positions on record ({db_path})"))
    except Exception as e:
        results.append(_fail("Database: trade_memory.sqlite",
            f"{type(e).__name__}: {e}"))

    # Feature store DB
    fs_path = _env("FEATURE_STORE_DB") or "data/feature_store.sqlite"
    try:
        conn = sqlite3.connect(fs_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS opportunities "
            "(id TEXT PRIMARY KEY, created_at TEXT)"
        )
        conn.commit()
        conn.close()
        results.append(_pass("Database: feature_store.sqlite",
            f"schema OK ({fs_path})"))
    except Exception as e:
        results.append(_fail("Database: feature_store.sqlite",
            f"{type(e).__name__}: {e}"))

    return results


def check_filesystem(mode: str) -> List[CheckResult]:
    """Ensure required directories exist and are writable."""
    results = []

    required_dirs = [
        "logs",
        "data",
        "state",
        "reports",
        "models/saved",
        "experiments",
        "memory",
        "pids",
    ]

    for d in required_dirs:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)

        # Test write permission
        test_file = path / ".preflight_write_test"
        try:
            test_file.write_text("ok")
            test_file.unlink()
            results.append(_pass(f"Directory: {d}/", "writable"))
        except PermissionError:
            results.append(_fail(f"Directory: {d}/",
                f"EXISTS but not writable — check permissions on {path}"))
        except Exception as e:
            results.append(_fail(f"Directory: {d}/",
                f"{type(e).__name__}: {e}"))

    return results


def check_logging(mode: str) -> List[CheckResult]:
    """Verify log file can be created and appended to."""
    results = []

    log_path = Path(_env("BOT_LOG_PATH") or "logs/bot.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with log_path.open("a") as f:
            from datetime import datetime, timezone
            f.write(f"[PREFLIGHT] {datetime.now(timezone.utc).isoformat()} mode={mode}\n")
        results.append(_pass("Logging: logs/bot.log",
            f"append OK ({log_path})"))
    except Exception as e:
        results.append(_fail("Logging: logs/bot.log",
            f"Cannot write: {type(e).__name__}: {e}"))

    return results


def check_safety_config() -> List[CheckResult]:
    """Verify critical risk controls are configured and valid."""
    results = []

    try:
        from config.config import CONFIG

        checks = [
            ("max_daily_drawdown_pct",  0.001, 0.5,  "max daily drawdown"),
            ("max_total_loss_pct",       0.01,  1.0,  "max total loss"),
            ("max_open_positions",        1,    100,  "max open positions"),
            ("max_position_pct",         0.01,  1.0,  "max position % of capital"),
            ("confidence_threshold",     0.01,  1.0,  "confidence threshold"),
            ("capital",                  1.0,   1e9,  "trading capital"),
        ]

        for key, lo, hi, label in checks:
            val = CONFIG.get(key)
            if val is None:
                results.append(_fail(f"Safety: {label}",
                    f"CONFIG['{key}'] is not set"))
                continue
            try:
                v = float(val)
                if lo <= v <= hi:
                    results.append(_pass(f"Safety: {label}", f"{v}"))
                else:
                    results.append(_fail(f"Safety: {label}",
                        f"{v} is outside acceptable range [{lo}, {hi}]"))
            except (TypeError, ValueError):
                results.append(_fail(f"Safety: {label}",
                    f"Cannot parse CONFIG['{key}'] = {val!r}"))

    except Exception as e:
        results.append(_fail("Safety config",
            f"Could not load CONFIG: {type(e).__name__}: {e}"))

    return results


def check_shadow_safety() -> List[CheckResult]:
    """Verify SHADOW mode cannot submit real orders."""
    results = []
    try:
        from core.shadow_mode import ShadowModeTracker, TradingMode
        tracker = ShadowModeTracker.__new__(ShadowModeTracker)
        tracker._mode   = TradingMode.SHADOW
        tracker._active = True
        if not tracker.should_submit_order():
            results.append(_pass("Shadow mode: order submission blocked",
                "should_submit_order() returns False in SHADOW mode"))
        else:
            results.append(_fail("Shadow mode: order submission blocked",
                "CRITICAL: should_submit_order() returns True in SHADOW mode"))
    except Exception as e:
        results.append(_warn("Shadow mode: order submission blocked",
            f"Could not verify: {e}"))
    return results


def check_paper_safety() -> List[CheckResult]:
    """Verify PAPER mode is wired to the paper broker."""
    results = []
    base_url = _env("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets"
    is_paper = "paper-api.alpaca.markets" in base_url
    if is_paper:
        results.append(_pass("Paper mode: paper broker URL confirmed",
            f"ALPACA_BASE_URL = {base_url}"))
    else:
        results.append(_fail("Paper mode: paper broker URL check",
            f"Expected paper-api.alpaca.markets, got: {base_url}"))
    return results


def check_config_validation() -> List[CheckResult]:
    """Run the ConfigValidator over the loaded CONFIG."""
    results = []
    try:
        from config.config import CONFIG
        from config.validator import validate_config
        is_valid = validate_config(CONFIG, verbose=False)
        if is_valid:
            results.append(_pass("Config validation", "all checks passed"))
        else:
            results.append(_warn("Config validation",
                "some checks failed — run 'python scripts/validate.py' for details"))
    except Exception as e:
        results.append(_warn("Config validation",
            f"Could not run: {type(e).__name__}: {e}"))
    return results


def check_broker_reconciliation(mode: str) -> List[CheckResult]:
    """Startup broker reconciliation (read-only)."""
    results = []

    if mode not in ("PAPER", "LIVE"):
        results.append(_pass("Broker reconciliation",
            f"SKIPPED (mode={mode})"))
        return results

    try:
        import alpaca_trade_api as tradeapi
        key    = _env("ALPACA_API_KEY")
        secret = _env("ALPACA_API_SECRET")
        base   = _env("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets"

        if not key or not secret:
            results.append(_warn("Broker reconciliation",
                "API credentials not set — skipping"))
            return results

        api = tradeapi.REST(key, secret, base_url=base, api_version="v2")

        # Read broker positions
        broker_positions = {p.symbol: float(p.qty) for p in api.list_positions()}
        broker_orders    = [o for o in api.list_orders(status="open")]

        # Read local positions
        db_path = _env("TRADE_MEMORY_DB") or "data/trade_memory.sqlite"
        local_open = {}
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT symbol, size FROM positions WHERE status='OPEN'"
            ).fetchall()
            conn.close()
            local_open = {r[0]: float(r[1]) for r in rows}
        except Exception:
            pass

        # Check for discrepancies
        broker_syms = set(broker_positions)
        local_syms  = set(local_open)
        only_broker = broker_syms - local_syms
        only_local  = local_syms - broker_syms

        if not only_broker and not only_local:
            results.append(_pass("Broker reconciliation",
                f"positions aligned (broker={len(broker_positions)}, "
                f"local={len(local_open)}, open orders={len(broker_orders)})"))
        else:
            detail = []
            if only_broker:
                detail.append(f"broker-only positions: {only_broker}")
            if only_local:
                detail.append(f"local-only positions: {only_local}")
            results.append(_warn("Broker reconciliation",
                "; ".join(detail) +
                " — review manually before trading"))

    except Exception as e:
        results.append(_warn("Broker reconciliation",
            f"Could not complete: {type(e).__name__}: {str(e)[:120]}"))

    return results


def check_stale_data() -> List[CheckResult]:
    """Warn if the most recent candle in the DB is too old."""
    results = []
    # This is a lightweight check — warn if market data hasn't been updated recently.
    # We just report the warning; stale data is caught at runtime by DataQualityMonitor.
    try:
        db_path = _env("TRADE_MEMORY_DB") or "data/trade_memory.sqlite"
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT MAX(exit_time) FROM positions"
            ).fetchone()
            last = row[0] if row and row[0] else None
        except Exception:
            last = None
        conn.close()

        if last:
            from datetime import datetime, timezone
            try:
                dt = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                if age_h > 48:
                    results.append(_warn("Stale data check",
                        f"Last recorded position exit was {age_h:.0f}h ago. "
                        f"Bot will fetch fresh data on startup."))
                else:
                    results.append(_pass("Stale data check",
                        f"Last position exit {age_h:.1f}h ago"))
            except Exception:
                results.append(_pass("Stale data check", "No recent positions recorded"))
        else:
            results.append(_pass("Stale data check",
                "No position history — fresh install"))

    except Exception as e:
        results.append(_warn("Stale data check",
            f"Could not check: {e}"))

    return results


def check_smoke_pipeline(mode: str) -> List[CheckResult]:
    """
    Lightweight smoke test of the decision pipeline (--smoke flag).
    No real orders. No real capital at risk.
    """
    results = []
    results.append(_pass("Smoke: starting pipeline checks", ""))

    # Config load
    try:
        from config.config import CONFIG, get_all_symbols
        syms = get_all_symbols()
        results.append(_pass("Smoke: config load",
            f"symbols={len(syms)} universe"))
    except Exception as e:
        results.append(_fail("Smoke: config load", str(e)))

    # TradeMemory
    try:
        from core.trade_memory import TradeMemory
        tm = TradeMemory()
        results.append(_pass("Smoke: TradeMemory init"))
    except Exception as e:
        results.append(_warn("Smoke: TradeMemory init", str(e)))

    # FeatureStore
    try:
        from core.feature_store import FeatureStore
        fs = FeatureStore()
        results.append(_pass("Smoke: FeatureStore init"))
    except Exception as e:
        results.append(_warn("Smoke: FeatureStore init", str(e)))

    # Validation engine
    try:
        from validation.engine import ValidationEngine
        ve = ValidationEngine()
        result = ve.evaluate([], label="smoke")
        results.append(_pass("Smoke: ValidationEngine", "evaluate([]) OK"))
    except Exception as e:
        results.append(_warn("Smoke: ValidationEngine", str(e)))

    # Shadow mode tracker
    try:
        from core.shadow_mode import ShadowModeTracker, TradingMode
        tracker = ShadowModeTracker()
        tracker.set_mode(TradingMode.SHADOW)
        assert not tracker.should_submit_order()
        results.append(_pass("Smoke: ShadowModeTracker",
            "SHADOW → should_submit_order()=False verified"))
    except Exception as e:
        results.append(_fail("Smoke: ShadowModeTracker", str(e)))

    # DataQualityMonitor
    try:
        import pandas as pd, numpy as np
        from core.data_quality import DataQualityMonitor
        dates = pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC")
        close = 50000 + np.arange(20, dtype=float)
        df = pd.DataFrame({
            "open": close, "high": close + 10, "low": close - 10,
            "close": close, "volume": np.ones(20) * 100,
        }, index=dates)
        dqm = DataQualityMonitor()
        res = dqm.check(df, symbol="BTC", timeframe="15m")
        if res.passed:
            results.append(_pass("Smoke: DataQualityMonitor", "clean OHLCV passes"))
        else:
            results.append(_warn("Smoke: DataQualityMonitor",
                f"clean data flagged: {res.issues}"))
    except Exception as e:
        results.append(_warn("Smoke: DataQualityMonitor", str(e)))

    return results


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def run_preflight(requested_mode: Optional[str] = None, smoke: bool = False) -> int:
    """
    Run all preflight checks.
    Returns 0 if ready, 1 if NO-GO.
    """
    _WIDE = 70

    print()
    print("=" * _WIDE)
    print("  PREFLIGHT STARTUP CHECK")
    print("  Trading Bot — Validation / Hardening / Paper-Trading")
    print("=" * _WIDE)

    all_results: List[CheckResult] = []

    # ── 1. Python version ───────────────────────────────────────────────────
    print("\n[1] Python version")
    r = check_python_version()
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 2. Execution mode ───────────────────────────────────────────────────
    print("\n[2] Execution mode")
    r, resolved_mode = check_execution_mode(requested_mode)
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 3. Dependencies ─────────────────────────────────────────────────────
    print("\n[3] Dependencies")
    r = check_dependencies()
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 4. Environment variables ────────────────────────────────────────────
    print("\n[4] Environment variables")
    r = check_env_vars(resolved_mode)
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 5. Broker / mode alignment ──────────────────────────────────────────
    print("\n[5] Broker / mode alignment")
    r = check_broker_mode(resolved_mode)
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 6. Filesystem ───────────────────────────────────────────────────────
    print("\n[6] Filesystem / directories")
    r = check_filesystem(resolved_mode)
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 7. Logging ──────────────────────────────────────────────────────────
    print("\n[7] Logging")
    r = check_logging(resolved_mode)
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 8. Database ─────────────────────────────────────────────────────────
    print("\n[8] Database initialization")
    r = check_database(resolved_mode)
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 9. Safety config ────────────────────────────────────────────────────
    print("\n[9] Safety configuration")
    r = check_safety_config()
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 10. Mode-specific safety ────────────────────────────────────────────
    print("\n[10] Mode-specific safety")
    if resolved_mode == "SHADOW":
        r = check_shadow_safety()
    elif resolved_mode in ("PAPER", "LIVE"):
        r = check_paper_safety()
        if resolved_mode == "PAPER":
            r += check_shadow_safety()   # also verify shadow guard is present
    else:
        r = [_pass("Mode-specific safety", f"BACKTEST — no broker interaction")]
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 11. Config validation ───────────────────────────────────────────────
    print("\n[11] Configuration validation")
    r = check_config_validation()
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 12. Kronos ──────────────────────────────────────────────────────────
    print("\n[12] Kronos availability")
    r = check_kronos(resolved_mode)
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 13. API connectivity ────────────────────────────────────────────────
    print("\n[13] API connectivity")
    r = check_api_connectivity(resolved_mode)
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 14. Broker reconciliation ───────────────────────────────────────────
    print("\n[14] Broker reconciliation")
    r = check_broker_reconciliation(resolved_mode)
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 15. Stale data ──────────────────────────────────────────────────────
    print("\n[15] Stale data check")
    r = check_stale_data()
    all_results.extend(r)
    for cr in r: print(cr)

    # ── 16. Smoke test (optional) ───────────────────────────────────────────
    if smoke:
        print("\n[16] Smoke test (--smoke)")
        r = check_smoke_pipeline(resolved_mode)
        all_results.extend(r)
        for cr in r: print(cr)

    # ── Summary ─────────────────────────────────────────────────────────────
    failures = [cr for cr in all_results if cr.is_fail]
    warnings = [cr for cr in all_results if cr.status == "WARN"]
    passed   = [cr for cr in all_results if cr.status == "PASS"]

    print()
    print("=" * _WIDE)
    print(f"  SUMMARY: {len(passed)} passed, {len(warnings)} warnings, {len(failures)} failures")
    print("=" * _WIDE)

    if failures:
        print()
        print("  FAILURES — must be resolved before starting:")
        for f in failures:
            print(f"    ✗ {f.name}")
            if f.detail:
                print(f"      {f.detail}")
        print()
        print("=" * _WIDE)
        print("  NO-GO")
        print("=" * _WIDE)
        return 1

    # All checks passed (warnings allowed)
    ready_label = {
        "PAPER":    "PAPER TRADING READY",
        "SHADOW":   "SHADOW TRADING READY",
        "BACKTEST": "BACKTEST READY",
        "LIVE":     "LIVE TRADING READY",
    }.get(resolved_mode, "READY")

    print()
    if warnings:
        print(f"  {len(warnings)} warning(s) — review above before trading")
    print()
    print("=" * _WIDE)
    print(f"  {ready_label}")
    print("=" * _WIDE)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Startup preflight check for the trading bot.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["BACKTEST", "PAPER", "SHADOW", "LIVE"],
        default=None,
        help="Override execution mode (default: read EXECUTION_MODE env var)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run lightweight smoke test of the decision pipeline",
    )
    args = parser.parse_args()

    sys.exit(run_preflight(requested_mode=args.mode, smoke=args.smoke))


if __name__ == "__main__":
    main()
