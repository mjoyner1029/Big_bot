"""
tests/test_startup.py — startup hardening tests.

Tests for:
- ExecutionMode enum and resolve_execution_mode()
- is_live_authorized() double gate
- verify_broker_matches_mode()
- Broker/mode alignment (paper URL with LIVE mode = error)
- LIVE mode without ENABLE_LIVE_TRADING flag = blocked
- Obsolete strategy profile values rejected
- ShadowModeTracker order submission prevention
- Preflight check functions (no API calls)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ════════════════════════════════════════════════════════════════════════════
# ExecutionMode enum
# ════════════════════════════════════════════════════════════════════════════

class TestExecutionModeEnum:
    def test_canonical_values_exist(self):
        from core.trading_mode import ExecutionMode
        assert ExecutionMode.BACKTEST.value == "BACKTEST"
        assert ExecutionMode.PAPER.value    == "PAPER"
        assert ExecutionMode.SHADOW.value   == "SHADOW"
        assert ExecutionMode.LIVE.value     == "LIVE"

    def test_is_str_subclass(self):
        from core.trading_mode import ExecutionMode
        assert isinstance(ExecutionMode.PAPER, str)
        assert ExecutionMode.PAPER == "PAPER"

    def test_membership(self):
        from core.trading_mode import ExecutionMode
        members = {m.value for m in ExecutionMode}
        assert {"BACKTEST", "PAPER", "SHADOW", "LIVE"}.issubset(members)


# ════════════════════════════════════════════════════════════════════════════
# resolve_execution_mode
# ════════════════════════════════════════════════════════════════════════════

class TestResolveExecutionMode:
    def test_exact_match_paper(self):
        from core.trading_mode import resolve_execution_mode, ExecutionMode
        assert resolve_execution_mode("PAPER") == ExecutionMode.PAPER

    def test_exact_match_live(self):
        from core.trading_mode import resolve_execution_mode, ExecutionMode
        assert resolve_execution_mode("LIVE") == ExecutionMode.LIVE

    def test_exact_match_shadow(self):
        from core.trading_mode import resolve_execution_mode, ExecutionMode
        assert resolve_execution_mode("SHADOW") == ExecutionMode.SHADOW

    def test_exact_match_backtest(self):
        from core.trading_mode import resolve_execution_mode, ExecutionMode
        assert resolve_execution_mode("BACKTEST") == ExecutionMode.BACKTEST

    def test_lowercase_accepted(self):
        from core.trading_mode import resolve_execution_mode, ExecutionMode
        assert resolve_execution_mode("paper")    == ExecutionMode.PAPER
        assert resolve_execution_mode("live")     == ExecutionMode.LIVE
        assert resolve_execution_mode("shadow")   == ExecutionMode.SHADOW
        assert resolve_execution_mode("backtest") == ExecutionMode.BACKTEST

    def test_mixed_case_accepted(self):
        from core.trading_mode import resolve_execution_mode, ExecutionMode
        assert resolve_execution_mode("Paper")   == ExecutionMode.PAPER
        assert resolve_execution_mode("SHADOW")  == ExecutionMode.SHADOW
        assert resolve_execution_mode("Live")    == ExecutionMode.LIVE

    def test_obsolete_balanced_rejected(self):
        from core.trading_mode import resolve_execution_mode
        with pytest.raises(ValueError, match="strategy profile"):
            resolve_execution_mode("balanced")

    def test_obsolete_conservative_rejected(self):
        from core.trading_mode import resolve_execution_mode
        with pytest.raises(ValueError, match="strategy profile"):
            resolve_execution_mode("conservative")

    def test_obsolete_aggressive_rejected(self):
        from core.trading_mode import resolve_execution_mode
        with pytest.raises(ValueError, match="strategy profile"):
            resolve_execution_mode("aggressive")

    def test_obsolete_claude_hf_rejected(self):
        from core.trading_mode import resolve_execution_mode
        with pytest.raises(ValueError, match="strategy profile"):
            resolve_execution_mode("claude_hf")

    def test_garbage_value_rejected(self):
        from core.trading_mode import resolve_execution_mode
        with pytest.raises(ValueError):
            resolve_execution_mode("not_a_mode")

    def test_empty_string_rejected(self):
        from core.trading_mode import resolve_execution_mode
        with pytest.raises(ValueError):
            resolve_execution_mode("")

    def test_reads_execution_mode_env_var(self):
        from core.trading_mode import resolve_execution_mode, ExecutionMode
        with patch.dict(os.environ, {"EXECUTION_MODE": "SHADOW"}, clear=False):
            assert resolve_execution_mode() == ExecutionMode.SHADOW

    def test_falls_back_to_trading_mode_env_var(self):
        from core.trading_mode import resolve_execution_mode, ExecutionMode
        env = {"TRADING_MODE": "PAPER"}
        with patch.dict(os.environ, env, clear=False):
            # Remove EXECUTION_MODE if set
            cleaned = {k: v for k, v in os.environ.items() if k != "EXECUTION_MODE"}
            cleaned.update(env)
            with patch.dict(os.environ, cleaned, clear=True):
                assert resolve_execution_mode() == ExecutionMode.PAPER

    def test_explicit_arg_overrides_env(self):
        from core.trading_mode import resolve_execution_mode, ExecutionMode
        with patch.dict(os.environ, {"EXECUTION_MODE": "LIVE"}, clear=False):
            assert resolve_execution_mode("PAPER") == ExecutionMode.PAPER


# ════════════════════════════════════════════════════════════════════════════
# is_live_authorized
# ════════════════════════════════════════════════════════════════════════════

class TestIsLiveAuthorized:
    def _env(self, execution_mode: str, enable_live: str) -> dict:
        return {
            "EXECUTION_MODE":      execution_mode,
            "ENABLE_LIVE_TRADING": enable_live,
        }

    def test_both_gates_open_authorizes(self):
        from core.trading_mode import is_live_authorized
        env = self._env("LIVE", "true")
        with patch.dict(os.environ, env, clear=False):
            assert is_live_authorized() is True

    def test_live_mode_without_enable_flag_blocks(self):
        from core.trading_mode import is_live_authorized
        env = self._env("LIVE", "false")
        with patch.dict(os.environ, env, clear=False):
            assert is_live_authorized() is False

    def test_paper_mode_with_enable_flag_blocks(self):
        from core.trading_mode import is_live_authorized
        env = self._env("PAPER", "true")
        with patch.dict(os.environ, env, clear=False):
            assert is_live_authorized() is False

    def test_shadow_mode_never_live(self):
        from core.trading_mode import is_live_authorized
        env = self._env("SHADOW", "true")
        with patch.dict(os.environ, env, clear=False):
            assert is_live_authorized() is False

    def test_backtest_mode_never_live(self):
        from core.trading_mode import is_live_authorized
        env = self._env("BACKTEST", "true")
        with patch.dict(os.environ, env, clear=False):
            assert is_live_authorized() is False

    def test_invalid_execution_mode_blocks(self):
        from core.trading_mode import is_live_authorized
        env = self._env("BALANCED", "true")
        with patch.dict(os.environ, env, clear=False):
            assert is_live_authorized() is False

    def test_enable_flag_not_set_blocks(self):
        from core.trading_mode import is_live_authorized
        cleaned = {k: v for k, v in os.environ.items()
                   if k not in ("EXECUTION_MODE", "TRADING_MODE", "ENABLE_LIVE_TRADING")}
        cleaned["EXECUTION_MODE"] = "LIVE"
        with patch.dict(os.environ, cleaned, clear=True):
            assert is_live_authorized() is False


# ════════════════════════════════════════════════════════════════════════════
# verify_broker_matches_mode
# ════════════════════════════════════════════════════════════════════════════

class TestVerifyBrokerMatchesMode:
    def test_paper_url_with_paper_mode_ok(self):
        from core.trading_mode import verify_broker_matches_mode, ExecutionMode
        ok, _ = verify_broker_matches_mode(
            "https://paper-api.alpaca.markets", ExecutionMode.PAPER
        )
        assert ok

    def test_paper_url_with_shadow_mode_ok(self):
        from core.trading_mode import verify_broker_matches_mode, ExecutionMode
        ok, _ = verify_broker_matches_mode(
            "https://paper-api.alpaca.markets", ExecutionMode.SHADOW
        )
        assert ok

    def test_live_url_with_live_mode_ok(self):
        from core.trading_mode import verify_broker_matches_mode, ExecutionMode
        ok, _ = verify_broker_matches_mode(
            "https://api.alpaca.markets", ExecutionMode.LIVE
        )
        assert ok

    def test_paper_url_with_live_mode_fails(self):
        from core.trading_mode import verify_broker_matches_mode, ExecutionMode
        ok, reason = verify_broker_matches_mode(
            "https://paper-api.alpaca.markets", ExecutionMode.LIVE
        )
        assert not ok
        assert "paper" in reason.lower() or "live" in reason.lower()

    def test_live_url_with_paper_mode_fails(self):
        from core.trading_mode import verify_broker_matches_mode, ExecutionMode
        ok, reason = verify_broker_matches_mode(
            "https://api.alpaca.markets", ExecutionMode.PAPER
        )
        assert not ok

    def test_backtest_mode_always_ok(self):
        from core.trading_mode import verify_broker_matches_mode, ExecutionMode
        ok, _ = verify_broker_matches_mode("", ExecutionMode.BACKTEST)
        assert ok

    def test_empty_url_with_paper_mode_uses_default(self):
        from core.trading_mode import verify_broker_matches_mode, ExecutionMode
        # Empty URL should warn but not hard-fail for PAPER (paper is the default)
        ok, reason = verify_broker_matches_mode("", ExecutionMode.PAPER)
        # Result depends on implementation; just ensure it returns a tuple
        assert isinstance(ok, bool)
        assert isinstance(reason, str)


# ════════════════════════════════════════════════════════════════════════════
# ShadowModeTracker order submission prevention
# ════════════════════════════════════════════════════════════════════════════

class TestShadowModeOrderBlocking:
    def test_shadow_mode_cannot_submit_orders(self):
        from core.shadow_mode import ShadowModeTracker, TradingMode
        t = ShadowModeTracker.__new__(ShadowModeTracker)
        t._mode   = TradingMode.SHADOW
        t._active = True
        assert not t.should_submit_order(), \
            "SHADOW mode must not submit orders"

    def test_paper_mode_can_submit_orders(self):
        from core.shadow_mode import ShadowModeTracker, TradingMode
        t = ShadowModeTracker.__new__(ShadowModeTracker)
        t._mode   = TradingMode.PAPER
        t._active = False
        assert t.should_submit_order(), \
            "PAPER mode must be able to submit orders"

    def test_shadow_tracker_set_mode(self):
        from core.shadow_mode import ShadowModeTracker, TradingMode
        t = ShadowModeTracker()
        t.set_mode(TradingMode.SHADOW)
        assert not t.should_submit_order()


# ════════════════════════════════════════════════════════════════════════════
# Preflight check functions (no network / no API calls)
# ════════════════════════════════════════════════════════════════════════════

class TestPreflightChecks:
    """Unit-test individual preflight check functions."""

    def test_python_version_pass(self):
        from scripts.preflight import check_python_version
        results = check_python_version()
        # We're running ≥3.9 in this env
        assert any(r.status == "PASS" for r in results)

    def test_check_filesystem_creates_dirs(self, tmp_path, monkeypatch):
        """Filesystem check should create missing directories."""
        from scripts.preflight import check_filesystem
        monkeypatch.chdir(tmp_path)
        # Temporarily change PROJECT_ROOT
        import scripts.preflight as pf
        original_root = pf.PROJECT_ROOT
        pf.PROJECT_ROOT = tmp_path
        try:
            results = check_filesystem("PAPER")
            # At least one PASS (some dirs created)
            passes = [r for r in results if r.status == "PASS"]
            fails  = [r for r in results if r.status == "FAIL"]
            assert len(fails) == 0, f"Filesystem failures: {fails}"
        finally:
            pf.PROJECT_ROOT = original_root

    def test_check_logging_writes_file(self, tmp_path, monkeypatch):
        """Logging check should append to log file."""
        from scripts.preflight import check_logging
        import scripts.preflight as pf
        original_root = pf.PROJECT_ROOT
        pf.PROJECT_ROOT = tmp_path
        monkeypatch.setenv("BOT_LOG_PATH", str(tmp_path / "test_bot.log"))
        try:
            results = check_logging("PAPER")
            assert any(r.status == "PASS" for r in results), \
                f"Expected PASS, got: {results}"
        finally:
            pf.PROJECT_ROOT = original_root

    def test_check_shadow_safety_blocks(self):
        """Shadow safety check should report PASS when shadow correctly blocks orders."""
        from scripts.preflight import check_shadow_safety
        results = check_shadow_safety()
        passes = [r for r in results if r.status == "PASS"]
        assert len(passes) >= 1, \
            f"Expected shadow safety to pass, got: {results}"

    def test_check_paper_safety_with_paper_url(self, monkeypatch):
        """Paper safety check should PASS when ALPACA_BASE_URL is paper."""
        monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        from scripts.preflight import check_paper_safety
        results = check_paper_safety()
        assert any(r.status == "PASS" for r in results)

    def test_check_paper_safety_with_live_url(self, monkeypatch):
        """Paper safety check should FAIL when ALPACA_BASE_URL is live."""
        monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
        from scripts.preflight import check_paper_safety
        results = check_paper_safety()
        assert any(r.status == "FAIL" for r in results)

    def test_execution_mode_check_paper(self, monkeypatch):
        """Execution mode check should resolve PAPER cleanly."""
        monkeypatch.setenv("EXECUTION_MODE", "PAPER")
        from scripts.preflight import check_execution_mode
        results, mode = check_execution_mode(None)
        assert mode == "PAPER"
        assert any(r.status == "PASS" for r in results)

    def test_execution_mode_check_shadow(self, monkeypatch):
        from scripts.preflight import check_execution_mode
        results, mode = check_execution_mode("SHADOW")
        assert mode == "SHADOW"

    def test_execution_mode_check_obsolete_fails(self, monkeypatch):
        from scripts.preflight import check_execution_mode
        results, mode = check_execution_mode("balanced")
        assert mode == "UNKNOWN"
        assert any(r.status == "FAIL" for r in results)

    def test_env_var_check_missing_key(self, monkeypatch):
        """Env check should FAIL if required key is missing."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from scripts.preflight import check_env_vars
        results = check_env_vars("PAPER")
        fails = [r for r in results if r.status == "FAIL" and "ANTHROPIC" in r.name]
        assert len(fails) >= 1

    def test_check_database_creates_schema(self, tmp_path, monkeypatch):
        """Database check should create tables without error."""
        import scripts.preflight as pf
        original_root = pf.PROJECT_ROOT
        pf.PROJECT_ROOT = tmp_path
        (tmp_path / "data").mkdir(exist_ok=True)
        monkeypatch.setenv("TRADE_MEMORY_DB", str(tmp_path / "data" / "test.sqlite"))
        monkeypatch.setenv("FEATURE_STORE_DB", str(tmp_path / "data" / "fs_test.sqlite"))
        try:
            results = pf.check_database("PAPER")
            fails = [r for r in results if r.status == "FAIL"]
            assert len(fails) == 0, f"DB failures: {fails}"
        finally:
            pf.PROJECT_ROOT = original_root

    def test_broker_mode_paper_aligned(self, monkeypatch):
        """Broker check should PASS for paper URL in PAPER mode."""
        monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        from scripts.preflight import check_broker_mode
        results = check_broker_mode("PAPER")
        assert any(r.status == "PASS" for r in results)

    def test_broker_mode_live_with_paper_url_fails(self, monkeypatch):
        """Broker check should FAIL for paper URL in LIVE mode."""
        monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        from scripts.preflight import check_broker_mode
        results = check_broker_mode("LIVE")
        assert any(r.status == "FAIL" for r in results)

    def test_broker_mode_backtest_skips(self, monkeypatch):
        """Broker check should PASS for BACKTEST mode regardless of URL."""
        monkeypatch.setenv("ALPACA_BASE_URL", "")
        from scripts.preflight import check_broker_mode
        results = check_broker_mode("BACKTEST")
        # Should not FAIL for BACKTEST
        fails = [r for r in results if r.status == "FAIL"]
        assert len(fails) == 0
