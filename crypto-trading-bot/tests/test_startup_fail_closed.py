"""Startup fail-closed regression tests (spec §14-23): core accounting/DB
failures must prevent normal startup via the ACTUAL production initialization
method; optional-source failures stay nonfatal.
"""
import sqlite3
import uuid

import pytest

from core.trade_history import (
    CoreAccountingInitializationError,
    TradeMemorySchemaError,
)
from ultimate_bot_v3_llm import LLMTradingBot


def bot_skeleton():
    """Real class, no heavyweight __init__ — the method under test IS the
    production _initialize_core_accounting."""
    return LLMTradingBot.__new__(LLMTradingBot)


def healthy_db(tmp_path):
    db = str(tmp_path / f"ok_{uuid.uuid4().hex}.sqlite")
    from core.trade_attribution import TradeAttributionStore
    from core.trade_memory import TradeMemory
    TradeMemory(db)
    TradeAttributionStore(db)
    return db


# ── Baseline: healthy DB starts, health becomes HEALTHY ───────────────────────


class TestHealthyStartup:
    def test_healthy_db_initializes(self, tmp_path):
        bot = bot_skeleton()
        bot._initialize_core_accounting(healthy_db(tmp_path))
        assert bot._db_health == "HEALTHY"
        assert bot.paper_evidence_tracker is not None


# ── §14: sqlite3.DatabaseError — the key regression test ──────────────────────


class TestDatabaseError:
    def test_corrupt_database_fails_closed(self, tmp_path):
        db = str(tmp_path / "corrupt.sqlite")
        with open(db, "wb") as f:                       # malformed disk image
            f.write(b"this is not a sqlite database at all" * 100)
        bot = bot_skeleton()
        with pytest.raises(RuntimeError) as exc_info:
            bot._initialize_core_accounting(db)
        assert bot._db_health == "CRITICAL"
        # original exception preserved via chaining (spec §11)
        assert isinstance(exc_info.value.__cause__, sqlite3.DatabaseError)

    def test_injected_database_error_fails_closed(self, tmp_path, monkeypatch):
        import core.trade_history as th
        monkeypatch.setattr(
            th, "migrate_legacy_position_size",
            lambda db: (_ for _ in ()).throw(
                sqlite3.DatabaseError("database disk image is malformed")))
        bot = bot_skeleton()
        with pytest.raises(RuntimeError):
            bot._initialize_core_accounting(healthy_db(tmp_path))
        assert bot._db_health == "CRITICAL"


# ── §15: IntegrityError ───────────────────────────────────────────────────────


class TestIntegrityError:
    def test_integrity_error_fails_closed(self, tmp_path, monkeypatch):
        import core.trade_history as th
        monkeypatch.setattr(
            th, "data_integrity_scan",
            lambda db: (_ for _ in ()).throw(
                sqlite3.IntegrityError("constraint violated")))
        bot = bot_skeleton()
        with pytest.raises(RuntimeError) as exc_info:
            bot._initialize_core_accounting(healthy_db(tmp_path))
        assert bot._db_health == "CRITICAL"
        assert isinstance(exc_info.value.__cause__, sqlite3.IntegrityError)


# ── §16: required migration failure ───────────────────────────────────────────


class TestMigrationFailure:
    def test_failed_migration_fails_closed(self, tmp_path, monkeypatch):
        import core.trade_history as th
        monkeypatch.setattr(
            th, "migrate_legacy_position_size",
            lambda db: (_ for _ in ()).throw(
                sqlite3.OperationalError("migration write failed")))
        bot = bot_skeleton()
        with pytest.raises(RuntimeError):               # OperationalError ⊂ DatabaseError
            bot._initialize_core_accounting(healthy_db(tmp_path))
        assert bot._db_health == "CRITICAL"

    def test_unexpected_migration_exception_propagates(self, tmp_path,
                                                       monkeypatch):
        """No generic soft-warning branch exists: unknown errors also stop
        startup (fail closed by default), just unclassified."""
        import core.trade_history as th
        monkeypatch.setattr(
            th, "migrate_legacy_position_size",
            lambda db: (_ for _ in ()).throw(ValueError("unexpected")))
        bot = bot_skeleton()
        with pytest.raises(ValueError):
            bot._initialize_core_accounting(healthy_db(tmp_path))


# ── §17: reconciliation database failure ──────────────────────────────────────


class TestReconciliationFailure:
    def test_reconcile_database_error_fails_closed(self, tmp_path, monkeypatch):
        from core.paper_evidence import PaperEvidenceTracker
        monkeypatch.setattr(
            PaperEvidenceTracker, "reconcile",
            lambda self: (_ for _ in ()).throw(
                sqlite3.DatabaseError("evidence table unreadable")))
        bot = bot_skeleton()
        with pytest.raises(RuntimeError):
            bot._initialize_core_accounting(healthy_db(tmp_path))
        assert bot._db_health == "CRITICAL"

    def test_tracker_constructor_failure_fails_closed_in_paper_mode(
            self, tmp_path, monkeypatch):
        from core import paper_evidence
        monkeypatch.setattr(
            paper_evidence.PaperEvidenceTracker, "__init__",
            lambda self, db: (_ for _ in ()).throw(
                sqlite3.DatabaseError("cannot open evidence store")))
        bot = bot_skeleton()
        with pytest.raises(RuntimeError):
            bot._initialize_core_accounting(healthy_db(tmp_path))
        assert bot._db_health == "CRITICAL"


# ── §18: malformed required schema via the real init path ─────────────────────


class TestSchemaError:
    def test_missing_required_column_fails_closed(self, tmp_path):
        db = str(tmp_path / "broken_schema.sqlite")
        with sqlite3.connect(db) as conn:               # trade_memory w/o size_dollars
            conn.execute("CREATE TABLE trade_memory (id INTEGER PRIMARY KEY, "
                         "symbol TEXT)")
        bot = bot_skeleton()
        with pytest.raises(RuntimeError) as exc_info:
            bot._initialize_core_accounting(db)
        assert bot._db_health == "CRITICAL"
        assert isinstance(exc_info.value.__cause__,
                          CoreAccountingInitializationError)

    def test_misleading_soft_message_removed(self):
        import inspect

        import ultimate_bot_v3_llm as bot_mod
        assert "failed soft" not in inspect.getsource(
            bot_mod.LLMTradingBot._initialize_core_accounting)
        assert "Schema health check failed soft" not in inspect.getsource(bot_mod)


# ── §19: optional source failure stays nonfatal ───────────────────────────────


class TestOptionalSourceIsolation:
    def test_optional_source_outage_does_not_touch_startup(self, tmp_path):
        """Optional intelligence is not initialized in the core path at all;
        a dead USAspending source degrades the registry, not accounting."""
        from data.external_intelligence import (
            AVAILABLE,
            ExternalSourceRegistry,
            RawEventStore,
        )
        from data.external_sources import USASpendingSource

        class Exploding(USASpendingSource):
            source_name = "exploding_optional"

            def availability(self):
                return AVAILABLE

            def fetch_since(self, since):
                raise ConnectionError("provider down")

        db = healthy_db(tmp_path)
        registry = ExternalSourceRegistry(event_store=RawEventStore(db))
        registry.register(Exploding())
        registry.refresh_all("2026-01-01")              # must not raise
        bot = bot_skeleton()
        bot._initialize_core_accounting(db)             # core unaffected
        assert bot._db_health == "HEALTHY"


# ── §20-21: order gate + safe close path ──────────────────────────────────────


class TestOrderGate:
    def test_no_orders_when_critical(self):
        bot = bot_skeleton()
        bot._db_health = "CRITICAL"
        # real production entry point — returns None before touching anything
        result = bot.execute_trade("BTC-USD", {"signal": "BUY"})
        assert result is None

    def test_no_orders_when_degraded(self):
        bot = bot_skeleton()
        bot._db_health = "DEGRADED_READ_ONLY"
        assert bot.execute_trade("BTC-USD", {"signal": "BUY"}) is None

    def test_close_path_not_gated(self):
        """Risk reduction stays possible: _close_position may SET the health
        state on write failure, but never gates/blocks on it."""
        import inspect
        src = inspect.getsource(LLMTradingBot._close_position)
        assert "!= 'HEALTHY'" not in src               # no entry-style gate
        assert "BLOCKED" not in src


# ── §24: runtime recovery untouched ───────────────────────────────────────────


class TestRecoveryIntact:
    def test_periodic_recovery_still_present(self):
        import inspect
        src = inspect.getsource(LLMTradingBot.run_trading_cycle)
        assert "recovered" in src and "_last_core_health_check" in src
