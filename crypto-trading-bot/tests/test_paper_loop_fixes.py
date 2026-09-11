"""Tests for the six paper-loop hardening fixes: real bar intervals, full-
fidelity resolution, MFE/MAE capture, signal-decay vs holding separation,
fail-closed accounting, and loud reconciliation.
"""
import sqlite3
import uuid

import pytest

from core.paper_evidence import PaperEvidenceTracker
from core.trade_history import TradeMemorySchemaError
from tests.test_paper_loop import enriched_candidate


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / f"fix_{uuid.uuid4().hex}.sqlite")


class TestBarInterval:
    def test_bot_helper_maps_timeframes(self):
        from ultimate_bot_v3_llm import LLMTradingBot
        c = enriched_candidate("tf1")
        c.features["timeframe"] = "5m"
        assert LLMTradingBot._bar_interval_seconds(
            type("B", (), {"alpha_library": None})(), c) == 300.0
        c.features["timeframe"] = "1h"
        assert LLMTradingBot._bar_interval_seconds(
            type("B", (), {"alpha_library": None})(), c) == 3600.0

    def test_unknown_timeframe_defaults_daily(self):
        from ultimate_bot_v3_llm import LLMTradingBot

        class Lib:
            def get(self, alpha_id):
                return {}
        c = enriched_candidate("tf2")
        bot = type("B", (), {"alpha_library": Lib()})()
        assert LLMTradingBot._bar_interval_seconds(bot, c) == 86_400.0


class TestFullFidelityResolution:
    def test_resolution_stores_components_and_extremes(self, db):
        tracker = PaperEvidenceTracker(db)
        tracker.record_prediction(enriched_candidate("f1"))
        tracker.resolve("f1", realized_net_return=0.01, realized_net_pnl=100,
                        realized_execution_cost_bps=9.5,
                        realized_holding_hours=26.0,
                        realized_mfe_pct=2.1, realized_mae_pct=-0.8,
                        realized_fee_bps=4.0, realized_slippage_bps=5.5,
                        close_reason="take_profit")
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT realized_mfe_pct, realized_mae_pct, realized_fee_bps, "
                "realized_slippage_bps, realized_holding_hours "
                "FROM paper_evidence WHERE candidate_id='f1'").fetchone()
        assert row == (2.1, -0.8, 4.0, 5.5, 26.0)


class TestSignalDecayVsHolding:
    def test_holding_time_is_not_decay_calibration(self, db):
        tracker = PaperEvidenceTracker(db)
        for i in range(12):
            tracker.record_prediction(
                enriched_candidate(f"d{i}", half_life_bars=1.0),
                bar_interval_seconds=86_400.0)
            tracker.resolve(f"d{i}", realized_net_return=0.01,
                            realized_net_pnl=100, realized_holding_hours=24.0)
        # holding hours exist but NO measured decay → honest unavailability
        cal = tracker.half_life_calibration()
        assert cal["verdict"] == "REALIZED_HALF_LIFE_UNAVAILABLE"
        diag = tracker.holding_vs_half_life()
        assert diag["n"] == 12
        assert "not signal decay" in diag["note"]

    def test_measured_decay_enables_calibration(self, db):
        tracker = PaperEvidenceTracker(db)
        for i in range(12):
            tracker.record_prediction(
                enriched_candidate(f"m{i}", half_life_bars=1.0),
                bar_interval_seconds=86_400.0)
            tracker.resolve(f"m{i}", realized_net_return=0.01,
                            realized_net_pnl=100)
            tracker.record_realized_decay(f"m{i}", 43_200.0)   # measured: 12h
        cal = tracker.half_life_calibration()
        assert cal["ratio"] == pytest.approx(0.5)   # decayed 2x faster than predicted


class TestFailClosedAccounting:
    def test_reconcile_raises_on_schema_mismatch(self, db):
        tracker = PaperEvidenceTracker(db)
        with sqlite3.connect(db) as conn:      # trade_memory WITHOUT size_dollars
            conn.execute("CREATE TABLE trade_memory (id INTEGER PRIMARY KEY, "
                         "position_id INTEGER, exit_time TEXT)")
            conn.execute("CREATE TABLE trade_attribution "
                         "(trade_memory_id INTEGER, alpha_id TEXT)")
        with pytest.raises(TradeMemorySchemaError):
            tracker.reconcile()                # loud, never silently clean

    def test_reconcile_ok_when_tables_missing(self, db):
        # no trade_memory at all = legitimately no history yet
        recon = PaperEvidenceTracker(db).reconcile()
        assert recon["resolved"] == 0 and recon["flags"] == []

    def test_entry_is_fail_closed_in_bot_source(self):
        import inspect

        from ultimate_bot_v3_llm import LLMTradingBot
        src = inspect.getsource(LLMTradingBot.run_trading_cycle)
        # no position may open without its prediction record
        assert "CRITICAL_ACCOUNTING_FAILURE" in src
        assert src.index("record_prediction") < src.index(
            "pos_id = self.execute_trade")
        # both failure branches skip the trade
        assert src.count("continue", src.index("for d in accepted:"),
                         src.index("pos_id = self.execute_trade")) >= 2


class TestSummaryReport:
    def test_insufficient_evidence_is_honest(self, db):
        report = PaperEvidenceTracker(db).summary_report(30)
        assert "INSUFFICIENT_FORWARD_EVIDENCE" in report

    def test_report_shows_predicted_vs_realized(self, db):
        tracker = PaperEvidenceTracker(db)
        for i in range(30):
            c = enriched_candidate(f"s{i}", alpha_id="temporal_x")
            c.expected_net_return = 0.0021
            tracker.record_prediction(c, family="TEMPORAL")
            tracker.resolve(f"s{i}", realized_net_return=0.0015,
                            realized_net_pnl=15.0,
                            realized_execution_cost_bps=7.1)
        report = tracker.summary_report(30)
        assert "Resolved trades:        30" in report
        assert "+21.0 bps" in report          # predicted mean
        assert "+15.0 bps" in report          # realized mean
        assert "TEMPORAL:" in report
        assert "Actual execution cost:    7.1 bps" in report


class TestProductionDecayMeasurement:
    def _resolved_evidence(self, db, cid="dm1", direction="long"):
        tracker = PaperEvidenceTracker(db)
        c = enriched_candidate(cid)
        c.direction = direction
        tracker.record_prediction(c, family="TEMPORAL",
                                  bar_interval_seconds=86_400.0)
        tracker.resolve(cid, realized_net_return=0.01, realized_net_pnl=100)
        return tracker

    @staticmethod
    def _df(prices):
        # bars anchored at the candidate's signal timestamp (2026-09-01 UTC)
        import pandas as pd
        idx = pd.date_range("2026-09-01", periods=len(prices), freq="1D",
                            tz="UTC")
        return pd.DataFrame({"close": prices}, index=idx)

    def test_decay_measured_from_price_path(self, db):
        tracker = self._resolved_evidence(db)
        # signal peaks +2% at bar 2, decays below +1% by bar 4
        df = self._df([100.0, 101.5, 102.0, 101.4, 100.8, 100.5, 100.2])
        result = tracker.measure_realized_decay(lambda s: df)
        assert result["measured"] == 1
        cal_rows = tracker.half_life_calibration(min_samples=1)
        assert cal_rows["realized_half_life_seconds"] == pytest.approx(
            4 * 86_400.0)                     # decayed to ≤half-peak at bar 4

    def test_insufficient_history_left_unset(self, db):
        tracker = self._resolved_evidence(db, cid="dm2")
        df = self._df([100.0, 100.5])
        result = tracker.measure_realized_decay(lambda s: df)
        assert result["measured"] == 0        # honest: nothing invented
        assert tracker.half_life_calibration()["verdict"] == \
            "REALIZED_HALF_LIFE_UNAVAILABLE"

    def test_undedecayed_signal_left_unset(self, db):
        tracker = self._resolved_evidence(db, cid="dm3")
        df = self._df([100.0 * (1.01 ** i) for i in range(10)])  # still climbing
        assert tracker.measure_realized_decay(lambda s: df)["measured"] == 0


class TestRuntimeFailClosed:
    def test_execute_trade_gated_on_critical_health(self):
        import inspect

        from ultimate_bot_v3_llm import LLMTradingBot
        src = inspect.getsource(LLMTradingBot.execute_trade)
        assert "_db_health" in src
        # gate sits before the actual broker call, not just the docstring
        assert src.index("_db_health") < src.index("self.broker.submit_and_wait")

    def test_runtime_schema_error_marks_critical(self):
        import inspect

        from ultimate_bot_v3_llm import LLMTradingBot
        src = inspect.getsource(LLMTradingBot.run_alpha_pipeline)
        assert "TradeMemorySchemaError" in src
        assert "'CRITICAL'" in src
        assert "return []" in src              # no candidates on broken accounting


class TestMFEMAEProductionCapture:
    def test_manage_loop_persists_excursions(self):
        import inspect

        from ultimate_bot_v3_llm import LLMTradingBot
        src = inspect.getsource(LLMTradingBot.manage_open_positions)
        assert "mfe_pct=" in src and "mae_pct=" in src
        assert "update_position" in src
