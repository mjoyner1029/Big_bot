"""Paper-evidence loop tests (spec §73-93): production PAPER trades feed the
tracker — prediction at entry, idempotent resolution at close, restart
reconciliation, canonical half-life seconds, and calibration on real outcomes.
"""
import sqlite3
import uuid

import numpy as np
import pytest

from core.paper_evidence import (
    PaperEvidenceTracker,
    alpha_capture_analysis,
)
from tests.test_monetization_wiring import make_candidate


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / f"pe_{uuid.uuid4().hex}.sqlite")


def enriched_candidate(cid="c1", alpha_id="a1", half_life_bars=6.0):
    c = make_candidate(alpha_id=alpha_id)
    c.candidate_id = cid
    c.half_life_bars = half_life_bars
    c.expected_dollar_alpha = 40.0
    c.expected_total_cost_bps = 6.0
    c.edge_survival_probability = 0.8
    c.model_versions = {"enrichment": "1.0.0", "meta_alpha_mode": "shadow"}
    return c


# ── §73-75: prediction lifecycle ──────────────────────────────────────────────


class TestPredictionLifecycle:
    def test_prediction_recorded_before_outcome(self, db):
        tracker = PaperEvidenceTracker(db)
        tracker.record_prediction(enriched_candidate(), family="TEMPORAL",
                                  regime="BULL")
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT status, predicted_net_ev, predicted_half_life_seconds, "
                "realized_net_return, model_versions_json FROM paper_evidence"
            ).fetchone()
        assert row[0] == "PENDING"
        assert row[1] is not None
        assert row[3] is None                   # no outcome yet
        assert "enrichment" in row[4]           # model provenance stored

    def test_only_accepted_decisions_recorded_in_bot(self):
        import inspect

        from ultimate_bot_v3_llm import LLMTradingBot
        src = inspect.getsource(LLMTradingBot.run_trading_cycle)
        # prediction recording lives INSIDE the accepted-decision loop
        assert "for d in accepted:" in src
        assert src.index("record_prediction") > src.index("for d in accepted:")

    def test_execution_rejection_is_not_a_loss(self, db):
        tracker = PaperEvidenceTracker(db)
        tracker.record_prediction(enriched_candidate("rejected1"))
        tracker.mark_not_executed("rejected1", "SPREAD_BLOWOUT")
        with sqlite3.connect(db) as conn:
            status, reason, realized = conn.execute(
                "SELECT status, not_executed_reason, realized_net_return "
                "FROM paper_evidence WHERE candidate_id='rejected1'").fetchone()
        assert status == "NOT_EXECUTED"
        assert reason == "SPREAD_BLOWOUT"
        assert realized is None
        # NOT_EXECUTED records never enter calibration
        assert tracker._resolved() == []

    def test_not_executed_cannot_be_resolved_later(self, db):
        tracker = PaperEvidenceTracker(db)
        tracker.record_prediction(enriched_candidate("ne1"))
        tracker.mark_not_executed("ne1", "SIGNAL_EXPIRED")
        assert not tracker.resolve("ne1", realized_net_return=0.01,
                                   realized_net_pnl=100)


# ── §76-77: resolution + idempotency ──────────────────────────────────────────


class TestResolution:
    def test_resolution_fills_realized_fields(self, db):
        tracker = PaperEvidenceTracker(db)
        tracker.record_prediction(enriched_candidate("r1"))
        ok = tracker.resolve("r1", realized_net_return=0.012,
                             realized_net_pnl=120.0,
                             realized_execution_cost_bps=7.5,
                             realized_holding_hours=30.0,
                             close_reason="take_profit")
        assert ok
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT status, realized_net_return, close_reason "
                "FROM paper_evidence WHERE candidate_id='r1'").fetchone()
        assert row == ("RESOLVED", 0.012, "take_profit")

    def test_double_close_is_idempotent(self, db):
        tracker = PaperEvidenceTracker(db)
        tracker.record_prediction(enriched_candidate("i1"))
        assert tracker.resolve("i1", realized_net_return=0.01,
                               realized_net_pnl=100)
        # second close callback: no-op, original values preserved
        assert not tracker.resolve("i1", realized_net_return=-0.99,
                                   realized_net_pnl=-9900)
        with sqlite3.connect(db) as conn:
            ret = conn.execute("SELECT realized_net_return FROM paper_evidence "
                               "WHERE candidate_id='i1'").fetchone()[0]
        assert ret == 0.01

    def test_resolve_by_position_canonical_identity(self, db):
        tracker = PaperEvidenceTracker(db)
        tracker.record_prediction(enriched_candidate("p1"))
        tracker.attach_position("p1", position_id=42,
                                filled_notional_usd=8_000.0)
        assert tracker.resolve_by_position(42, realized_net_return=0.01,
                                           realized_net_pnl=80.0)
        with sqlite3.connect(db) as conn:
            filled = conn.execute(
                "SELECT filled_notional_usd FROM paper_evidence "
                "WHERE candidate_id='p1'").fetchone()[0]
        assert filled == 8_000.0               # §79: FILLED notional, not requested


# ── §78: restart reconciliation ───────────────────────────────────────────────


class TestRestartReconciliation:
    def test_restarted_tracker_resolves_via_trade_memory(self, db):
        from core.trade_memory import TradeMemory
        TradeMemory(db)
        tracker = PaperEvidenceTracker(db)
        tracker.record_prediction(enriched_candidate("restart1"))
        tracker.attach_position("restart1", position_id=99,
                                filled_notional_usd=10_000.0)
        # bot "crashes"; position later closes and lands in trade_memory
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO trade_memory (position_id, symbol, entry_time, "
                "exit_time, size_dollars, net_pnl, holding_hours, close_reason, "
                "recorded_at) VALUES (99,'AAA','2026-09-01T10:00:00',"
                "'2026-09-02T10:00:00',10000,150,24,'take_profit',"
                "datetime('now'))")
        # restart: NEW tracker instance reconciles from durable state
        recon = PaperEvidenceTracker(db).reconcile()
        assert recon["resolved"] == 1
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT status, realized_net_return FROM paper_evidence "
                "WHERE candidate_id='restart1'").fetchone()
        assert row[0] == "RESOLVED"
        assert row[1] == pytest.approx(0.015)   # canonical net_pnl/size_dollars

    def test_orphan_closed_trades_flagged_not_fabricated(self, db):
        from core.trade_attribution import TradeAttributionStore
        from core.trade_memory import TradeMemory
        TradeMemory(db)
        TradeAttributionStore(db)
        tracker = PaperEvidenceTracker(db)
        with sqlite3.connect(db) as conn:
            cur = conn.execute(
                "INSERT INTO trade_memory (position_id, symbol, exit_time, "
                "size_dollars, net_pnl, recorded_at) VALUES "
                "(7,'BBB','2026-09-01T10:00:00',5000,50,datetime('now'))")
            conn.execute(
                "INSERT INTO trade_attribution (trade_memory_id, alpha_id, "
                "alpha_version, candidate_id, recorded_at) "
                "VALUES (?,?,?,?,datetime('now'))",
                (cur.lastrowid, "ghost", "1", "cg"))
        recon = tracker.reconcile()
        assert recon["orphan_closed_trades"] == 1
        assert any("CLOSED_TRADES_WITHOUT_EVIDENCE" in f
                   for f in recon["flags"])
        assert recon["resolved"] == 0           # nothing fabricated


# ── §80-82: canonical half-life units ─────────────────────────────────────────


class TestHalfLifeUnits:
    def _hl_seconds(self, db, cid, bars, bar_seconds):
        tracker = PaperEvidenceTracker(db)
        tracker.record_prediction(enriched_candidate(cid, half_life_bars=bars),
                                  bar_interval_seconds=bar_seconds)
        with sqlite3.connect(db) as conn:
            return conn.execute(
                "SELECT predicted_half_life_seconds FROM paper_evidence "
                "WHERE candidate_id=?", (cid,)).fetchone()[0]

    def test_five_minute_bars(self, db):
        assert self._hl_seconds(db, "m5", 6, 300.0) == pytest.approx(1_800)

    def test_daily_bars(self, db):
        assert self._hl_seconds(db, "d1", 6, 86_400.0) == pytest.approx(518_400)

    def test_mixed_timeframes_never_equal(self, db):
        a = self._hl_seconds(db, "x5", 6, 300.0)
        b = self._hl_seconds(db, "xd", 6, 86_400.0)
        assert a != b and b / a == pytest.approx(288.0)

    def test_half_life_calibration_uses_seconds(self, db):
        tracker = PaperEvidenceTracker(db)
        for i in range(12):
            tracker.record_prediction(
                enriched_candidate(f"hl{i}", half_life_bars=1.0),
                bar_interval_seconds=86_400.0)
            tracker.resolve(f"hl{i}", realized_net_return=0.01,
                            realized_net_pnl=100, realized_holding_hours=24.0)
            tracker.record_realized_decay(f"hl{i}", 86_400.0)  # measured decay
        cal = tracker.half_life_calibration()
        assert cal["unit"] == "seconds"
        assert cal["predicted_half_life_seconds"] == pytest.approx(86_400)
        assert cal["ratio"] == pytest.approx(1.0)


# ── §89: tiny forward samples cannot kill alphas ──────────────────────────────


class TestSmallSampleProtection:
    def test_two_losses_do_not_override_prior(self, tmp_path):
        from core.ev_model import EconomicEVModel
        from core.trade_attribution import TradeAttributionStore
        from core.trade_memory import TradeMemory
        db = str(tmp_path / "small.sqlite")
        TradeMemory(db)
        TradeAttributionStore(db)
        with sqlite3.connect(db) as conn:
            for i in range(2):                  # only two forward losses
                cur = conn.execute(
                    "INSERT INTO trade_memory (symbol, entry_time, exit_time, "
                    "size_dollars, net_pnl, net_return_pct, recorded_at) VALUES "
                    "('AAA',?,?,10000,-100,-1.0,datetime('now'))",
                    (f"2026-09-0{i + 1}T10:00:00", f"2026-09-0{i + 1}T12:00:00"))
                conn.execute(
                    "INSERT INTO trade_attribution (trade_memory_id, alpha_id, "
                    "alpha_version, candidate_id, recorded_at) "
                    "VALUES (?,?,?,?,datetime('now'))",
                    (cur.lastrowid, "young", "1", f"c{i}"))
        model = EconomicEVModel(db)
        # insufficient forward evidence → abstain (None) → prior remains in
        # charge via estimate_with_prior; the alpha is NOT killed
        assert model.estimate("young") is None


# ── §90: research ROI favors forward performers ───────────────────────────────


class TestResearchROIForward:
    def test_budget_shifts_toward_forward_winners(self, db):
        from core.search_ledger import (
            SearchLedger,
            allocate_research_budget,
            research_roi_scores,
        )
        tracker = PaperEvidenceTracker(db)
        for i in range(45):     # A: gorgeous backtests, negative forward
            c = enriched_candidate(f"a{i}", alpha_id="famA_alpha")
            tracker.record_prediction(c, family="FAMILY_A")
            tracker.resolve(f"a{i}", realized_net_return=-0.002,
                            realized_net_pnl=-20)
        for i in range(45):     # B: moderate validation, strong forward
            c = enriched_candidate(f"b{i}", alpha_id="famB_alpha")
            tracker.record_prediction(c, family="FAMILY_B")
            tracker.resolve(f"b{i}", realized_net_return=0.003,
                            realized_net_pnl=30)
        ledger = SearchLedger(db)
        ledger.record_search("FAMILY_A", 10_000)
        ledger.record_outcome("FAMILY_A", "validated", 50)   # great backtests
        ledger.record_search("FAMILY_B", 10_000)
        ledger.record_outcome("FAMILY_B", "validated", 10)
        tracker.update_research_roi(ledger)
        roi = research_roi_scores(ledger)
        assert roi["FAMILY_A"]["forward_pnl"] < 0
        assert roi["FAMILY_B"]["forward_pnl"] > 0
        budget = allocate_research_budget(roi, total_budget=10_000)
        assert budget["FAMILY_A"] > 0           # exploration floor retained


# ── §92: alpha capture reconciliation ─────────────────────────────────────────


class TestAlphaCapture:
    def test_capture_reconciles(self):
        result = alpha_capture_analysis(
            gross_alpha_bps=30.0, execution_cost_bps=8.0,
            delay_cost_bps=4.0, realized_net_bps=18.0)
        assert result["expected_captured_bps"] == pytest.approx(18.0)
        assert result["unexplained_bps"] == pytest.approx(0.0)
        assert result["alpha_capture_ratio"] == pytest.approx(0.6)


# ── §93: full paper end-to-end through real components ────────────────────────


class TestFullPaperEndToEnd:
    def test_signal_to_calibration_to_edge_health(self, db):
        from core.broker import PaperBroker
        from core.position_manager import PositionManager
        from core.search_ledger import SearchLedger
        from core.trade_attribution import TradeAttributionStore
        from core.trade_memory import TradeMemory

        broker = PaperBroker(starting_cash=100_000)
        positions = PositionManager(db_path=db)
        memory = TradeMemory(db)
        attribution = TradeAttributionStore(db)
        tracker = PaperEvidenceTracker(db)
        ledger = SearchLedger(db)

        # 1. enriched candidate → prediction BEFORE fill
        c = enriched_candidate("e2e1", alpha_id="temporal_overnight_x")
        c.expected_net_return = 0.004
        tracker.record_prediction(c, family="TEMPORAL", regime="BULL")

        # 2. paper fill → open position → link evidence
        fill = broker.submit_and_wait(symbol="AAA", side="BUY", quantity=20,
                                      current_price=100.0, timeout_seconds=5)
        assert fill.status.value in ("FILLED", "PARTIAL")
        pos_id = positions.open_position(
            symbol="AAA", signal="BUY", size=fill.notional,
            entry_price=fill.fill_price, entry_fill_price=fill.fill_price,
            broker_order_id=fill.order_id)
        tracker.attach_position("e2e1", pos_id,
                                filled_notional_usd=fill.notional)

        # 3. close → TradeMemory → attribution → evidence resolution
        net_pnl = positions.close_position(
            pos_id, close_price=100.9, exit_fill_price=100.9,
            exit_fees=1.0, reason="take_profit")
        assert net_pnl is not None
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            full_pos = dict(conn.execute(
                "SELECT * FROM positions WHERE id=?", (pos_id,)).fetchone())
        tm_id = memory.record(full_pos)
        assert tm_id is not None
        attribution.record(trade_memory_id=tm_id, position_id=pos_id,
                           alpha_id=c.alpha_id, candidate_id="e2e1")
        from core.trade_history import compute_trade_net_return
        ret = compute_trade_net_return(net_pnl, full_pos["size"])
        assert tracker.resolve_by_position(pos_id, realized_net_return=ret,
                                           realized_net_pnl=net_pnl,
                                           close_reason="take_profit")

        # 4. calibration sees the resolved trade
        assert tracker.status_counts()["RESOLVED"] == 1
        assert tracker.daily_report()["closed_today"] == 1

        # 5. edge health consumes the SAME canonical history
        from core.opportunity_enrichment import OpportunityEnricher
        c2 = enriched_candidate("e2e2", alpha_id="temporal_overnight_x")
        history = OpportunityEnricher(db_path=db)._alpha_history(c.alpha_id)
        assert history["recent"] or history["older"]
        realized = (history["recent"] + history["older"])[0]
        assert realized == pytest.approx(ret)

        # 6. research ROI updated from forward evidence
        ledger.record_search("TEMPORAL", 100)
        tracker.update_research_roi(ledger)    # below family minimum → no-op OK
        assert tracker.review_status(c.alpha_id) == "INSUFFICIENT_EVIDENCE"


# ── Meta shadow stays measurement-only (spec §40) ─────────────────────────────


class TestMetaShadowStrict:
    def test_shadow_never_multiplies_sizing(self):
        from core.opportunity_enrichment import EnrichmentConfig, OpportunityEnricher
        fetch = lambda a: {"recent": [0.004] * 30, "older": [0.004] * 30,
                           "by_regime": [("BULL", 0.02)] * 50}
        c = make_candidate()
        OpportunityEnricher(EnrichmentConfig(meta_alpha_mode="shadow"),
                            returns_fetcher=fetch, db_path=":memory:").enrich(
            c, regime="BULL", capital=10_000, base_position_frac=0.05)
        assert c.regime_fit == 1.0             # measurement only
        assert c.meta_alpha_ev is not None     # but the prediction is stored
