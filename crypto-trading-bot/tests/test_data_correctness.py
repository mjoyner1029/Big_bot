"""Data-correctness tests (spec §16-22, §85-96): the REAL production schema
feeds normalized returns, correlations, risk, edge health, and paper evidence.
No hand-mocked schemas — tables come from production init code.
"""
import sqlite3
import uuid

import numpy as np
import pytest

from core.paper_evidence import PaperEvidenceTracker
from core.trade_history import (
    DatabaseSchemaHealthCheck,
    TradeMemorySchemaError,
    bucketed_alpha_returns,
    choose_risk_resolution,
    compute_trade_net_return,
    data_integrity_scan,
    fetch_attributed_returns,
    migrate_legacy_position_size,
)


@pytest.fixture
def real_db(tmp_path):
    """Production-initialized database: TradeMemory + TradeAttributionStore
    create the REAL schema (spec §16: never mock it manually)."""
    db = str(tmp_path / f"real_{uuid.uuid4().hex}.sqlite")
    from core.trade_attribution import TradeAttributionStore
    from core.trade_memory import TradeMemory
    TradeMemory(db)
    TradeAttributionStore(db)
    return db


def insert_trade(db, *, alpha_id="a1", symbol="AAA", size_dollars=10_000.0,
                 net_pnl=100.0, entry="2026-08-01T10:00:00",
                 exit_="2026-08-01T18:00:00", regime="BULL",
                 holding_hours=8.0, position_id=None, direction="long"):
    with sqlite3.connect(db) as conn:
        cur = conn.execute(
            "INSERT INTO trade_memory (symbol, entry_time, exit_time, "
            "entry_price, size_dollars, quantity, net_pnl, net_return_pct, "
            "gross_pnl, total_fees, market_regime, holding_hours, direction, "
            "position_id, recorded_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
            (symbol, entry, exit_, 100.0, size_dollars,
             size_dollars / 100.0, net_pnl,
             (net_pnl / size_dollars * 100) if size_dollars else None,
             net_pnl + 5.0, 5.0, regime, holding_hours, direction, position_id))
        tid = cur.lastrowid
        conn.execute(
            "INSERT INTO trade_attribution (trade_memory_id, alpha_id, "
            "alpha_version, candidate_id, recorded_at) "
            "VALUES (?,?,?,?,datetime('now'))",
            (tid, alpha_id, "1", f"c{tid}"))
    return tid


# ── §17-20: canonical normalization against the real schema ───────────────────


class TestCanonicalNormalization:
    def test_positive_return_is_pnl_over_size_dollars(self, real_db):
        insert_trade(real_db, size_dollars=10_000, net_pnl=100)
        trades = fetch_attributed_returns(real_db)
        assert len(trades) == 1
        assert trades[0].return_fraction == pytest.approx(0.01)   # +1%, NOT /price

    def test_negative_return(self, real_db):
        insert_trade(real_db, size_dollars=20_000, net_pnl=-400)
        t = fetch_attributed_returns(real_db)[0]
        assert t.return_fraction == pytest.approx(-0.02)

    def test_short_trade_not_double_inverted(self, real_db):
        insert_trade(real_db, size_dollars=10_000, net_pnl=300, direction="short")
        t = fetch_attributed_returns(real_db)[0]
        assert t.return_fraction == pytest.approx(0.03)    # net_pnl already signed

    def test_partial_fill_uses_filled_notional(self, real_db):
        # requested $20k, filled $8k → size_dollars stores FILLED notional
        insert_trade(real_db, size_dollars=8_000, net_pnl=80)
        t = fetch_attributed_returns(real_db)[0]
        assert t.return_fraction == pytest.approx(0.01)

    def test_multi_leg_uses_gross_capital(self):
        # pair trade: $5k long + $5k short, combined PnL $100 → 1% not 2%
        assert compute_trade_net_return(100.0, 5_000.0,
                                        gross_capital_usd=10_000.0) == \
            pytest.approx(0.01)

    def test_zero_notional_never_divides(self, real_db):
        insert_trade(real_db, size_dollars=0.0, net_pnl=50)
        t = fetch_attributed_returns(real_db)[0]
        assert t.return_fraction is None
        assert "INVALID_NOTIONAL" in t.flags

    def test_absurd_return_flagged_not_used(self, real_db):
        insert_trade(real_db, size_dollars=1.0, net_pnl=1_000.0)  # +100,000%
        t = fetch_attributed_returns(real_db)[0]
        assert t.return_fraction is None
        assert "ABSURD_RETURN" in t.flags


# ── §22, §85: schema mismatch fails LOUDLY ────────────────────────────────────


class TestSchemaFailLoud:
    def test_invalid_column_raises_not_empty(self, real_db):
        from core.trade_history import _run_query
        with pytest.raises(TradeMemorySchemaError):
            _run_query(real_db,
                       "SELECT tm.position_size FROM trade_memory tm")

    def test_missing_table_is_expected_empty(self, tmp_path):
        from core.trade_history import _run_query
        db = str(tmp_path / "empty.sqlite")
        sqlite3.connect(db).close()
        assert _run_query(db, "SELECT * FROM trade_memory") == []

    def test_health_check_passes_on_production_schema(self, real_db):
        report = DatabaseSchemaHealthCheck(real_db).run()
        assert report["ok"]
        assert report["schema_version"] is not None
        assert report["tables"]["trade_memory"] == "ok"

    def test_health_check_flags_broken_table(self, tmp_path):
        db = str(tmp_path / "broken.sqlite")
        with sqlite3.connect(db) as conn:      # trade_memory WITHOUT size_dollars
            conn.execute("CREATE TABLE trade_memory (id INTEGER PRIMARY KEY, "
                         "symbol TEXT)")
        report = DatabaseSchemaHealthCheck(db).run()
        assert not report["ok"]
        assert any("TRADE_MEMORY_SCHEMA_MISMATCH" in c
                   for c in report["critical"])

    def test_enricher_schema_failure_is_conservative(self, tmp_path):
        """Broken history must never mean zero risk: candidate is unsizable."""
        db = str(tmp_path / "brk.sqlite")
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE trade_memory (id INTEGER PRIMARY KEY)")
            conn.execute("CREATE TABLE trade_attribution "
                         "(trade_memory_id INTEGER, alpha_id TEXT)")
        from core.alpha_signal_engine import ReasonCode
        from core.opportunity_enrichment import OpportunityEnricher
        from tests.test_monetization_wiring import make_candidate
        c = make_candidate()
        OpportunityEnricher(db_path=db).enrich(
            c, regime="BULL", capital=10_000, base_position_frac=0.05)
        assert c.survival_multiplier == 0.0
        assert ReasonCode.EDGE_DECAY in c.reason_codes


# ── §21, §86: real history flows into portfolio risk + allocator ──────────────


class TestRealHistoryReachesRisk:
    def test_portfolio_path_exact_math(self):
        from core.portfolio_paths import build_return_matrix, portfolio_return_series
        _, matrix = build_return_matrix({
            "A": {"d1": 0.01, "d2": -0.02},
            "B": {"d1": -0.01, "d2": -0.02}})
        series = portfolio_return_series(matrix, {"A": 0.5, "B": 0.5})
        assert series[0] == pytest.approx(0.0)
        assert series[1] == pytest.approx(-0.02)

    def test_production_correlations_populated_from_real_rows(self, real_db):
        rng = np.random.default_rng(0)
        shock = rng.normal(0, 0.02, 40)
        for i in range(40):
            day = f"2026-06-{i % 28 + 1:02d}T1{i % 8}:00:00"
            for alpha, mult in (("a1", 1.0), ("a2", 0.9)):
                pnl = (0.001 + mult * shock[i]) * 10_000
                insert_trade(real_db, alpha_id=alpha, symbol=f"S{alpha}",
                             size_dollars=10_000, net_pnl=pnl,
                             entry=day, exit_=day.replace("T1", "T2"))
        from core.portfolio_paths import ProductionCorrelationService
        svc = ProductionCorrelationService(real_db)
        m = svc.matrices()
        assert m["ordinary"], "ordinary correlations must be populated"
        key = ("a1", "a2") if ("a1", "a2") in m["ordinary"] else ("a2", "a1")
        assert m["ordinary"][key] > 0.5        # shrunk but clearly positive
        assert m["downside"], "downside correlations must be populated"
        assert m["clusters"]["nominal_alphas"] == 2

        from core.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator()
        svc.populate_allocator(alloc)
        assert alloc.alpha_correlations and alloc.downside_correlations

    def test_min_sample_policy_blocks_tiny_correlations(self, real_db):
        for alpha in ("x1", "x2"):
            for i in range(3):                 # only 3 overlapping days
                insert_trade(real_db, alpha_id=alpha,
                             size_dollars=10_000, net_pnl=100,
                             entry=f"2026-06-0{i + 1}T10:00:00",
                             exit_=f"2026-06-0{i + 1}T12:00:00")
        from core.portfolio_paths import ProductionCorrelationService
        m = ProductionCorrelationService(real_db).matrices()
        assert m["ordinary"] == {}             # conservative: no estimate

    def test_edge_health_receives_real_history(self, real_db):
        for i in range(30):
            insert_trade(real_db, alpha_id="healthy", size_dollars=10_000,
                         net_pnl=80, entry=f"2026-05-{i % 28 + 1:02d}T10:00:00",
                         exit_=f"2026-05-{i % 28 + 1:02d}T18:00:00")
        from core.opportunity_enrichment import OpportunityEnricher
        from tests.test_monetization_wiring import make_candidate
        c = make_candidate(alpha_id="healthy")
        OpportunityEnricher(db_path=real_db).enrich(
            c, regime="BULL", capital=10_000, base_position_frac=0.05)
        assert c.edge_survival_probability is not None
        assert c.survival_multiplier == 1.0    # consistent +0.8% → HEALTHY

    def test_old_loader_vs_corrected_loader(self, real_db):
        """The OLD query (position_size) found ZERO history on this schema;
        the corrected loader finds all of it."""
        for i in range(10):
            insert_trade(real_db, size_dollars=10_000, net_pnl=100,
                         entry=f"2026-04-{i + 1:02d}T10:00:00",
                         exit_=f"2026-04-{i + 1:02d}T12:00:00")
        with pytest.raises(sqlite3.OperationalError):
            with sqlite3.connect(real_db) as conn:   # the old broken query
                conn.execute("SELECT tm.net_pnl, tm.position_size, tm.entry_price "
                             "FROM trade_memory tm").fetchall()
        corrected = fetch_attributed_returns(real_db)
        assert len(corrected) == 10            # real history restored


# ── §28-31: risk resolution ───────────────────────────────────────────────────


class TestRiskResolution:
    def test_intraday_alphas_get_hourly_buckets(self, real_db):
        for i in range(70):                    # median hold 1h → hourly
            insert_trade(real_db, alpha_id="fast", size_dollars=10_000,
                         net_pnl=50, holding_hours=1.0,
                         entry=f"2026-06-{i % 28 + 1:02d}T{i % 20 + 1:02d}:00:00",
                         exit_=f"2026-06-{i % 28 + 1:02d}T{i % 20 + 3:02d}:00:00")
        assert choose_risk_resolution(real_db) == "HOURLY"
        buckets = bucketed_alpha_returns(real_db, resolution="HOURLY")
        assert len(buckets["fast"]) > 20       # hour-level, not day-level

    def test_multiday_alphas_stay_daily(self, real_db):
        for i in range(70):
            insert_trade(real_db, alpha_id="slow", size_dollars=10_000,
                         net_pnl=50, holding_hours=72.0,
                         entry=f"2026-06-{i % 28 + 1:02d}T10:00:00",
                         exit_=f"2026-06-{i % 28 + 1:02d}T12:00:00")
        assert choose_risk_resolution(real_db) == "DAILY"


# ── §79-81: migration ─────────────────────────────────────────────────────────


class TestLegacyMigration:
    def _legacy_db(self, tmp_path):
        db = str(tmp_path / "legacy.sqlite")
        from core.trade_memory import TradeMemory
        TradeMemory(db)
        with sqlite3.connect(db) as conn:
            conn.execute("ALTER TABLE trade_memory ADD COLUMN position_size REAL")
            # unambiguous: position_size == quantity * entry_price → dollars
            conn.execute(
                "INSERT INTO trade_memory (symbol, entry_price, quantity, "
                "position_size, net_pnl, recorded_at) "
                "VALUES ('AAA', 100.0, 50.0, 5000.0, 50.0, datetime('now'))")
            # ambiguous: units cannot be determined
            conn.execute(
                "INSERT INTO trade_memory (symbol, entry_price, quantity, "
                "position_size, net_pnl, recorded_at) "
                "VALUES ('BBB', 100.0, 50.0, 777.0, 50.0, datetime('now'))")
        return db

    def test_migration_preserves_and_flags(self, tmp_path):
        db = self._legacy_db(tmp_path)
        result = migrate_legacy_position_size(db)
        assert result["migrated"] == 1
        assert result["flagged"] == 1
        with sqlite3.connect(db) as conn:
            ok = conn.execute("SELECT size_dollars FROM trade_memory "
                              "WHERE symbol='AAA'").fetchone()[0]
            flagged = conn.execute("SELECT needs_unit_review FROM trade_memory "
                                   "WHERE symbol='BBB'").fetchone()[0]
            rows = conn.execute("SELECT COUNT(*) FROM trade_memory").fetchone()[0]
        assert ok == pytest.approx(5000.0)     # dollars preserved economically
        assert flagged == 1                    # ambiguous marked, never guessed
        assert rows == 2                       # nothing destroyed

    def test_migration_idempotent(self, tmp_path):
        db = self._legacy_db(tmp_path)
        migrate_legacy_position_size(db)
        second = migrate_legacy_position_size(db)
        assert second["migrated"] == 0         # nothing re-migrated


# ── §76, §95-96: data integrity ───────────────────────────────────────────────


class TestDataIntegrity:
    def test_integrity_scan_flags_bad_rows(self, real_db):
        insert_trade(real_db, size_dollars=0.0, net_pnl=10)          # zero size
        insert_trade(real_db, size_dollars=1.0, net_pnl=1_000.0)     # unit error
        insert_trade(real_db, size_dollars=10_000, net_pnl=100,
                     entry="2026-08-02T10:00:00", exit_="2026-08-01T10:00:00")
        report = data_integrity_scan(real_db)
        flags = {i["flag"] for i in report["issues"]}
        assert {"MISSING_OR_ZERO_NOTIONAL", "ABSURD_RETURN",
                "TIMESTAMPS_REVERSED"} <= flags
        assert report["scanned"] == 3          # flagged, never deleted

    def test_schema_itself_blocks_duplicate_positions(self, real_db):
        insert_trade(real_db, size_dollars=10_000, net_pnl=50, position_id=7)
        with pytest.raises(sqlite3.IntegrityError):   # UNIQUE(position_id)
            insert_trade(real_db, size_dollars=10_000, net_pnl=60, position_id=7)


# ── §52-59, §91: paper evidence + calibration ─────────────────────────────────


class TestPaperEvidence:
    def _record(self, tracker, cid, alpha, ev, p_pos, realized, family="MOM"):
        from tests.test_monetization_wiring import make_candidate
        c = make_candidate(alpha_id=alpha)
        c.candidate_id = cid
        c.expected_net_return = ev
        c.probability_positive = p_pos
        c.expected_dollar_alpha = ev * 500
        c.expected_total_cost_bps = 6.0
        tracker.record_prediction(c, family=family, regime="BULL")
        tracker.resolve(cid, realized_net_return=realized,
                        realized_net_pnl=realized * 500,
                        realized_execution_cost_bps=7.0,
                        realized_holding_hours=24.0)

    def test_calibrated_predictions_pass(self, tmp_path):
        tracker = PaperEvidenceTracker(str(tmp_path / "pe.sqlite"))
        for i in range(30):                    # predicted +20bps, realized +20bps
            self._record(tracker, f"good{i}", "a1", 0.0020, 0.6, 0.0020)
        cal = tracker.ev_calibration()
        bucket = cal["buckets"]["20-40bps"]
        assert bucket["calibrated"]
        assert bucket["realized_mean"] == pytest.approx(0.0020)

    def test_severe_miscalibration_detected(self, tmp_path):
        tracker = PaperEvidenceTracker(str(tmp_path / "pe.sqlite"))
        for i in range(30):                    # predicted +50bps, realized −10bps
            self._record(tracker, f"bad{i}", "a2", 0.0050, 0.7, -0.0010)
        bucket = tracker.ev_calibration()["buckets"]["40-+bps"]
        assert not bucket["calibrated"]
        assert bucket["calibration_error"] > 0.005
        prob = tracker.probability_calibration()["70%-80%"]
        assert not prob["calibrated"]          # 70% predicted, 0% realized

    def test_execution_and_survival_calibration(self, tmp_path):
        tracker = PaperEvidenceTracker(str(tmp_path / "pe.sqlite"))
        from tests.test_monetization_wiring import make_candidate
        for i in range(15):
            c = make_candidate(alpha_id="surv_hi")
            c.candidate_id = f"hi{i}"
            c.edge_survival_probability = 0.8
            c.expected_total_cost_bps = 6.0
            tracker.record_prediction(c, family="F")
            tracker.resolve(f"hi{i}", realized_net_return=0.004,
                            realized_net_pnl=40,
                            realized_execution_cost_bps=9.0)
            c2 = make_candidate(alpha_id="surv_lo")
            c2.candidate_id = f"lo{i}"
            c2.edge_survival_probability = 0.2
            c2.expected_total_cost_bps = 6.0
            tracker.record_prediction(c2, family="F")
            tracker.resolve(f"lo{i}", realized_net_return=-0.002,
                            realized_net_pnl=-20,
                            realized_execution_cost_bps=9.0)
        surv = tracker.survival_calibration()
        assert surv["ordering_correct"]
        execution = tracker.execution_calibration()
        assert execution["realized_cost_bps"] == pytest.approx(9.0)
        assert execution["cost_underestimate_bps"] > 0

    def test_no_auto_live_promotion(self, tmp_path):
        tracker = PaperEvidenceTracker(str(tmp_path / "pe.sqlite"))
        for i in range(25):                    # excellent paper results
            self._record(tracker, f"star{i}", "star", 0.003, 0.7, 0.004)
        assert tracker.review_status("star") == "LIVE_ELIGIBLE_REVIEW"

    def test_insufficient_evidence_status(self, tmp_path):
        tracker = PaperEvidenceTracker(str(tmp_path / "pe.sqlite"))
        for i in range(5):                     # five trades prove nothing
            self._record(tracker, f"few{i}", "few", 0.003, 0.7, 0.01)
        assert tracker.review_status("few") == "INSUFFICIENT_EVIDENCE"

    def test_research_roi_forward_penalty(self, tmp_path):
        from core.search_ledger import SearchLedger, source_roi_report
        db = str(tmp_path / "roi.sqlite")
        tracker = PaperEvidenceTracker(db)
        for i in range(45):                    # great backtests, bad forward
            self._record(tracker, f"pf{i}", "pretty", 0.005, 0.7, -0.002,
                         family="PRETTY_FAMILY")
        ledger = SearchLedger(db)
        ledger.record_search("PRETTY_FAMILY", 5_000)
        tracker.update_research_roi(ledger)
        report = source_roi_report(ledger)
        assert report["PRETTY_FAMILY"]["forward_pnl"] < 0   # penalized


# ── §92: forward evidence overrides historical prior ──────────────────────────


class TestForwardOverridesHistory:
    def test_negative_forward_evidence_beats_positive_prior(self, real_db):
        # 40 attributed forward trades at −10 bps for an alpha whose
        # historical OOS prior claimed +50 bps
        for i in range(40):
            insert_trade(real_db, alpha_id="fading", size_dollars=10_000,
                         net_pnl=-10.0,
                         entry=f"2026-03-{i % 28 + 1:02d}T10:00:00",
                         exit_=f"2026-03-{i % 28 + 1:02d}T14:00:00")
        from core.ev_model import EconomicEVModel
        est = EconomicEVModel(real_db).estimate("fading")
        assert est is not None
        assert est.expected_net_return < 0     # forward evidence wins
        assert est.sample_size == 40


# ── §89: research scale with 2,000 instruments ────────────────────────────────


class TestResearchScaleLarge:
    def test_2000_instruments_all_enter_batching(self, tmp_path):
        import pandas as pd

        from core.discovery_detectors import DiscoveryDetector
        from core.research_campaign import CampaignConfig, ResearchCampaignRunner

        class Counting(DiscoveryDetector):
            family = "PRICE"
            seen: set = set()

            def scan(self, data):
                Counting.seen.update(data.keys())
                return []

        Counting.seen = set()
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=210, freq="1D")
        base = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 210)))
        data = {}
        for i in range(2000):                  # cheap variations of one path
            close = base * (1 + (i % 7) * 0.001)
            data[f"U{i:04d}"] = pd.DataFrame({
                "open": close, "high": close * 1.004, "low": close * 0.996,
                "close": close, "volume": np.full(210, 1e6)}, index=idx)
        runner = ResearchCampaignRunner(
            str(tmp_path / "scale.sqlite"),
            config=CampaignConfig(instrument_batch_size=100),
            detectors_factory=lambda limits=None, tracker=None:
                [Counting(limits, tracker)])
        report = runner.run(data)
        assert report["batches"] == 20
        assert len(Counting.seen) == 2000      # nothing truncated
