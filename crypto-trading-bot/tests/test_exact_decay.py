"""Exact point-in-time decay tests (spec §52-70): timezone-aware timestamp
alignment, structured statuses, retry semantics, calibration exclusion, and
fail-closed core accounting.
"""
import sqlite3
import uuid
from datetime import datetime, timezone

import pandas as pd
import pytest

from core.paper_evidence import PaperEvidenceTracker
from core.time_norm import TimezoneNormalizationError, to_utc
from tests.test_paper_loop import enriched_candidate


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / f"decay_{uuid.uuid4().hex}.sqlite")


def tracked(db, cid="c1", signal_time="2026-09-10T19:55:00+00:00",
            asset_class="stock", direction="long", half_life_bars=6.0):
    tracker = PaperEvidenceTracker(db)
    c = enriched_candidate(cid, half_life_bars=half_life_bars)
    c.signal_time = signal_time
    c.asset_class = asset_class
    c.direction = direction
    tracker.record_prediction(c, family="TEMPORAL",
                              bar_interval_seconds=300.0)
    tracker.resolve(cid, realized_net_return=0.01, realized_net_pnl=100)
    return tracker


def bars(times, prices, tz="UTC"):
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz=tz) if pd.Timestamp(t).tzinfo
                            is None else pd.Timestamp(t) for t in times])
    return pd.DataFrame({"close": prices}, index=idx)


# ── to_utc unit behavior (spec §3-4, §54) ─────────────────────────────────────


class TestTimestampNormalization:
    def test_aware_utc_passthrough(self):
        t = to_utc("2026-09-10T22:45:00+00:00")
        assert t.tzinfo is not None and t.hour == 22

    def test_eastern_converts_to_utc(self):
        t = to_utc("2026-09-10T15:55:00-04:00")
        assert t.hour == 19 and t.minute == 55      # 15:55 ET = 19:55 UTC

    def test_naive_uses_configured_source_timezone(self):
        t = to_utc("2026-09-10 15:55:00", source_timezone="US/Eastern")
        assert t.hour == 19 and t.minute == 55

    def test_naive_without_policy_is_structured_error(self):
        with pytest.raises(TimezoneNormalizationError):
            to_utc(datetime(2026, 9, 10, 15, 55), source_timezone=None)

    def test_epoch_milliseconds(self):
        ms = 1_788_000_000_000
        assert to_utc(ms) == to_utc(ms / 1000)

    def test_garbage_is_structured_error(self):
        with pytest.raises(TimezoneNormalizationError):
            to_utc("not a timestamp")


# ── §52-53: exact intraday alignment, Micron-like case ────────────────────────


class TestExactIntradayAlignment:
    def test_earlier_same_day_bars_excluded(self, db):
        """Signal 15:55 ET (=19:55 UTC). Same-day earlier bars 09:30/12:00/
        15:50 must NEVER leak into the decay curve; baseline is 15:55."""
        tracker = tracked(db, signal_time="2026-09-10T15:55:00-04:00")
        df = bars(
            ["2026-09-10 13:30", "2026-09-10 16:00", "2026-09-10 19:50",
             "2026-09-10 19:55", "2026-09-10 20:00", "2026-09-11 13:30",
             "2026-09-11 14:30", "2026-09-11 15:30"],
            # earlier bars have a HUGE move that would corrupt the curve
            [50.0, 60.0, 99.0, 100.0, 102.0, 101.0, 100.4, 100.2])
        result = tracker.measure_realized_decay(lambda s: df)
        assert result["statuses"].get("SUCCESS") == 1
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT baseline_timestamp, baseline_price "
                "FROM decay_measurements").fetchone()
        assert row[0].startswith("2026-09-10T19:55")   # never 13:30/09:30
        assert row[1] == 100.0                          # not 50.0

    def test_et_signal_utc_bars_align(self, db):
        tracker = tracked(db, signal_time="2026-09-10T15:55:00-04:00")
        df = bars(["2026-09-10 19:54", "2026-09-10 19:55", "2026-09-10 20:00",
                   "2026-09-10 20:05", "2026-09-10 20:10"],
                  [98.0, 100.0, 101.0, 100.4, 100.1])
        tracker.measure_realized_decay(lambda s: df)
        with sqlite3.connect(db) as conn:
            baseline = conn.execute("SELECT baseline_price FROM "
                                    "decay_measurements").fetchone()[0]
        assert baseline == 100.0                        # 19:54 bar excluded

    def test_crypto_intraday_next_bar(self, db):
        """Signal 14:37:20 → first eligible 5m bar is 14:40 (spec §9)."""
        tracker = tracked(db, signal_time="2026-09-10T14:37:20+00:00",
                          asset_class="crypto")
        df = bars(["2026-09-10 14:35", "2026-09-10 14:40", "2026-09-10 14:45",
                   "2026-09-10 14:50", "2026-09-10 14:55"],
                  [99.0, 100.0, 101.0, 100.4, 100.1])
        tracker.measure_realized_decay(lambda s: df)
        with sqlite3.connect(db) as conn:
            row = conn.execute("SELECT baseline_timestamp FROM "
                               "decay_measurements").fetchone()
        assert row[0].startswith("2026-09-10T14:40")


# ── §55-56: session boundaries ────────────────────────────────────────────────


class TestSessionBoundaries:
    def test_weekend_equity_signal_starts_monday(self, db):
        # Saturday signal; no weekend equity bars exist → Monday baseline
        tracker = tracked(db, signal_time="2026-09-12T10:00:00+00:00")
        df = bars(["2026-09-11 15:30", "2026-09-14 13:30", "2026-09-14 14:30",
                   "2026-09-14 15:30", "2026-09-14 16:30"],
                  [95.0, 100.0, 101.5, 100.6, 100.2])
        tracker.measure_realized_decay(lambda s: df)
        with sqlite3.connect(db) as conn:
            row = conn.execute("SELECT baseline_timestamp, session_policy "
                               "FROM decay_measurements").fetchone()
        assert row[0].startswith("2026-09-14")          # Friday bar excluded
        assert row[1] == "RTH"                          # session policy stored

    def test_crypto_24_7_sunday(self, db):
        tracker = tracked(db, signal_time="2026-09-13T02:00:00+00:00",
                          asset_class="crypto")
        df = bars(["2026-09-13 02:00", "2026-09-13 02:05", "2026-09-13 02:10",
                   "2026-09-13 02:15", "2026-09-13 02:20"],
                  [100.0, 101.0, 101.5, 100.7, 100.2])
        result = tracker.measure_realized_decay(lambda s: df)
        assert result["statuses"].get("SUCCESS") == 1
        with sqlite3.connect(db) as conn:
            policy = conn.execute("SELECT session_policy FROM "
                                  "decay_measurements").fetchone()[0]
        assert policy == "24/7"


# ── §57-60: structured outcomes ───────────────────────────────────────────────


class TestDecayStatuses:
    def test_half_life_not_observed(self, db):
        tracker = tracked(db, signal_time="2026-09-01T00:00:00+00:00")
        # rising through the FULL horizon (horizon_bars=5 → 6 eligible bars)
        times = [f"2026-09-0{i + 1} 00:00" for i in range(7)]
        df = bars(times, [100.0 * (1.01 ** i) for i in range(7)])
        result = tracker.measure_realized_decay(lambda s: df, horizon_bars=5)
        assert result["statuses"].get("HALF_LIFE_NOT_OBSERVED") == 1

    def test_no_realized_signal_peak(self, db):
        tracker = tracked(db, signal_time="2026-09-01T00:00:00+00:00")
        times = [f"2026-09-0{i + 1} 00:00" for i in range(6)]
        df = bars(times, [100.0, 99.0, 98.5, 98.0, 97.0, 96.5])  # never up
        result = tracker.measure_realized_decay(lambda s: df)
        assert result["statuses"].get("NO_REALIZED_SIGNAL_PEAK") == 1

    def test_insufficient_bars_pending_then_success(self, db):
        tracker = tracked(db, signal_time="2026-09-01T00:00:00+00:00")
        few = bars(["2026-09-01 00:00", "2026-09-02 00:00"], [100.0, 101.0])
        result = tracker.measure_realized_decay(lambda s: few)
        assert result["statuses"].get("INSUFFICIENT_FORWARD_BARS") == 1
        # more bars arrive later → the SAME record upgrades to SUCCESS
        times = [f"2026-09-0{i + 1} 00:00" for i in range(6)]
        full = bars(times, [100.0, 101.5, 102.0, 101.4, 100.8, 100.5])
        result = tracker.measure_realized_decay(lambda s: full)
        assert result["statuses"].get("SUCCESS") == 1
        with sqlite3.connect(db) as conn:
            n = conn.execute("SELECT COUNT(*) FROM decay_measurements").fetchone()[0]
            status = conn.execute("SELECT measurement_status FROM "
                                  "decay_measurements").fetchone()[0]
        assert n == 1 and status == "SUCCESS"           # idempotent upgrade

    def test_provider_error_retryable(self, db):
        tracker = tracked(db, signal_time="2026-09-01T00:00:00+00:00")

        def boom(symbol):
            raise ConnectionError("provider down")
        result = tracker.measure_realized_decay(boom)
        assert result["statuses"].get("PROVIDER_ERROR") == 1
        # retry with a working provider succeeds on the same record
        times = [f"2026-09-0{i + 1} 00:00" for i in range(6)]
        df = bars(times, [100.0, 101.5, 102.0, 101.4, 100.8, 100.5])
        assert tracker.measure_realized_decay(lambda s: df)["measured"] == 1

    def test_data_integrity_error(self, db):
        tracker = tracked(db, signal_time="2026-09-01T00:00:00+00:00")
        times = [f"2026-09-0{i + 1} 00:00" for i in range(5)]
        df = bars(times, [100.0, float("nan"), 102.0, 101.0, 100.5])
        result = tracker.measure_realized_decay(lambda s: df)
        assert result["statuses"].get("DATA_INTEGRITY_ERROR") == 1

    def test_success_is_final_not_remeasured(self, db):
        tracker = tracked(db, signal_time="2026-09-01T00:00:00+00:00")
        times = [f"2026-09-0{i + 1} 00:00" for i in range(6)]
        df = bars(times, [100.0, 101.5, 102.0, 101.4, 100.8, 100.5])
        assert tracker.measure_realized_decay(lambda s: df)["measured"] == 1
        again = tracker.measure_realized_decay(lambda s: df)
        assert again["candidates"] == 0                 # nothing re-selected


# ── §67-68: calibration inclusion/exclusion ───────────────────────────────────


class TestCalibrationExclusion:
    def test_only_success_feeds_half_life_calibration(self, db):
        tracker = PaperEvidenceTracker(db)
        outcomes = (
            [("ok", [100.0, 101.5, 102.0, 101.4, 100.8, 100.5])] * 4
            + [("nopeak", [100.0, 99.0, 98.0, 97.5, 97.0, 96.5])] * 3
            + [("short", [100.0, 101.0])] * 3)
        for i, (kind, prices) in enumerate(outcomes):
            cid = f"{kind}{i}"
            c = enriched_candidate(cid, half_life_bars=4.0)
            c.signal_time = "2026-09-01T00:00:00+00:00"
            tracker.record_prediction(c, family="F",
                                      bar_interval_seconds=86_400.0)
            tracker.resolve(cid, realized_net_return=0.01, realized_net_pnl=10)
            times = [f"2026-09-0{j + 1} 00:00" for j in range(len(prices))]
            tracker.measure_realized_decay(
                lambda s, _p=prices, _t=times: bars(_t, _p), max_rows=1000)
        report = tracker.decay_exclusion_report()
        assert report["SUCCESS"] == 4
        assert report["NO_REALIZED_SIGNAL_PEAK"] == 3
        assert report["INSUFFICIENT_FORWARD_BARS"] == 3
        cal = tracker.half_life_calibration(min_samples=1)
        assert cal["n"] == 4                            # ONLY successes

    def test_health_report_shows_decay_pipeline(self, db):
        tracker = tracked(db, signal_time="2026-09-01T00:00:00+00:00")
        tracker.measure_realized_decay(
            lambda s: bars(["2026-09-01 00:00", "2026-09-02 00:00"],
                           [100.0, 101.0]))
        health = tracker.evidence_health_report()
        assert health["decay_pending"] == 1
        assert health["decay_successful"] == 0
        assert "RESOLVED" in health["status_counts"]


# ── §61-65: fail-closed core accounting ───────────────────────────────────────


class TestFailClosedCore:
    def test_gate_blocks_any_non_healthy_state(self):
        import inspect

        from ultimate_bot_v3_llm import LLMTradingBot
        src = inspect.getsource(LLMTradingBot.execute_trade)
        assert "!= 'HEALTHY'" in src                    # DEGRADED also blocks

    def test_trade_memory_write_failure_degrades(self):
        import inspect

        from ultimate_bot_v3_llm import LLMTradingBot
        src = inspect.getsource(LLMTradingBot._close_position)
        assert "DEGRADED_READ_ONLY" in src
        assert "CRITICAL_ACCOUNTING_FAILURE" in src

    def test_periodic_health_check_in_cycle(self):
        import inspect

        from ultimate_bot_v3_llm import LLMTradingBot
        src = inspect.getsource(LLMTradingBot.run_trading_cycle)
        assert "DatabaseSchemaHealthCheck" in src
        assert "_last_core_health_check" in src

    def test_architecture_freeze_marker(self):
        from config.config import ARCHITECTURE_PHASE, PAPER_EVIDENCE_MODE
        assert ARCHITECTURE_PHASE == "STABLE"
        assert isinstance(PAPER_EVIDENCE_MODE, bool)


# ── Snapshots + reliability ───────────────────────────────────────────────────


class TestSnapshotsAndReliability:
    def test_daily_snapshot_persists_and_upserts(self, db):
        tracker = PaperEvidenceTracker(db)
        for i in range(25):
            c = enriched_candidate(f"sn{i}")
            c.expected_net_return = 0.002
            tracker.record_prediction(c, family="F")
            tracker.resolve(f"sn{i}", realized_net_return=0.0019,
                            realized_net_pnl=19.0)
        tracker.snapshot_daily()
        tracker.snapshot_daily()                        # idempotent per day
        with sqlite3.connect(db) as conn:
            rows = conn.execute("SELECT snapshot_date, resolved_count, ev_slope "
                                "FROM calibration_snapshots").fetchall()
        assert len(rows) == 1
        assert rows[0][1] == 25

    def test_reliability_report_grades(self, db):
        tracker = PaperEvidenceTracker(db)
        rel = tracker.model_reliability_report()
        assert rel["EconomicEVModel"] == "INSUFFICIENT_EVIDENCE"
        for i in range(30):                             # calibrated slope ≈ 0.95
            c = enriched_candidate(f"r{i}")
            c.expected_net_return = 0.001 + i * 0.0001
            tracker.record_prediction(c, family="F")
            tracker.resolve(f"r{i}",
                            realized_net_return=0.95 * (0.001 + i * 0.0001),
                            realized_net_pnl=10.0)
        rel = tracker.model_reliability_report()
        assert rel["EconomicEVModel"] == "STRONG"
        assert rel["MetaAlphaShadow"] == "INSUFFICIENT_EVIDENCE"


# ── §72: deterministic decay replay ───────────────────────────────────────────


class TestDeterministicReplay:
    def test_known_half_life_recovered(self, db):
        """Signal value peaks at +2% two hours in, decays below +1% at hour 5
        → realized half-life must be exactly 5 hours from the signal."""
        tracker = tracked(db, signal_time="2026-09-01T00:00:00+00:00",
                          asset_class="crypto")
        times = [f"2026-09-01 0{i}:00" for i in range(8)]
        prices = [100.0, 101.0, 102.0, 101.6, 101.2, 100.9, 100.5, 100.2]
        result = tracker.measure_realized_decay(lambda s: bars(times, prices))
        assert result["measured"] == 1
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT realized_half_life_seconds, peak_time_seconds "
                "FROM decay_measurements").fetchone()
        assert row[0] == pytest.approx(5 * 3600)        # ≤half-peak at 05:00
        assert row[1] == pytest.approx(2 * 3600)        # peak at 02:00
