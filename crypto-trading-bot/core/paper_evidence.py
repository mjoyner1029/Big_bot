"""PaperEvidenceTracker: predicted vs realized for every resolved PAPER trade.

The forward-evidence loop (spec §52-75):
    PREDICT → OBSERVE REALIZED → MEASURE CALIBRATION → UPDATE CONFIDENCE

Nothing here promotes to live capital — strong paper evidence produces
LIVE_ELIGIBLE_REVIEW status at most (spec §63).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import statistics as st
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

# Minimum evidence before conclusions (spec §61)
MIN_RESOLVED_PER_ALPHA = 20
MIN_RESOLVED_PER_FAMILY = 40
MIN_CALENDAR_DAYS = 14

EV_BUCKETS_BPS = ((0, 10), (10, 20), (20, 40), (40, 10_000))
PROB_BUCKETS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01))


class PaperEvidenceTracker:
    def __init__(self, db_path: str = "data/trade_memory.sqlite") -> None:
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    candidate_id TEXT UNIQUE, alpha_id TEXT, family TEXT,
                    regime TEXT,
                    predicted_net_ev REAL, predicted_ev_lower_bound REAL,
                    predicted_p_positive REAL, predicted_dollar_alpha REAL,
                    predicted_half_life_bars REAL,
                    predicted_execution_cost_bps REAL,
                    predicted_survival REAL, predicted_regime_fit REAL,
                    realized_net_return REAL, realized_net_pnl REAL,
                    realized_execution_cost_bps REAL, realized_holding_hours REAL,
                    realized_mfe_pct REAL, realized_mae_pct REAL,
                    resolved_at TEXT
                )""")
            # Idempotent additive migration (spec §67-72)
            cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(paper_evidence)").fetchall()}
            for col, decl in (
                    ("status", "TEXT DEFAULT 'PENDING'"),
                    ("position_id", "INTEGER"),
                    ("filled_notional_usd", "REAL"),
                    ("predicted_half_life_seconds", "REAL"),
                    ("bar_interval_seconds", "REAL"),
                    ("not_executed_reason", "TEXT"),
                    ("close_reason", "TEXT"),
                    ("model_versions_json", "TEXT"),
                    ("meta_alpha_ev", "REAL"),
                    ("realized_half_life_seconds", "REAL"),
                    ("realized_fee_bps", "REAL"),
                    ("realized_slippage_bps", "REAL"),
                    ("symbol", "TEXT"),
                    ("direction", "TEXT"),
                    ("signal_time", "TEXT"),
                    ("session_policy", "TEXT")):
                if col not in cols:
                    conn.execute(f"ALTER TABLE paper_evidence ADD COLUMN {col} {decl}")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decay_measurements (
                    candidate_id TEXT PRIMARY KEY,
                    alpha_id TEXT,
                    measured_at TEXT NOT NULL,
                    signal_timestamp TEXT,
                    baseline_timestamp TEXT,
                    baseline_price REAL,
                    baseline_semantics TEXT,
                    session_policy TEXT,
                    bar_interval_seconds REAL,
                    bars_available INTEGER,
                    measurement_status TEXT NOT NULL,
                    failure_reason TEXT,
                    predicted_half_life_seconds REAL,
                    realized_half_life_seconds REAL,
                    peak_time_seconds REAL,
                    max_observation_horizon_seconds REAL
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pe_alpha "
                         "ON paper_evidence(alpha_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pe_position "
                         "ON paper_evidence(position_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pe_status "
                         "ON paper_evidence(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pe_family "
                         "ON paper_evidence(family)")

    # ── Record / resolve ──────────────────────────────────────────────────────

    def record_prediction(self, candidate: Any, *, family: str = "",
                          regime: str = "unknown",
                          position_id: Optional[int] = None,
                          filled_notional_usd: Optional[float] = None,
                          bar_interval_seconds: float = 86_400.0) -> None:
        """One prediction per ACCEPTED paper trade, recorded BEFORE the outcome
        is known. Half-life is canonically stored in SECONDS — bars are
        converted with the ACTUAL bar interval (spec §17-21)."""
        hl_seconds = (candidate.half_life_bars * bar_interval_seconds
                      if candidate.half_life_bars is not None else None)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO paper_evidence (recorded_at, candidate_id, "
                "alpha_id, family, regime, predicted_net_ev, "
                "predicted_ev_lower_bound, predicted_p_positive, "
                "predicted_dollar_alpha, predicted_half_life_bars, "
                "predicted_execution_cost_bps, predicted_survival, "
                "predicted_regime_fit, status, position_id, filled_notional_usd, "
                "predicted_half_life_seconds, bar_interval_seconds, "
                "model_versions_json, meta_alpha_ev, symbol, direction, "
                "signal_time, session_policy) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (_utcnow(), candidate.candidate_id, candidate.alpha_id, family,
                 regime, candidate.expected_net_return, candidate.ev_lower_bound,
                 candidate.probability_positive, candidate.expected_dollar_alpha,
                 candidate.half_life_bars, candidate.expected_total_cost_bps,
                 candidate.edge_survival_probability, candidate.regime_fit,
                 "PENDING", position_id, filled_notional_usd,
                 hl_seconds, bar_interval_seconds,
                 json.dumps(getattr(candidate, "model_versions", {}) or {}),
                 candidate.meta_alpha_ev, candidate.symbol,
                 candidate.direction, candidate.signal_time,
                 "24/7" if candidate.asset_class == "crypto" else "RTH"))

    def mark_not_executed(self, candidate_id: str, reason: str) -> None:
        """Allocator accepted but execution rejected — NEVER a resolved loss."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE paper_evidence SET status='NOT_EXECUTED', "
                "not_executed_reason=? WHERE candidate_id=? AND resolved_at IS NULL",
                (reason, candidate_id))

    def resolve(self, candidate_id: str, *, realized_net_return: float,
                realized_net_pnl: float,
                realized_execution_cost_bps: Optional[float] = None,
                realized_holding_hours: Optional[float] = None,
                realized_mfe_pct: Optional[float] = None,
                realized_mae_pct: Optional[float] = None,
                realized_fee_bps: Optional[float] = None,
                realized_slippage_bps: Optional[float] = None,
                close_reason: Optional[str] = None) -> bool:
        """Idempotent: only PENDING, unresolved evidence resolves — repeated
        close callbacks are no-ops (spec §12)."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE paper_evidence SET realized_net_return=?, "
                "realized_net_pnl=?, realized_execution_cost_bps=?, "
                "realized_holding_hours=?, realized_mfe_pct=?, "
                "realized_mae_pct=?, realized_fee_bps=?, realized_slippage_bps=?, "
                "close_reason=?, status='RESOLVED', "
                "resolved_at=? WHERE candidate_id=? AND resolved_at IS NULL "
                "AND status='PENDING'",
                (realized_net_return, realized_net_pnl,
                 realized_execution_cost_bps, realized_holding_hours,
                 realized_mfe_pct, realized_mae_pct, realized_fee_bps,
                 realized_slippage_bps, close_reason,
                 _utcnow(), candidate_id))
            return cur.rowcount > 0

    def record_realized_decay(self, candidate_id: str,
                              realized_half_life_seconds: float) -> None:
        """Realized SIGNAL decay from post-signal price path — measured
        separately from trade holding time (they are different things)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE paper_evidence SET realized_half_life_seconds=? "
                "WHERE candidate_id=?",
                (realized_half_life_seconds, candidate_id))

    def _save_decay_diagnostic(self, candidate_id: str, alpha_id: str,
                               **fields) -> None:
        """Idempotent upsert — repeated measurement runs update one record."""
        cols = ("signal_timestamp", "baseline_timestamp", "baseline_price",
                "baseline_semantics", "session_policy", "bar_interval_seconds",
                "bars_available", "measurement_status", "failure_reason",
                "predicted_half_life_seconds", "realized_half_life_seconds",
                "peak_time_seconds", "max_observation_horizon_seconds")
        values = [fields.get(c) for c in cols]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT INTO decay_measurements (candidate_id, alpha_id, "
                f"measured_at, {', '.join(cols)}) "
                f"VALUES (?,?,?,{','.join('?' * len(cols))}) "
                f"ON CONFLICT(candidate_id) DO UPDATE SET "
                + ", ".join(f"{c}=excluded.{c}" for c in cols)
                + ", measured_at=excluded.measured_at",
                [candidate_id, alpha_id, _utcnow()] + values)

    # Decay statuses: PENDING-retryable vs final-diagnostic (spec §18-22)
    DECAY_RETRYABLE = ("INSUFFICIENT_FORWARD_BARS", "PROVIDER_ERROR",
                       "MARKET_DATA_UNAVAILABLE")

    def measure_realized_decay(self, fetch_fn, *, max_rows: int = 50,
                               horizon_bars: int = 15,
                               source_timezone: str = "UTC",
                               min_forward_bars: int = 4) -> Dict[str, Any]:
        """PRODUCTION decay measurement with EXACT point-in-time alignment.

        Bars are selected by parsed timezone-aware UTC timestamps:
            bar_ts >= signal_ts
        — never by calendar-date strings, so earlier same-day bars can never
        leak into the post-signal curve. Baseline = close of the first
        eligible bar at/after the signal (baseline_semantics recorded).
        Every outcome gets a structured status; only retryable statuses are
        re-attempted on later runs. No generic exception swallowing.
        """
        from core.time_norm import TimezoneNormalizationError, to_utc
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT pe.candidate_id, pe.alpha_id, pe.symbol, pe.direction, "
                "COALESCE(pe.signal_time, pe.recorded_at), "
                "pe.bar_interval_seconds, pe.predicted_half_life_seconds, "
                "pe.session_policy, dm.measurement_status "
                "FROM paper_evidence pe LEFT JOIN decay_measurements dm "
                "ON dm.candidate_id = pe.candidate_id "
                "WHERE pe.status='RESOLVED' AND pe.symbol IS NOT NULL "
                "AND (dm.measurement_status IS NULL OR dm.measurement_status "
                f"IN ({','.join('?' * len(self.DECAY_RETRYABLE))})) LIMIT ?",
                list(self.DECAY_RETRYABLE) + [max_rows]).fetchall()

        counts: Dict[str, int] = {}

        def finish(cid, aid, status, reason=None, **extra):
            counts[status] = counts.get(status, 0) + 1
            self._save_decay_diagnostic(
                cid, aid, measurement_status=status, failure_reason=reason,
                **extra)

        for (cid, aid, symbol, direction, signal_raw, bar_s, pred_hl,
             session_policy, _prev) in rows:
            bar_s = float(bar_s or 86_400)
            common = {"session_policy": session_policy,
                      "bar_interval_seconds": bar_s,
                      "predicted_half_life_seconds": pred_hl,
                      "baseline_semantics": "first_eligible_bar_close"}
            try:
                signal_ts = to_utc(signal_raw, source_timezone="UTC")
            except (TimezoneNormalizationError, ValueError, TypeError) as e:
                finish(cid, aid, "TIMESTAMP_PARSE_ERROR", str(e)[:200], **common)
                continue
            common["signal_timestamp"] = signal_ts.isoformat()

            try:
                df = fetch_fn(symbol)
            except Exception as e:              # provider boundary — retryable
                finish(cid, aid, "PROVIDER_ERROR", str(e)[:200], **common)
                continue
            if df is None or len(df) == 0:
                finish(cid, aid, "MARKET_DATA_UNAVAILABLE", None, **common)
                continue

            closes = df["close"] if "close" in df.columns else df.iloc[:, 0]
            try:
                bar_ts = [to_utc(t, source_timezone=source_timezone)
                          for t in closes.index]
            except TimezoneNormalizationError as e:
                finish(cid, aid, "TIMEZONE_ERROR", str(e)[:200], **common)
                continue
            except (ValueError, TypeError) as e:
                finish(cid, aid, "TIMESTAMP_PARSE_ERROR", str(e)[:200], **common)
                continue

            # EXACT selection: first valid observation AT/AFTER the signal
            eligible = [(ts, float(v)) for ts, v in zip(bar_ts, closes)
                        if ts >= signal_ts]
            eligible = eligible[: horizon_bars + 1]
            if any(not (v > 0) or v != v for _, v in eligible):
                finish(cid, aid, "DATA_INTEGRITY_ERROR",
                       "non-finite or non-positive close", **common)
                continue
            if len(eligible) < min_forward_bars:
                finish(cid, aid, "INSUFFICIENT_FORWARD_BARS",
                       f"{len(eligible)} bars available — pending", **common,
                       bars_available=len(eligible))
                continue

            baseline_ts, baseline_px = eligible[0]
            common["baseline_timestamp"] = baseline_ts.isoformat()
            common["baseline_price"] = baseline_px
            common["bars_available"] = len(eligible)
            sign = 1.0 if str(direction) == "long" else -1.0
            curve = [(ts, sign * (px / baseline_px - 1.0))
                     for ts, px in eligible[1:]]
            horizon_s = (curve[-1][0] - signal_ts).total_seconds()
            common["max_observation_horizon_seconds"] = horizon_s

            peak_val = max(v for _, v in curve)
            if peak_val <= 0:                   # never developed — explicit
                finish(cid, aid, "NO_REALIZED_SIGNAL_PEAK", None, **common)
                continue
            peak_i = next(i for i, (_, v) in enumerate(curve) if v == peak_val)
            peak_s = (curve[peak_i][0] - signal_ts).total_seconds()
            common["peak_time_seconds"] = peak_s
            half_ts = None
            for ts, v in curve[peak_i:]:
                if v <= peak_val / 2:
                    half_ts = ts
                    break
            if half_ts is None:
                if len(eligible) <= horizon_bars:   # more bars may still come
                    finish(cid, aid, "INSUFFICIENT_FORWARD_BARS",
                           "not yet decayed within available bars", **common)
                else:                               # full horizon observed
                    finish(cid, aid, "HALF_LIFE_NOT_OBSERVED", None, **common)
                continue
            realized_s = (half_ts - signal_ts).total_seconds()
            self.record_realized_decay(cid, realized_s)
            finish(cid, aid, "SUCCESS", None, **common,
                   realized_half_life_seconds=realized_s)

        measured = counts.get("SUCCESS", 0)
        return {"measured": measured,
                "skipped": sum(counts.values()) - measured,
                "candidates": len(rows), "statuses": counts}

    def decay_exclusion_report(self) -> Dict[str, int]:
        """All measurement outcomes by status — the calibration denominator is
        never hidden (spec §24-25)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT measurement_status, COUNT(*) FROM decay_measurements "
                "GROUP BY measurement_status").fetchall()
        return {r[0]: r[1] for r in rows}

    def resolve_by_position(self, position_id: int, *,
                            realized_net_return: float,
                            realized_net_pnl: float,
                            realized_execution_cost_bps: Optional[float] = None,
                            realized_holding_hours: Optional[float] = None,
                            realized_mfe_pct: Optional[float] = None,
                            realized_mae_pct: Optional[float] = None,
                            realized_fee_bps: Optional[float] = None,
                            realized_slippage_bps: Optional[float] = None,
                            close_reason: Optional[str] = None) -> bool:
        """Canonical identity: evidence links prediction → position → trade
        memory by position_id, surviving restarts (spec §8, §14)."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT candidate_id FROM paper_evidence WHERE position_id=? "
                "AND resolved_at IS NULL AND status='PENDING'",
                (position_id,)).fetchone()
        if not row:
            return False
        return self.resolve(row[0], realized_net_return=realized_net_return,
                            realized_net_pnl=realized_net_pnl,
                            realized_execution_cost_bps=realized_execution_cost_bps,
                            realized_holding_hours=realized_holding_hours,
                            realized_mfe_pct=realized_mfe_pct,
                            realized_mae_pct=realized_mae_pct,
                            realized_fee_bps=realized_fee_bps,
                            realized_slippage_bps=realized_slippage_bps,
                            close_reason=close_reason)

    def reconcile(self) -> Dict[str, Any]:
        """Restart/crash recovery (spec §16): resolve PENDING evidence whose
        position already closed in trade_memory; flag orphans. Missing
        predictions are NEVER fabricated. Schema failures raise loudly —
        broken accounting must never look like clean reconciliation."""
        from core.trade_history import _run_query, compute_trade_net_return
        resolved = 0
        flags: List[str] = []
        pending = _run_query(
            self.db_path,
            "SELECT pe.candidate_id, pe.position_id, tm.net_pnl, "
            "tm.size_dollars, tm.holding_hours, tm.close_reason "
            "FROM paper_evidence pe JOIN trade_memory tm "
            "ON tm.position_id = pe.position_id "
            "WHERE pe.status='PENDING' AND pe.resolved_at IS NULL "
            "AND tm.exit_time IS NOT NULL")
        orphan_rows = _run_query(
            self.db_path,
            "SELECT COUNT(*) FROM trade_memory tm "
            "JOIN trade_attribution ta ON ta.trade_memory_id = tm.id "
            "WHERE tm.exit_time IS NOT NULL AND tm.position_id IS NOT NULL "
            "AND tm.position_id NOT IN "
            "(SELECT position_id FROM paper_evidence "
            " WHERE position_id IS NOT NULL)")
        orphan_trades = orphan_rows[0][0] if orphan_rows else 0
        for cid, pos_id, pnl, size, hold, reason in pending:
            ret = compute_trade_net_return(pnl, size)
            if ret is None:
                flags.append(f"EVIDENCE_RECONCILIATION_REQUIRED:{cid}")
                continue
            if self.resolve(cid, realized_net_return=ret, realized_net_pnl=pnl,
                            realized_holding_hours=hold, close_reason=reason):
                resolved += 1
        if orphan_trades:
            flags.append(f"CLOSED_TRADES_WITHOUT_EVIDENCE:{orphan_trades}")
        return {"resolved": resolved, "flags": flags,
                "orphan_closed_trades": orphan_trades}

    def _resolved(self, where: str = "", args: Sequence = ()) -> List[tuple]:
        q = ("SELECT alpha_id, family, regime, predicted_net_ev, "
             "predicted_p_positive, predicted_dollar_alpha, "
             "predicted_half_life_bars, predicted_execution_cost_bps, "
             "predicted_survival, realized_net_return, realized_net_pnl, "
             "realized_execution_cost_bps, realized_holding_hours "
             "FROM paper_evidence WHERE resolved_at IS NOT NULL "
             "AND status='RESOLVED' ")
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(q + where, list(args)).fetchall()

    # ── Calibration reports (spec §54-59) ─────────────────────────────────────

    def ev_calibration(self, min_per_bucket: int = 10) -> Dict[str, Any]:
        rows = self._resolved()
        buckets: Dict[str, Dict[str, Any]] = {}
        for lo, hi in EV_BUCKETS_BPS:
            preds, reals = [], []
            for r in rows:
                p = (r[3] or 0.0) * 10_000
                if lo <= p < hi and r[9] is not None:
                    preds.append(r[3])
                    reals.append(r[9])
            key = f"{lo}-{hi if hi < 10_000 else '+'}bps"
            if len(preds) < min_per_bucket:
                buckets[key] = {"n": len(preds), "verdict": "insufficient_sample"}
                continue
            pm, rm = st.mean(preds), st.mean(reals)
            buckets[key] = {
                "n": len(preds), "predicted_mean": pm, "realized_mean": rm,
                "calibration_error": pm - rm,
                "calibrated": abs(pm - rm) <= max(abs(pm) * 0.5, 0.0005),
            }
        return {"buckets": buckets, "total_resolved": len(rows)}

    def probability_calibration(self, min_per_bucket: int = 10) -> Dict[str, Any]:
        rows = self._resolved()
        out: Dict[str, Any] = {}
        for lo, hi in PROB_BUCKETS:
            sample = [r for r in rows
                      if r[4] is not None and lo <= r[4] < hi
                      and r[9] is not None]
            key = f"{lo:.0%}-{min(hi, 1.0):.0%}"
            if len(sample) < min_per_bucket:
                out[key] = {"n": len(sample), "verdict": "insufficient_sample"}
                continue
            actual = sum(1 for r in sample if r[9] > 0) / len(sample)
            predicted = st.mean(r[4] for r in sample)
            out[key] = {"n": len(sample), "predicted_p": predicted,
                        "actual_positive_rate": actual,
                        "calibrated": abs(predicted - actual) <= 0.15}
        return out

    def execution_calibration(self, min_samples: int = 10) -> Dict[str, Any]:
        rows = [r for r in self._resolved()
                if r[7] is not None and r[11] is not None]
        if len(rows) < min_samples:
            return {"n": len(rows), "verdict": "insufficient_sample"}
        pred = st.mean(r[7] for r in rows)
        real = st.mean(r[11] for r in rows)
        return {"n": len(rows), "predicted_cost_bps": pred,
                "realized_cost_bps": real,
                "cost_underestimate_bps": real - pred,
                "calibrated": abs(real - pred) <= max(pred * 0.5, 2.0)}

    def dollar_alpha_calibration(self, min_samples: int = 10) -> Dict[str, Any]:
        by_alpha: Dict[str, Dict[str, float]] = {}
        for r in self._resolved():
            if r[5] is None or r[10] is None:
                continue
            d = by_alpha.setdefault(r[0], {"predicted": 0.0, "realized": 0.0,
                                           "n": 0})
            d["predicted"] += r[5]
            d["realized"] += r[10]
            d["n"] += 1
        return {a: {**v, "capture_ratio": (v["realized"] / v["predicted"]
                                           if v["predicted"] else None),
                    "meaningful": v["n"] >= min_samples}
                for a, v in by_alpha.items()}

    def half_life_calibration(self, min_samples: int = 10) -> Dict[str, Any]:
        """Predicted vs REALIZED SIGNAL DECAY in canonical seconds. Holding
        time is NOT signal decay — without measured decay the honest answer
        is REALIZED_HALF_LIFE_UNAVAILABLE (spec fix #4)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT predicted_half_life_seconds, realized_half_life_seconds "
                "FROM paper_evidence WHERE resolved_at IS NOT NULL AND "
                "predicted_half_life_seconds IS NOT NULL AND "
                "realized_half_life_seconds IS NOT NULL").fetchall()
        if len(rows) < min_samples:
            return {"n": len(rows),
                    "verdict": "REALIZED_HALF_LIFE_UNAVAILABLE"}
        pred_s = st.mean(r[0] for r in rows)
        real_s = st.mean(r[1] for r in rows)
        return {"n": len(rows), "predicted_half_life_seconds": pred_s,
                "realized_half_life_seconds": real_s,
                "ratio": real_s / pred_s if pred_s else None,
                "unit": "seconds"}

    def holding_vs_half_life(self, min_samples: int = 10) -> Dict[str, Any]:
        """Diagnostic only: how holding time compares to predicted half-life.
        This is NOT decay calibration."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT predicted_half_life_seconds, realized_holding_hours "
                "FROM paper_evidence WHERE resolved_at IS NOT NULL AND "
                "predicted_half_life_seconds IS NOT NULL AND "
                "realized_holding_hours IS NOT NULL").fetchall()
        if len(rows) < min_samples:
            return {"n": len(rows), "verdict": "insufficient_sample"}
        return {"n": len(rows),
                "mean_holding_seconds": st.mean(r[1] * 3600 for r in rows),
                "mean_predicted_half_life_seconds": st.mean(r[0] for r in rows),
                "note": "diagnostic — holding time is not signal decay"}

    def calibration_slope(self, min_samples: int = 20) -> Dict[str, Any]:
        """realized ~ predicted regression: ideal slope ≈ 1 (spec §33)."""
        rows = [(r[3], r[9]) for r in self._resolved()
                if r[3] is not None and r[9] is not None]
        if len(rows) < min_samples:
            return {"n": len(rows), "verdict": "insufficient_sample"}
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        mx, my = st.mean(xs), st.mean(ys)
        var = sum((x - mx) ** 2 for x in xs)
        slope = (sum((x - mx) * (y - my) for x, y in rows) / var
                 if var > 0 else None)
        return {"n": len(rows), "slope": slope,
                "predicted_mean": mx, "realized_mean": my}

    def brier_score(self, min_samples: int = 20) -> Dict[str, Any]:
        rows = [(r[4], 1.0 if r[9] > 0 else 0.0) for r in self._resolved()
                if r[4] is not None and r[9] is not None]
        if len(rows) < min_samples:
            return {"n": len(rows), "verdict": "insufficient_sample"}
        return {"n": len(rows),
                "brier": st.mean((p - o) ** 2 for p, o in rows)}

    def survival_calibration(self, min_per_bucket: int = 10) -> Dict[str, Any]:
        rows = [r for r in self._resolved() if r[8] is not None
                and r[9] is not None]
        high = [r[9] for r in rows if r[8] >= 0.6]
        low = [r[9] for r in rows if r[8] < 0.4]
        if len(high) < min_per_bucket or len(low) < min_per_bucket:
            return {"n_high": len(high), "n_low": len(low),
                    "verdict": "insufficient_sample"}
        return {"n_high": len(high), "n_low": len(low),
                "high_survival_mean_return": st.mean(high),
                "low_survival_mean_return": st.mean(low),
                "ordering_correct": st.mean(high) > st.mean(low)}

    # ── Family report + funnel yields (spec §65-68) ───────────────────────────

    def family_report(self) -> Dict[str, Dict[str, Any]]:
        by_family: Dict[str, List[tuple]] = {}
        for r in self._resolved():
            by_family.setdefault(r[1] or "unknown", []).append(r)
        out = {}
        for family, rows in by_family.items():
            reals = [r[9] for r in rows if r[9] is not None]
            preds = [r[3] for r in rows if r[3] is not None]
            if not reals:
                continue
            sd = st.pstdev(reals) if len(reals) > 1 else 0.0
            out[family] = {
                "paper_trades": len(rows),
                "predicted_ev_mean": st.mean(preds) if preds else None,
                "realized_ev_mean": st.mean(reals),
                "realized_pnl_usd": sum(r[10] or 0.0 for r in rows),
                "sharpe_proxy": st.mean(reals) / sd if sd > 0 else None,
                "meaningful": len(rows) >= MIN_RESOLVED_PER_FAMILY,
            }
        return out

    def forward_yield(self) -> Dict[str, Any]:
        """forward-profitable PAPER alphas / resolved alphas, per family."""
        by_alpha: Dict[str, List[float]] = {}
        fam_of: Dict[str, str] = {}
        for r in self._resolved():
            if r[9] is not None:
                by_alpha.setdefault(r[0], []).append(r[9])
                fam_of[r[0]] = r[1] or "unknown"
        by_family: Dict[str, Dict[str, int]] = {}
        for alpha_id, rets in by_alpha.items():
            if len(rets) < MIN_RESOLVED_PER_ALPHA:
                continue
            f = by_family.setdefault(fam_of[alpha_id],
                                     {"resolved_alphas": 0, "profitable": 0})
            f["resolved_alphas"] += 1
            if st.mean(rets) > 0:
                f["profitable"] += 1
        return {fam: {**v, "forward_yield": (v["profitable"] / v["resolved_alphas"]
                                             if v["resolved_alphas"] else None)}
                for fam, v in by_family.items()}

    def update_research_roi(self, ledger) -> None:
        """Forward PAPER results feed research ROI: pretty backtests with poor
        forward P&L lose budget (spec §69); exploration keeps them alive."""
        for family, stats in self.family_report().items():
            if stats["meaningful"]:
                ledger.record_outcome(
                    family, "forward", stats["paper_trades"],
                    forward_pnl=stats["realized_pnl_usd"])

    def evidence_health_report(self) -> Dict[str, Any]:
        """PAPER evidence health: full lifecycle + decay pipeline counts."""
        decay = self.decay_exclusion_report()
        with sqlite3.connect(self.db_path) as conn:
            orphaned = conn.execute(
                "SELECT COUNT(*) FROM paper_evidence WHERE status='PENDING' "
                "AND position_id IS NULL AND recorded_at < datetime('now','-1 day')"
            ).fetchone()[0]
        return {
            "status_counts": self.status_counts(),
            "orphaned_pending": orphaned,
            "decay_successful": decay.get("SUCCESS", 0),
            "decay_pending": sum(decay.get(s, 0) for s in self.DECAY_RETRYABLE),
            "decay_failed": sum(v for k, v in decay.items()
                                if k != "SUCCESS" and k not in self.DECAY_RETRYABLE),
            "decay_by_status": decay,
        }

    def snapshot_daily(self) -> None:
        """Persist the daily calibration snapshot (spec §43-45)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_snapshots (
                    snapshot_date TEXT PRIMARY KEY,
                    resolved_count INTEGER,
                    ev_slope REAL, brier REAL,
                    exec_cost_error_bps REAL,
                    half_life_samples INTEGER, half_life_ratio REAL,
                    net_pnl_usd REAL,
                    report_json TEXT
                )""")
            slope = self.calibration_slope()
            brier = self.brier_score()
            execution = self.execution_calibration()
            hl = self.half_life_calibration()
            resolved = self._resolved()
            conn.execute(
                "INSERT INTO calibration_snapshots (snapshot_date, "
                "resolved_count, ev_slope, brier, exec_cost_error_bps, "
                "half_life_samples, half_life_ratio, net_pnl_usd, report_json) "
                "VALUES (date('now'),?,?,?,?,?,?,?,?) "
                "ON CONFLICT(snapshot_date) DO UPDATE SET "
                "resolved_count=excluded.resolved_count, "
                "ev_slope=excluded.ev_slope, brier=excluded.brier, "
                "exec_cost_error_bps=excluded.exec_cost_error_bps, "
                "half_life_samples=excluded.half_life_samples, "
                "half_life_ratio=excluded.half_life_ratio, "
                "net_pnl_usd=excluded.net_pnl_usd, "
                "report_json=excluded.report_json",
                (len(resolved), slope.get("slope"), brier.get("brier"),
                 execution.get("cost_underestimate_bps"),
                 hl.get("n", 0), hl.get("ratio"),
                 sum(r[10] or 0.0 for r in resolved),
                 json.dumps({"ev": self.ev_calibration(),
                             "decay": self.decay_exclusion_report()},
                            default=str)))

    def model_reliability_report(self) -> Dict[str, str]:
        """INSUFFICIENT_EVIDENCE / POORLY_CALIBRATED / ACCEPTABLE / STRONG per
        model — measurement only, never a bypass of hard risk gates."""
        def grade(n, ok, strong=False):
            if n < MIN_RESOLVED_PER_ALPHA:
                return "INSUFFICIENT_EVIDENCE"
            if not ok:
                return "POORLY_CALIBRATED"
            return "STRONG" if strong else "ACCEPTABLE"

        slope = self.calibration_slope()
        s_val = slope.get("slope")
        ev_ok = s_val is not None and 0.5 <= s_val <= 1.5
        ev_strong = s_val is not None and 0.8 <= s_val <= 1.2
        execution = self.execution_calibration()
        exec_ok = execution.get("calibrated", False)
        hl = self.half_life_calibration()
        hl_ok = hl.get("ratio") is not None and 0.5 <= (hl.get("ratio") or 0) <= 2.0
        surv = self.survival_calibration()
        return {
            "EconomicEVModel": grade(slope.get("n", 0), ev_ok, ev_strong),
            "ExecutionModel": grade(execution.get("n", 0), exec_ok),
            "OpportunityHalfLifeEstimator": grade(hl.get("n", 0), hl_ok),
            "EdgeSurvivalModel": grade(
                surv.get("n_high", 0) + surv.get("n_low", 0),
                surv.get("ordering_correct", False)),
            "MetaAlphaShadow": "INSUFFICIENT_EVIDENCE",   # promoted by gate only
        }

    def review_status(self, alpha_id: str) -> str:
        """Even excellent paper evidence yields REVIEW, never auto-live."""
        rows = [r for r in self._resolved("AND alpha_id=?", (alpha_id,))
                if r[9] is not None]
        if len(rows) < MIN_RESOLVED_PER_ALPHA:
            return "INSUFFICIENT_EVIDENCE"
        mean_ret = st.mean(r[9] for r in rows)
        if mean_ret > 0:
            return "LIVE_ELIGIBLE_REVIEW"     # human review — never auto-live
        return "UNDERPERFORMING"

    def attach_position(self, candidate_id: str, position_id: int,
                        filled_notional_usd: Optional[float] = None) -> None:
        """Link the prediction to the actual filled position (partial fills
        store the FILLED notional — spec §7-8)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE paper_evidence SET position_id=?, filled_notional_usd=? "
                "WHERE candidate_id=?",
                (position_id, filled_notional_usd, candidate_id))

    def status_counts(self) -> Dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT COALESCE(status,'PENDING'), COUNT(*) "
                                "FROM paper_evidence GROUP BY status").fetchall()
        return {r[0]: r[1] for r in rows}

    def daily_report(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            today = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(realized_net_pnl),0) FROM "
                "paper_evidence WHERE substr(resolved_at,1,10)=date('now')"
            ).fetchone()
            new_today = conn.execute(
                "SELECT COUNT(*) FROM paper_evidence "
                "WHERE substr(recorded_at,1,10)=date('now')").fetchone()[0]
        return {
            "date": _utcnow()[:10],
            "status_counts": self.status_counts(),
            "new_predictions_today": new_today,
            "closed_today": today[0],
            "net_pnl_today_usd": today[1],
            "ev_calibration": self.ev_calibration(),
            "execution_calibration": self.execution_calibration(),
        }

    def weekly_report(self) -> Dict[str, Any]:
        rets = [r[9] for r in self._resolved() if r[9] is not None]
        sd = st.pstdev(rets) if len(rets) > 1 else 0.0
        return {
            "resolved_total": len(rets),
            "net_pnl_usd": sum(r[10] or 0.0 for r in self._resolved()),
            "mean_net_return": st.mean(rets) if rets else None,
            "sharpe_proxy": (st.mean(rets) / sd if rets and sd > 0 else None),
            "ev_calibration": self.ev_calibration(),
            "probability_calibration": self.probability_calibration(),
            "calibration_slope": self.calibration_slope(),
            "brier": self.brier_score(),
            "dollar_alpha_calibration": self.dollar_alpha_calibration(),
            "execution_calibration": self.execution_calibration(),
            "half_life_calibration": self.half_life_calibration(),
            "survival_calibration": self.survival_calibration(),
            "family_report": self.family_report(),
            "forward_yield": self.forward_yield(),
        }


    def summary_report(self, days: int = 30) -> str:
        """The report that matters: predicted vs realized, plain text."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT family, predicted_net_ev, realized_net_return, "
                "predicted_execution_cost_bps, realized_execution_cost_bps, "
                "predicted_dollar_alpha, realized_net_pnl "
                "FROM paper_evidence WHERE status='RESOLVED' AND "
                "resolved_at >= datetime('now', ?)", (f"-{days} day",)).fetchall()
        lines = [f"PAPER — {days} DAYS", "",
                 f"Resolved trades:        {len(rows)}"]
        if len(rows) < MIN_RESOLVED_PER_ALPHA:
            lines.append("Status:                 INSUFFICIENT_FORWARD_EVIDENCE")
            return "\n".join(lines)
        preds = [r[1] for r in rows if r[1] is not None]
        reals = [r[2] for r in rows if r[2] is not None]
        lines += [
            "",
            f"Predicted mean EV:      {st.mean(preds) * 10_000:+.1f} bps",
            f"Realized mean return:   {st.mean(reals) * 10_000:+.1f} bps",
            "",
            f"EV calibration slope:   {self.calibration_slope().get('slope')}",
            f"Brier score:            {self.brier_score().get('brier')}",
        ]
        pc = [r[3] for r in rows if r[3] is not None]
        rc = [r[4] for r in rows if r[4] is not None]
        if pc and rc:
            lines += ["",
                      f"Predicted execution cost: {st.mean(pc):.1f} bps",
                      f"Actual execution cost:    {st.mean(rc):.1f} bps"]
        pd_ = sum(r[5] or 0.0 for r in rows)
        rp = sum(r[6] or 0.0 for r in rows)
        lines += ["",
                  f"Expected dollar Alpha:  ${pd_:,.0f}",
                  f"Realized net P&L:       ${rp:,.0f}", ""]
        for fam, stats in sorted(self.family_report().items()):
            if stats.get("predicted_ev_mean") is None:
                continue
            lines += [f"{fam}:",
                      f"  pred {stats['predicted_ev_mean'] * 10_000:+.0f} bps",
                      f"  real {stats['realized_ev_mean'] * 10_000:+.0f} bps"]
        surv = self.survival_calibration()
        if "ordering_correct" in surv:
            lines += ["", "Edge survival: "
                      + ("high-survival bucket materially better"
                         if surv["ordering_correct"] else
                         "WARNING — survival ordering NOT confirmed")]
        return "\n".join(lines)


def alpha_capture_analysis(*, gross_alpha_bps: float,
                           execution_cost_bps: float,
                           delay_cost_bps: float = 0.0,
                           capacity_loss_bps: float = 0.0,
                           realized_net_bps: float) -> Dict[str, float]:
    """Where the theoretical edge was lost (spec §63-64, test §92)."""
    accounted = gross_alpha_bps - execution_cost_bps - delay_cost_bps \
        - capacity_loss_bps
    return {
        "theoretical_gross_bps": gross_alpha_bps,
        "execution_cost_bps": execution_cost_bps,
        "delay_cost_bps": delay_cost_bps,
        "capacity_loss_bps": capacity_loss_bps,
        "expected_captured_bps": accounted,
        "realized_net_bps": realized_net_bps,
        "unexplained_bps": realized_net_bps - accounted,
        "alpha_capture_ratio": (realized_net_bps / gross_alpha_bps
                                if gross_alpha_bps else None),
    }
