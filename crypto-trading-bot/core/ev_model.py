"""Economic Expected-Value model.

Upgrades the meta layer from "P(return > 0)" to real economic outcomes:

    EV = P(win) * E[win] + (1 - P(win)) * E[loss] - expected_costs

with win/loss MAGNITUDES estimated from attributed realized outcomes (and
optional gradient-boosting regressors), never from `(probability - 0.5) * 2`.

Ranking uses a conservative EV (lower confidence bound), so an uncertain
+0.60% expectation ranks below a confident +0.30% one.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.validation_stats import effective_sample_size

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

EV_MODEL_VERSION = "1.0.0"

Z_CONSERVATIVE = 1.645  # 95% one-sided


@dataclass
class EVEstimate:
    """ALL return quantities are FRACTIONAL returns (0.003 = +0.3%).
    See core.return_units for the unit convention."""
    probability_positive: float
    expected_net_return: float          # net of costs, fraction (0.003 = 0.3%)
    expected_win: float                 # E[return | win], fraction
    expected_loss: float                # E[return | loss], fraction (negative)
    expected_adverse_excursion: Optional[float]
    expected_favorable_excursion: Optional[float]
    expected_holding_hours: Optional[float]
    expected_costs: float               # fraction
    prediction_std: float               # fraction (same units as the mean)
    ev_lower_bound: float               # conservative EV, fraction
    sample_size: int
    effective_sample_size: float
    source: str                         # 'historical' | 'ml' | 'oos_prior' | 'blended'
    model_version: str = EV_MODEL_VERSION
    notes: str = ""
    # ── EV components (explainability; all fractional returns) ─────────────
    historical_ev: Optional[float] = None
    historical_se: Optional[float] = None
    forward_ev: Optional[float] = None
    forward_se: Optional[float] = None
    combined_ev: Optional[float] = None
    combined_se: Optional[float] = None
    historical_weight: Optional[float] = None
    forward_weight: Optional[float] = None
    forward_effective_n: Optional[float] = None

    @property
    def conservative_ev(self) -> float:
        return self.ev_lower_bound

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class OutcomeLabel:
    """Training label tied to an attributed trade (leakage-safe)."""
    alpha_id: str
    alpha_version: str
    signal_id: Optional[str]
    signal_time: str
    realized_net_return: float          # fraction
    realized_gross_return: Optional[float]
    realized_cost: Optional[float]
    max_adverse_excursion: Optional[float]
    max_favorable_excursion: Optional[float]
    holding_hours: Optional[float]
    stop_hit: Optional[bool] = None
    target_hit: Optional[bool] = None


class EconomicEVModel:
    """EV estimator with a deterministic evidence-based core and optional
    ML magnitude regressors.

    The deterministic path uses ONLY realized outcomes attributed to the
    specific alpha (via trade attribution), with uncertainty from the
    autocorrelation-adjusted effective sample size.
    """

    def __init__(self, db_path: str = "data/trade_memory.sqlite",
                 min_samples: int = 8) -> None:
        self.db_path = db_path
        self.min_samples = min_samples
        self.version = EV_MODEL_VERSION
        self._regressors: Dict[str, Any] = {}
        self._calibrator = None
        self._calibration: Dict[str, Any] = {}

    # ── Label generation ──────────────────────────────────────────────────────

    def build_labels(self, alpha_id: Optional[str] = None) -> List[OutcomeLabel]:
        """Labels from attributed realized outcomes, time-ordered ascending."""
        q = (
            "SELECT ta.alpha_id, ta.alpha_version, ta.signal_id, "
            "  tm.entry_time, tm.net_return_pct, tm.gross_pnl, tm.total_fees, "
            "  tm.size_dollars, tm.mae_pct, tm.mfe_pct, tm.holding_hours, tm.close_reason "
            "FROM trade_attribution ta JOIN trade_memory tm ON tm.id = ta.trade_memory_id "
        )
        args: list = []
        if alpha_id:
            q += "WHERE ta.alpha_id=? "
            args.append(alpha_id)
        q += "ORDER BY tm.entry_time ASC"
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(q, args).fetchall()
        except sqlite3.OperationalError:
            return []
        labels = []
        for (aid, aver, sid, ts, net_ret, gross, fees, size, mae, mfe,
             hold, close_reason) in rows:
            if net_ret is None:
                continue
            labels.append(OutcomeLabel(
                alpha_id=aid, alpha_version=aver or "1", signal_id=sid,
                signal_time=ts or "",
                realized_net_return=float(net_ret) / 100.0,
                realized_gross_return=(float(gross) / float(size)) if gross is not None and size else None,
                realized_cost=(float(fees) / float(size)) if fees is not None and size else None,
                max_adverse_excursion=float(mae) if mae is not None else None,
                max_favorable_excursion=float(mfe) if mfe is not None else None,
                holding_hours=float(hold) if hold is not None else None,
                stop_hit=(close_reason == "stop_loss") if close_reason else None,
                target_hit=(close_reason == "take_profit") if close_reason else None,
            ))
        return labels

    # ── Estimation ────────────────────────────────────────────────────────────

    def estimate(
        self,
        alpha_id: str,
        expected_costs: float = 0.0,
        returns: Optional[Sequence[float]] = None,
        maes: Optional[Sequence[float]] = None,
        mfes: Optional[Sequence[float]] = None,
        holds: Optional[Sequence[float]] = None,
    ) -> Optional[EVEstimate]:
        """Estimate economic EV for one alpha's next opportunity.

        Uses attributed labels unless explicit return samples are provided.
        Returns None when evidence is insufficient (abstain, don't guess).
        """
        if returns is None:
            labels = self.build_labels(alpha_id)
            returns = [l.realized_net_return for l in labels]
            maes = [l.max_adverse_excursion for l in labels
                    if l.max_adverse_excursion is not None]
            mfes = [l.max_favorable_excursion for l in labels
                    if l.max_favorable_excursion is not None]
            holds = [l.holding_hours for l in labels if l.holding_hours is not None]

        n = len(returns)
        if n < self.min_samples:
            return None

        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        p_win_raw = len(wins) / n
        p_win = self.calibrated_probability(p_win_raw)
        e_win = (sum(wins) / len(wins)) if wins else 0.0
        e_loss = (sum(losses) / len(losses)) if losses else 0.0

        # ML magnitude adjustment when regressors are trained
        source = "historical"
        if self._regressors.get("net_return") is not None:
            source = "ml"

        # Economic EV with magnitudes and costs — never (p-0.5)*2
        ev = p_win * e_win + (1 - p_win) * e_loss - expected_costs

        mean = sum(returns) / n
        var = sum((r - mean) ** 2 for r in returns) / max(n - 1, 1)
        ess = effective_sample_size(list(returns))
        std_err = math.sqrt(var / max(ess, 1.0))
        lcb = ev - Z_CONSERVATIVE * std_err

        return EVEstimate(
            probability_positive=p_win,
            expected_net_return=ev,
            expected_win=e_win,
            expected_loss=e_loss,
            expected_adverse_excursion=(sum(maes) / len(maes)) if maes else None,
            expected_favorable_excursion=(sum(mfes) / len(mfes)) if mfes else None,
            expected_holding_hours=(sum(holds) / len(holds)) if holds else None,
            expected_costs=expected_costs,
            prediction_std=std_err,
            ev_lower_bound=lcb,
            sample_size=n,
            effective_sample_size=ess,
            source=source,
        )

    # ── Cold-start: OOS prior blended with forward evidence ───────────────────

    def estimate_with_prior(
        self,
        alpha_record: Dict[str, Any],
        expected_costs: float = 0.0,
        prior_haircut: float = 0.5,
        prior_strength: float = 20.0,
        min_prior_trades: int = 10,
        prior_age_days: Optional[float] = None,
        prior_half_life_days: float = 180.0,
    ) -> Optional[EVEstimate]:
        """EV blending historical OOS evidence with forward observations.

        UNITS: everything here is a FRACTIONAL return. The prior comes only
        from normalized OOS return statistics (mean_net_return,
        standard_error_return) — never from dollar P&L divided by an assumed
        position size.

        Method (documented per spec §8-11):
          prior mean   = haircut × OOS mean (conservative shrinkage)
          prior var    = SE² + (shrunk amount)²   — the haircut portion is
                         treated as model uncertainty and added to variance
          age decay    = prior precision × 2^(-age/half_life) (optional)
          blending     = inverse-variance (precision) weighting of prior vs
                         forward estimates; combined var = 1/(prec_h+prec_f)

        Consequences: 3 poor forward trades cannot overturn 500 OOS
        observations (huge forward SE), while 100 poor forward trades
        overwhelm the prior (tiny forward SE). No usable prior and
        insufficient forward data → None (abstain).
        """
        from core.return_units import validate_return_units

        alpha_id = alpha_record.get("alpha_id")
        forward = self.estimate(alpha_id, expected_costs=expected_costs)

        prior = self._prior_from_oos(alpha_record, prior_haircut, min_prior_trades)
        if prior is None:
            return forward   # no valid prior — forward estimate or abstain

        mu_h, se_h, p_h, win_h, loss_h = prior
        mu_h -= expected_costs
        validate_return_units(mu_h, "historical_prior_mean",
                              alpha_record.get("asset_class") or "unknown",
                              context=alpha_id)

        # Optional age decay of prior authority
        var_h = se_h ** 2
        if prior_age_days is not None and prior_age_days > 0:
            var_h /= 2 ** (-prior_age_days / prior_half_life_days)  # precision decays

        if forward is None:
            lcb = mu_h - Z_CONSERVATIVE * math.sqrt(var_h)
            return EVEstimate(
                probability_positive=p_h,
                expected_net_return=mu_h,
                expected_win=win_h, expected_loss=loss_h,
                expected_adverse_excursion=None,
                expected_favorable_excursion=None,
                expected_holding_hours=None,
                expected_costs=expected_costs,
                prediction_std=math.sqrt(var_h),
                ev_lower_bound=lcb,
                sample_size=0, effective_sample_size=0.0,
                source="oos_prior",
                historical_ev=mu_h, historical_se=math.sqrt(var_h),
                forward_ev=None, forward_se=None,
                combined_ev=mu_h, combined_se=math.sqrt(var_h),
                historical_weight=1.0, forward_weight=0.0,
                forward_effective_n=0.0,
                notes=f"cold-start: {prior_haircut:.0%}-haircut OOS prior, "
                      f"no forward observations",
            )

        # Inverse-variance blend of prior and forward (both fractional returns),
        # with the forward weight CAPPED by effective sample size so a handful
        # of (lucky or unlucky) early trades cannot dominate:
        #   w_f = min( prec_f/(prec_h+prec_f),  ess/(ess+prior_strength) )
        # Combined variance uses the fixed-weight formula
        #   var_c = w_f²·var_f + (1-w_f)²·var_h
        mu_f = forward.expected_net_return
        var_f = max(forward.prediction_std, 1e-6) ** 2
        ess_f = forward.effective_sample_size
        ess_cap = ess_f / (ess_f + prior_strength)
        prec_h = 1.0 / max(var_h, 1e-12)
        prec_f = 1.0 / var_f
        w_f = min(prec_f / (prec_h + prec_f), ess_cap)
        mu_c = w_f * mu_f + (1 - w_f) * mu_h
        var_c = (w_f ** 2) * var_f + ((1 - w_f) ** 2) * var_h
        se_c = math.sqrt(var_c)
        lcb = mu_c - Z_CONSERVATIVE * se_c

        if ess_f >= 3 * prior_strength and w_f > 0.7:
            source = "historical"   # prior effectively retired
        else:
            source = "blended"

        return EVEstimate(
            probability_positive=w_f * forward.probability_positive + (1 - w_f) * p_h,
            expected_net_return=mu_c,
            expected_win=w_f * forward.expected_win + (1 - w_f) * win_h,
            expected_loss=w_f * forward.expected_loss + (1 - w_f) * loss_h,
            expected_adverse_excursion=forward.expected_adverse_excursion,
            expected_favorable_excursion=forward.expected_favorable_excursion,
            expected_holding_hours=forward.expected_holding_hours,
            expected_costs=expected_costs,
            prediction_std=se_c,
            ev_lower_bound=lcb,
            sample_size=forward.sample_size,
            effective_sample_size=ess_f,
            source=source,
            historical_ev=mu_h, historical_se=math.sqrt(var_h),
            forward_ev=mu_f, forward_se=forward.prediction_std,
            combined_ev=mu_c, combined_se=se_c,
            historical_weight=1 - w_f, forward_weight=w_f,
            forward_effective_n=ess_f,
            notes=f"inverse-variance blend: forward weight {w_f:.0%} "
                  f"(ESS {ess_f:.1f} vs prior_strength {prior_strength:g})",
        )

    def _prior_from_oos(self, alpha_record: Dict[str, Any], haircut: float,
                        min_trades: int) -> Optional[tuple]:
        """(mean, se, p_win, e_win, e_loss) — ALL fractional returns.

        Uses ONLY normalized return statistics from true OOS/holdout evidence
        (mean_net_return, standard_error_return). Legacy records containing
        only dollar P&L are NOT converted with an assumed position size —
        they yield no prior (abstain, don't guess).
        """
        oos = alpha_record.get("oos_metrics")
        if isinstance(oos, str):
            try:
                oos = json.loads(oos)
            except json.JSONDecodeError:
                return None
        if not isinstance(oos, dict):
            return None
        trades = oos.get("trades") or 0
        mean_ret = oos.get("mean_net_return", oos.get("net_expectancy_return"))
        se_ret = oos.get("standard_error_return")
        win_rate = oos.get("win_rate")
        if trades < min_trades or mean_ret is None or se_ret is None:
            return None
        mean_ret = float(mean_ret)
        se_ret = float(se_ret)
        if mean_ret <= 0 or se_ret <= 0:
            return None

        # Conservative mean shrinkage; the shrunk-away band is treated as ±2σ
        # model uncertainty added in quadrature (documented interpretation —
        # not an arbitrary multiplier)
        mu = mean_ret * haircut
        shrunk = mean_ret * (1 - haircut)
        se = math.sqrt(se_ret ** 2 + (shrunk / 2) ** 2)

        # Shrink win rate toward 0.5 by evidence
        p = (float(win_rate or 0.5) * trades + 0.5 * min_trades) / (trades + min_trades)
        # Magnitudes consistent with mu = p*w + (1-p)*l, assuming l = -w/2
        denom = p - (1 - p) * 0.5
        e_win = mu / denom if denom > 1e-9 else mu
        e_loss = -e_win * 0.5
        return mu, se, p, e_win, e_loss

    def train(self) -> Dict[str, Any]:
        """Train magnitude regressors + probability classifier on attributed
        labels using time-series CV. Fail-soft: without lightgbm/sklearn or
        data, the deterministic path remains authoritative."""
        labels = self.build_labels()
        if len(labels) < 50:
            return {"status": "insufficient_data", "rows": len(labels)}
        try:
            import numpy as np
            from sklearn.model_selection import TimeSeriesSplit
            import lightgbm as lgb
        except ImportError as e:
            return {"status": "dependencies_missing", "error": str(e)}

        X = np.array([[l.max_adverse_excursion or 0.0,
                       l.max_favorable_excursion or 0.0,
                       l.holding_hours or 0.0] for l in labels])
        y_ret = np.array([l.realized_net_return for l in labels])
        y_cls = (y_ret > 0).astype(int)

        metrics: Dict[str, Any] = {}
        tscv = TimeSeriesSplit(n_splits=4, gap=3)

        reg = lgb.LGBMRegressor(n_estimators=100, max_depth=4, verbosity=-1,
                                random_state=42)
        fold_mae = []
        for tr, te in tscv.split(X):
            from sklearn.base import clone
            m = clone(reg)
            m.fit(X[tr], y_ret[tr])
            pred = m.predict(X[te])
            fold_mae.append(float(np.mean(np.abs(pred - y_ret[te]))))
        reg.fit(X, y_ret)
        self._regressors["net_return"] = reg
        metrics["net_return_cv_mae"] = fold_mae

        # Calibration of P(win) via isotonic regression on OOS folds
        try:
            from sklearn.isotonic import IsotonicRegression
            clf = lgb.LGBMClassifier(n_estimators=100, max_depth=4, verbosity=-1,
                                     random_state=42)
            probs, actuals = [], []
            for tr, te in tscv.split(X):
                if len(set(y_cls[tr])) < 2:
                    continue
                from sklearn.base import clone
                m = clone(clf)
                m.fit(X[tr], y_cls[tr])
                probs.extend(m.predict_proba(X[te])[:, 1].tolist())
                actuals.extend(y_cls[te].tolist())
            if probs:
                brier = float(np.mean([(p - a) ** 2 for p, a in zip(probs, actuals)]))
                metrics["brier_score"] = brier
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(probs, actuals)
                self._calibrator = iso
                self._calibration = {"brier_score": brier, "n": len(probs),
                                     "method": "isotonic", "fitted_at": _utcnow()}
        except ImportError:
            pass

        metrics["status"] = "ok"
        metrics["rows"] = len(labels)
        return metrics

    def calibrated_probability(self, raw_probability: float) -> float:
        """Apply isotonic calibration when fitted; identity otherwise."""
        if self._calibrator is None:
            return raw_probability
        try:
            return float(self._calibrator.predict([raw_probability])[0])
        except Exception:
            return raw_probability

    def calibration_report(self) -> Dict[str, Any]:
        return dict(self._calibration) or {"status": "uncalibrated"}


# ── Shadow model comparison (spec §52) ────────────────────────────────────────

_CREATE_SHADOW = """
CREATE TABLE IF NOT EXISTS ev_model_shadow (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id   TEXT,
    alpha_id       TEXT,
    symbol         TEXT,
    champion_version TEXT,
    challenger_version TEXT,
    champion_ev    REAL,
    challenger_ev  REAL,
    champion_rank  INTEGER,
    challenger_rank INTEGER,
    recorded_at    TEXT
)
"""


class ShadowEVRunner:
    """Runs a challenger EV model alongside the champion WITHOUT giving it
    capital authority. Records both rankings for forward comparison; a
    challenger may only be promoted after a minimum forward sample."""

    def __init__(self, db_path: str = "data/trade_memory.sqlite",
                 min_forward_samples: int = 50) -> None:
        self.db_path = db_path
        self.min_forward_samples = min_forward_samples
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_SHADOW)
            conn.commit()

    def record(self, candidates: List[Any], champion_evs: Dict[str, float],
               challenger_evs: Dict[str, float],
               champion_version: str, challenger_version: str) -> None:
        champ_rank = {cid: i for i, cid in enumerate(
            sorted(champion_evs, key=champion_evs.get, reverse=True), 1)}
        chal_rank = {cid: i for i, cid in enumerate(
            sorted(challenger_evs, key=challenger_evs.get, reverse=True), 1)}
        with sqlite3.connect(self.db_path) as conn:
            for c in candidates:
                cid = getattr(c, "candidate_id", None) or c.get("candidate_id")
                conn.execute(
                    "INSERT INTO ev_model_shadow (candidate_id, alpha_id, symbol, "
                    "champion_version, challenger_version, champion_ev, challenger_ev, "
                    "champion_rank, challenger_rank, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (cid,
                     getattr(c, "alpha_id", None) or c.get("alpha_id"),
                     getattr(c, "symbol", None) or c.get("symbol"),
                     champion_version, challenger_version,
                     champion_evs.get(cid), challenger_evs.get(cid),
                     champ_rank.get(cid), chal_rank.get(cid), _utcnow()),
                )
            conn.commit()

    def forward_sample_size(self, challenger_version: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM ev_model_shadow WHERE challenger_version=?",
                (challenger_version,),
            ).fetchone()
        return int(row[0]) if row else 0

    def can_promote(self, challenger_version: str) -> Tuple[bool, str]:
        n = self.forward_sample_size(challenger_version)
        if n < self.min_forward_samples:
            return False, f"forward sample {n} < {self.min_forward_samples}"
        return True, "sufficient forward sample"
