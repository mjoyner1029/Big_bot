"""Modular discovery detectors.

Each detector scans historical data for one anomaly family and emits
STRUCTURED hypotheses whose entry conditions use the validated condition DSL
(machine-readable — feed straight into the experiment pipeline / alpha
library, never natural language).

Search discipline (spec §40-41):
    Stage 1 single-feature effects → Stage 2 pair interactions (only on
    promising branches), bounded by SearchLimits; every test is recorded for
    Benjamini-Hochberg FDR within its family.

Detectors that need data providers we don't have (earnings surprise, funding,
basis, open interest, news/filings) implement the interface but report
`requires_data` instead of fabricating data.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core.validation_stats import (
    HypothesisFamilyTracker,
    effective_sample_size,
    sign_test_p_value,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchLimits:
    max_features_per_hypothesis: int = 2
    max_parameter_variants: int = 6
    max_interaction_depth: int = 2
    max_hypotheses_per_family: int = 50
    minimum_effective_sample: int = 25
    stage2_min_stage1_expectancy: float = 0.0   # only expand positive branches


@dataclass
class DiscoveredHypothesis:
    """Structured, machine-readable hypothesis."""
    family: str
    subfamily: str
    symbol: str
    direction: str                        # 'long' | 'short'
    entry_conditions: List[Dict[str, Any]]   # condition-DSL dicts
    holding_bars: int
    sample_size: int
    effective_sample: float
    mean_return: float
    p_value: float
    description: str = ""
    hypothesis_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class DiscoveryDetector(ABC):
    """Interface every detector implements."""

    family: str = "UNKNOWN"
    requires_data: List[str] = []          # unavailable data providers, if any

    def __init__(self, limits: Optional[SearchLimits] = None,
                 tracker: Optional[HypothesisFamilyTracker] = None) -> None:
        self.limits = limits or SearchLimits()
        self.tracker = tracker or HypothesisFamilyTracker()

    @abstractmethod
    def scan(self, data: Dict[str, pd.DataFrame]) -> List[DiscoveredHypothesis]:
        """Scan historical data; emit hypotheses that clear minimum evidence."""

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _emit(self, symbol: str, subfamily: str, direction: str,
              conditions: List[Dict], returns: Sequence[float],
              holding_bars: int, description: str,
              **metadata) -> Optional[DiscoveredHypothesis]:
        # Accept pandas Series to preserve observation timestamps
        sample_times: List[str] = []
        if isinstance(returns, pd.Series):
            if isinstance(returns.index, pd.DatetimeIndex):
                sample_times = [str(t) for t in returns.index[:500]]
            returns = returns.tolist()
        n = len(returns)
        if n < self.limits.minimum_effective_sample:
            return None
        ess = effective_sample_size(list(returns))
        if ess < self.limits.minimum_effective_sample:
            return None
        signed = list(returns) if direction == "long" else [-r for r in returns]
        p = sign_test_p_value(signed)
        mean = sum(signed) / n
        if mean <= 0:
            return None
        hyp_cat = "|".join(sorted(
            f"{c.get('feature')}{c.get('op')}{c.get('value')}"
            for c in conditions if isinstance(c.get("value"), str)))
        hyp_id = f"{self.family}:{subfamily}:{symbol}:{direction}:{hyp_cat}"
        self.tracker.record(hyp_id, family=f"{self.family}:{subfamily}",
                            p_value=p, metric=mean)
        if self.tracker.n_trials(f"{self.family}:{subfamily}") > self.limits.max_hypotheses_per_family:
            return None
        metadata = dict(metadata)
        metadata["returns_sample"] = list(signed[:500])  # for downstream validation
        metadata["sample_times"] = sample_times
        return DiscoveredHypothesis(
            family=self.family, subfamily=subfamily, symbol=symbol,
            direction=direction, entry_conditions=conditions,
            holding_bars=holding_bars, sample_size=n, effective_sample=ess,
            mean_return=mean, p_value=p, description=description,
            hypothesis_id=hyp_id,
            metadata=metadata,
        )

    @staticmethod
    def _daily_returns(df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(df["close"], errors="coerce").pct_change()


class TemporalAnomalyDetector(DiscoveryDetector):
    """Day-of-week, overnight (close→open) and turn-of-month effects."""

    family = "TEMPORAL"

    def scan(self, data: Dict[str, pd.DataFrame]) -> List[DiscoveredHypothesis]:
        out: List[DiscoveredHypothesis] = []
        for symbol, df in data.items():
            if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 60:
                continue
            rets = self._daily_returns(df)

            # Stage 1: day-of-week effects
            promising_days = []
            for day in ("MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY",
                        "SATURDAY", "SUNDAY"):
                mask = df.index.strftime("%A").str.upper() == day
                day_rets = rets[mask].dropna()
                for direction in ("long", "short"):
                    h = self._emit(
                        symbol, "day_of_week", direction,
                        [{"feature": "day_of_week", "op": "==", "value": day}],
                        day_rets, holding_bars=1,
                        description=f"{symbol} {day} {direction} effect",
                    )
                    if h:
                        out.append(h)
                        promising_days.append(day)

            # Overnight close→open (needs open data)
            if "open" in df.columns:
                o = pd.to_numeric(df["open"], errors="coerce")
                c = pd.to_numeric(df["close"], errors="coerce")
                overnight = ((o - c.shift(1)) / c.shift(1)).dropna()
                for direction in ("long", "short"):
                    h = self._emit(
                        symbol, "overnight", direction, [],
                        overnight, holding_bars=1,
                        description=f"{symbol} close→open {direction} drift",
                    )
                    if h:
                        out.append(h)

            # Stage 2: pair interaction — promising day × high RVOL (bounded)
            if self.limits.max_interaction_depth >= 2:
                v = pd.to_numeric(df.get("volume"), errors="coerce") if "volume" in df.columns else None
                if v is not None:
                    rvol = v / v.rolling(20).mean().shift(1)
                    for day in promising_days[: self.limits.max_parameter_variants]:
                        mask = (df.index.strftime("%A").str.upper() == day) & (rvol > 2)
                        combo = rets[mask].dropna()
                        h = self._emit(
                            symbol, "day_x_rvol", "long",
                            [{"feature": "day_of_week", "op": "==", "value": day},
                             {"feature": "relative_volume_20d", "op": ">", "value": 2}],
                            combo, holding_bars=1,
                            description=f"{symbol} {day} + RVOL>2 interaction",
                        )
                        if h:
                            out.append(h)
        return out


class GapDetector(DiscoveryDetector):
    """Gap up/down continuation vs reversal, segmented by magnitude."""

    family = "EVENT"

    def scan(self, data: Dict[str, pd.DataFrame]) -> List[DiscoveredHypothesis]:
        out: List[DiscoveredHypothesis] = []
        for symbol, df in data.items():
            if "open" not in df.columns or len(df) < 60:
                continue
            o = pd.to_numeric(df["open"], errors="coerce")
            c = pd.to_numeric(df["close"], errors="coerce")
            gap = (o - c.shift(1)) / c.shift(1)
            day_ret = (c - o) / o   # open→close after the gap
            for lo, hi, name in ((0.01, 0.05, "gap_up_small"),
                                 (-0.05, -0.01, "gap_down_small")):
                mask = (gap >= lo) & (gap <= hi)
                sample = day_ret[mask].dropna()
                for direction in ("long", "short"):
                    h = self._emit(
                        symbol, name, direction,
                        [{"feature": "gap_pct", "op": "BETWEEN", "value": [lo, hi]}],
                        sample, holding_bars=1,
                        description=f"{symbol} {name} {direction} (continuation/reversal)",
                    )
                    if h:
                        out.append(h)
        return out


class RelativeVolumeDetector(DiscoveryDetector):
    """RVOL threshold effects on forward returns."""

    family = "MOMENTUM"

    def scan(self, data: Dict[str, pd.DataFrame]) -> List[DiscoveredHypothesis]:
        out: List[DiscoveredHypothesis] = []
        thresholds = [1.5, 2.0, 3.0, 5.0][: self.limits.max_parameter_variants]
        for symbol, df in data.items():
            if "volume" not in df.columns or len(df) < 60:
                continue
            v = pd.to_numeric(df["volume"], errors="coerce")
            rvol = v / v.rolling(20).mean().shift(1)
            c = pd.to_numeric(df["close"], errors="coerce")
            fwd = c.shift(-1) / c - 1   # next-bar forward return
            for t in thresholds:
                sample = fwd[rvol > t].dropna()
                for direction in ("long", "short"):
                    h = self._emit(
                        symbol, f"rvol_{t}x", direction,
                        [{"feature": "relative_volume_20d", "op": ">", "value": t}],
                        sample, holding_bars=1,
                        description=f"{symbol} RVOL>{t} {direction} forward effect",
                    )
                    if h:
                        out.append(h)
        return out


class VolatilityCompressionDetector(DiscoveryDetector):
    """Range/volatility compression → subsequent expansion/return effects."""

    family = "VOLATILITY"

    def scan(self, data: Dict[str, pd.DataFrame]) -> List[DiscoveredHypothesis]:
        out: List[DiscoveredHypothesis] = []
        for symbol, df in data.items():
            if len(df) < 80:
                continue
            c = pd.to_numeric(df["close"], errors="coerce")
            vol20 = c.pct_change().rolling(20).std()
            vol_pctile = vol20.rolling(60).rank(pct=True)
            fwd5 = c.shift(-5) / c - 1
            compressed = fwd5[vol_pctile < 0.2].dropna()
            for direction in ("long", "short"):
                h = self._emit(
                    symbol, "compression", direction,
                    [{"feature": "realized_volatility", "op": "<", "value": float(vol20.quantile(0.2) or 0)}],
                    compressed, holding_bars=5,
                    description=f"{symbol} vol compression {direction} 5-bar effect",
                )
                if h:
                    out.append(h)
        return out


class CalendarEffectsDetector(DiscoveryDetector):
    """Turn-of-month / month-start / month-end effects."""

    family = "TEMPORAL"

    def scan(self, data: Dict[str, pd.DataFrame]) -> List[DiscoveredHypothesis]:
        out: List[DiscoveredHypothesis] = []
        for symbol, df in data.items():
            if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 120:
                continue
            rets = self._daily_returns(df)
            day = df.index.day
            month_end = df.index.is_month_end | (day >= 28)
            month_start = day <= 3
            for mask, name in ((month_end, "month_end"), (month_start, "month_start")):
                sample = rets[mask].dropna()
                for direction in ("long", "short"):
                    h = self._emit(
                        symbol, name, direction, [],
                        sample, holding_bars=1,
                        description=f"{symbol} {name} {direction} calendar effect",
                    )
                    if h:
                        out.append(h)
        return out


class CrossAssetLeadLagDetector(DiscoveryDetector):
    """Lagged leader returns predicting follower returns (lagged features ONLY)."""

    family = "RELATIVE_VALUE"

    def __init__(self, pairs: Optional[List[tuple]] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pairs = pairs or []   # [(leader, follower), ...]

    def scan(self, data: Dict[str, pd.DataFrame]) -> List[DiscoveredHypothesis]:
        out: List[DiscoveredHypothesis] = []
        for leader, follower in self.pairs:
            ldf, fdf = data.get(leader), data.get(follower)
            if ldf is None or fdf is None:
                continue
            lret = self._daily_returns(ldf)
            fret = self._daily_returns(fdf)
            joined = pd.concat([lret.rename("l"), fret.rename("f")], axis=1).dropna()
            if len(joined) < 60:
                continue
            # STRICTLY lagged: leader's PREVIOUS bar vs follower's current bar
            lagged_leader = joined["l"].shift(1)
            follower_now = joined["f"]
            sample = follower_now[lagged_leader > 0.01].dropna()
            h = self._emit(
                follower, f"leadlag_{leader}", "long",
                [{"feature": "market_relative_return", "op": ">", "value": 0.01}],
                sample, holding_bars=1,
                description=f"{leader} up >1% (t-1) → {follower} (t) continuation",
                leader=leader,
            )
            if h:
                out.append(h)
        return out


# ── Detectors requiring data providers we do not fabricate ────────────────────


class EarningsEffectDetector(DiscoveryDetector):
    family = "EVENT"
    requires_data = ["earnings_calendar", "earnings_surprise"]

    def scan(self, data):
        logger.info("EarningsEffectDetector: skipped — requires earnings data provider")
        return []


class CryptoFundingDetector(DiscoveryDetector):
    family = "STRUCTURAL_CRYPTO"
    requires_data = ["funding_rates"]

    def scan(self, data):
        logger.info("CryptoFundingDetector: skipped — requires funding-rate provider")
        return []


class BasisDetector(DiscoveryDetector):
    family = "STRUCTURAL_CRYPTO"
    requires_data = ["perp_prices", "futures_prices"]

    def scan(self, data):
        logger.info("BasisDetector: skipped — requires derivatives data provider")
        return []


class LiquidationDetector(DiscoveryDetector):
    family = "STRUCTURAL_CRYPTO"
    requires_data = ["open_interest", "liquidations"]

    def scan(self, data):
        logger.info("LiquidationDetector: skipped — requires OI/liquidation provider")
        return []


class EventFeatureInterface(DiscoveryDetector):
    """Generic hook for future news / SEC filings / analyst revisions /
    insider / buyback / index-change / economic-release feeds."""
    family = "EVENT"
    requires_data = ["event_feed"]

    def scan(self, data):
        return []


def default_detectors(limits: Optional[SearchLimits] = None,
                      tracker: Optional[HypothesisFamilyTracker] = None,
                      leadlag_pairs: Optional[List[tuple]] = None
                      ) -> List[DiscoveryDetector]:
    limits = limits or SearchLimits()
    tracker = tracker or HypothesisFamilyTracker()
    return [
        TemporalAnomalyDetector(limits, tracker),
        GapDetector(limits, tracker),
        RelativeVolumeDetector(limits, tracker),
        VolatilityCompressionDetector(limits, tracker),
        CalendarEffectsDetector(limits, tracker),
        CrossAssetLeadLagDetector(pairs=leadlag_pairs or [("BTC-USD", "ETH-USD")],
                                  limits=limits, tracker=tracker),
        EarningsEffectDetector(limits, tracker),
        CryptoFundingDetector(limits, tracker),
        BasisDetector(limits, tracker),
        LiquidationDetector(limits, tracker),
    ]


def run_discovery_scan(
    data: Dict[str, pd.DataFrame],
    detectors: Optional[List[DiscoveryDetector]] = None,
) -> Dict[str, Any]:
    """Run all detectors; apply per-family FDR; return a structured report."""
    from core.validation_stats import benjamini_hochberg

    tracker = HypothesisFamilyTracker()
    detectors = detectors or default_detectors(tracker=tracker)
    all_hyps: List[DiscoveredHypothesis] = []
    skipped: List[str] = []
    for det in detectors:
        if det.requires_data:
            skipped.append(f"{det.__class__.__name__} (needs {det.requires_data})")
            continue
        det.tracker = tracker
        try:
            all_hyps.extend(det.scan(data))
        except Exception as e:
            logger.error(f"{det.__class__.__name__} failed: {e}")

    # FDR within each family — only survivors go forward
    survivors = []
    by_family: Dict[str, List[DiscoveredHypothesis]] = {}
    for h in all_hyps:
        by_family.setdefault(f"{h.family}:{h.subfamily}", []).append(h)
    for family, hyps in by_family.items():
        passed = benjamini_hochberg([h.p_value for h in hyps], alpha=0.05)
        survivors.extend(h for h, ok in zip(hyps, passed) if ok)

    return {
        "universe_scanned": sorted(data.keys()),
        "hypotheses_tested": tracker.n_trials(),
        "families_tested": len(by_family),
        "candidates": len(all_hyps),
        "fdr_survivors": survivors,
        "detectors_skipped_missing_data": skipped,
    }
