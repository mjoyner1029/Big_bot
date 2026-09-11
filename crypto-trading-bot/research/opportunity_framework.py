from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MarketAnomaly:
    """Standardized market edge candidate emitted by any discovery detector."""

    anomaly_id: str
    detector_name: str
    market: str
    symbols: List[str]
    discovered_at: str = field(default_factory=_utcnow)
    hypothesis: Dict[str, Any] = field(default_factory=dict)
    entry_condition: str = ""
    exit_condition: str = ""
    holding_period: str = "1d"
    sample_size: int = 0
    expected_return: float = 0.0
    median_return: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    statistical_significance: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    estimated_costs: float = 0.0
    net_expected_return: float = 0.0
    regime_information: Dict[str, Any] = field(default_factory=dict)
    event_concentration: float = 0.0
    discovery_score: float = 0.0
    status: str = "DISCOVERED"


class OpportunityDetector:
    """Base detector plugin for market opportunity discovery."""

    detector_name: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def required_data(self) -> Sequence[str]:
        return ("open", "high", "low", "close", "volume")

    def validate_inputs(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        missing = set(self.required_data()) - set(df.columns)
        return not missing

    def generate_hypotheses(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        return []

    def scan(self, universe: Dict[str, pd.DataFrame]) -> List[MarketAnomaly]:
        raise NotImplementedError


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _max_drawdown(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    cumulative = (1.0 + values).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _sharpe_ratio(series: pd.Series, periods_per_year: int = 252) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return 0.0
    mean = values.mean()
    std = values.std(ddof=1)
    if std <= 0:
        return 0.0
    return float((mean / std) * math.sqrt(periods_per_year))


def _sortino_ratio(series: pd.Series, periods_per_year: int = 252) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return 0.0
    mean = values.mean()
    downside = values[values < 0]
    if downside.empty:
        return 0.0
    downside_std = downside.std(ddof=1)
    if downside_std <= 0:
        return 0.0
    return float((mean / downside_std) * math.sqrt(periods_per_year))


def _profit_factor(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    gains = values[values > 0].sum()
    losses = abs(values[values < 0].sum())
    if losses <= 0:
        return 1.0 if gains > 0 else 0.0
    return float(gains / losses)


def _build_anomaly(
    detector_name: str,
    symbol: str,
    market: str,
    return_series: pd.Series,
    hypothesis: Dict[str, Any],
    entry_condition: str,
    exit_condition: str,
    holding_period: str,
    *,
    regime: Optional[Dict[str, Any]] = None,
    event_concentration: float = 0.0,
    discovery_status: str = "DISCOVERED",
    base_score: float = 70.0,
) -> Optional[MarketAnomaly]:
    if return_series is None or return_series.empty:
        return None
    values = pd.to_numeric(return_series, errors="coerce").dropna()
    if values.empty:
        return None
    sample_size = int(len(values))
    expected_return = float(values.mean())
    median_return = float(values.median())
    volatility = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    sharpe = _sharpe_ratio(values)
    sortino = _sortino_ratio(values)
    drawdown = _max_drawdown(values)
    win_rate = float((values > 0).mean())
    profit_factor = _profit_factor(values)
    significance = 0.05 if sample_size >= 30 else 0.10
    if expected_return > 0 and sample_size >= 60:
        significance = max(0.001, min(0.05, 1.0 / sample_size))
    score = max(0.0, min(100.0, base_score + expected_return * 500 + sharpe * 12 + win_rate * 15 - abs(drawdown) * 60))
    if sample_size < 30:
        score *= 0.8
    if event_concentration > 0.45:
        score -= 12.0
    score = max(0.0, min(100.0, score))

    return MarketAnomaly(
        anomaly_id=f"{detector_name}-{symbol}-{uuid.uuid4().hex[:8]}",
        detector_name=detector_name,
        market=market,
        symbols=[symbol],
        hypothesis=hypothesis,
        entry_condition=entry_condition,
        exit_condition=exit_condition,
        holding_period=holding_period,
        sample_size=sample_size,
        expected_return=expected_return,
        median_return=median_return,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        statistical_significance=significance,
        confidence_interval=(expected_return - 2 * volatility, expected_return + 2 * volatility),
        estimated_costs=max(0.001, abs(expected_return) * 0.25),
        net_expected_return=max(expected_return - max(0.001, abs(expected_return) * 0.25), -1.0),
        regime_information=regime or {"regime": "unknown"},
        event_concentration=event_concentration,
        discovery_score=round(score, 2),
        status=discovery_status if score >= 55 else "VALIDATING",
    )


class TemporalAnomalyDetector(OpportunityDetector):
    detector_name = "temporal"

    def scan(self, universe: Dict[str, pd.DataFrame]) -> List[MarketAnomaly]:
        anomalies: List[MarketAnomaly] = []
        for symbol, df in universe.items():
            if not self.validate_inputs(df):
                continue
            close = pd.to_numeric(df["close"], errors="coerce")
            open_price = pd.to_numeric(df["open"], errors="coerce")
            prev_close = close.shift(1)
            overnight = (open_price / prev_close) - 1.0
            overnight = overnight.dropna()
            if overnight.empty:
                continue
            anomaly = _build_anomaly(
                detector_name=self.detector_name,
                symbol=symbol,
                market="equity",
                return_series=overnight,
                hypothesis={"feature": "close_to_open", "window": "overnight", "condition": "open > previous_close"},
                entry_condition="close_to_open > 0",
                exit_condition="next_open <= previous_close",
                holding_period="overnight",
                regime={"regime": "unknown"},
                event_concentration=0.10,
                discovery_status="DISCOVERED",
                base_score=72.0,
            )
            if anomaly is not None:
                anomalies.append(anomaly)
        return anomalies


class MomentumDetector(OpportunityDetector):
    detector_name = "momentum"

    def scan(self, universe: Dict[str, pd.DataFrame]) -> List[MarketAnomaly]:
        anomalies: List[MarketAnomaly] = []
        for symbol, df in universe.items():
            if not self.validate_inputs(df):
                continue
            close = pd.to_numeric(df["close"], errors="coerce")
            volume = pd.to_numeric(df["volume"], errors="coerce")
            returns = close.pct_change().dropna()
            vol_ratio = volume / volume.shift(20).rolling(3).mean()
            momentum = (returns * vol_ratio.fillna(1.0)).dropna()
            if momentum.empty:
                continue
            anomaly = _build_anomaly(
                detector_name=self.detector_name,
                symbol=symbol,
                market="equity",
                return_series=momentum,
                hypothesis={"feature": "relative_volume_x_return", "window": "5d", "condition": "volume_ratio > 2"},
                entry_condition="relative_volume > 2 and return_5d > 0",
                exit_condition="trend fails or return_5d <= 0",
                holding_period="5d",
                regime={"regime": "risk_on"},
                event_concentration=0.15,
                discovery_status="DISCOVERED",
                base_score=68.0,
            )
            if anomaly is not None:
                anomalies.append(anomaly)
        return anomalies


class MeanReversionDetector(OpportunityDetector):
    detector_name = "mean_reversion"

    def scan(self, universe: Dict[str, pd.DataFrame]) -> List[MarketAnomaly]:
        anomalies: List[MarketAnomaly] = []
        for symbol, df in universe.items():
            if not self.validate_inputs(df):
                continue
            close = pd.to_numeric(df["close"], errors="coerce")
            rolling_mean = close.rolling(20).mean()
            spread = (close - rolling_mean) / rolling_mean
            reversion = (-spread).dropna()
            if reversion.empty:
                continue
            anomaly = _build_anomaly(
                detector_name=self.detector_name,
                symbol=symbol,
                market="equity",
                return_series=reversion,
                hypothesis={"feature": "distance_from_mean", "window": "20d", "condition": "spread < -1.5 std"},
                entry_condition="abs(close - rolling_mean) > 1.5 * rolling_std",
                exit_condition="close reverts to rolling_mean",
                holding_period="10d",
                regime={"regime": "sideways"},
                event_concentration=0.20,
                discovery_status="VALIDATING",
                base_score=60.0,
            )
            if anomaly is not None:
                anomalies.append(anomaly)
        return anomalies


class VolumeGapDetector(OpportunityDetector):
    detector_name = "volume_gap"

    def scan(self, universe: Dict[str, pd.DataFrame]) -> List[MarketAnomaly]:
        anomalies: List[MarketAnomaly] = []
        for symbol, df in universe.items():
            if not self.validate_inputs(df):
                continue
            open_price = pd.to_numeric(df["open"], errors="coerce")
            prev_close = pd.to_numeric(df["close"], errors="coerce").shift(1)
            volume = pd.to_numeric(df["volume"], errors="coerce")
            gap = ((open_price / prev_close) - 1.0).dropna()
            rel_vol = (volume / volume.rolling(20).mean()).dropna()
            combined = (gap * rel_vol).dropna()
            if combined.empty:
                continue
            anomaly = _build_anomaly(
                detector_name=self.detector_name,
                symbol=symbol,
                market="equity",
                return_series=combined,
                hypothesis={"feature": "gap_x_volume", "window": "1d", "condition": "relative_volume > 3 and gap_pct > 0"},
                entry_condition="relative_volume > 3 and gap_pct > 0",
                exit_condition="gap fills or reversal occurs",
                holding_period="1d",
                regime={"regime": "high_vol"},
                event_concentration=0.08,
                discovery_status="PAPER_CANDIDATE",
                base_score=78.0,
            )
            if anomaly is not None:
                anomalies.append(anomaly)
        return anomalies


class OpportunityScanner:
    """Registry-backed opportunity discovery runner."""

    def __init__(self) -> None:
        self.registry: Dict[str, OpportunityDetector] = {}
        self._register_default_detectors()

    def _register_default_detectors(self) -> None:
        detectors = [
            TemporalAnomalyDetector(),
            MomentumDetector(),
            MeanReversionDetector(),
            VolumeGapDetector(),
        ]
        for detector in detectors:
            self.registry[detector.detector_name] = detector

    def register(self, detector: OpportunityDetector) -> None:
        self.registry[detector.detector_name] = detector

    def scan_universe(self, universe: Dict[str, pd.DataFrame]) -> List[MarketAnomaly]:
        anomalies: List[MarketAnomaly] = []
        for detector in self.registry.values():
            anomalies.extend(detector.scan(universe))
        anomalies.sort(key=lambda item: item.discovery_score, reverse=True)
        return anomalies

    def paper_only_gate(self, anomaly: MarketAnomaly) -> bool:
        disallowed_live = {"LIVE_ELIGIBLE", "LIVE_LIMITED", "LIVE_SCALED"}
        return anomaly.status not in disallowed_live


def compute_opportunity_score(
    *,
    net_expectancy: float,
    sharpe: float,
    sortino: float,
    drawdown: float,
    profit_factor: float,
    sample_size: int,
    significance: float,
    oos_sharpe: float,
    parameter_robustness: float,
    cost_robustness: float,
    regime_stability: float,
    liquidity: float,
    execution_feasibility: float,
    event_concentration: float,
    recent_performance: float,
    multiple_testing_adjusted: bool,
) -> float:
    """Return a normalized score from 0–100."""
    score = 0.0
    score += min(20.0, max(0.0, net_expectancy * 1000.0))
    score += min(15.0, max(0.0, sharpe * 8.0))
    score += min(10.0, max(0.0, sortino * 5.0))
    score += min(15.0, max(0.0, (1.0 - min(drawdown, 1.0)) * 15.0))
    score += min(10.0, max(0.0, (profit_factor - 1.0) * 10.0))
    score += min(10.0, sample_size / 60.0)
    score += min(10.0, max(0.0, (1.0 - min(significance, 1.0)) * 10.0))
    score += min(10.0, max(0.0, oos_sharpe * 6.0))
    score += min(10.0, max(0.0, parameter_robustness * 10.0))
    score += min(10.0, max(0.0, cost_robustness * 10.0))
    score += min(10.0, max(0.0, regime_stability * 10.0))
    score += min(10.0, max(0.0, liquidity * 10.0))
    score += min(10.0, max(0.0, execution_feasibility * 10.0))
    score += min(10.0, max(0.0, (1.0 - min(event_concentration, 1.0)) * 10.0))
    score += min(10.0, max(0.0, recent_performance * 10.0))
    if multiple_testing_adjusted:
        score += 10.0
    return round(max(0.0, min(100.0, score)), 2)


def benjamini_hochberg_qvalues(pvalues: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg FDR-adjusted q-values."""
    arr = np.asarray(pvalues, dtype=float)
    if arr.size == 0:
        return []
    m = arr.size
    order = np.argsort(arr)
    sorted_p = arr[order]
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for idx in range(m - 1, -1, -1):
        rank = idx + 1
        running = min(running, sorted_p[idx] * m / rank)
        adjusted[idx] = running
    out = np.empty(m, dtype=float)
    out[order] = adjusted
    return [float(x) for x in out]


class HoldoutManager:
    """Guard rails for the final holdout window."""

    def __init__(self) -> None:
        self.final_holdout: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

    def freeze_holdout(self, start: str, end: str) -> None:
        self.final_holdout = (pd.Timestamp(start), pd.Timestamp(end))

    def is_final_holdout_accessible(self, date: str) -> bool:
        if self.final_holdout is None:
            return True
        start, end = self.final_holdout
        target = pd.Timestamp(date)
        return not (start <= target <= end)


__all__ = [
    "MarketAnomaly",
    "OpportunityDetector",
    "OpportunityScanner",
    "TemporalAnomalyDetector",
    "MomentumDetector",
    "MeanReversionDetector",
    "VolumeGapDetector",
    "compute_opportunity_score",
    "benjamini_hochberg_qvalues",
    "HoldoutManager",
]
