from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.signal_flipper import AssetClass, Signal, SignalType
from core.strategy_base import StrategyBase
from core.strategy_registry import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class OvernightEdgeConfig:
    """Configuration for overnight edge research and validation."""

    enabled: bool = True
    paper_only: bool = True
    allow_live: bool = False
    min_history_days: int = 252
    min_price: float = 5.0
    min_avg_daily_dollar_volume: float = 20_000_000.0
    min_observations: int = 126
    min_net_overnight_cagr: float = 0.0
    min_sharpe: float = 1.0
    min_profit_factor: float = 1.15
    max_drawdown: float = 0.5
    min_out_of_sample_sharpe: float = 0.0
    max_top_5_profit_concentration: float = 0.40
    max_top_10_profit_concentration: float = 0.60
    max_portfolio_overnight_exposure_pct: float = 0.20
    max_single_overnight_position_pct: float = 0.10
    lookbacks: Tuple[int, ...] = (20, 60, 126, 252, 504)
    transaction_cost_bps: float = 10.0
    entry_slippage_bps: float = 15.0
    exit_slippage_bps: float = 25.0
    close_entry_window_minutes: int = 15
    open_exit_window_minutes: int = 15
    live_trading_disabled_reason: str = "Overnight edge strategy remains paper-only until validation is complete."


@dataclass
class OvernightEdgeSignal:
    ticker: str
    timestamp: str
    direction: str
    score: float
    confidence: float
    expected_return: float
    expected_risk: float
    holding_period: str
    entry_window: str
    exit_window: str
    edge_type: str
    reason_codes: List[str] = field(default_factory=list)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or math.isnan(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def validate_overnight_inputs(df: pd.DataFrame) -> List[str]:
    """Return validation issues for an overnight-edge dataset."""
    issues: List[str] = []
    if df is None or df.empty:
        return ["dataframe is empty"]

    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        issues.append(f"missing required columns: {missing}")

    if df.index.has_duplicates:
        issues.append("duplicate timestamps detected")

    if not df.index.is_monotonic_increasing:
        issues.append("timestamp index is not sorted ascending")

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            cleaned = _coerce_numeric(df[col])
            if (cleaned <= 0).any():
                issues.append(f"non-positive values in {col}")

    if "volume" in df.columns:
        volume = _coerce_numeric(df["volume"])
        if (volume < 0).any():
            issues.append("negative volume values detected")

    if "open" in df.columns and "close" in df.columns:
        open_s = _coerce_numeric(df["open"])
        close_s = _coerce_numeric(df["close"])
        if ((open_s.isna()) | (close_s.isna())).any():
            issues.append("NaN price values in OHLC data")

    return issues


def compute_overnight_returns(df: pd.DataFrame) -> pd.Series:
    """Return = current day open / previous day close - 1."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    out = pd.DataFrame(df.copy())
    prev_close = _coerce_numeric(out.get("previous_close", out["close"].shift(1)))
    open_price = _coerce_numeric(out["open"])
    overnight = (open_price / prev_close) - 1.0
    return overnight.rename("overnight_return")


def compute_intraday_returns(df: pd.DataFrame) -> pd.Series:
    """Return = current day close / current day open - 1."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    open_price = _coerce_numeric(df["open"])
    close_price = _coerce_numeric(df["close"])
    return ((close_price / open_price) - 1.0).rename("intraday_return")


def compute_close_to_close_returns(df: pd.DataFrame) -> pd.Series:
    """Return = current day close / previous day close - 1."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    close_price = _coerce_numeric(df["close"])
    prev_close = close_price.shift(1)
    return ((close_price / prev_close) - 1.0).rename("close_to_close_return")


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    cumulative = (1.0 + values).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _annualized_volatility(series: pd.Series, periods_per_year: int = 252) -> float:
    if series.empty:
        return 0.0
    returns = pd.to_numeric(series, errors="coerce").dropna()
    if len(returns) < 2:
        return 0.0
    std = returns.std(ddof=1)
    if pd.isna(std) or std <= 0:
        return 0.0
    return float(std * math.sqrt(periods_per_year))


def _sharpe_ratio(series: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    returns = pd.to_numeric(series, errors="coerce").dropna()
    if returns.empty or len(returns) < 2:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=1)
    if pd.isna(std) or std <= 0:
        return 0.0
    excess = mean - risk_free
    return float((excess / std) * math.sqrt(periods_per_year))


def _sortino_ratio(series: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    returns = pd.to_numeric(series, errors="coerce").dropna()
    if returns.empty or len(returns) < 2:
        return 0.0
    mean = returns.mean()
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if downside_std <= 0:
        return 0.0
    excess = mean - risk_free
    return float((excess / downside_std) * math.sqrt(periods_per_year))


def _profit_factor(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    gains = values[values > 0].sum()
    losses = abs(values[values < 0].sum())
    return float(_safe_divide(gains, losses)) if losses > 0 else (1.0 if gains > 0 else 0.0)


def _cagr(series: pd.Series, periods_per_year: int = 252) -> float:
    returns = pd.to_numeric(series, errors="coerce").dropna()
    if returns.empty:
        return 0.0
    cumulative = (1.0 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float(cumulative ** (1.0 / n_years) - 1.0)


def _win_rate(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float((values > 0).mean())


def _skew(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 3:
        return 0.0
    return float(values.skew())


def _kurtosis(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 4:
        return 0.0
    return float(values.kurtosis())


def classify_edge_type(general_edge: bool, earnings_driven: bool) -> str:
    if general_edge and earnings_driven:
        return "MIXED"
    if general_edge:
        return "GENERAL"
    if earnings_driven:
        return "EARNINGS_DRIVEN"
    return "NONE"


class OvernightEdgeScanner:
    """Research and validation layer for overnight-return anomalies."""

    def __init__(self, config: Optional[OvernightEdgeConfig] = None):
        self.config = config or OvernightEdgeConfig()
        self.logger = logging.getLogger("research.overnight_edge")

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.index.name is None:
            df.index.name = "timestamp"
        df = df.sort_index()
        df["previous_close"] = df["close"].shift(1)
        df["overnight_return"] = compute_overnight_returns(df)
        df["intraday_return"] = compute_intraday_returns(df)
        df["close_to_close_return"] = compute_close_to_close_returns(df)
        return df

    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        issues = validate_overnight_inputs(df)
        if issues:
            self.logger.warning("Input validation issues: %s", issues)

        prepared = self._prepare_df(df)
        overnight = prepared["overnight_return"].dropna()
        intraday = prepared["intraday_return"].dropna()
        close_to_close = prepared["close_to_close_return"].dropna()

        overnight_cum = float((1.0 + overnight).prod() - 1.0) if not overnight.empty else 0.0
        intraday_cum = float((1.0 + intraday).prod() - 1.0) if not intraday.empty else 0.0
        close_to_close_cum = float((1.0 + close_to_close).prod() - 1.0) if not close_to_close.empty else 0.0

        overnight_mean = float(overnight.mean()) if not overnight.empty else 0.0
        intraday_mean = float(intraday.mean()) if not intraday.empty else 0.0
        close_mean = float(close_to_close.mean()) if not close_to_close.empty else 0.0

        panel = {
            "sample_size": int(len(overnight.dropna())),
            "overnight_mean": overnight_mean,
            "intraday_mean": intraday_mean,
            "close_to_close_mean": close_mean,
            "overnight_cumulative_return": overnight_cum,
            "intraday_cumulative_return": intraday_cum,
            "close_to_close_cumulative_return": close_to_close_cum,
            "overnight_cagr": _cagr(overnight, periods_per_year=252),
            "intraday_cagr": _cagr(intraday, periods_per_year=252),
            "close_to_close_cagr": _cagr(close_to_close, periods_per_year=252),
            "overnight_annualized_volatility": _annualized_volatility(overnight, periods_per_year=252),
            "intraday_annualized_volatility": _annualized_volatility(intraday, periods_per_year=252),
            "overnight_sharpe": _sharpe_ratio(overnight, periods_per_year=252),
            "intraday_sharpe": _sharpe_ratio(intraday, periods_per_year=252),
            "overnight_sortino": _sortino_ratio(overnight, periods_per_year=252),
            "intraday_sortino": _sortino_ratio(intraday, periods_per_year=252),
            "overnight_win_rate": _win_rate(overnight),
            "intraday_win_rate": _win_rate(intraday),
            "overnight_profit_factor": _profit_factor(overnight),
            "intraday_profit_factor": _profit_factor(intraday),
            "overnight_max_drawdown": _max_drawdown(overnight),
            "intraday_max_drawdown": _max_drawdown(intraday),
            "overnight_calmar": _safe_divide(_cagr(overnight, periods_per_year=252), abs(_max_drawdown(overnight))) if _max_drawdown(overnight) else 0.0,
            "overnight_skew": _skew(overnight),
            "overnight_kurtosis": _kurtosis(overnight),
            "overnight_intraday_ratio": _safe_divide(abs(overnight_cum), abs(intraday_cum)),
            "top_1_night_contribution": 0.0,
            "top_5_night_contribution": 0.0,
            "top_10_night_contribution": 0.0,
            "earnings_night_contribution": 0.0,
            "non_earnings_night_performance": 0.0,
        }

        if not overnight.empty:
            cumulative = (1.0 + overnight).cumprod() - 1.0
            sorted_contrib = cumulative.abs().sort_values(ascending=False)
            n = len(sorted_contrib)
            if n:
                panel["top_1_night_contribution"] = float(sorted_contrib.iloc[0] / cumulative.abs().sum()) if cumulative.abs().sum() > 0 else 0.0
                panel["top_5_night_contribution"] = float(sorted_contrib.iloc[: max(1, min(5, n))].sum() / cumulative.abs().sum()) if cumulative.abs().sum() > 0 else 0.0
                panel["top_10_night_contribution"] = float(sorted_contrib.iloc[: max(1, min(10, n))].sum() / cumulative.abs().sum()) if cumulative.abs().sum() > 0 else 0.0

        panel["percentage_of_total_return_attributable_to_overnight"] = _safe_divide(abs(overnight_cum), abs(close_to_close_cum))
        panel["general_edge"] = panel["overnight_sharpe"] >= self.config.min_sharpe and panel["sample_size"] >= self.config.min_observations
        panel["earnings_driven"] = panel["top_1_night_contribution"] > 0.25 or panel["top_5_night_contribution"] > 0.60
        panel["edge_type"] = classify_edge_type(panel["general_edge"], panel["earnings_driven"])
        panel["net_ready"] = panel["overnight_cagr"] > self.config.min_net_overnight_cagr
        return panel

    def _score_metrics(self, metrics: Dict[str, Any]) -> float:
        score = 0.0
        sample_size = metrics.get("sample_size", 0)
        score += min(15.0, max(0.0, sample_size / max(1, self.config.min_observations)) * 15.0)

        overnight_cagr = metrics.get("overnight_cagr", 0.0)
        score += min(25.0, max(0.0, overnight_cagr) * 1000.0)

        sharpe = metrics.get("overnight_sharpe", 0.0)
        score += min(20.0, max(0.0, sharpe) * 15.0)

        sortino = metrics.get("overnight_sortino", 0.0)
        score += min(10.0, max(0.0, sortino) * 8.0)

        win_rate = metrics.get("overnight_win_rate", 0.0)
        score += min(10.0, win_rate * 20.0)

        profit_factor = metrics.get("overnight_profit_factor", 0.0)
        score += min(10.0, max(0.0, profit_factor - 1.0) * 20.0)

        drawdown = metrics.get("overnight_max_drawdown", 0.0)
        if drawdown < -0.20:
            score -= 15.0
        elif drawdown < 0.0:
            score -= 5.0

        top5 = metrics.get("top_5_night_contribution", 0.0)
        if top5 > self.config.max_top_5_profit_concentration:
            score -= 20.0

        ratio = metrics.get("overnight_intraday_ratio", 0.0)
        score += min(10.0, max(0.0, ratio) * 8.0)

        score = max(0.0, min(100.0, score))
        return round(score, 2)

    def qualifies(self, metrics: Dict[str, Any]) -> bool:
        if metrics.get("sample_size", 0) < self.config.min_observations:
            return False
        if metrics.get("overnight_profit_factor", 0.0) < self.config.min_profit_factor:
            return False
        if abs(metrics.get("overnight_max_drawdown", 0.0)) > self.config.max_drawdown:
            return False
        if metrics.get("top_5_night_contribution", 0.0) > self.config.max_top_5_profit_concentration:
            return False
        if metrics.get("top_10_night_contribution", 0.0) > self.config.max_top_10_profit_concentration:
            return False
        if metrics.get("overnight_cagr", 0.0) < self.config.min_net_overnight_cagr:
            return False
        return True

    def scan_universe(self, universe: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Rank candidate tickers for overnight edge research."""
        results: List[Dict[str, Any]] = []
        for ticker, df in universe.items():
            try:
                if df is None or df.empty:
                    continue
                if validate_overnight_inputs(df):
                    self.logger.info("Skipping %s due to input issues: %s", ticker, validate_overnight_inputs(df))
                    continue
                metrics = self.compute_metrics(df)
                score = self._score_metrics(metrics)
                recommendation = "REJECT"
                if self.qualifies(metrics):
                    if score >= 75 and metrics.get("overnight_sharpe", 0.0) >= self.config.min_sharpe:
                        recommendation = "PAPER_TRADE"
                    elif score >= 60:
                        recommendation = "RESEARCH"
                    else:
                        recommendation = "WATCH"
                result = {
                    "ticker": ticker,
                    "score": round(score, 2),
                    "rank": 0,
                    "recommendation": recommendation,
                    "metrics": metrics,
                    "net_overnight_cagr": metrics.get("overnight_cagr", 0.0),
                    "sharpe": metrics.get("overnight_sharpe", 0.0),
                    "sortino": metrics.get("overnight_sortino", 0.0),
                    "win_rate": metrics.get("overnight_win_rate", 0.0),
                    "profit_factor": metrics.get("overnight_profit_factor", 0.0),
                    "max_drawdown": metrics.get("overnight_max_drawdown", 0.0),
                    "top_5_profit_concentration": metrics.get("top_5_night_concentration", 0.0),
                    "overnight_vs_intraday_ratio": metrics.get("overnight_intraday_ratio", 0.0),
                    "sample_size": metrics.get("sample_size", 0),
                    "edge_type": metrics.get("edge_type", "NONE"),
                    "out_of_sample_score": 0.0,
                    "regime_stability": 0.0,
                }
                results.append(result)
            except Exception as exc:  # pragma: no cover - resilient scanner
                self.logger.exception("Overnight edge scan failed for %s: %s", ticker, exc)

        results.sort(key=lambda item: item["score"], reverse=True)
        for idx, row in enumerate(results, start=1):
            row["rank"] = idx
        return results

    def walk_forward_split(self, df: pd.DataFrame, train_ratio: float = 0.6, validation_ratio: float = 0.2) -> List[Dict[str, Any]]:
        """Return chronological train/validation/test folds that avoid look-ahead bias."""
        clean = self._prepare_df(df)
        if clean.empty:
            return []

        n = len(clean)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))
        if train_end <= 10 or val_end <= train_end:
            return [{"train_start": 0, "train_end": n - 1, "test_start": 0, "test_end": n - 1, "window_label": "single_window"}]

        folds = []
        for start in range(0, n - 30, max(5, int((n - 30) / 3))):
            train_end_i = min(start + train_end, n - 1)
            test_start = train_end_i + 1
            test_end = min(test_start + max(10, (n - test_start) // 2), n - 1)
            if test_start >= n:
                break
            folds.append({
                "train_start": start,
                "train_end": train_end_i,
                "test_start": test_start,
                "test_end": test_end,
                "window_label": f"fold_{len(folds) + 1}",
            })
        return folds

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        """Generate a long-only overnight signal subject to paper-only lockout."""
        if not self.config.enabled:
            return Signal.no_trade(symbol=symbol, strategy_name="overnight_edge", reason="Overnight edge scanner disabled")
        if self.config.paper_only and not self.config.allow_live:
            return Signal.no_trade(symbol=symbol, strategy_name="overnight_edge", reason=self.config.live_trading_disabled_reason)

        if df is None or df.empty:
            return Signal.no_trade(symbol=symbol, strategy_name="overnight_edge", reason="No market data")

        issues = validate_overnight_inputs(df)
        if issues:
            return Signal.no_trade(symbol=symbol, strategy_name="overnight_edge", reason="; ".join(issues))

        metrics = self.compute_metrics(df)
        if not self.qualifies(metrics):
            return Signal.no_trade(symbol=symbol, strategy_name="overnight_edge", reason="Failed overnight-edge qualification")

        score = self._score_metrics(metrics)
        if score < 60:
            return Signal.no_trade(symbol=symbol, strategy_name="overnight_edge", reason=f"Score {score} below threshold")

        entry_price = float(df["close"].iloc[-1])
        expected_return = float(metrics.get("overnight_cagr", 0.0))
        risk = max(abs(metrics.get("overnight_max_drawdown", 0.0)), 0.01)
        stop_loss = entry_price * (1.0 - risk)

        return Signal(
            symbol=symbol,
            signal=SignalType.BUY,
            confidence=min(100.0, score),
            entry=entry_price,
            stop_loss=stop_loss,
            targets=[entry_price * (1.0 + max(expected_return, 0.02))],
            strategy_name="overnight_edge",
            reason="Long overnight candidate passes validation, liquidity, and cost filters.",
            asset_class=AssetClass.STOCK if not str(symbol).upper().endswith(("USDT", "USD")) else AssetClass.CRYPTO,
            metadata={
                "edge_score": round(score, 2),
                "edge_type": metrics.get("edge_type", "NONE"),
                "metrics": metrics,
                "paper_only": self.config.paper_only,
                "allow_live": self.config.allow_live,
            },
        )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Scan overnight edge candidates")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    parser.add_argument("--ticker", default=None, help="Optional ticker to analyze from a synthetic dataset")
    args = parser.parse_args()

    scanner = OvernightEdgeScanner()
    sample = {
        "TEST": pd.DataFrame({
            "open": np.linspace(100.0, 120.0, 260),
            "high": np.linspace(101.0, 121.0, 260),
            "low": np.linspace(99.0, 119.0, 260),
            "close": np.linspace(100.0, 140.0, 260),
            "volume": 1_000_000,
        })
    }
    ranked = scanner.scan_universe(sample)
    if args.json:
        print(json.dumps(ranked[:5], default=str, indent=2))
    else:
        print("Ticker | Score | Recommendation | Edge Type")
        print("-" * 72)
        for row in ranked[:10]:
            print(f"{row['ticker']} | {row['score']} | {row['recommendation']} | {row['edge_type']}")


if __name__ == "__main__":
    _cli()


@StrategyRegistry.register("overnight_edge")
class OvernightEdgeStrategy(StrategyBase):
    """Strategy wrapper for overnight-edge research and paper-only validation."""

    name = "overnight_edge"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.scanner = OvernightEdgeScanner(OvernightEdgeConfig(**(config or {})))

    def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Signal:
        df = data.get("df")
        if df is None:
            return self._no_trade(symbol, reason="Missing DataFrame in overnight edge strategy")
        return self.scanner.generate_signal(symbol, df)
