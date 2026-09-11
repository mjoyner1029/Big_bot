"""Market data-feed providers for event/derivatives features.

Concrete providers use real public sources where available:
  - Earnings dates: yfinance (when installed)
  - Funding rates / open interest / basis: Binance public REST (no credentials)

Everything else (SEC filings, corporate actions, analyst changes,
liquidations) exposes the interface and reports unavailability — features are
NEVER fabricated. All providers fail soft: on any error they return None/[]
so a missing feed degrades to feature-unavailable (alphas abstain).
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CACHE: Dict[str, tuple] = {}
_CACHE_TTL = 900  # 15 min


def _cached(key: str, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _CACHE_TTL:
        return hit[1]
    value = fn()
    _CACHE[key] = (now, value)
    return value


@dataclass
class EventRecord:
    symbol: str
    event_type: str            # 'earnings' | 'filing' | 'split' | 'dividend' | ...
    event_time: str
    payload: Dict[str, Any] = field(default_factory=dict)


class DataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def available(self) -> bool: ...


# ── Earnings ──────────────────────────────────────────────────────────────────


class EarningsProvider(DataProvider):
    name = "earnings"

    def available(self) -> bool:
        return False

    def days_to_next_earnings(self, symbol: str) -> Optional[float]:
        return None

    def recent_earnings_events(self, symbol: str, limit: int = 8) -> List[EventRecord]:
        return []


class YFinanceEarningsProvider(EarningsProvider):
    """Earnings dates via yfinance. Applies only to equities."""

    def available(self) -> bool:
        try:
            import yfinance  # noqa: F401
            return True
        except ImportError:
            return False

    def days_to_next_earnings(self, symbol: str) -> Optional[float]:
        if "-" in symbol:   # crypto pairs have no earnings
            return None

        def fetch():
            try:
                import pandas as pd
                import yfinance as yf
                dates = yf.Ticker(symbol).get_earnings_dates(limit=8)
                if dates is None or dates.empty:
                    return None
                now = pd.Timestamp.now(tz=dates.index.tz) if dates.index.tz \
                    else pd.Timestamp.now()
                future = dates.index[dates.index > now]
                if len(future) == 0:
                    return None
                return float((future.min() - now).total_seconds() / 86400.0)
            except Exception as e:
                logger.debug(f"YFinanceEarningsProvider: {symbol}: {e}")
                return None

        return _cached(f"earn_next:{symbol}", fetch)

    def recent_earnings_events(self, symbol: str, limit: int = 8) -> List[EventRecord]:
        if "-" in symbol:
            return []

        def fetch():
            try:
                import yfinance as yf
                dates = yf.Ticker(symbol).get_earnings_dates(limit=limit)
                if dates is None or dates.empty:
                    return []
                out = []
                for ts, row in dates.iterrows():
                    out.append(EventRecord(
                        symbol=symbol, event_type="earnings",
                        event_time=str(ts),
                        payload={k: (None if _isnan(v) else v)
                                 for k, v in row.to_dict().items()},
                    ))
                return out
            except Exception as e:
                logger.debug(f"YFinanceEarningsProvider events: {symbol}: {e}")
                return []

        return _cached(f"earn_events:{symbol}", fetch)


def _isnan(v) -> bool:
    try:
        import math
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return False


# ── Crypto derivatives: funding / open interest / basis ──────────────────────


class FundingProvider(DataProvider):
    name = "funding"

    def available(self) -> bool:
        return False

    def current_funding_rate(self, symbol: str) -> Optional[float]:
        return None

    def funding_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        return []


class BinanceDerivativesProvider(FundingProvider):
    """Funding, open interest and perp/spot basis via Binance public REST.

    No credentials required. Symbols use the bot's `BTC-USD` style and are
    mapped to Binance's `BTCUSDT` perps.
    """

    BASE = "https://fapi.binance.com"
    SPOT = "https://api.binance.com"

    def available(self) -> bool:
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _perp_symbol(symbol: str) -> Optional[str]:
        if not symbol.endswith("-USD"):
            return None
        return symbol.replace("-USD", "USDT")

    def _get(self, url: str, params: Dict) -> Optional[Any]:
        try:
            import requests
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception as e:
            logger.debug(f"BinanceDerivativesProvider: {url}: {e}")
            return None

    def current_funding_rate(self, symbol: str) -> Optional[float]:
        perp = self._perp_symbol(symbol)
        if not perp:
            return None

        def fetch():
            data = self._get(f"{self.BASE}/fapi/v1/premiumIndex", {"symbol": perp})
            if not data or "lastFundingRate" not in data:
                return None
            try:
                return float(data["lastFundingRate"])
            except (TypeError, ValueError):
                return None

        return _cached(f"funding:{perp}", fetch)

    def funding_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        perp = self._perp_symbol(symbol)
        if not perp:
            return []

        def fetch():
            data = self._get(f"{self.BASE}/fapi/v1/fundingRate",
                             {"symbol": perp, "limit": min(limit, 1000)})
            if not isinstance(data, list):
                return []
            return [{"time": d.get("fundingTime"),
                     "rate": float(d.get("fundingRate", 0))} for d in data]

        return _cached(f"funding_hist:{perp}:{limit}", fetch)

    def open_interest(self, symbol: str) -> Optional[float]:
        perp = self._perp_symbol(symbol)
        if not perp:
            return None

        def fetch():
            data = self._get(f"{self.BASE}/fapi/v1/openInterest", {"symbol": perp})
            if not data or "openInterest" not in data:
                return None
            try:
                return float(data["openInterest"])
            except (TypeError, ValueError):
                return None

        return _cached(f"oi:{perp}", fetch)

    def perp_spot_basis(self, symbol: str) -> Optional[float]:
        """(perp mark - spot) / spot. None when either quote is unavailable."""
        perp = self._perp_symbol(symbol)
        if not perp:
            return None

        def fetch():
            mark = self._get(f"{self.BASE}/fapi/v1/premiumIndex", {"symbol": perp})
            spot = self._get(f"{self.SPOT}/api/v3/ticker/price", {"symbol": perp})
            try:
                mark_p = float(mark["markPrice"])
                spot_p = float(spot["price"])
                return (mark_p - spot_p) / spot_p if spot_p > 0 else None
            except (TypeError, KeyError, ValueError):
                return None

        return _cached(f"basis:{perp}", fetch)


# ── Interfaces without live data yet (honest unavailability) ─────────────────


class SECFilingsProvider(DataProvider):
    name = "sec_filings"

    def available(self) -> bool:
        return False

    def recent_filings(self, symbol: str) -> List[EventRecord]:
        return []


class CorporateActionsProvider(DataProvider):
    name = "corporate_actions"

    def available(self) -> bool:
        return False

    def recent_actions(self, symbol: str) -> List[EventRecord]:
        return []


class AnalystChangesProvider(DataProvider):
    name = "analyst_changes"

    def available(self) -> bool:
        return False

    def recent_changes(self, symbol: str) -> List[EventRecord]:
        return []


class LiquidationsProvider(DataProvider):
    name = "liquidations"

    def available(self) -> bool:
        return False

    def recent_liquidations(self, symbol: str) -> List[EventRecord]:
        return []


# ── Context assembly for the alpha pipeline ───────────────────────────────────


class MarketContextBuilder:
    """Builds per-symbol context features (earnings_days_away, funding_rate,
    open_interest, basis) from whichever providers are actually available.
    Unavailable feeds simply leave features as None — alphas abstain."""

    def __init__(
        self,
        earnings: Optional[EarningsProvider] = None,
        derivatives: Optional[BinanceDerivativesProvider] = None,
    ) -> None:
        self.earnings = earnings if earnings is not None else YFinanceEarningsProvider()
        self.derivatives = derivatives if derivatives is not None \
            else BinanceDerivativesProvider()

    def build(self, symbols: List[str],
              base_context: Optional[Dict[str, Dict[str, Any]]] = None
              ) -> Dict[str, Dict[str, Any]]:
        context: Dict[str, Dict[str, Any]] = {s: dict((base_context or {}).get(s, {}))
                                              for s in symbols}
        earnings_ok = self.earnings.available()
        deriv_ok = self.derivatives.available()
        for symbol in symbols:
            ctx = context[symbol]
            if earnings_ok and "-" not in symbol:
                ctx.setdefault("earnings_days_away",
                               self.earnings.days_to_next_earnings(symbol))
            if deriv_ok and symbol.endswith("-USD"):
                ctx.setdefault("funding_rate",
                               self.derivatives.current_funding_rate(symbol))
                ctx.setdefault("open_interest",
                               self.derivatives.open_interest(symbol))
                ctx.setdefault("basis", self.derivatives.perp_spot_basis(symbol))
        return context

    def status(self) -> Dict[str, bool]:
        return {
            "earnings": self.earnings.available(),
            "derivatives": self.derivatives.available(),
            "sec_filings": SECFilingsProvider().available(),
            "corporate_actions": CorporateActionsProvider().available(),
            "analyst_changes": AnalystChangesProvider().available(),
            "liquidations": LiquidationsProvider().available(),
        }
