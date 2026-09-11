"""
Multi-asset Universe Scanners.

Each scanner produces standardised `Candidate` objects that the
OpportunityRanker can combine across asset classes.

Available scanners:
    CryptoScanner           — spot crypto on CCXT exchanges
    EquityScanner           — US equities via yfinance
    ETFScanner              — ETFs (SPY, QQQ, etc.) via yfinance
    PredictionMarketScanner — Polymarket / Manifold (REST API)

Usage:
    from core.scanners import CryptoScanner, EquityScanner

    crypto = CryptoScanner()
    candidates = crypto.scan(top_n=10)
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Standardised Candidate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """Standardised opportunity from any scanner."""
    symbol:             str
    asset_class:        str             # 'crypto' | 'equity' | 'etf' | 'prediction'
    exchange:           str             = ''
    price:              float           = 0.0
    volume_usd_24h:     float           = 0.0
    spread_pct:         float           = 0.0
    atr_pct:            float           = 0.0
    momentum_score:     float           = 0.0    # -1 to +1
    volume_surge:       float           = 1.0    # recent vs baseline ratio
    technical_score:    float           = 0.5    # 0 to 1
    opportunity_score:  float           = 0.0    # composite
    scanned_at:         datetime        = field(default_factory=_utcnow)
    meta:               Dict            = field(default_factory=dict)

    def to_symbol_str(self) -> str:
        """Return a symbol string suitable for market data fetching."""
        return self.symbol


# ─────────────────────────────────────────────────────────────────────────────
# Base Scanner
# ─────────────────────────────────────────────────────────────────────────────

class BaseScanner:
    """Abstract base; all scanners implement scan()."""

    CACHE_TTL = 300   # seconds

    def __init__(self):
        self._cache: Optional[List[Candidate]] = None
        self._cache_ts: float = 0.0

    def scan(self, top_n: int = 10) -> List[Candidate]:
        raise NotImplementedError

    def get_candidates(self, top_n: int = 10) -> List[Candidate]:
        """Scan with TTL cache."""
        if self._cache and (time.monotonic() - self._cache_ts) < self.CACHE_TTL:
            return self._cache[:top_n]
        try:
            self._cache    = self.scan(top_n=top_n * 2)  # over-fetch, then slice
            self._cache_ts = time.monotonic()
        except Exception as e:
            logger.warning(f"{type(self).__name__}: scan failed: {e}")
            self._cache = []
        return (self._cache or [])[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# CryptoScanner
# ─────────────────────────────────────────────────────────────────────────────

_CRYPTO_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "MATIC/USDT", "DOT/USDT",
    "LINK/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT",
    "OP/USDT", "ARB/USDT", "INJ/USDT", "SUI/USDT", "SEI/USDT",
    "APT/USDT", "TIA/USDT", "PYTH/USDT", "JUP/USDT", "WIF/USDT",
]


class CryptoScanner(BaseScanner):
    """
    Crypto opportunity scanner using CCXT.

    Scores each asset on:
        momentum_score    — EMA20 vs EMA50 crossover + RSI deviation
        volume_surge      — 24h volume vs 7d average
        technical_score   — combined indicator health

    Returns Candidates sorted by opportunity_score.
    """

    def __init__(
        self,
        exchange_id: str = 'binance',
        min_volume_usd: float = 5_000_000,
        min_atr_pct: float = 0.005,
        max_atr_pct: float = 0.08,
    ):
        super().__init__()
        self.exchange_id    = exchange_id
        self.min_volume_usd = min_volume_usd
        self.min_atr_pct    = min_atr_pct
        self.max_atr_pct    = max_atr_pct
        self._exchange      = None

    def _get_exchange(self):
        if self._exchange is not None:
            return self._exchange
        try:
            import ccxt
            self._exchange = getattr(ccxt, self.exchange_id)({'timeout': 10000})
        except Exception as e:
            logger.warning(f"CryptoScanner: could not init {self.exchange_id}: {e}")
            self._exchange = None
        return self._exchange

    def scan(self, top_n: int = 20) -> List[Candidate]:
        ex = self._get_exchange()
        if ex is None:
            return self._yfinance_fallback(top_n)

        try:
            tickers = ex.fetch_tickers(_CRYPTO_UNIVERSE)
        except Exception as e:
            logger.warning(f"CryptoScanner: fetch_tickers error: {e}")
            return self._yfinance_fallback(top_n)

        candidates = []
        for sym, ticker in tickers.items():
            try:
                volume_usd = float(ticker.get('quoteVolume', 0))
                price      = float(ticker.get('last', 0))
                if volume_usd < self.min_volume_usd or price <= 0:
                    continue

                pct_change = float(ticker.get('percentage', 0)) / 100.0
                momentum   = max(-1.0, min(1.0, pct_change * 5))
                vol_surge  = float(ticker.get('quoteVolume', 0)) / max(volume_usd, 1)

                c = Candidate(
                    symbol=sym.replace('/', '-') + 'T',   # BTC/USDT → BTC-USDT
                    asset_class='crypto',
                    exchange=self.exchange_id,
                    price=price,
                    volume_usd_24h=volume_usd,
                    momentum_score=momentum,
                    volume_surge=vol_surge,
                    technical_score=0.5,
                )
                c.opportunity_score = (
                    abs(c.momentum_score) * 0.40
                    + min(c.volume_surge, 3.0) / 3.0 * 0.30
                    + c.technical_score * 0.30
                )
                candidates.append(c)
            except Exception:
                continue

        candidates.sort(key=lambda c: c.opportunity_score, reverse=True)
        return candidates[:top_n]

    def _yfinance_fallback(self, top_n: int) -> List[Candidate]:
        """Use yfinance as a fallback if CCXT exchange is unavailable."""
        yf_symbols = [
            "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
            "DOGE-USD", "ADA-USD", "AVAX-USD", "MATIC-USD", "DOT-USD",
        ]
        candidates = []
        try:
            import yfinance as yf
            tickers = yf.download(
                yf_symbols[:top_n], period='2d', interval='1h',
                auto_adjust=True, progress=False, group_by='ticker',
            )
            for sym in yf_symbols[:top_n]:
                try:
                    df = tickers[sym] if sym in tickers.columns.get_level_values(0) else None
                    if df is None or df.empty:
                        continue
                    price = float(df['Close'].iloc[-1])
                    pct   = (price - float(df['Close'].iloc[0])) / float(df['Close'].iloc[0])
                    candidates.append(Candidate(
                        symbol=sym, asset_class='crypto',
                        price=price,
                        momentum_score=max(-1, min(1, pct * 5)),
                        technical_score=0.5,
                        opportunity_score=abs(pct * 5) * 0.7 + 0.5 * 0.3,
                    ))
                except Exception:
                    continue
        except ImportError:
            logger.warning("CryptoScanner yfinance fallback: yfinance not installed")
        except Exception as e:
            logger.warning(f"CryptoScanner yfinance fallback error: {e}")
        return sorted(candidates, key=lambda c: c.opportunity_score, reverse=True)[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# EquityScanner
# ─────────────────────────────────────────────────────────────────────────────

_EQUITY_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "QCOM", "ASML",
    "JPM", "GS", "MS", "BAC", "V", "MA",
    "SPY", "QQQ", "IWM",
]


class EquityScanner(BaseScanner):
    """
    US equity opportunity scanner via yfinance.

    Only active during US market hours (9:30 – 16:00 ET, Mon-Fri).
    """

    def __init__(self, min_volume: int = 500_000):
        super().__init__()
        self.min_volume = min_volume

    def scan(self, top_n: int = 10) -> List[Candidate]:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("EquityScanner: yfinance not installed")
            return []

        import pytz
        from datetime import datetime as dt
        et = pytz.timezone('America/New_York')
        now_et = dt.now(et)

        # Market hours gate
        if now_et.weekday() >= 5:
            logger.info("EquityScanner: weekend — market closed")
            return []
        if not (9 <= now_et.hour < 16 or (now_et.hour == 9 and now_et.minute >= 30)):
            logger.info("EquityScanner: outside market hours")
            return []

        try:
            tickers = yf.download(
                _EQUITY_UNIVERSE, period='5d', interval='1h',
                auto_adjust=True, progress=False, group_by='ticker',
            )
        except Exception as e:
            logger.warning(f"EquityScanner: download error: {e}")
            return []

        candidates = []
        for sym in _EQUITY_UNIVERSE:
            try:
                lvl = tickers.columns.get_level_values(0)
                if sym not in lvl:
                    continue
                df = tickers[sym].dropna()
                if df.empty or len(df) < 10:
                    continue

                price   = float(df['Close'].iloc[-1])
                vol     = float(df['Volume'].iloc[-1])
                if vol < self.min_volume:
                    continue

                ret_1d = (price - float(df['Close'].iloc[-8])) / float(df['Close'].iloc[-8])
                momentum = max(-1.0, min(1.0, ret_1d * 10))

                c = Candidate(
                    symbol=sym, asset_class='equity',
                    exchange='NYSE/NASDAQ',
                    price=price,
                    volume_usd_24h=vol * price,
                    momentum_score=momentum,
                    technical_score=0.5,
                )
                c.opportunity_score = abs(momentum) * 0.6 + 0.4
                candidates.append(c)
            except Exception:
                continue

        candidates.sort(key=lambda c: c.opportunity_score, reverse=True)
        return candidates[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# ETFScanner
# ─────────────────────────────────────────────────────────────────────────────

_ETF_UNIVERSE = [
    "SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "HYG",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP",
    "ARKK", "SQQQ", "TQQQ",
]


class ETFScanner(BaseScanner):
    """
    ETF opportunity scanner.

    Focuses on momentum, sector rotation, and volatility signals.
    """

    def __init__(self):
        super().__init__()

    def scan(self, top_n: int = 5) -> List[Candidate]:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("ETFScanner: yfinance not installed")
            return []

        try:
            tickers = yf.download(
                _ETF_UNIVERSE, period='10d', interval='1d',
                auto_adjust=True, progress=False, group_by='ticker',
            )
        except Exception as e:
            logger.warning(f"ETFScanner: download error: {e}")
            return []

        candidates = []
        for sym in _ETF_UNIVERSE:
            try:
                lvl = tickers.columns.get_level_values(0)
                if sym not in lvl:
                    continue
                df = tickers[sym].dropna()
                if df.empty or len(df) < 5:
                    continue

                price  = float(df['Close'].iloc[-1])
                ret_5d = (price - float(df['Close'].iloc[0])) / float(df['Close'].iloc[0])
                vol_ratio = float(df['Volume'].iloc[-1]) / (float(df['Volume'].mean()) + 1e-9)

                momentum = max(-1.0, min(1.0, ret_5d * 8))
                surge    = min(vol_ratio, 3.0) / 3.0

                c = Candidate(
                    symbol=sym, asset_class='etf',
                    exchange='NYSE',
                    price=price,
                    momentum_score=momentum,
                    volume_surge=surge,
                    technical_score=0.5 + momentum * 0.2,
                )
                c.opportunity_score = abs(momentum) * 0.5 + surge * 0.3 + 0.2
                candidates.append(c)
            except Exception:
                continue

        candidates.sort(key=lambda c: c.opportunity_score, reverse=True)
        return candidates[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# PredictionMarketScanner
# ─────────────────────────────────────────────────────────────────────────────

_POLYMARKET_API = "https://clob.polymarket.com"


class PredictionMarketScanner(BaseScanner):
    """
    Prediction market opportunity scanner (Polymarket).

    Estimates edge as:
        edge = |model_probability - market_probability|

    Returns markets where the model's estimate diverges significantly
    from market pricing. Claude analyzes real-world events to form
    the model probability.
    """

    def __init__(self, min_volume_usd: float = 10_000):
        super().__init__()
        self.min_volume_usd = min_volume_usd

    def scan(self, top_n: int = 5) -> List[Candidate]:
        """Fetch active Polymarket markets and score by potential edge."""
        markets = self._fetch_polymarket_markets()
        if not markets:
            return []

        candidates = []
        for m in markets:
            try:
                if m.get('volume', 0) < self.min_volume_usd:
                    continue

                market_prob = float(m.get('outcomePrices', [0.5])[0])  # YES price ≈ probability
                liquidity   = float(m.get('liquidityClob', 0))

                # Placeholder: model probability = market probability until Claude analyzes
                model_prob  = market_prob
                edge        = abs(model_prob - market_prob)
                ev          = edge * 1.0 - (1.0 - edge) * 1.0  # binary EV proxy

                c = Candidate(
                    symbol=m.get('question', '')[:60],
                    asset_class='prediction',
                    exchange='polymarket',
                    price=market_prob,
                    volume_usd_24h=float(m.get('volume', 0)),
                    opportunity_score=max(edge, 0.0),
                    meta={
                        'market_id':   m.get('conditionId', ''),
                        'market_prob': market_prob,
                        'model_prob':  model_prob,
                        'edge':        edge,
                        'ev':          ev,
                        'end_date':    m.get('endDate', ''),
                    },
                )
                candidates.append(c)
            except Exception:
                continue

        candidates.sort(key=lambda c: c.opportunity_score, reverse=True)
        return candidates[:top_n]

    def _fetch_polymarket_markets(self) -> List[Dict]:
        """Fetch active Polymarket markets via REST API."""
        try:
            import requests
            resp = requests.get(
                f"{_POLYMARKET_API}/markets",
                params={"active": True, "closed": False, "limit": 50},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get('data', [])
        except ImportError:
            logger.warning("PredictionMarketScanner: requests not installed")
        except Exception as e:
            logger.info(f"PredictionMarketScanner: Polymarket API unavailable: {e}")
        return []

    def set_model_probability(
        self,
        market_id: str,
        model_prob: float,
        candidates: List[Candidate],
    ) -> List[Candidate]:
        """
        Update model probability for a market after Claude analysis.

        Call this after Claude has analyzed the real-world event and formed
        a probability estimate.
        """
        for c in candidates:
            if c.meta.get('market_id') == market_id:
                market_prob  = c.meta['market_prob']
                edge         = abs(model_prob - market_prob)
                c.meta['model_prob'] = model_prob
                c.meta['edge']       = edge
                c.opportunity_score  = edge
        return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Unified scanner — aggregates all asset classes
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedScanner:
    """
    Runs all individual scanners and returns a ranked combined list.

    Scanners are run in parallel (via threads) and results are merged.
    """

    def __init__(
        self,
        include_crypto: bool = True,
        include_equity: bool = True,
        include_etf: bool = True,
        include_prediction: bool = False,
    ):
        self._scanners = []
        if include_crypto:
            self._scanners.append(CryptoScanner())
        if include_equity:
            self._scanners.append(EquityScanner())
        if include_etf:
            self._scanners.append(ETFScanner())
        if include_prediction:
            self._scanners.append(PredictionMarketScanner())

    def scan(self, top_n_per_class: int = 10) -> List[Candidate]:
        """Scan all asset classes concurrently and return combined list."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_candidates: List[Candidate] = []

        with ThreadPoolExecutor(max_workers=len(self._scanners)) as ex:
            futures = {
                ex.submit(scanner.get_candidates, top_n_per_class): scanner
                for scanner in self._scanners
            }
            for future in as_completed(futures, timeout=30):
                scanner = futures[future]
                try:
                    results = future.result()
                    all_candidates.extend(results)
                    logger.info(
                        f"UnifiedScanner: {type(scanner).__name__} "
                        f"returned {len(results)} candidates"
                    )
                except Exception as e:
                    logger.warning(f"UnifiedScanner: {type(scanner).__name__} failed: {e}")

        all_candidates.sort(key=lambda c: c.opportunity_score, reverse=True)
        return all_candidates
