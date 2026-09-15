"""
Universe Scanner — dynamic multi-asset opportunity discovery.

Replaces hardcoded symbol lists with a ranked candidate pipeline:

    ┌───────────────────────────────────────────┐
    │  Asset Universe (crypto / stocks / ETFs)  │
    │  fetched from CCXT + Alpaca               │
    └────────────┬──────────────────────────────┘
                 │
    ┌────────────▼──────────────────────────────┐
    │  Filters (run in parallel)                │
    │   • Liquidity   (min 24h volume)          │
    │   • Spread      (max bid-ask %)           │
    │   • Volatility  (ATR / price within band) │
    │   • Catalyst    (upcoming events)         │
    └────────────┬──────────────────────────────┘
                 │
    ┌────────────▼──────────────────────────────┐
    │  Opportunity Ranking                      │
    │   • Momentum score                        │
    │   • Volatility score                      │
    │   • Volume surge score                    │
    │   • Technical setup score                 │
    └────────────┬──────────────────────────────┘
                 │
    ┌────────────▼──────────────────────────────┐
    │  Top-N Candidates → deep analysis         │
    └───────────────────────────────────────────┘

Usage:
    scanner = UniverseScanner(config)
    candidates = scanner.scan(top_n=10)
    # → list of Candidate sorted by opportunity score
"""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScannerConfig:
    # Liquidity filter
    min_volume_usd_24h:   float = 5_000_000      # $5M daily volume minimum
    # Intrabar-range filter (% of price). NOTE: this is a liquidity-risk proxy
    # from OHLC bar range — it is NOT bid/ask spread (no quote data here).
    max_bar_range_pct:    float = 0.05            # 5% max intrabar high-low range
    # Volatility filter (ATR as % of price)
    # This scanner consumes 5-minute bars. A 0.5% floor was calibrated like
    # an hourly threshold and rejected liquid majors in ordinary markets.
    min_atr_pct:          float = 0.0005          # 0.05% min 5m ATR
    max_atr_pct:          float = 0.08            # 8% max ATR (too wild = unmanageable)
    # Momentum window (bars)
    momentum_period:      int   = 14
    # Universe composition
    include_crypto:       bool  = True
    include_stocks:       bool  = False           # enable when Alpaca stock data wired
    include_etfs:         bool  = False
    # Cache TTL (seconds) to avoid hammering exchange APIs
    cache_ttl_seconds:    int   = 300             # 5 minutes
    # Top-N per scan
    top_n:                int   = 10


# ─────────────────────────────────────────────────────────────────────────────
# Candidate dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    symbol:           str
    asset_class:      str          # "crypto" | "stock" | "etf"
    price:            float
    volume_usd_24h:   float
    bar_range_pct:    float        # intrabar (high-low)/price — NOT bid/ask spread
    atr_pct:          float
    momentum_score:   float        # -1 to +1
    volume_surge:     float        # ratio vs 20-bar avg
    technical_score:  float        # 0 to 1 composite
    opportunity_score: float       # final composite (higher = better opportunity)
    notes:            List[str] = field(default_factory=list)

    def __repr__(self):
        return (
            f"Candidate({self.symbol} opp={self.opportunity_score:.2f} "
            f"vol=${self.volume_usd_24h/1e6:.0f}M atr={self.atr_pct:.1%})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Default crypto universe
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CRYPTO_UNIVERSE = [
    # Large cap
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    # Mid cap
    "ADA-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD",
    "ATOM-USD", "NEAR-USD", "FTM-USD", "ALGO-USD", "VET-USD",
    # High momentum / volatile
    "DOGE-USD", "SHIB-USD", "PEPE-USD",
    # DeFi
    "UNI-USD", "AAVE-USD", "CRV-USD", "MKR-USD",
    # Layer 2 / infrastructure
    "ARB-USD", "OP-USD", "IMX-USD",
]


# ─────────────────────────────────────────────────────────────────────────────
# UniverseScanner
# ─────────────────────────────────────────────────────────────────────────────

class UniverseScanner:
    """Scans the full asset universe and returns ranked trading candidates."""

    def __init__(self, config: Optional[ScannerConfig] = None):
        self.config  = config or ScannerConfig()
        self._cache: Dict[str, Tuple[datetime, List[Candidate]]] = {}
        self.last_rejections: Dict[str, str] = {}

    # ── Main entry point ──────────────────────────────────────────────────────

    def scan(self, top_n: Optional[int] = None) -> List[Candidate]:
        """
        Run a full universe scan.

        Returns candidates sorted by opportunity_score (descending).
        Results are cached for config.cache_ttl_seconds.
        """
        top_n = top_n or self.config.top_n
        cache_key = f"scan_{top_n}"

        # Return cached result if fresh
        if cache_key in self._cache:
            cached_at, cached_result = self._cache[cache_key]
            age = (datetime.now(timezone.utc) - cached_at).total_seconds()
            if age < self.config.cache_ttl_seconds:
                logger.debug(f"UniverseScanner: returning cached result ({age:.0f}s old)")
                return cached_result[:top_n]

        logger.info("UniverseScanner: starting full universe scan...")
        self.last_rejections = {}
        all_candidates = []

        if self.config.include_crypto:
            crypto_candidates = self._scan_crypto()
            all_candidates.extend(crypto_candidates)
            logger.info(f"UniverseScanner: {len(crypto_candidates)} crypto candidates")

        # Sort by opportunity score (descending)
        all_candidates.sort(key=lambda c: c.opportunity_score, reverse=True)

        self._cache[cache_key] = (datetime.now(timezone.utc), all_candidates)

        top = all_candidates[:top_n]
        logger.info(
            f"UniverseScanner: top {len(top)} candidates: "
            + ", ".join(f"{c.symbol}({c.opportunity_score:.2f})" for c in top)
        )
        if self.last_rejections:
            logger.info("UniverseScanner rejection reasons: %s", self.last_rejections)
        return top

    def get_symbols(self, top_n: Optional[int] = None) -> List[str]:
        """Convenience wrapper — returns just the symbol list."""
        return [c.symbol for c in self.scan(top_n=top_n)]

    # ── Crypto scanner ────────────────────────────────────────────────────────

    def _scan_crypto(self) -> List[Candidate]:
        """Scan the crypto universe. Falls back gracefully if exchange is unavailable."""
        from data.fetcher import fetch_latest_market_data

        candidates = []
        for symbol in DEFAULT_CRYPTO_UNIVERSE:
            try:
                candidate = self._evaluate_symbol(symbol, asset_class="crypto")
                if candidate is not None:
                    candidates.append(candidate)
            except Exception as e:
                self.last_rejections[symbol] = f"SCAN_ERROR: {e}"
                logger.warning(f"UniverseScanner: skipping {symbol} — {e}")

        return candidates

    # ── Symbol evaluation ─────────────────────────────────────────────────────

    def _evaluate_symbol(self, symbol: str, asset_class: str) -> Optional[Candidate]:
        """
        Fetch data for one symbol, apply filters, and compute scores.
        Returns None if the symbol fails any filter.
        """
        from data.fetcher import fetch_latest_market_data
        df = fetch_latest_market_data(symbol, period="5d", interval="5m")

        def reject(reason):
            self.last_rejections[symbol] = reason
            return None

        if df is None or len(df) < 50:
            return reject("INSUFFICIENT_HISTORY")

        price  = float(df["close"].iloc[-1])
        if price <= 0:
            return reject("INVALID_PRICE")

        # ── Liquidity filter ──────────────────────────────────────────────────
        # Approximate 24h volume in USD from 5m bars (288 bars = 1 day)
        recent = df.iloc[-288:] if len(df) >= 288 else df
        vol_usd_24h = float((recent["volume"] * recent["close"]).sum())
        if vol_usd_24h < self.config.min_volume_usd_24h:
            return reject(f"LOW_VOLUME: ${vol_usd_24h:,.0f} < ${self.config.min_volume_usd_24h:,.0f}")

        # ── Intrabar-range filter (OHLC liquidity-risk proxy, NOT bid/ask) ────
        last_bar      = df.iloc[-1]
        bar_range_pct = (float(last_bar["high"]) - float(last_bar["low"])) / price
        if bar_range_pct > self.config.max_bar_range_pct:
            return reject(f"BAR_RANGE: {bar_range_pct:.3%} > {self.config.max_bar_range_pct:.3%}")

        # ── ATR-based volatility filter ───────────────────────────────────────
        atr_pct = self._compute_atr_pct(df, price, period=14)
        if atr_pct < self.config.min_atr_pct or atr_pct > self.config.max_atr_pct:
            return reject(f"ATR_OUT_OF_RANGE: {atr_pct:.3%}, allowed "
                          f"{self.config.min_atr_pct:.3%}–{self.config.max_atr_pct:.3%}")

        # ── Scoring ───────────────────────────────────────────────────────────
        momentum     = self._momentum_score(df)
        volume_surge = self._volume_surge(df)
        technical    = self._technical_score(df)

        # Opportunity score: weighted composite
        # Absolute momentum is valuable for trend-following
        # Volume surge confirms institutional interest
        # Technical score measures setup quality
        opp_score = (
            abs(momentum) * 0.40
            + min(volume_surge / 3.0, 1.0) * 0.30
            + technical * 0.30
        )

        notes = []
        if volume_surge > 2.0:
            notes.append(f"volume surge {volume_surge:.1f}x")
        if abs(momentum) > 0.5:
            direction = "bullish" if momentum > 0 else "bearish"
            notes.append(f"strong {direction} momentum")

        return Candidate(
            symbol=symbol,
            asset_class=asset_class,
            price=price,
            volume_usd_24h=vol_usd_24h,
            bar_range_pct=bar_range_pct,
            atr_pct=atr_pct,
            momentum_score=momentum,
            volume_surge=volume_surge,
            technical_score=technical,
            opportunity_score=opp_score,
            notes=notes,
        )

    # ── Technical scoring helpers ─────────────────────────────────────────────

    def _compute_atr_pct(self, df: pd.DataFrame, price: float, period: int = 14) -> float:
        """ATR as percentage of current price."""
        try:
            high  = df["high"].values
            low   = df["low"].values
            close = df["close"].values
            trs = [
                max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                for i in range(1, len(df))
            ]
            atr = np.mean(trs[-period:]) if len(trs) >= period else np.mean(trs)
            return float(atr / price)
        except Exception:
            return 0.0

    def _momentum_score(self, df: pd.DataFrame) -> float:
        """
        Momentum score in [-1, +1].
        Combines: price ROC, EMA alignment, RSI normalization.
        """
        try:
            close = df["close"].values
            n = len(close)
            if n < 50:
                return 0.0

            # 14-bar rate of change
            roc = (close[-1] - close[-14]) / close[-14] if close[-14] > 0 else 0.0
            roc_score = np.clip(roc * 10, -1, 1)  # scale: ±10% ROC → ±1

            # EMA alignment (20 vs 50)
            ema20 = pd.Series(close).ewm(span=20).mean().iloc[-1]
            ema50 = pd.Series(close).ewm(span=50).mean().iloc[-1]
            ema_score = 1.0 if (ema20 > ema50 and close[-1] > ema20) else \
                       -1.0 if (ema20 < ema50 and close[-1] < ema20) else 0.0

            # RSI (0–100) → normalized to [-1, +1]
            delta   = np.diff(close[-15:])
            gains   = delta[delta > 0]
            losses  = -delta[delta < 0]
            avg_g   = gains.mean()   if len(gains)   > 0 else 0.001
            avg_l   = losses.mean()  if len(losses)  > 0 else 0.001
            rs      = avg_g / avg_l
            rsi     = 100 - 100 / (1 + rs)
            rsi_score = (rsi - 50) / 50  # center on 0

            return float(np.clip(roc_score * 0.4 + ema_score * 0.4 + rsi_score * 0.2, -1, 1))
        except Exception:
            return 0.0

    def _volume_surge(self, df: pd.DataFrame) -> float:
        """Ratio of recent 5-bar volume to 20-bar average."""
        try:
            vol = df["volume"].values
            if len(vol) < 25:
                return 1.0
            recent_avg = vol[-5:].mean()
            baseline   = vol[-25:-5].mean()
            return float(recent_avg / baseline) if baseline > 0 else 1.0
        except Exception:
            return 1.0

    def _technical_score(self, df: pd.DataFrame) -> float:
        """
        0–1 composite technical setup quality.
        Checks: trend clarity, momentum alignment, volume confirmation.
        """
        score = 0.0
        try:
            close  = df["close"].values
            volume = df["volume"].values

            if len(close) < 50:
                return 0.5

            ema20 = pd.Series(close).ewm(span=20).mean().iloc[-1]
            ema50 = pd.Series(close).ewm(span=50).mean().iloc[-1]

            # Clear trend: EMAs aligned
            if ema20 > ema50 * 1.005 or ema20 < ema50 * 0.995:
                score += 0.3

            # Price above/below both EMAs (trend clarity)
            if close[-1] > ema20 > ema50 or close[-1] < ema20 < ema50:
                score += 0.3

            # Volume confirmation
            if volume[-5:].mean() > volume[-20:].mean() * 1.2:
                score += 0.2

            # Recent swing: price has moved >1% in last 3 bars
            if abs(close[-1] - close[-4]) / close[-4] > 0.01:
                score += 0.2

        except Exception:
            return 0.5

        return float(np.clip(score, 0.0, 1.0))
