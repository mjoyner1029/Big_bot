"""Correlation & exposure management — prevents correlated position stacking.

Professional traders never treat 4 long crypto positions as 4 independent
bets.  This module:
  • Groups symbols into correlation clusters (crypto-major, crypto-alt,
    tech-mega, broad-market, etc.)
  • Limits net directional exposure per cluster
  • Computes rolling pairwise correlation from recent price data
  • Blocks new positions that would push cluster exposure beyond limits
"""
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config.config import CONFIG, is_crypto

# ── Static correlation clusters ───────────────────────────────────
# Assets in the same cluster are assumed to be highly correlated and
# their combined exposure is capped.
CLUSTER_MAP: Dict[str, str] = {
    # Crypto — majors move together (~0.8+ correlation)
    "BTC-USD": "crypto_major",
    "ETH-USD": "crypto_major",
    "SOL-USD": "crypto_major",
    "BNB-USD": "crypto_major",
    "AVAX-USD": "crypto_major",

    # Crypto — alts are correlated with majors but have idiosyncratic risk
    "XRP-USD": "crypto_alt",
    "ADA-USD": "crypto_alt",
    "DOGE-USD": "crypto_alt",
    "DOT-USD": "crypto_alt",
    "LINK-USD": "crypto_alt",

    # Tech megacaps
    "AAPL": "tech_mega",
    "MSFT": "tech_mega",
    "NVDA": "tech_mega",
    "GOOG": "tech_mega",
    "AMZN": "tech_mega",
    "META": "tech_mega",
    "TSLA": "tech_mega",

    # Broad market ETFs
    "SPY": "broad_market",
    "QQQ": "broad_market",

    # Financials
    "JPM": "financials",
}

# Default cluster for unknown symbols
_DEFAULT_CRYPTO_CLUSTER = "crypto_other"
_DEFAULT_STOCK_CLUSTER = "stock_other"


def get_cluster(symbol: str) -> str:
    """Return the correlation cluster for a symbol."""
    if symbol in CLUSTER_MAP:
        return CLUSTER_MAP[symbol]
    return _DEFAULT_CRYPTO_CLUSTER if is_crypto(symbol) else _DEFAULT_STOCK_CLUSTER


# ── Asset-class level grouping ────────────────────────────────────

def get_asset_class_group(symbol: str) -> str:
    """Broad grouping: 'crypto' or 'stock'."""
    return "crypto" if is_crypto(symbol) else "stock"


# ── Exposure calculation ──────────────────────────────────────────

def compute_cluster_exposure(
    open_positions: List[Dict],
    current_prices: Dict[str, float],
    total_equity: float,
) -> Dict[str, Dict]:
    """Compute directional exposure per cluster.

    Returns:
        {cluster_name: {
            'long_pct': float,   # % of equity in long positions
            'short_pct': float,  # % of equity in short positions
            'net_pct': float,    # net directional exposure
            'gross_pct': float,  # total absolute exposure
            'count': int,        # number of positions
            'symbols': [str],    # symbols in this cluster
        }}
    """
    if total_equity <= 0:
        return {}

    clusters: Dict[str, Dict] = defaultdict(
        lambda: {"long_pct": 0.0, "short_pct": 0.0, "net_pct": 0.0,
                 "gross_pct": 0.0, "count": 0, "symbols": []}
    )

    for pos in open_positions:
        sym = pos["symbol"]
        cluster = get_cluster(sym)
        price = current_prices.get(sym, pos["entry_price"])
        value = pos["qty"] * price
        pct = value / total_equity

        clusters[cluster]["count"] += 1
        clusters[cluster]["symbols"].append(sym)

        if pos["side"] == "buy":
            clusters[cluster]["long_pct"] += pct
        else:
            clusters[cluster]["short_pct"] += pct

    # Compute net and gross
    for c in clusters.values():
        c["net_pct"] = c["long_pct"] - c["short_pct"]
        c["gross_pct"] = c["long_pct"] + c["short_pct"]

    return dict(clusters)


def compute_asset_class_exposure(
    open_positions: List[Dict],
    current_prices: Dict[str, float],
    total_equity: float,
) -> Dict[str, float]:
    """Compute total directional exposure per asset class (crypto / stock)."""
    if total_equity <= 0:
        return {}

    exposure: Dict[str, float] = defaultdict(float)
    for pos in open_positions:
        sym = pos["symbol"]
        group = get_asset_class_group(sym)
        price = current_prices.get(sym, pos["entry_price"])
        value = pos["qty"] * price / total_equity
        if pos["side"] == "buy":
            exposure[group] += value
        else:
            exposure[group] -= value

    return dict(exposure)


def can_add_to_cluster(
    symbol: str,
    side: str,
    position_pct: float,
    open_positions: List[Dict],
    current_prices: Dict[str, float],
    total_equity: float,
) -> Tuple[bool, str]:
    """Check if adding a new position would exceed cluster exposure limits.

    Args:
        symbol:          The symbol to add.
        side:            'buy' or 'sell'.
        position_pct:    The new position's % of equity.
        open_positions:  Current open positions.
        current_prices:  Latest prices.
        total_equity:    Total portfolio equity.
    Returns:
        (allowed: bool, reason: str)
    """
    max_cluster_pct = CONFIG.get("max_cluster_exposure_pct", 0.40)
    max_asset_class_pct = CONFIG.get("max_asset_class_exposure_pct", 0.60)

    cluster = get_cluster(symbol)
    cluster_exposure = compute_cluster_exposure(open_positions, current_prices, total_equity)

    # Check cluster limit
    if cluster in cluster_exposure:
        current_gross = cluster_exposure[cluster]["gross_pct"]
        new_gross = current_gross + position_pct
        if new_gross > max_cluster_pct:
            reason = (
                f"cluster '{cluster}' would be {new_gross*100:.1f}% "
                f"(limit={max_cluster_pct*100:.0f}%)"
            )
            logging.warning(f"[Correlation] BLOCKED {symbol}: {reason}")
            return False, reason

    # Check same-direction stacking in cluster
    max_same_dir = CONFIG.get("max_same_direction_per_cluster", 2)
    if cluster in cluster_exposure:
        same_dir_count = 0
        for pos in open_positions:
            if get_cluster(pos["symbol"]) == cluster and pos["side"] == side:
                same_dir_count += 1
        if same_dir_count >= max_same_dir:
            reason = (
                f"already {same_dir_count} {side} positions in cluster '{cluster}' "
                f"(limit={max_same_dir})"
            )
            logging.warning(f"[Correlation] BLOCKED {symbol}: {reason}")
            return False, reason

    # Check asset-class exposure
    asset_class = get_asset_class_group(symbol)
    ac_exposure = compute_asset_class_exposure(open_positions, current_prices, total_equity)
    current_ac = ac_exposure.get(asset_class, 0.0)
    if side == "buy":
        new_ac = current_ac + position_pct
    else:
        new_ac = current_ac - position_pct

    if abs(new_ac) > max_asset_class_pct:
        reason = (
            f"asset class '{asset_class}' would be {abs(new_ac)*100:.1f}% "
            f"(limit={max_asset_class_pct*100:.0f}%)"
        )
        logging.warning(f"[Correlation] BLOCKED {symbol}: {reason}")
        return False, reason

    return True, "ok"


# ── Dynamic rolling correlation ───────────────────────────────────

def compute_rolling_correlation(
    price_data: Dict[str, pd.DataFrame],
    window: int = 48,
) -> Optional[pd.DataFrame]:
    """Compute a pairwise correlation matrix from recent close prices.

    Args:
        price_data: {symbol: OHLCV DataFrame} — at least 'Close' column.
        window:     Number of bars for the rolling window (default 48h).
    Returns:
        Correlation matrix DataFrame, or None if insufficient data.
    """
    closes = {}
    for sym, df in price_data.items():
        if df is not None and "Close" in df.columns and len(df) >= window:
            closes[sym] = df["Close"].tail(window).pct_change().dropna()

    if len(closes) < 2:
        return None

    combined = pd.DataFrame(closes)
    return combined.corr()


def find_highly_correlated(
    symbol: str,
    open_positions: List[Dict],
    price_data: Dict[str, pd.DataFrame],
    threshold: float = 0.75,
) -> List[Tuple[str, float]]:
    """Find open positions that are highly correlated with a candidate symbol.

    Returns:
        List of (symbol, correlation) pairs exceeding the threshold.
    """
    held_symbols = [p["symbol"] for p in open_positions]
    if not held_symbols or symbol not in price_data:
        return []

    corr_matrix = compute_rolling_correlation(price_data)
    if corr_matrix is None or symbol not in corr_matrix.columns:
        # Fall back to static cluster correlation
        candidate_cluster = get_cluster(symbol)
        return [
            (s, 0.85)  # estimated cluster correlation
            for s in held_symbols
            if get_cluster(s) == candidate_cluster
        ]

    highly_corr = []
    for held_sym in held_symbols:
        if held_sym in corr_matrix.columns:
            corr_val = corr_matrix.loc[symbol, held_sym]
            if abs(corr_val) >= threshold:
                highly_corr.append((held_sym, round(corr_val, 3)))

    return highly_corr
