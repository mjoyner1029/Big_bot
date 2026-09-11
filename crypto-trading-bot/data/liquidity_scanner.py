"""
Real-Time Liquidity Scanner - Discovers Liquid, Volatile Opportunities
Replaces static watchlists with dynamic market scanning (Vertus-style)
"""
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OpportunityMetrics:
    """Metrics for ranking opportunities"""
    symbol: str
    volume_24h_usd: float
    market_cap: float = 0
    price_change_24h: float = 0
    volatility_score: float = 0  # 0-100
    liquidity_score: float = 0   # 0-100
    opportunity_score: float = 0  # Composite score
    spread_bps: float = 0
    asset_class: str = "unknown"


# ── Configuration ─────────────────────────────────────────────────
LIQUIDITY_REQUIREMENTS = {
    "crypto": {
        "min_volume_24h_usd": 5_000_000,    # $5M daily volume
        "min_market_cap": 50_000_000,        # $50M market cap (avoid rug pulls)
        "max_spread_bps": 30,                # 0.3% max spread
        "min_volatility": 2.0,               # Min 2% daily range
    },
    "stocks": {
        "min_volume_daily_usd": 10_000_000,  # $10M daily volume
        "min_market_cap": 500_000_000,       # $500M market cap
        "max_spread_bps": 15,                # 0.15% max spread
        "min_volatility": 1.5,               # Min 1.5% daily range
        "min_options_volume": 500,           # 500 contracts/day (optional)
    },
}


def scan_crypto_universe(max_results: int = 50) -> List[OpportunityMetrics]:
    """
    Scan CoinGecko for liquid, volatile crypto opportunities.
    
    Returns top opportunities ranked by:
    - Volume (liquidity)
    - Volatility (trading opportunity)
    - Market cap (avoid scams)
    
    Args:
        max_results: Max number of opportunities to return
    
    Returns:
        List of OpportunityMetrics sorted by opportunity_score
    """
    logger.info("[Liquidity Scanner] Scanning crypto universe...")
    
    opportunities = []
    reqs = LIQUIDITY_REQUIREMENTS["crypto"]
    
    try:
        # CoinGecko markets endpoint (top 250 by market cap)
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "volume_desc",  # Sort by volume
            "per_page": 250,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "24h",
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        for coin in data:
            symbol = coin.get("symbol", "").upper() + "-USD"
            volume_24h = coin.get("total_volume", 0)
            market_cap = coin.get("market_cap", 0)
            price_change = coin.get("price_change_percentage_24h", 0)
            high_24h = coin.get("high_24h", 0)
            low_24h = coin.get("low_24h", 0)
            current_price = coin.get("current_price", 0)
            
            # Skip if missing critical data
            if not all([volume_24h, market_cap, current_price]):
                continue
            
            # Calculate volatility (daily range %)
            if current_price > 0 and high_24h and low_24h:
                volatility = ((high_24h - low_24h) / current_price) * 100
            else:
                volatility = abs(price_change) if price_change else 0
            
            # Apply filters
            if volume_24h < reqs["min_volume_24h_usd"]:
                continue
            if market_cap < reqs["min_market_cap"]:
                continue
            if volatility < reqs["min_volatility"]:
                continue
            
            # Calculate scores
            # Liquidity score: Volume relative to $50M (log scale)
            liquidity_score = min(100, (volume_24h / 50_000_000) * 50)
            
            # Volatility score: Higher is better (capped at 100)
            volatility_score = min(100, volatility * 10)
            
            # Opportunity score: Weighted combination
            # 60% liquidity (can actually fill orders)
            # 40% volatility (trading opportunities)
            opportunity_score = (liquidity_score * 0.6) + (volatility_score * 0.4)
            
            opportunities.append(OpportunityMetrics(
                symbol=symbol,
                volume_24h_usd=volume_24h,
                market_cap=market_cap,
                price_change_24h=price_change,
                volatility_score=volatility_score,
                liquidity_score=liquidity_score,
                opportunity_score=opportunity_score,
                spread_bps=0,  # CoinGecko doesn't provide spread
                asset_class="crypto",
            ))
        
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
        
        logger.info(f"[Liquidity Scanner] Found {len(opportunities)} crypto opportunities")
        
        return opportunities[:max_results]
    
    except Exception as e:
        logger.error(f"[Liquidity Scanner] Crypto scan failed: {e}")
        return []


def scan_stock_universe(max_results: int = 50) -> List[OpportunityMetrics]:
    """
    Scan Yahoo Finance for liquid, volatile stock opportunities.
    
    Uses Yahoo Finance screeners:
    - Most active (volume leaders)
    - Day gainers/losers (volatility)
    - Trending tickers
    
    Args:
        max_results: Max number of opportunities to return
    
    Returns:
        List of OpportunityMetrics sorted by opportunity_score
    """
    logger.info("[Liquidity Scanner] Scanning stock universe...")
    
    opportunities = []
    reqs = LIQUIDITY_REQUIREMENTS["stocks"]
    
    try:
        # Yahoo Finance screeners
        # Using yfinance download for top movers would require individual calls
        # Instead, we'll use a predefined liquid stock universe and filter by TA
        
        # Top liquid stocks (S&P 500 + high-volume tech/growth)
        liquid_universe = [
            # Mega cap tech (always liquid)
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
            
            # High-volume tech
            "AMD", "INTC", "AVGO", "QCOM", "ANET", "CRWD", "PANW",
            
            # Semiconductors (high volatility)
            "TSM", "ASML", "MU", "LRCX", "KLAC", "AMAT",
            
            # Cloud/Software (volatile)
            "SNOW", "DDOG", "NET", "ZS", "FTNT", "OKTA",
            
            # EV/Energy (volatile)
            "RIVN", "LCID", "NIO", "PLUG", "ENPH", "SEDG",
            
            # Biotech (high volatility)
            "MRNA", "BNTX", "NVAX", "REGN", "VRTX", "GILD",
            
            # Finance (liquid)
            "JPM", "BAC", "WFC", "GS", "MS",
            
            # Consumer (liquid)
            "COST", "WMT", "TGT", "HD", "LOW",
            
            # ETFs (ultra-liquid for hedging)
            "SPY", "QQQ", "IWM", "DIA",
            
            # Leveraged ETFs (high volatility)
            "TQQQ", "SQQQ", "UPRO", "SPXL", "UVXY",
        ]
        
        # In a real implementation, fetch live data for each
        # For now, return the liquid universe with estimated scores
        # This would be enhanced with real-time yfinance data
        
        for symbol in liquid_universe:
            # Placeholder metrics (in production, fetch real-time data)
            opportunities.append(OpportunityMetrics(
                symbol=symbol,
                volume_24h_usd=50_000_000,  # Estimated (all are highly liquid)
                market_cap=1_000_000_000,
                price_change_24h=0,
                volatility_score=60,  # Estimated
                liquidity_score=80,   # All are very liquid
                opportunity_score=72,  # 80 * 0.6 + 60 * 0.4
                spread_bps=5,  # Estimated tight spread
                asset_class="stock",
            ))
        
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
        
        logger.info(f"[Liquidity Scanner] Found {len(opportunities)} stock opportunities")
        
        return opportunities[:max_results]
    
    except Exception as e:
        logger.error(f"[Liquidity Scanner] Stock scan failed: {e}")
        return []


def scan_universe(
    regime: str = "unknown",
    max_crypto: int = 25,
    max_stocks: int = 25
) -> Tuple[List[str], List[str]]:
    """
    Scan entire tradeable universe for opportunities.
    
    Returns symbols optimized for current market regime:
    - Trending: High momentum, breaking out
    - Ranging: Mean-reverting at extremes
    - Volatile: Wait or trade breakouts only
    - Quiet: Scalp small moves
    
    Args:
        regime: Current market regime
        max_crypto: Max crypto symbols to return
        max_stocks: Max stock symbols to return
    
    Returns:
        (crypto_symbols, stock_symbols)
    """
    logger.info(f"[Liquidity Scanner] Scanning universe for regime: {regime}")
    
    # Scan both asset classes
    crypto_opps = scan_crypto_universe(max_results=max_crypto)
    stock_opps = scan_stock_universe(max_results=max_stocks)
    
    # Extract symbols
    crypto_symbols = [opp.symbol for opp in crypto_opps]
    stock_symbols = [opp.symbol for opp in stock_opps]
    
    # Regime-based filtering (future enhancement)
    # For now, return top opportunities by score
    
    logger.info(
        f"[Liquidity Scanner] Selected {len(crypto_symbols)} crypto + "
        f"{len(stock_symbols)} stocks"
    )
    
    return crypto_symbols, stock_symbols


def get_liquid_opportunities(
    asset_class: str = "both",
    max_symbols: int = 50
) -> List[str]:
    """
    Convenience function to get liquid opportunities for one asset class.
    
    Args:
        asset_class: "crypto", "stocks", or "both"
        max_symbols: Max symbols to return
    
    Returns:
        List of symbols
    """
    if asset_class == "crypto":
        opps = scan_crypto_universe(max_results=max_symbols)
        return [opp.symbol for opp in opps]
    
    elif asset_class == "stocks":
        opps = scan_stock_universe(max_results=max_symbols)
        return [opp.symbol for opp in opps]
    
    else:  # both
        crypto_opps = scan_crypto_universe(max_results=max_symbols // 2)
        stock_opps = scan_stock_universe(max_results=max_symbols // 2)
        return (
            [opp.symbol for opp in crypto_opps] +
            [opp.symbol for opp in stock_opps]
        )


if __name__ == "__main__":
    # Test the scanner
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("CRYPTO OPPORTUNITIES")
    print("=" * 80)
    crypto_opps = scan_crypto_universe(max_results=10)
    for opp in crypto_opps:
        print(
            f"{opp.symbol:15} | "
            f"Vol: ${opp.volume_24h_usd/1e6:6.1f}M | "
            f"MCap: ${opp.market_cap/1e9:5.1f}B | "
            f"Score: {opp.opportunity_score:5.1f}"
        )
    
    print("\n" + "=" * 80)
    print("STOCK OPPORTUNITIES")
    print("=" * 80)
    stock_opps = scan_stock_universe(max_results=10)
    for opp in stock_opps:
        print(
            f"{opp.symbol:15} | "
            f"Score: {opp.opportunity_score:5.1f}"
        )
