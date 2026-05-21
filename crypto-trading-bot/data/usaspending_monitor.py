"""
USAspending.gov Contract Monitor
=================================
Monitors federal contract awards in real-time for trading signals.

Strategy: Detect major government contracts to small/mid-cap companies
BEFORE mainstream media reports them. Public data, but real-time monitoring
provides an information edge.
"""

import requests
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

logger = logging.getLogger(__name__)


class USASpendingMonitor:
    """
    Monitors USAspending.gov for federal contract awards that could
    impact stock prices before the information becomes widely known.
    """
    
    BASE_URL = "https://api.usaspending.gov/api/v2"
    
    def __init__(
        self,
        min_award_amount: float = 50_000_000,  # $50M minimum
        max_award_amount: float = 500_000_000,  # $500M max (avoid mega-caps)
        lookback_days: int = 7,  # Check last 7 days (not just 1 day)
        cache_hours: int = 1,  # Cache results for 1 hour
    ):
        """
        Args:
            min_award_amount: Minimum contract value to track ($50M default)
            max_award_amount: Maximum contract value (avoid mega-cap news)
            lookback_days: How far back to check for new awards
            cache_hours: How long to cache results to avoid duplicate signals
        """
        self.min_award_amount = min_award_amount
        self.max_award_amount = max_award_amount
        self.lookback_days = lookback_days
        self.cache_hours = cache_hours
        
        # Track recently seen awards to avoid duplicate signals
        self._seen_awards = {}  # award_id -> timestamp
        self._last_fetch = None
        self._cached_results = []
        
    def fetch_recent_awards(self) -> List[Dict]:
        """
        Fetch recent federal contract awards from USAspending.gov.
        
        Returns:
            List of contract awards with company name, amount, date, etc.
        """
        # Use cache if available and fresh
        if self._last_fetch and self._cached_results:
            elapsed = (datetime.now() - self._last_fetch).total_seconds() / 3600
            if elapsed < self.cache_hours:
                logger.info(f"Using cached contract data ({elapsed:.1f}h old)")
                return self._cached_results
        
        try:
            # Calculate date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
            
            # USAspending API v2 - Search for recent awards
            endpoint = f"{self.BASE_URL}/search/spending_by_award/"
            
            # Correct API v2 format based on official docs
            payload = {
                "filters": {
                    "time_period": [
                        {
                            "start_date": start_date,
                            "end_date": end_date,
                            "date_type": "action_date"  # When contract was signed
                        }
                    ],
                    "award_type_codes": [
                        "B",   # Purchase Order (new equipment/services)
                        "C",   # Delivery Order (specific task orders)
                        "D",   # Definitive Contract (firm contracts)
                        # Excluding "A" (BPA calls) as they're often modifications
                    ],
                    "award_amounts": [
                        {
                            "lower_bound": self.min_award_amount,
                            "upper_bound": self.max_award_amount
                        }
                    ]
                },
                "fields": [
                    "Award ID",
                    "Recipient Name",
                    "recipient_id",  # DUNS or UEI
                    "Award Amount",
                    "Award Type",
                    "Awarding Agency",
                    "Start Date",
                    "Description",
                    "naics_code",
                    "naics_description"
                ],
                "page": 1,
                "limit": 100,  # Top 100 recent awards
                "sort": "Award Amount",
                "order": "desc"
            }
            
            logger.info(f"Fetching contracts from {start_date} to {end_date} (${self.min_award_amount/1e6:.0f}M - ${self.max_award_amount/1e6:.0f}M)")
            
            response = requests.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            awards = []
            if "results" in data:
                for result in data["results"]:
                    # Extract fields (API returns various formats)
                    award = {
                        "award_id": result.get("Award ID", result.get("internal_id", "")),
                        "company_name": result.get("Recipient Name", result.get("recipient_name", "")),
                        "amount": float(result.get("Award Amount", result.get("total_obligation", 0))),
                        "award_type": result.get("Award Type", result.get("type", "")),
                        "agency": result.get("Awarding Agency", result.get("awarding_agency_name", "")),
                        "date": result.get("Start Date", result.get("period_of_performance_start_date", "")),
                        "description": str(result.get("Description", result.get("description", "")))[:200],
                    }
                    
                    # Skip if amount is outside our range (double-check)
                    if award["amount"] < self.min_award_amount or award["amount"] > self.max_award_amount:
                        continue
                    
                    # Filter out if we've seen this recently
                    award_id = award["award_id"]
                    if award_id and award_id in self._seen_awards:
                        last_seen = self._seen_awards[award_id]
                        hours_since = (datetime.now() - last_seen).total_seconds() / 3600
                        if hours_since < 24:  # Don't repeat within 24h
                            continue
                    
                    awards.append(award)
                    if award_id:
                        self._seen_awards[award_id] = datetime.now()
                
                logger.info(f"Found {len(awards)} new contract awards (total results: {data.get('page_metadata', {}).get('total', 'unknown')})")
            else:
                logger.warning(f"No 'results' key in USAspending response. Keys: {list(data.keys())}")
            
            # Update cache
            self._cached_results = awards
            self._last_fetch = datetime.now()
            
            return awards
            
        except requests.exceptions.RequestException as e:
            logger.error(f"USAspending API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"API Error Details: {error_detail}")
                except:
                    logger.error(f"API Response Text: {e.response.text[:500]}")
            return []
        except Exception as e:
            logger.error(f"Error fetching contract awards: {e}", exc_info=True)
            return []
    
    def clean_company_name(self, name: str) -> str:
        """
        Clean company name for ticker matching.
        Remove common suffixes, legal terms, etc.
        """
        if not name:
            return ""
        
        # Convert to uppercase for matching
        name = name.upper().strip()
        
        # Remove common suffixes
        suffixes = [
            " INC", " INC.", " INCORPORATED",
            " LLC", " LLC.", " L.L.C.",
            " CORP", " CORP.", " CORPORATION",
            " LTD", " LTD.", " LIMITED",
            " CO", " CO.", " COMPANY",
            " PLC", " P.L.C.",
            " LP", " L.P.",
            " THE",
        ]
        
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
        
        return name
    
    def get_trading_signals(
        self,
        stock_watchlist: List[str],
        ticker_to_company_map: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Generate trading signals from recent contract awards.
        
        Args:
            stock_watchlist: List of stock tickers we're monitoring
            ticker_to_company_map: Optional mapping of ticker -> company name
        
        Returns:
            List of trading signals with ticker, signal, reason, amount
        """
        awards = self.fetch_recent_awards()
        
        if not awards:
            return []
        
        signals = []
        
        for award in awards:
            company_name = self.clean_company_name(award["company_name"])
            amount = award["amount"]
            
            # Try to match to a ticker in our watchlist
            matched_ticker = None
            
            # If we have a ticker->company mapping, use it
            if ticker_to_company_map:
                for ticker, company in ticker_to_company_map.items():
                    if ticker in stock_watchlist:
                        clean_mapped = self.clean_company_name(company)
                        # Fuzzy match - company name contains or matches
                        if company_name in clean_mapped or clean_mapped in company_name:
                            matched_ticker = ticker
                            break
            
            # Fallback: Simple substring matching
            if not matched_ticker:
                for ticker in stock_watchlist:
                    # Remove common suffix like -USD
                    base_ticker = ticker.split("-")[0]
                    if base_ticker in company_name:
                        matched_ticker = ticker
                        break
            
            if matched_ticker:
                signal = {
                    "ticker": matched_ticker,
                    "signal": "BUY",
                    "strength": "STRONG",  # Government contracts are high-confidence signals
                    "reason": f"Gov contract: ${amount/1e6:.1f}M from {award['agency']}",
                    "source": "USAspending.gov",
                    "contract_amount": amount,
                    "award_date": award["date"],
                    "description": award["description"],
                    "timestamp": datetime.now().isoformat(),
                }
                
                signals.append(signal)
                logger.info(f"CONTRACT SIGNAL: {matched_ticker} won ${amount/1e6:.1f}M contract")
        
        return signals


def create_ticker_company_map() -> Dict[str, str]:
    """
    Create a mapping of ticker symbols to company names.
    This helps match USAspending awards to stock tickers.
    
    You can expand this with:
    - SEC Edgar API lookups
    - Financial data provider APIs (Alpha Vantage, etc.)
    - Manual curated list for key defense/gov contractors
    """
    
    # Common government contractors (defense, medical, tech)
    # This is a starter list - expand with your watchlist
    ticker_map = {
        # Defense contractors
        "LMT": "LOCKHEED MARTIN",
        "RTX": "RAYTHEON TECHNOLOGIES",
        "NOC": "NORTHROP GRUMMAN",
        "GD": "GENERAL DYNAMICS",
        "BA": "BOEING",
        "HII": "HUNTINGTON INGALLS",
        "LHX": "L3HARRIS TECHNOLOGIES",
        "TXT": "TEXTRON",
        "HWM": "HOWMET AEROSPACE",
        
        # Medical/Healthcare
        "UNH": "UNITEDHEALTH",
        "CVS": "CVS HEALTH",
        "CI": "CIGNA",
        "HUM": "HUMANA",
        "ANTM": "ANTHEM",
        "MOH": "MOLINA HEALTHCARE",
        "CNC": "CENTENE",
        
        # Technology contractors
        "PLTR": "PALANTIR",
        "CSCO": "CISCO",
        "ORCL": "ORACLE",
        "IBM": "IBM",
        "MSFT": "MICROSOFT",
        "GOOGL": "GOOGLE",
        "AMZN": "AMAZON",
        
        # Small/mid caps often missed
        "KTOS": "KRATOS DEFENSE",
        "AVAV": "AEROVIRONMENT",
        "ATRO": "ASTRONICS",
        "PKE": "PARK AEROSPACE",
        "CW": "CURTISS-WRIGHT",
        "MRCY": "MERCURY SYSTEMS",
        "NPTN": "NEOVASC INC",
    }
    
    return ticker_map


# Convenience function for easy integration
def get_contract_signals(
    stock_watchlist: List[str],
    min_amount: float = 50_000_000,
    lookback_days: int = 7
) -> List[Dict]:
    """
    Quick function to get trading signals from recent government contracts.
    
    Args:
        stock_watchlist: List of tickers to monitor
        min_amount: Minimum contract value ($50M default)
        lookback_days: Number of days to look back (7 default)
    
    Returns:
        List of trading signals
    """
    monitor = USASpendingMonitor(
        min_award_amount=min_amount,
        lookback_days=lookback_days
    )
    ticker_map = create_ticker_company_map()
    return monitor.get_trading_signals(stock_watchlist, ticker_map)


if __name__ == "__main__":
    # Test the monitor
    logging.basicConfig(level=logging.INFO)
    
    test_watchlist = ["LMT", "RTX", "PLTR", "KTOS", "MSFT", "AAPL"]
    
    print("Testing USAspending.gov contract monitor...")
    signals = get_contract_signals(test_watchlist, min_amount=10_000_000)  # Lower for testing
    
    if signals:
        print(f"\nFound {len(signals)} contract signals:")
        for sig in signals:
            print(f"  • {sig['ticker']}: {sig['reason']}")
    else:
        print("\nWARNING: No matching contracts found in recent period")
