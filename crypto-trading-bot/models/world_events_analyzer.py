"""World Events Analyzer — comprehensive news and events impact assessment.

Uses LLM (Claude) to analyze:
  • Breaking news and headlines
  • Geopolitical events and their market impact
  • Economic indicators and central bank actions
  • Regulatory changes (especially for crypto)
  • Technology developments
  • Social sentiment trends

The analyzer provides structured impact assessments that the autonomous
agent uses to make informed trading decisions.
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple

from models.llm_analyst import _call_claude, _get_client
from models.news_fetcher import fetch_all_headlines
from config.config import CONFIG, is_crypto


class WorldEventsAnalyzer:
    """Comprehensive world events and news impact analyzer.
    
    Provides deep analysis beyond simple sentiment scoring.
    """
    
    def __init__(self):
        self.last_analysis: Dict[str, Any] = {}
        self.last_analysis_time: Optional[datetime] = None
        self.cache_duration = timedelta(minutes=15)  # Cache analysis for 15 min
    
    def analyze_market_environment(
        self,
        symbols: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis of current market environment.
        
        Args:
            symbols: Specific symbols to analyze (None = general market)
            force_refresh: Bypass cache and fetch fresh analysis
        
        Returns:
            Dict with:
              - overall_sentiment: -1 to +1
              - magnitude: 0 to 1 (how significant current events are)
              - key_events: List of major events
              - risk_factors: List of risks
              - opportunities: List of opportunities
              - crypto_specific: Dict of crypto-relevant info
              - stock_specific: Dict of stock-relevant info
              - summary: Human-readable summary
              - recommendations: Trading recommendations
        """
        # Check cache
        if (not force_refresh and 
            self.last_analysis_time and 
            datetime.now(timezone.utc) - self.last_analysis_time < self.cache_duration):
            logging.info("[WorldEvents] Using cached analysis")
            return self.last_analysis
        
        client = _get_client()
        if client is None:
            return self._default_analysis()
        
        # Fetch latest news from all sources
        logging.info("[WorldEvents] Fetching latest news and events...")
        all_headlines = fetch_all_headlines(symbol=None, is_crypto=True, max_total=100)
        
        if not all_headlines:
            logging.warning("[WorldEvents] No headlines fetched")
            return self._default_analysis()
        
        # Use LLM to analyze
        analysis = self._llm_analyze_world_events(all_headlines, symbols)
        
        if analysis:
            self.last_analysis = analysis
            self.last_analysis_time = datetime.now(timezone.utc)
            logging.info(
                f"[WorldEvents] Analysis complete: "
                f"sentiment={analysis.get('overall_sentiment', 0):.2f}, "
                f"magnitude={analysis.get('magnitude', 0):.2f}"
            )
            return analysis
        
        return self._default_analysis()
    
    def _llm_analyze_world_events(
        self,
        headlines: List[str],
        symbols: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Use Claude to perform deep analysis of world events."""
        
        # Prepare context
        symbol_context = ""
        if symbols:
            symbol_context = f"\nSpecific focus: {', '.join(symbols[:10])}"
        
        system = (
            "You are an expert financial analyst specializing in geopolitical "
            "and macroeconomic impact on markets. Analyze news and world events "
            "to assess their impact on trading decisions.\n\n"
            "Provide a comprehensive JSON analysis with these keys:\n"
            "- overall_sentiment: float from -1.0 (very bearish) to +1.0 (very bullish)\n"
            "- magnitude: float from 0.0 (negligible) to 1.0 (major market-moving events)\n"
            "- key_events: array of 3-5 most important events\n"
            "- risk_factors: array of current risks (geopolitical, economic, regulatory)\n"
            "- opportunities: array of potential opportunities\n"
            "- crypto_specific: {sentiment: float, key_factors: array, regulatory_risk: float}\n"
            "- stock_specific: {sentiment: float, key_factors: array, fed_impact: float}\n"
            "- summary: string (2-3 sentences)\n"
            "- recommendations: {stance: 'aggressive'|'cautious'|'defensive', focus_areas: array}\n\n"
            "Return ONLY valid JSON. No other text."
        )
        
        # Sample headlines to stay within token limits
        sample_size = 100
        if len(headlines) > sample_size:
            # Take a mix: most recent + random sample
            headlines = headlines[:50] + headlines[-50:]
        
        user = f"Analyze these recent headlines and world events:{symbol_context}\n\n"
        user += "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines[:100]))
        user += "\n\nProvide comprehensive market impact analysis as JSON."
        
        raw = _call_claude(system, user, max_tokens=2048, temperature=0.4)
        if raw is None:
            return None
        
        try:
            # Try to extract JSON from response
            # Sometimes Claude wraps it in markdown
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            
            analysis = json.loads(raw.strip())
            
            # Validate structure
            required_keys = ["overall_sentiment", "magnitude", "summary"]
            if not all(k in analysis for k in required_keys):
                logging.warning("[WorldEvents] LLM response missing required keys")
                return None
            
            return analysis
            
        except json.JSONDecodeError as e:
            logging.warning(f"[WorldEvents] Could not parse LLM response: {e}")
            logging.debug(f"[WorldEvents] Raw response: {raw[:500]}")
            return None
    
    def _default_analysis(self) -> Dict[str, Any]:
        """Return neutral default when LLM unavailable."""
        return {
            "overall_sentiment": 0.0,
            "magnitude": 0.0,
            "key_events": [],
            "risk_factors": [],
            "opportunities": [],
            "crypto_specific": {"sentiment": 0.0, "key_factors": [], "regulatory_risk": 0.0},
            "stock_specific": {"sentiment": 0.0, "key_factors": [], "fed_impact": 0.0},
            "summary": "No analysis available (LLM not configured or news fetch failed)",
            "recommendations": {"stance": "cautious", "focus_areas": []},
        }
    
    def get_symbol_specific_analysis(
        self,
        symbol: str,
        general_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get symbol-specific impact from world events.
        
        Args:
            symbol: The symbol to analyze
            general_analysis: Previously fetched general analysis (to avoid redundant calls)
        
        Returns:
            Symbol-specific insights
        """
        if general_analysis is None:
            general_analysis = self.analyze_market_environment(symbols=[symbol])
        
        # Extract relevant subset
        if is_crypto(symbol):
            specific = general_analysis.get("crypto_specific", {})
        else:
            specific = general_analysis.get("stock_specific", {})
        
        return {
            "symbol": symbol,
            "sentiment_score": specific.get("sentiment", general_analysis.get("overall_sentiment", 0.0)),
            "magnitude": general_analysis.get("magnitude", 0.0),
            "key_factors": specific.get("key_factors", []),
            "summary": general_analysis.get("summary", ""),
            "recommended_stance": general_analysis.get("recommendations", {}).get("stance", "cautious"),
        }
    
    def should_pause_trading(self, analysis: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Determine if trading should be paused due to extreme events.
        
        Returns:
            (should_pause, reason)
        """
        if analysis is None:
            analysis = self.last_analysis
        
        if not analysis:
            return False, ""
        
        magnitude = analysis.get("magnitude", 0.0)
        sentiment = analysis.get("overall_sentiment", 0.0)
        
        # Pause if magnitude is very high (major market-moving events)
        # and sentiment is extremely negative
        if magnitude > 0.85 and sentiment < -0.7:
            return True, "Major negative market event detected (high magnitude, very bearish)"
        
        # Check for specific risk factors
        risk_factors = analysis.get("risk_factors", [])
        critical_risks = [
            "market crash", "exchange hack", "regulatory ban",
            "geopolitical crisis", "financial crisis", "bank run"
        ]
        
        for factor in risk_factors:
            factor_lower = str(factor).lower()
            if any(critical in factor_lower for critical in critical_risks):
                return True, f"Critical risk detected: {factor}"
        
        return False, ""
    
    def get_adaptive_parameters(
        self,
        analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Get trading parameter adjustments based on world events.
        
        Returns:
            Dict with multiplicative adjustments:
              - position_size_mult: multiply position size by this
              - confidence_mult: multiply confidence threshold by this
              - tp_distance_mult: multiply take-profit distance by this
        """
        if analysis is None:
            analysis = self.last_analysis
        
        if not analysis:
            return {"position_size_mult": 1.0, "confidence_mult": 1.0, "tp_distance_mult": 1.0}
        
        magnitude = analysis.get("magnitude", 0.0)
        sentiment = analysis.get("overall_sentiment", 0.0)
        stance = analysis.get("recommendations", {}).get("stance", "cautious")
        
        adjustments = {
            "position_size_mult": 1.0,
            "confidence_mult": 1.0,
            "tp_distance_mult": 1.0,
        }
        
        # High magnitude + negative sentiment = reduce size, tighten stops
        if magnitude > 0.7:
            if sentiment < -0.5:
                adjustments["position_size_mult"] = 0.5  # halve positions
                adjustments["tp_distance_mult"] = 0.8  # tighter take profits
            elif sentiment > 0.5:
                adjustments["position_size_mult"] = 1.2  # increase positions
                adjustments["tp_distance_mult"] = 1.2  # wider take profits
        
        # Stance-based adjustments
        if stance == "defensive":
            adjustments["position_size_mult"] *= 0.7
            adjustments["confidence_mult"] = 1.1  # require higher confidence
        elif stance == "aggressive":
            adjustments["position_size_mult"] *= 1.3
            adjustments["confidence_mult"] = 0.95  # slightly lower threshold
        
        return adjustments


# Singleton instance
_world_events_analyzer: Optional[WorldEventsAnalyzer] = None


def get_world_events_analyzer() -> WorldEventsAnalyzer:
    """Get or create the global world events analyzer instance."""
    global _world_events_analyzer
    if _world_events_analyzer is None:
        _world_events_analyzer = WorldEventsAnalyzer()
    return _world_events_analyzer
