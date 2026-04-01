"""Sentiment analysis module — supports crypto & stocks.

Headlines are sourced from multiple free feeds (see models/news_fetcher.py):
  • Google News RSS, CryptoPanic, Finviz, Reddit, NewsAPI (optional)

Scoring layers:
  • VADER  — fast lexicon-based compound score
  • Claude — deeper contextual analysis (if ANTHROPIC_API_KEY set)
  The final score blends both when the LLM is available.
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Optional
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.config import CONFIG, get_news_terms_for
from models.news_fetcher import fetch_all_headlines

_analyzer = SentimentIntensityAnalyzer()


# ── Headline fetchers ─────────────────────────────────────────────

def _fetch_newsapi_headlines(query: str, hours: int = 24) -> List[str]:
    """Fetch headlines from NewsAPI.org (free tier ≈ 100 req/day)."""
    api_key = CONFIG.get("news_api_key", "")
    if not api_key:
        return []

    from_date = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 30,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        headlines = [a["title"] for a in articles if a.get("title")]
        logging.info(f"[Sentiment] Fetched {len(headlines)} headlines from NewsAPI for '{query}'")
        return headlines
    except Exception as e:
        logging.warning(f"[Sentiment] NewsAPI request failed: {e}")
        return []


def _fetch_google_news_rss(query: str) -> List[str]:
    """Scrape Google News RSS as a fallback headline source."""
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "lxml")
        titles = [item.text for item in soup.find_all("title")][2:]
        logging.info(f"[Sentiment] Fetched {len(titles)} headlines from Google News RSS for '{query}'")
        return titles[:30]
    except Exception as e:
        logging.warning(f"[Sentiment] Google News RSS failed: {e}")
        return []


def fetch_headlines(symbol: Optional[str] = None,
                    query_terms: Optional[List[str]] = None) -> List[str]:
    """
    Aggregate headlines for a symbol from ALL available sources
    (Google News, CryptoPanic, Reddit, Finviz, NewsAPI).

    Falls back to the legacy single-source approach if the multi-source
    fetcher fails entirely.
    """
    # Primary: use the multi-source parallel fetcher
    try:
        headlines = fetch_all_headlines(symbol=symbol)
        if headlines:
            return headlines
    except Exception as e:
        logging.warning(f"[Sentiment] Multi-source fetcher failed: {e}")

    # Fallback: legacy approach (Google News RSS per query term)
    if query_terms is None:
        if symbol:
            query_terms = get_news_terms_for(symbol)
        else:
            query_terms = CONFIG.get("news_query_terms_crypto", ["crypto"])

    hours = CONFIG.get("sentiment_lookback_hours", 24)
    all_headlines: List[str] = []

    for term in query_terms:
        headlines = _fetch_newsapi_headlines(term, hours=hours)
        if not headlines:
            headlines = _fetch_google_news_rss(term)
        all_headlines.extend(headlines)

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for h in all_headlines:
        key = h.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(h)

    logging.info(f"[Sentiment] {len(unique)} unique headlines collected (fallback)")
    return unique


# ── Scoring ───────────────────────────────────────────────────────

def score_text(text: str) -> float:
    """Return VADER compound score for a single string."""
    return _analyzer.polarity_scores(text)["compound"]


def analyze_sentiment(symbol: Optional[str] = None,
                      text: Optional[str] = None) -> Optional[float]:
    """
    Produce an aggregate sentiment score.

    Modes:
      • text supplied  → score only that string (testing)
      • symbol supplied → fetch live headlines for that asset
      • neither         → fetch general crypto headlines

    When Claude is available the score is a 60/40 blend of
    LLM analysis (richer context) and VADER (speed/coverage).

    Returns:
        Float in [-1, 1].  Positive → bullish,  Negative → bearish.
    """
    if text is not None:
        return score_text(text)

    try:
        headlines = fetch_headlines(symbol=symbol)
        if not headlines:
            logging.warning("[Sentiment] No headlines found — returning None (data unavailable)")
            return None

        # VADER scores
        vader_scores = [score_text(h) for h in headlines]
        vader_mean = sum(vader_scores) / len(vader_scores)

        # Claude LLM sentiment (optional)
        llm_score = None
        if CONFIG.get("use_llm") and CONFIG.get("anthropic_api_key"):
            try:
                from models.llm_analyst import llm_sentiment_analysis
                llm_score = llm_sentiment_analysis(headlines, symbol=symbol)
            except Exception as e:
                logging.warning(f"[Sentiment] LLM sentiment call failed: {e}")

        if llm_score is not None:
            blended = 0.6 * llm_score + 0.4 * vader_mean
            logging.info(
                f"[Sentiment] VADER={vader_mean:.4f}  LLM={llm_score:.4f}  "
                f"blended={blended:.4f}  ({len(headlines)} headlines)"
            )
            return round(blended, 4)

        logging.info(f"[Sentiment] VADER-only score: {vader_mean:.4f} ({len(headlines)} headlines)")
        return round(vader_mean, 4)

    except Exception as e:
        logging.error(f"[Sentiment] Analysis failed: {e}", exc_info=True)
        return None
