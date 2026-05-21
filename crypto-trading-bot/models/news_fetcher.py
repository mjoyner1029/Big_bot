"""Multi-source news headline fetcher — free, no API key required.

Aggregates headlines from:
  1. Google News RSS (primary — no rate limits, no key needed)
  2. CryptoPanic RSS (crypto-specific, free tier)
  3. Finviz news scraper (stock-specific)
  4. Reddit /r/cryptocurrency and /r/wallstreetbets (via JSON API)
  5. NewsAPI (if NEWS_API_KEY is set — higher quality)

All sources are tried in parallel via ThreadPoolExecutor.
"""
import logging
import re
import time
import warnings
import requests

# Silence BS4 XML-as-HTML warning for RSS parsing
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

from config.config import CONFIG


_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) CryptoBot/1.0",
})

# Simple cache: (query_key, timestamp) -> headlines
_cache: Dict[str, tuple] = {}
_CACHE_TTL = 300  # 5 minutes


def _cached(key: str) -> Optional[List[str]]:
    if key in _cache:
        headlines, ts = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            return headlines
    return None


def _set_cache(key: str, headlines: List[str]) -> None:
    _cache[key] = (headlines, time.time())


# ── Google News RSS ───────────────────────────────────────────────

def fetch_google_news(query: str, max_results: int = 40) -> List[str]:
    """Fetch headlines from Google News RSS (free, no API key)."""
    cache_key = f"google:{query}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = _session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "lxml")
        titles = [item.text.strip() for item in soup.find_all("title")][2:]  # skip RSS boilerplate
        results = titles[:max_results]
        _set_cache(cache_key, results)
        logging.info(f"[News] Google News: {len(results)} headlines for '{query}'")
        return results
    except Exception as e:
        logging.warning(f"[News] Google News failed for '{query}': {e}")
        return []


# ── CryptoPanic RSS ──────────────────────────────────────────────

def fetch_cryptopanic(max_results: int = 30) -> List[str]:
    """Fetch crypto headlines from CryptoPanic RSS feed."""
    cache_key = "cryptopanic"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    url = "https://cryptopanic.com/news/rss/"
    try:
        resp = _session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "lxml")
        titles = [item.text.strip() for item in soup.find_all("title")][1:]
        results = titles[:max_results]
        _set_cache(cache_key, results)
        logging.info(f"[News] CryptoPanic: {len(results)} headlines")
        return results
    except Exception as e:
        logging.warning(f"[News] CryptoPanic failed: {e}")
        return []


# ── Finviz stock news ─────────────────────────────────────────────

def fetch_finviz_news(symbol: str, max_results: int = 20) -> List[str]:
    """Scrape recent headlines for a stock from Finviz."""
    cache_key = f"finviz:{symbol}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    url = f"https://finviz.com/quote.ashx?t={symbol}"
    try:
        resp = _session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        news_table = soup.find(id="news-table")
        if not news_table:
            return []
        headlines = []
        for row in news_table.find_all("tr"):
            a_tag = row.find("a")
            if a_tag:
                headlines.append(a_tag.text.strip())
        results = headlines[:max_results]
        _set_cache(cache_key, results)
        logging.info(f"[News] Finviz: {len(results)} headlines for {symbol}")
        return results
    except Exception as e:
        logging.warning(f"[News] Finviz failed for {symbol}: {e}")
        return []


# ── Reddit JSON API ───────────────────────────────────────────────

def fetch_reddit_headlines(subreddit: str = "cryptocurrency",
                           max_results: int = 25) -> List[str]:
    """Fetch top post titles from a Reddit subreddit via JSON API."""
    cache_key = f"reddit:{subreddit}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={max_results}"
    try:
        resp = _session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        posts = data.get("data", {}).get("children", [])
        headlines = [p["data"]["title"] for p in posts if p.get("data", {}).get("title")]
        results = headlines[:max_results]
        _set_cache(cache_key, results)
        logging.info(f"[News] Reddit r/{subreddit}: {len(results)} headlines")
        return results
    except Exception as e:
        logging.warning(f"[News] Reddit r/{subreddit} failed: {e}")
        return []


# ── NewsAPI (optional, higher quality) ────────────────────────────

def fetch_newsapi(query: str, hours: int = 24, max_results: int = 30) -> List[str]:
    """Fetch from NewsAPI.org (requires NEWS_API_KEY)."""
    api_key = CONFIG.get("news_api_key", "")
    if not api_key:
        return []

    cache_key = f"newsapi:{query}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    from_date = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": max_results,
        "apiKey": api_key,
    }
    try:
        resp = _session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        headlines = [a["title"] for a in articles if a.get("title")]
        _set_cache(cache_key, headlines)
        logging.info(f"[News] NewsAPI: {len(headlines)} headlines for '{query}'")
        return headlines
    except Exception as e:
        logging.warning(f"[News] NewsAPI failed for '{query}': {e}")
        return []


# ── Aggregate fetcher ─────────────────────────────────────────────

def fetch_all_headlines(
    symbol: Optional[str] = None,
    is_crypto: bool = True,
    max_total: int = 100,
) -> List[str]:
    """
    Fetch headlines from ALL available sources in parallel.

    Returns deduplicated list sorted by source quality.
    """
    from config.config import is_crypto as _is_crypto
    if symbol:
        is_crypto = _is_crypto(symbol)

    tasks = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Google News — always
        if symbol:
            base_sym = symbol.replace("-USD", "").replace("-", " ")
            tasks.append(executor.submit(fetch_google_news, base_sym))
        if is_crypto:
            tasks.append(executor.submit(fetch_google_news, "cryptocurrency market"))
            tasks.append(executor.submit(fetch_cryptopanic))
            tasks.append(executor.submit(fetch_reddit_headlines, "cryptocurrency"))
        else:
            tasks.append(executor.submit(fetch_google_news, "stock market"))
            if symbol:
                tasks.append(executor.submit(fetch_finviz_news, symbol))
            tasks.append(executor.submit(fetch_reddit_headlines, "wallstreetbets"))
            tasks.append(executor.submit(fetch_reddit_headlines, "stocks"))

        # NewsAPI if key available
        if CONFIG.get("news_api_key"):
            query = symbol.replace("-USD", "") if symbol else ("crypto" if is_crypto else "stock market")
            tasks.append(executor.submit(fetch_newsapi, query))

    all_headlines: List[str] = []
    for future in as_completed(tasks):
        try:
            all_headlines.extend(future.result())
        except Exception as e:
            logging.warning(f"[News] Source fetch error: {e}")

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for h in all_headlines:
        key = re.sub(r"\s+", " ", h.strip().lower())
        if key not in seen and len(key) > 10:
            seen.add(key)
            unique.append(h.strip())

    logging.info(f"[News] Total unique headlines: {len(unique)} (from {len(all_headlines)} raw)")
    return unique[:max_total]
