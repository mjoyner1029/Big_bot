import pandas as pd
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict
from config.config import CONFIG
from datetime import datetime, timezone, timedelta
from trading.rate_limiter import get_yfinance_limiter
import requests

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency fallback
    yf = None

# ── In-memory data cache ──────────────────────────────────────────────────────
# Avoids re-fetching the same bars every cycle; crypto TTL is short (5 min),
# stocks TTL is 1 hour — Alpaca IEX updates hourly so there's no benefit
# refreshing more frequently, and longer TTLs reduce API load.
_DATA_CACHE: Dict[str, tuple] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_CRYPTO = 300    # 5 minutes
_CACHE_TTL_STOCK  = 3600   # 60 minutes — refresh once per hour
_CACHE_TTL_INDEX  = 1800   # 30 minutes for market indices (SPY, QQQ) — longer to reduce rate limits

def _cache_key(ticker: str, period: str, interval: str) -> str:
    return f"{ticker}|{period}|{interval}"

def _cache_get(key: str) -> Optional[pd.DataFrame]:
    with _CACHE_LOCK:
        if key in _DATA_CACHE:
            df, expires = _DATA_CACHE[key]
            if time.time() < expires:
                return df
            del _DATA_CACHE[key]
    return None

def _cache_set(key: str, df: pd.DataFrame, ttl: float) -> None:
    with _CACHE_LOCK:
        _DATA_CACHE[key] = (df.copy(), time.time() + ttl)

def clear_data_cache() -> None:
    """Flush the in-memory data cache (useful for testing)."""
    with _CACHE_LOCK:
        _DATA_CACHE.clear()

# ── Failed ticker blacklist ──────────────────────────────────────────────────
# Track tickers that consistently fail (delisted, invalid, etc.) to avoid wasting API calls
_FAILED_TICKER_BLACKLIST: Dict[str, float] = {}  # ticker -> timestamp when blacklisted
_BLACKLIST_LOCK = threading.Lock()
_BLACKLIST_TTL = 86400  # 24 hours - retry blacklisted tickers once per day

def _is_blacklisted(ticker: str) -> bool:
    """Check if a ticker is currently blacklisted."""
    with _BLACKLIST_LOCK:
        if ticker in _FAILED_TICKER_BLACKLIST:
            blacklisted_at = _FAILED_TICKER_BLACKLIST[ticker]
            if time.time() - blacklisted_at < _BLACKLIST_TTL:
                return True
            # Expired - remove from blacklist
            del _FAILED_TICKER_BLACKLIST[ticker]
    return False

def _blacklist_ticker(ticker: str, reason: str = "") -> None:
    """Add a ticker to the blacklist."""
    with _BLACKLIST_LOCK:
        _FAILED_TICKER_BLACKLIST[ticker] = time.time()
    logging.info(f"[Fetcher] Blacklisted {ticker} for {_BLACKLIST_TTL/3600:.1f}h: {reason}")

# ── yfinance rate-limit circuit breaker ──────────────────────────────────────
# When yfinance returns a rate-limit error, we back off for _YF_BACKOFF_SECONDS
# instead of hammering the endpoint once per symbol.
_YF_RATE_LIMIT_LOCK = threading.Lock()
_yf_rate_limited_until: float = 0.0   # epoch seconds; 0 means "not limited"
_YF_BACKOFF_SECONDS = 60              # how long to skip yfinance after a 429

def _yf_is_rate_limited() -> bool:
    with _YF_RATE_LIMIT_LOCK:
        return time.time() < _yf_rate_limited_until

def _yf_mark_rate_limited() -> None:
    with _YF_RATE_LIMIT_LOCK:
        global _yf_rate_limited_until
        _yf_rate_limited_until = time.time() + _YF_BACKOFF_SECONDS
        logging.warning(
            f"[Fetcher] yfinance rate-limited — skipping yf for {_YF_BACKOFF_SECONDS}s"
        )


def fetch_latest_market_data(
    ticker: Optional[str] = None,
    period: Optional[str] = None,
    interval: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch latest OHLCV market data.

    Priority order:
      1. In-memory cache (avoids re-fetching within TTL window)
      2. Primary broker feeds (Coinbase for crypto, Alpaca for stocks)
      3. yfinance single-symbol download (skipped if circuit-breaker is open)
      4. Fallback sources (Stooq daily for stocks)

    Args:
        ticker:   Symbol (default from config).
        period:   Look-back period string e.g. '3mo' (default from config).
        interval: Bar interval string e.g. '1h' (default from config).
    Returns:
        DataFrame with OHLCV data, or None on failure.
    """
    ticker = ticker or CONFIG.get("symbol", "ETH-USD")
    period = period or CONFIG.get("period", "3mo")
    interval = interval or CONFIG.get("interval", "1h")

    # 1. Cache hit — return immediately without any network call
    ckey = _cache_key(ticker, period, interval)
    cached = _cache_get(ckey)
    if cached is not None:
        return cached

    if yf is None:
        logging.debug("yfinance not installed; relying on broker-native/fallback data for %s", ticker)

    # 2. Prefer broker-native/public exchange data before yfinance to avoid rate limits.
    primary_df = _primary_fetch_market_data(ticker=ticker, period=period, interval=interval)
    if primary_df is not None and not primary_df.empty:
        primary_df = _add_ohlcv_alias_columns(primary_df)
        ttl = _CACHE_TTL_CRYPTO if "-" in ticker else _CACHE_TTL_STOCK
        _cache_set(ckey, primary_df, ttl)
        logging.info(f"Fetched {len(primary_df)} bars for {ticker} via primary provider")
        return primary_df

    # 3. yfinance — skip entirely when disabled or circuit-breaker is open
    if CONFIG.get("disable_yfinance", False):
        logging.debug(f"[Fetcher] yfinance disabled — using Alpaca/WebSocket only for {ticker}")
        return _fallback_fetch_market_data(ticker=ticker, period=period, interval=interval)
    
    if _yf_is_rate_limited():
        logging.debug(f"[Fetcher] yfinance circuit-breaker open — skipping yf for {ticker}")
        return _fallback_fetch_market_data(ticker=ticker, period=period, interval=interval)

    # Apply rate limiting only when falling back to yfinance.
    rate_limiter = get_yfinance_limiter()
    if CONFIG.get("rate_limiting_enabled", True):
        rate_limiter.wait_if_needed()

    if yf is None:
        return _fallback_fetch_market_data(ticker=ticker, period=period, interval=interval)

    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data is None or data.empty:
            # yfinance silently returns empty on rate-limit — detect via shared error dict
            try:
                err = str(getattr(__import__("yfinance.shared", fromlist=["_ERRORS"]), "_ERRORS", {}).get(ticker, ""))
                if "rate" in err.lower() or "429" in err or "too many" in err.lower():
                    _yf_mark_rate_limited()
            except Exception as e:
                logging.debug(f"[Fetcher] Could not inspect yfinance shared errors for {ticker}: {e}")
            logging.warning(f"No data returned for {ticker} ({period}, {interval}); trying fallback source")
            return _fallback_fetch_market_data(ticker=ticker, period=period, interval=interval)
        # Flatten multi-level columns if present (yfinance sometimes returns multi-index)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = _add_ohlcv_alias_columns(data)
        ttl = _CACHE_TTL_CRYPTO if "-" in ticker else _CACHE_TTL_STOCK
        _cache_set(ckey, data, ttl)
        logging.info(f"Fetched {len(data)} bars for {ticker} ({interval})")
        return data
    except Exception as e:
        error_str = str(e)
        if "rate" in error_str.lower() or "429" in error_str or "too many" in error_str.lower():
            _yf_mark_rate_limited()
        else:
            logging.error(f"Error fetching data for {ticker}: {e}")
        return _fallback_fetch_market_data(ticker=ticker, period=period, interval=interval)


def _primary_fetch_market_data(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """Prefer broker-native providers to reduce dependence on yfinance."""
    if "-" in ticker:
        crypto_df = _fetch_coinbase_candles(ticker=ticker, interval=interval)
        if crypto_df is not None and not crypto_df.empty:
            return crypto_df

    stock_df = _fetch_alpaca_stock_bars(ticker=ticker, period=period, interval=interval)
    if stock_df is not None and not stock_df.empty:
        # Reject stale data: Alpaca IEX feed can return months-old bars.
        # A last bar older than 72 h (covers weekends) means the feed has a gap —
        # fall through to yfinance which returns current data instead.
        last_bar_time = stock_df.index[-1]
        if hasattr(last_bar_time, 'tzinfo') and last_bar_time.tzinfo is None:
            last_bar_time = last_bar_time.tz_localize('UTC')
        age_hours = (datetime.now(timezone.utc) - last_bar_time.to_pydatetime()).total_seconds() / 3600
        if age_hours <= 72:
            return stock_df
        logging.warning(
            f"Alpaca data for {ticker} is {age_hours:.0f}h stale "
            f"(last bar: {last_bar_time}) — falling back to yfinance"
        )

    return None


def fetch_batch_market_data(
    symbols: List[str],
    period: str = "5d",
    interval: str = "1h",
    max_workers: int = 8,
) -> Dict[str, pd.DataFrame]:
    """Fetch market data for many symbols efficiently.

    Strategy:
    - Crypto (symbols containing '-'): parallel Coinbase REST calls via ThreadPoolExecutor
    - Stocks: single batched yfinance download (one HTTP request for all tickers),
      falling back to parallel Stooq if yfinance is rate-limited
    - All results cached with per-asset TTL so repeated calls within the same cycle
      are instant.

    Returns:
        Dict mapping symbol -> DataFrame (only symbols with valid data included).
    """
    # Filter out blacklisted tickers
    symbols = [s for s in symbols if not _is_blacklisted(s)]
    if not symbols:
        return {}
    
    result: Dict[str, pd.DataFrame] = {}
    crypto_syms = [s for s in symbols if "-" in s]
    stock_syms  = [s for s in symbols if "-" not in s]

    # ── Crypto: parallel Coinbase fetches ────────────────────────────────────
    def _fetch_crypto(sym: str):
        ckey = _cache_key(sym, period, interval)
        cached = _cache_get(ckey)
        if cached is not None:
            return sym, cached
        # Try primary (Coinbase) first, then yfinance if not rate-limited/disabled
        df = _fetch_coinbase_candles(ticker=sym, interval=interval)
        if df is None or df.empty:
            if not CONFIG.get("disable_yfinance", False) and not _yf_is_rate_limited() and yf is not None:
                try:
                    df = yf.download(sym, period=period, interval=interval, progress=False)
                    if df is not None and not df.empty and isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                except Exception as exc:
                    exc_str = str(exc).lower()
                    if "rate" in exc_str or "429" in str(exc):
                        _yf_mark_rate_limited()
                    # Blacklist if delisted/invalid
                    elif any(kw in exc_str for kw in ["delisted", "no price data", "no data found", "invalid"]):
                        _blacklist_ticker(sym, str(exc))
                    df = None
        if df is not None and not df.empty:
            df = _add_ohlcv_alias_columns(df)
            _cache_set(ckey, df, _CACHE_TTL_CRYPTO)
            return sym, df
        return sym, None

    if crypto_syms:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(crypto_syms))) as pool:
            for sym, df in pool.map(_fetch_crypto, crypto_syms):
                if df is not None and not df.empty:
                    result[sym] = df

    # ── Stocks: Alpaca primary, then yfinance batch fallback ──────────────────
    if stock_syms:
        # Return cached symbols immediately
        uncached = []
        for sym in stock_syms:
            ckey = _cache_key(sym, period, interval)
            cached = _cache_get(ckey)
            if cached is not None:
                result[sym] = cached
            else:
                uncached.append(sym)

        if uncached:
            # 1) Alpaca primary per-symbol fetch (parallel)
            def _fetch_alpaca_one(sym: str):
                df = _fetch_alpaca_stock_bars(ticker=sym, period=period, interval=interval)
                if df is not None and not df.empty:
                    # Guard against stale broker bars; if stale, let fallback sources try.
                    last_bar_time = df.index[-1]
                    if hasattr(last_bar_time, 'tzinfo') and last_bar_time.tzinfo is None:
                        last_bar_time = last_bar_time.tz_localize('UTC')
                    age_hours = (datetime.now(timezone.utc) - last_bar_time.to_pydatetime()).total_seconds() / 3600
                    if age_hours <= 72:
                        df = _add_ohlcv_alias_columns(df)
                        ckey = _cache_key(sym, period, interval)
                        # Use longer TTL for market indices (SPY, QQQ, DIA, IWM) to reduce rate limits
                        index_symbols = {"SPY", "QQQ", "DIA", "IWM", "VTI", "VOO"}
                        ttl = _CACHE_TTL_INDEX if sym.upper() in index_symbols else _CACHE_TTL_STOCK
                        _cache_set(ckey, df, ttl)
                        return sym, df
                return sym, None

            with ThreadPoolExecutor(max_workers=min(max_workers, len(uncached))) as pool:
                for sym, df in pool.map(_fetch_alpaca_one, uncached):
                    if df is not None and not df.empty:
                        result[sym] = df

            # 2) yfinance batch fallback for any symbols still missing
            still_missing = [s for s in uncached if s not in result]
            if still_missing:
                batch_ok = False
                if not CONFIG.get("disable_yfinance", False) and not _yf_is_rate_limited() and yf is not None:
                    try:
                        tickers_str = " ".join(still_missing)
                        raw = yf.download(
                            tickers_str,
                            period=period,
                            interval=interval,
                            progress=False,
                            group_by="ticker",
                            threads=False,
                        )
                        if raw is not None and not raw.empty:
                            batch_ok = True
                            for sym in still_missing:
                                try:
                                    if len(still_missing) == 1:
                                        sym_df = raw
                                    else:
                                        sym_df = raw[sym] if sym in raw.columns.get_level_values(0) else None
                                    if sym_df is not None and not sym_df.empty:
                                        sym_df = sym_df.dropna(how="all")
                                        if not sym_df.empty:
                                            sym_df = _add_ohlcv_alias_columns(sym_df)
                                            ckey = _cache_key(sym, period, interval)
                                            # Use longer TTL for market indices to reduce rate limits
                                            index_symbols = {"SPY", "QQQ", "DIA", "IWM", "VTI", "VOO"}
                                            ttl = _CACHE_TTL_INDEX if sym.upper() in index_symbols else _CACHE_TTL_STOCK
                                            _cache_set(ckey, sym_df, ttl)
                                            result[sym] = sym_df
                                        else:
                                            # DataFrame was all NaN - likely delisted/invalid
                                            _blacklist_ticker(sym, "all-NaN DataFrame from yfinance")
                                    else:
                                        # Check yfinance shared errors for this symbol
                                        try:
                                            import yfinance.shared as _yf_shared
                                            err = str(_yf_shared._ERRORS.get(sym, ""))
                                            if any(kw in err.lower() for kw in ["delisted", "no price data", "no data found", "invalid"]):
                                                _blacklist_ticker(sym, err)
                                        except:
                                            pass
                                except Exception as e:
                                    logging.debug(f"[Fetcher] Failed to parse batch data for {sym}: {e}")
                        else:
                            # Batch returned empty — check shared error dict for rate limit and delisted tickers
                            try:
                                import yfinance.shared as _yf_shared
                                for _sym in still_missing:
                                    err = str(_yf_shared._ERRORS.get(_sym, ""))
                                    if "rate" in err.lower() or "429" in err or "too many" in err.lower():
                                        _yf_mark_rate_limited()
                                        break
                                    # Blacklist tickers that are clearly delisted/invalid
                                    elif any(keyword in err.lower() for keyword in ["delisted", "no price data", "no data found", "invalid ticker"]):
                                        _blacklist_ticker(_sym, err)
                            except Exception as e:
                                logging.debug(f"[Fetcher] Could not inspect yfinance shared errors: {e}")
                    except Exception as exc:
                        if "rate" in str(exc).lower() or "429" in str(exc):
                            _yf_mark_rate_limited()
                        logging.warning(f"[Fetcher] Batch yfinance failed: {exc}")

                # 3) Stooq fallback for any symbols still missing
                if not batch_ok:
                    still_missing = [s for s in still_missing if s not in result]

                    def _fetch_stooq_one(sym: str):
                        df = _fetch_stooq_daily(ticker=sym)
                        if df is not None and not df.empty:
                            df = _add_ohlcv_alias_columns(df)
                            ckey = _cache_key(sym, period, interval)
                            _cache_set(ckey, df, _CACHE_TTL_STOCK)
                        return sym, df

                    if still_missing:
                        with ThreadPoolExecutor(max_workers=min(max_workers, len(still_missing))) as pool:
                            for sym, df in pool.map(_fetch_stooq_one, still_missing):
                                if df is not None and not df.empty:
                                    result[sym] = df

                # 4) Emergency fallback: stale Alpaca bars when yfinance is rate-limited
                #    and Stooq also returned nothing.  Better to trade on slightly stale
                #    data than to skip every stock entirely for days on end.
                truly_missing = [s for s in uncached if s not in result]
                if truly_missing and _yf_is_rate_limited():
                    def _fetch_alpaca_emergency(sym: str):
                        df = _fetch_alpaca_stock_bars(ticker=sym, period=period, interval=interval)
                        if df is not None and not df.empty:
                            df = _add_ohlcv_alias_columns(df)
                            ckey = _cache_key(sym, period, interval)
                            _cache_set(ckey, df, _CACHE_TTL_STOCK)
                            return sym, df
                        return sym, None

                    with ThreadPoolExecutor(max_workers=min(max_workers, len(truly_missing))) as pool:
                        for sym, df in pool.map(_fetch_alpaca_emergency, truly_missing):
                            if df is not None and not df.empty:
                                age = ""
                                try:
                                    last = df.index[-1]
                                    if hasattr(last, "tzinfo") and last.tzinfo is None:
                                        last = last.tz_localize("UTC")
                                    h = (datetime.now(timezone.utc) - last.to_pydatetime()).total_seconds() / 3600
                                    age = f" ({h:.0f}h stale)"
                                except Exception:
                                    pass
                                logging.warning(
                                    "[Fetcher] %s: using stale Alpaca data%s "
                                    "(yfinance rate-banned, Stooq empty — last-resort fallback)",
                                    sym, age,
                                )
                                result[sym] = df

    return result


def fetch_multiple_symbols(
    symbols: Optional[List[str]] = None,
    period: Optional[str] = None,
    interval: Optional[str] = None,
    max_workers: int = 8,
) -> Dict[str, pd.DataFrame]:
    """Fetch market data for multiple symbols in parallel.

    Uses ``fetch_batch_market_data`` under the hood which batches stock requests
    and parallelises crypto fetches, dramatically reducing cycle time vs. the
    old sequential loop.

    Returns:
        Dict mapping symbol -> DataFrame.
    """
    from config.config import get_all_symbols
    symbols = symbols or get_all_symbols()
    period = period or CONFIG.get("period", "3mo")
    interval = interval or CONFIG.get("interval", "1h")
    return fetch_batch_market_data(symbols, period=period, interval=interval, max_workers=max_workers)


def save_historical_data(
    df: pd.DataFrame,
    symbol: str,
    directory: str = "data/historical",
) -> str:
    """
    Persist a DataFrame to CSV in the historical data directory.

    Returns:
        The file path written.
    """
    os.makedirs(directory, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_sym = symbol.replace("/", "_").replace("-", "_")
    filepath = os.path.join(directory, f"{safe_sym}_{ts}.csv")
    df.to_csv(filepath)
    logging.info(f"Saved historical data to {filepath}")
    return filepath


def load_historical_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load a previously saved CSV into a DataFrame."""
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logging.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return None


def _fallback_fetch_market_data(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """Best-effort fallback data fetch when yfinance is unavailable or rate-limited.

    Crypto symbols use Coinbase public candles.
    Equity symbols use Stooq daily CSV.
    """
    crypto_df = _fetch_coinbase_candles(ticker=ticker, interval=interval)
    if crypto_df is not None and not crypto_df.empty:
        crypto_df = _add_ohlcv_alias_columns(crypto_df)
        logging.info(f"Fallback fetch succeeded via Coinbase for {ticker} ({interval})")
        return crypto_df

    stock_df = _fetch_stooq_daily(ticker=ticker)
    if stock_df is not None and not stock_df.empty:
        stock_df = _add_ohlcv_alias_columns(stock_df)
        logging.info(f"Fallback fetch succeeded via Stooq for {ticker} (daily)")
        return stock_df

    logging.warning(f"Fallback sources returned no data for {ticker}")
    return None


def _fetch_alpaca_stock_bars(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    if not ticker or "-" in ticker:
        return None

    api_key = CONFIG.get("alpaca_api_key", "")
    api_secret = CONFIG.get("alpaca_api_secret", "")
    if not api_key or not api_secret:
        return None

    timeframe_map = {
        "1m": "1Min",
        "5m": "5Min",
        "15m": "15Min",
        "1h": "1Hour",
        "1d": "1Day",
    }
    timeframe = timeframe_map.get(interval)
    if timeframe is None:
        return None

    # Alpaca IEX free tier only retains ~30–35 days of history.
    # Capping the lookback ensures we always get the most recent bars
    # rather than receiving stale April/old data when a longer period is
    # requested.  30 days of hourly bars (~180 candles) is more than enough
    # for all common technical indicators (RSI-14, BB-20, MACD-26 etc.).
    _IEX_MAX_DAYS = 30
    requested_start = _period_start(period)
    iex_min_start   = datetime.now(timezone.utc) - timedelta(days=_IEX_MAX_DAYS)
    start_ts = max(requested_start, iex_min_start)
    end_ts = datetime.now(timezone.utc)
    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    params = {
        "timeframe": timeframe,
        "start": start_ts.isoformat().replace("+00:00", "Z"),
        "end": end_ts.isoformat().replace("+00:00", "Z"),
        "limit": 10000,
        "feed": "iex",
        "adjustment": "raw",
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        bars = payload.get("bars", [])
        if not bars:
            return None

        df = pd.DataFrame(bars)
        df = df.rename(columns={
            "t": "time",
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
        })
        if "time" not in df.columns:
            return None
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["Open", "High", "Low", "Close"])
    except Exception as exc:
        logging.debug(f"Alpaca stock bars fetch failed for {ticker}: {exc}")
        return None


def _period_start(period: str) -> datetime:
    now = datetime.now(timezone.utc)
    mapping = {
        "1d": timedelta(days=1),
        "5d": timedelta(days=5),
        "1mo": timedelta(days=30),
        "3mo": timedelta(days=90),
        "6mo": timedelta(days=180),
        "1y": timedelta(days=365),
        "2y": timedelta(days=730),
    }
    return now - mapping.get(period, timedelta(days=90))


def _fetch_coinbase_candles(ticker: str, interval: str) -> Optional[pd.DataFrame]:
    if "-" not in ticker:
        return None

    granularity_map = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "1d": 86400,
    }
    granularity = granularity_map.get(interval, 3600)
    url = f"https://api.exchange.coinbase.com/products/{ticker}/candles"
    params = {"granularity": granularity}
    try:
        response = requests.get(url, params=params, timeout=6)
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list) or not rows:
            return None

        # Coinbase format: [time, low, high, open, close, volume]
        df = pd.DataFrame(rows, columns=["time", "Low", "High", "Open", "Close", "Volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.sort_values("time").set_index("time")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["Open", "High", "Low", "Close"])
    except Exception as exc:
        logging.debug(f"Coinbase fallback failed for {ticker}: {exc}")
        return None


def _fetch_stooq_daily(ticker: str) -> Optional[pd.DataFrame]:
    symbol = ticker.strip().lower()
    if not symbol:
        return None

    # Stooq symbols for US equities are typically suffixed with .us
    stooq_symbol = symbol if "." in symbol else f"{symbol}.us"
    url = "https://stooq.com/q/d/l/"
    params = {"s": stooq_symbol, "i": "d"}
    try:
        response = requests.get(url, params=params, timeout=4)
        response.raise_for_status()
        text = response.text.strip()
        if not text or text.startswith("No data"):
            return None

        from io import StringIO
        df = pd.read_csv(StringIO(text))
        if df.empty or "Date" not in df.columns:
            return None

        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.rename(columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        })
        df = df.set_index("Date")
        return df.dropna(subset=["Open", "High", "Low", "Close"]) 
    except Exception as exc:
        logging.debug(f"Stooq fallback failed for {ticker}: {exc}")
        return None


def _add_ohlcv_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure both uppercase and lowercase OHLCV aliases are available."""
    out = df.copy()
    alias_pairs = [
        ("Open", "open"),
        ("High", "high"),
        ("Low", "low"),
        ("Close", "close"),
        ("Volume", "volume"),
    ]
    for upper, lower in alias_pairs:
        if upper in out.columns and lower not in out.columns:
            # Handle multi-column case (malformed data)
            if isinstance(out[upper], pd.DataFrame):
                # Take first column if multiple
                out[lower] = out[upper].iloc[:, 0]
            else:
                out[lower] = out[upper]
        if lower in out.columns and upper not in out.columns:
            if isinstance(out[lower], pd.DataFrame):
                out[upper] = out[lower].iloc[:, 0]
            else:
                out[upper] = out[lower]
    return out
