"""Research history must cover the requested period before validation."""
import logging
import re
from datetime import datetime, timezone

import pandas as pd

from core.ohlcv import normalize_ohlcv

logger = logging.getLogger(__name__)


def history_start(period, end):
    match = re.fullmatch(r"(\d+)(d|mo|y)", period)
    if not match or int(match[1]) < 1:
        raise ValueError(f"Unsupported history period: {period}")
    unit = {"d": "days", "mo": "months", "y": "years"}[match[2]]
    return (pd.Timestamp(end) - pd.DateOffset(**{unit: int(match[1])})).to_pydatetime()


def verify_history_coverage(df, period="2y", now=None):
    now = now or datetime.now(timezone.utc)
    start = pd.Timestamp(history_start(period, now))
    if df is None or len(df) < 200 or not isinstance(df.index, pd.DatetimeIndex):
        return False, "missing history or fewer than 200 bars"
    idx = pd.to_datetime(df.index, utc=True)
    # Allow a weekend/holiday at either end; internal gaps are checked by DQ.
    tolerance = pd.Timedelta(days=4)
    ok = idx.min() <= start + tolerance and idx.max() >= pd.Timestamp(now) - tolerance
    return bool(ok), f"{len(df)} bars, {idx.min().isoformat()} to {idx.max().isoformat()}, requested {period}"


def fetch_research_history(symbol, period="2y", interval="1d"):
    from data.fetcher import _fetch_coinbase_candles, fetch_latest_market_data
    if "-" in symbol:
        df = _fetch_coinbase_candles(symbol, interval, period=period)
    else:
        df = fetch_latest_market_data(symbol, period=period, interval=interval)
    ok, detail = verify_history_coverage(df, period)
    logger.log(logging.INFO if ok else logging.WARNING,
               "Research coverage %s [%s]: %s", symbol, "PASS" if ok else "REJECT", detail)
    return normalize_ohlcv(df) if ok else None
