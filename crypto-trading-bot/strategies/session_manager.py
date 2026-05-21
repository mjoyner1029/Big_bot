"""Market session awareness — knows when markets are open, what session
we're in, and adjusts trading behavior accordingly.

Key features:
  • Stock market-hours enforcement (no stock trades outside 9:30-16:00 ET)
  • Session classification (pre-market, open, midday, power-hour, after-hours)
  • Time-of-day confidence adjustments (reduce size during midday chop)
  • Overnight risk awareness (flag stock positions held past close)
  • Market holiday calendar (US stock market)
"""
import logging
from datetime import datetime, time, date, timedelta
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from config.config import is_crypto

# ── US Eastern timezone ───────────────────────────────────────────
ET = ZoneInfo("America/New_York")

# ── US stock market holidays (static + computed) ──────────────────
# Format: set of (month, day) for fixed holidays; dynamic ones computed below.
_FIXED_HOLIDAYS: set = {
    (1, 1),   # New Year's Day
    (6, 19),  # Juneteenth
    (7, 4),   # Independence Day
    (12, 25), # Christmas Day
}


def _us_market_holidays(year: int) -> set:
    """Return a set of datetime.date objects for US stock market holidays."""
    holidays = set()

    # Fixed holidays (observed: if Sat → Fri, if Sun → Mon)
    for m, d in _FIXED_HOLIDAYS:
        dt = date(year, m, d)
        if dt.weekday() == 5:     # Saturday → observe Friday
            dt = dt - timedelta(days=1)
        elif dt.weekday() == 6:   # Sunday → observe Monday
            dt = dt + timedelta(days=1)
        holidays.add(dt)

    # MLK Day: 3rd Monday of January
    holidays.add(_nth_weekday(year, 1, 0, 3))
    # Presidents' Day: 3rd Monday of February
    holidays.add(_nth_weekday(year, 2, 0, 3))
    # Good Friday: 2 days before Easter Sunday
    holidays.add(_easter(year) - timedelta(days=2))
    # Memorial Day: last Monday of May
    holidays.add(_last_weekday(year, 5, 0))
    # Labor Day: 1st Monday of September
    holidays.add(_nth_weekday(year, 9, 0, 1))
    # Thanksgiving: 4th Thursday of November
    holidays.add(_nth_weekday(year, 11, 3, 4))

    return holidays


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth occurrence of a weekday in a given month."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of a weekday in a given month."""
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _easter(year: int) -> date:
    """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


# Cache holidays per year
_holiday_cache: Dict[int, set] = {}


def _get_holidays(year: int) -> set:
    if year not in _holiday_cache:
        _holiday_cache[year] = _us_market_holidays(year)
    return _holiday_cache[year]


# ── Session boundaries (Eastern Time) ────────────────────────────
PRE_MARKET_OPEN = time(4, 0)
MARKET_OPEN = time(9, 30)
MIDDAY_START = time(11, 30)
MIDDAY_END = time(13, 30)
POWER_HOUR_START = time(15, 0)
MARKET_CLOSE = time(16, 0)
AFTER_HOURS_END = time(20, 0)

# ── Session labels ────────────────────────────────────────────────
SESSION_PRE_MARKET = "pre_market"
SESSION_OPEN = "open"         # 9:30 → 11:30 — high volume, trends set
SESSION_MIDDAY = "midday"     # 11:30 → 13:30 — choppy, low volume
SESSION_AFTERNOON = "afternoon"  # 13:30 → 15:00
SESSION_POWER_HOUR = "power_hour"  # 15:00 → 16:00 — institutional activity
SESSION_AFTER_HOURS = "after_hours"
SESSION_CLOSED = "closed"
SESSION_CRYPTO_247 = "crypto_24_7"


# ── Confidence multipliers per session ────────────────────────────
# < 1.0 means we're more cautious; > 1.0 means market conditions favor action
_SESSION_CONFIDENCE_MULT: Dict[str, float] = {
    SESSION_PRE_MARKET: 0.0,     # No stock trading
    SESSION_OPEN: 1.05,          # Slightly boost — trends form here
    SESSION_MIDDAY: 0.80,        # Reduce — choppy, reversals common
    SESSION_AFTERNOON: 0.95,     # Normal
    SESSION_POWER_HOUR: 1.10,    # Institutional flows, strong moves
    SESSION_AFTER_HOURS: 0.0,    # No stock trading
    SESSION_CLOSED: 0.0,         # No stock trading
    SESSION_CRYPTO_247: 1.0,     # Crypto trades anytime
}


def get_current_session(symbol: str, now: Optional[datetime] = None) -> str:
    """Classify the current trading session for a given symbol.

    Args:
        symbol: Ticker string.
        now:    Override current time (for testing / backtesting).
    Returns:
        Session label string.
    """
    if is_crypto(symbol):
        return SESSION_CRYPTO_247

    now_et = (now or datetime.now(ET)).astimezone(ET)
    t = now_et.time()
    d = now_et.date()

    # Weekend?
    if d.weekday() >= 5:
        return SESSION_CLOSED

    # Holiday?
    if d in _get_holidays(d.year):
        return SESSION_CLOSED

    if t < PRE_MARKET_OPEN:
        return SESSION_CLOSED
    elif t < MARKET_OPEN:
        return SESSION_PRE_MARKET
    elif t < MIDDAY_START:
        return SESSION_OPEN
    elif t < MIDDAY_END:
        return SESSION_MIDDAY
    elif t < POWER_HOUR_START:
        return SESSION_AFTERNOON
    elif t < MARKET_CLOSE:
        return SESSION_POWER_HOUR
    elif t < AFTER_HOURS_END:
        return SESSION_AFTER_HOURS
    else:
        return SESSION_CLOSED


def is_trading_allowed(symbol: str, now: Optional[datetime] = None) -> Tuple[bool, str]:
    """Check if trading is allowed right now for this symbol.

    Returns:
        (allowed: bool, reason: str)
    """
    session = get_current_session(symbol, now=now)

    if is_crypto(symbol):
        return True, "crypto_24_7"

    if session in (SESSION_CLOSED, SESSION_PRE_MARKET, SESSION_AFTER_HOURS):
        return False, f"market_{session}"

    return True, session


def session_confidence_multiplier(symbol: str, now: Optional[datetime] = None) -> float:
    """Return a confidence multiplier based on the current session.

    During midday chop, the multiplier is <1.0, reducing the effective
    confidence and making the bot less likely to trade.  During power
    hour and the open, it's boosted slightly.
    """
    session = get_current_session(symbol, now=now)
    return _SESSION_CONFIDENCE_MULT.get(session, 1.0)


def should_flatten_overnight(symbol: str, now: Optional[datetime] = None) -> bool:
    """Return True if it's close to market close and the symbol is a stock.

    Professional day traders flatten stock positions before the close
    to avoid overnight gap risk.
    """
    if is_crypto(symbol):
        return False

    now_et = (now or datetime.now(ET)).astimezone(ET)
    t = now_et.time()

    # Final 10 minutes before close: 15:50 → 16:00
    return time(15, 50) <= t < MARKET_CLOSE


def minutes_to_close(symbol: str, now: Optional[datetime] = None) -> Optional[int]:
    """Return minutes until market close for stocks, None for crypto."""
    if is_crypto(symbol):
        return None

    now_et = (now or datetime.now(ET)).astimezone(ET)
    close_dt = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    delta = (close_dt - now_et).total_seconds() / 60.0
    return int(delta) if delta > 0 else 0


def get_session_info(symbol: str, now: Optional[datetime] = None) -> Dict:
    """Return a comprehensive session info dict for logging/decisions."""
    session = get_current_session(symbol, now=now)
    allowed, reason = is_trading_allowed(symbol, now=now)
    return {
        "symbol": symbol,
        "session": session,
        "trading_allowed": allowed,
        "reason": reason,
        "confidence_multiplier": session_confidence_multiplier(symbol, now=now),
        "should_flatten": should_flatten_overnight(symbol, now=now),
        "minutes_to_close": minutes_to_close(symbol, now=now),
    }
