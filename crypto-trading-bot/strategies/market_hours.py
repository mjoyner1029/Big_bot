"""Market Hours & Trading Session Management.

Comprehensive market hours tracking for stocks, crypto, and forex.
Prevents trading during closed markets and manages position lifecycle
across market sessions.

Features:
  • Real-time market status (open/closed/pre-market/after-hours)
  • Time zone aware (handles NYSE, NASDAQ, crypto 24/7)
  • Pre-market and after-hours detection
  • Holiday calendar integration
  • Automatic position flattening before market close
  • Weekend/holiday detection

Usage:
    from strategies.market_hours import is_market_open, get_market_status
    
    if is_market_open("AAPL"):
        # Place stock trade
        pass
    
    status = get_market_status("AAPL")
    if status["should_flatten"]:
        # Close positions before market close
        pass
"""
import logging
from datetime import datetime, time, timedelta
from typing import Dict, Any, Optional
import pytz

from config.config import CONFIG, is_crypto


# ── Market hours definitions ──────────────────────────────────────

# US stock market (NYSE/NASDAQ) - Eastern Time
STOCK_MARKET_OPEN = time(9, 30)   # 9:30 AM ET
STOCK_MARKET_CLOSE = time(16, 0)  # 4:00 PM ET
PREMARKET_OPEN = time(4, 0)       # 4:00 AM ET
AFTERHOURS_CLOSE = time(20, 0)    # 8:00 PM ET

# Time zones
ET = pytz.timezone('America/New_York')
UTC = pytz.UTC


# ── Market status checks ──────────────────────────────────────────

def is_market_open(
    symbol: str,
    include_extended_hours: bool = False,
) -> bool:
    """Check if the market for this symbol is currently open.
    
    Args:
        symbol: Trading symbol
        include_extended_hours: For stocks, include pre-market and after-hours
    
    Returns:
        True if market is open and trading is allowed
    """
    # Crypto markets are always open
    if is_crypto(symbol):
        return True
    
    # Stock markets have specific hours
    status = get_market_status(symbol)
    
    if include_extended_hours:
        return status["is_open"] or status["is_extended_hours"]
    
    return status["is_open"]


def get_market_status(symbol: str) -> Dict[str, Any]:
    """Get detailed market status for a symbol.
    
    Args:
        symbol: Trading symbol
    
    Returns:
        Dict with:
            - is_open: bool (regular hours)
            - is_extended_hours: bool (pre/after market)
            - is_closed: bool
            - is_weekend: bool
            - is_holiday: bool
            - next_open: datetime
            - next_close: datetime
            - should_flatten: bool (close positions soon)
            - minutes_to_close: int
    """
    asset_type = "crypto" if is_crypto(symbol) else "stock"
    
    # Crypto is always open
    if asset_type == "crypto":
        return {
            "is_open": True,
            "is_extended_hours": False,
            "is_closed": False,
            "is_weekend": False,
            "is_holiday": False,
            "next_open": None,
            "next_close": None,
            "should_flatten": False,
            "minutes_to_close": None,
            "market_phase": "24/7",
        }
    
    # Stock market status
    now_et = datetime.now(ET)
    current_time = now_et.time()
    current_date = now_et.date()
    weekday = now_et.weekday()  # 0 = Monday, 6 = Sunday
    
    # Check if weekend
    is_weekend = weekday >= 5  # Saturday or Sunday
    
    # Check if holiday (simplified - you might want to add a proper holiday calendar)
    is_holiday = _is_market_holiday(current_date)
    
    # Determine market phase
    is_regular_hours = False
    is_extended_hours = False
    market_phase = "closed"
    
    if not is_weekend and not is_holiday:
        if PREMARKET_OPEN <= current_time < STOCK_MARKET_OPEN:
            is_extended_hours = True
            market_phase = "pre-market"
        elif STOCK_MARKET_OPEN <= current_time < STOCK_MARKET_CLOSE:
            is_regular_hours = True
            market_phase = "regular"
        elif STOCK_MARKET_CLOSE <= current_time < AFTERHOURS_CLOSE:
            is_extended_hours = True
            market_phase = "after-hours"
    
    # Calculate next open/close times
    next_open = _get_next_market_open(now_et)
    next_close = _get_next_market_close(now_et)
    
    # Calculate minutes to close
    minutes_to_close = None
    if is_regular_hours:
        close_today = ET.localize(datetime.combine(current_date, STOCK_MARKET_CLOSE))
        minutes_to_close = int((close_today - now_et).total_seconds() / 60)
    
    # Should flatten positions?
    flatten_before_close_min = CONFIG.get("flatten_before_close_minutes", 30)
    should_flatten = (
        is_regular_hours and 
        minutes_to_close is not None and 
        minutes_to_close <= flatten_before_close_min and
        CONFIG.get("flatten_overnight", True)
    )
    
    return {
        "is_open": is_regular_hours,
        "is_extended_hours": is_extended_hours,
        "is_closed": not is_regular_hours and not is_extended_hours,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "next_open": next_open.isoformat() if next_open else None,
        "next_close": next_close.isoformat() if next_close else None,
        "should_flatten": should_flatten,
        "minutes_to_close": minutes_to_close,
        "market_phase": market_phase,
    }


def _is_market_holiday(date: datetime.date) -> bool:
    """Check if date is a US stock market holiday.
    
    This is a simplified version. For production, use a proper holiday calendar
    like pandas_market_calendars.
    """
    # Major US holidays (simplified)
    year = date.year
    
    holidays = [
        # New Year's Day
        datetime(year, 1, 1).date(),
        # Martin Luther King Jr. Day (3rd Monday in January)
        # Presidents' Day (3rd Monday in February)
        # Good Friday (varies)
        # Memorial Day (last Monday in May)
        # Independence Day
        datetime(year, 7, 4).date(),
        # Labor Day (1st Monday in September)
        # Thanksgiving (4th Thursday in November)
        # Christmas
        datetime(year, 12, 25).date(),
    ]
    
    # If holiday falls on weekend, it's usually observed on Friday or Monday
    # This is simplified - a real implementation would handle this properly
    
    return date in holidays


def _get_next_market_open(from_dt: datetime) -> Optional[datetime]:
    """Get the next market open time from given datetime."""
    current = from_dt
    
    # Check today first
    today_open = ET.localize(
        datetime.combine(current.date(), STOCK_MARKET_OPEN)
    )
    
    if current < today_open and current.weekday() < 5:
        if not _is_market_holiday(current.date()):
            return today_open
    
    # Check subsequent days
    for days_ahead in range(1, 8):  # Look up to a week ahead
        next_date = (current + timedelta(days=days_ahead)).date()
        next_dt = datetime.combine(next_date, STOCK_MARKET_OPEN)
        next_dt = ET.localize(next_dt)
        
        # Skip weekends and holidays
        if next_dt.weekday() < 5 and not _is_market_holiday(next_date):
            return next_dt
    
    return None


def _get_next_market_close(from_dt: datetime) -> Optional[datetime]:
    """Get the next market close time from given datetime."""
    current = from_dt
    
    # Check today first
    today_close = ET.localize(
        datetime.combine(current.date(), STOCK_MARKET_CLOSE)
    )
    
    if current < today_close and current.weekday() < 5:
        if not _is_market_holiday(current.date()):
            return today_close
    
    # Otherwise return next open day's close
    next_open = _get_next_market_open(from_dt)
    if next_open:
        return ET.localize(
            datetime.combine(next_open.date(), STOCK_MARKET_CLOSE)
        )
    
    return None


# ── Position flattening ───────────────────────────────────────────

def should_flatten_positions(symbol: str) -> tuple[bool, str]:
    """Check if positions should be flattened before market close.
    
    Args:
        symbol: Trading symbol
    
    Returns:
        (should_flatten, reason)
    """
    if not CONFIG.get("flatten_overnight", True):
        return False, ""
    
    if is_crypto(symbol):
        return False, "crypto trades 24/7"
    
    status = get_market_status(symbol)
    
    if status["should_flatten"]:
        return True, (
            f"Market closes in {status['minutes_to_close']} minutes - "
            f"flattening overnight positions"
        )
    
    return False, ""


def can_open_new_position(symbol: str) -> tuple[bool, str]:
    """Check if we can open a new position based on market hours.
    
    Args:
        symbol: Trading symbol
    
    Returns:
        (can_open, reason)
    """
    # Check if trading is enforced to market hours
    if not CONFIG.get("enforce_market_hours", True):
        return True, ""
    
    if is_crypto(symbol):
        return True, "crypto trades 24/7"
    
    status = get_market_status(symbol)
    
    if status["is_closed"] or status["is_weekend"] or status["is_holiday"]:
        return False, f"Market closed ({status['market_phase']})"
    
    # Don't open new positions too close to market close
    if status["minutes_to_close"] is not None:
        min_minutes_required = CONFIG.get("min_minutes_before_close", 60)
        if status["minutes_to_close"] < min_minutes_required:
            return False, (
                f"Too close to market close "
                f"({status['minutes_to_close']} min remaining)"
            )
    
    # Check if we allow extended hours trading
    allow_extended = CONFIG.get("allow_extended_hours_trading", False)
    if status["is_extended_hours"] and not allow_extended:
        return False, f"Extended hours trading disabled ({status['market_phase']})"
    
    return True, ""


def get_market_hours_summary(symbols: list = None) -> str:
    """Get formatted summary of market hours for given symbols.
    
    Args:
        symbols: List of symbols to check. If None, uses watchlist.
    """
    if symbols is None:
        from config.config import get_all_symbols
        symbols = get_all_symbols()
    
    # Group by asset type
    crypto_syms = [s for s in symbols if is_crypto(s)]
    stock_syms = [s for s in symbols if not is_crypto(s)]
    
    lines = ["╔════════════════════════════════════════════════╗"]
    lines.append("║          MARKET HOURS STATUS                   ║")
    lines.append("╠════════════════════════════════════════════════╣")
    
    # Crypto status
    if crypto_syms:
        lines.append("║ CRYPTO MARKETS:    🟢 Open 24/7               ║")
    
    # Stock status
    if stock_syms:
        # Just check one stock symbol for status
        status = get_market_status(stock_syms[0])
        
        if status["is_open"]:
            icon = "🟢"
            phase_text = "OPEN (Regular Hours)"
        elif status["is_extended_hours"]:
            icon = "🟡"
            phase_text = f"Extended Hours ({status['market_phase']})"
        else:
            icon = "🔴"
            phase_text = "CLOSED"
        
        lines.append(f"║ STOCK MARKETS:     {icon} {phase_text:<27}║")
        
        if status["minutes_to_close"]:
            lines.append(f"║   Closes in:       {status['minutes_to_close']:>3} minutes                  ║")
        
        if status["next_open"]:
            next_open_dt = datetime.fromisoformat(status["next_open"])
            next_open_str = next_open_dt.strftime("%Y-%m-%d %H:%M %Z")
            lines.append(f"║   Next open:       {next_open_str:<27}║")
    
    lines.append("╚════════════════════════════════════════════════╝")
    
    return "\n".join(lines)
