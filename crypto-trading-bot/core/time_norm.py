"""Canonical timestamp normalization: everything becomes timezone-aware UTC.

Naive datetimes are localized with an EXPLICITLY configured source timezone —
never the local machine's zone. Unresolvable inputs raise
TimezoneNormalizationError (a structured, catchable condition), never a
silent guess.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo


class TimezoneNormalizationError(ValueError):
    """Timestamp could not be normalized to UTC without guessing."""


_EPOCH_MS_THRESHOLD = 10_000_000_000       # > this = epoch milliseconds


def to_utc(value: Any, *, source_timezone: Optional[str] = "UTC") -> datetime:
    """Normalize str/datetime/epoch/pandas timestamps to aware-UTC datetime.

    Naive inputs use ``source_timezone`` (explicit policy); pass
    ``source_timezone=None`` to make naive input a structured error.
    """
    if value is None:
        raise TimezoneNormalizationError("timestamp is None")

    # pandas.Timestamp quacks like datetime (has tzinfo + to_pydatetime)
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()

    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > _EPOCH_MS_THRESHOLD:
            seconds /= 1000.0               # epoch milliseconds
        return datetime.fromtimestamp(seconds, tz=timezone.utc)

    if isinstance(value, str):
        raw = value.strip().replace("Z", "+00:00")
        try:
            value = datetime.fromisoformat(raw)
        except ValueError as e:
            raise TimezoneNormalizationError(
                f"unparseable timestamp {value!r}") from e

    if not isinstance(value, datetime):
        raise TimezoneNormalizationError(
            f"unsupported timestamp type {type(value).__name__}")

    if value.tzinfo is None:
        if source_timezone is None:
            raise TimezoneNormalizationError(
                "naive datetime with no configured source timezone")
        value = value.replace(tzinfo=ZoneInfo(source_timezone))
    return value.astimezone(timezone.utc)
