"""Canonical OHLCV columns without silently discarding conflicting data."""
import pandas as pd


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work.columns = [str(c).lower() for c in work.columns]
    for name in work.columns[work.columns.duplicated()].unique():
        aliases = work.loc[:, work.columns == name]
        first = aliases.iloc[:, 0]
        for i in range(1, aliases.shape[1]):
            other = aliases.iloc[:, i]
            if not (first.eq(other) | (first.isna() & other.isna())).all():
                raise ValueError(f"Conflicting OHLCV aliases: {name}")
    return work.loc[:, ~work.columns.duplicated()].copy()
