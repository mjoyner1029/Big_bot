import pandas as pd

def normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series to 0-1 range.

    Args:
        series: Input pandas Series.

    Returns:
        Normalized Series.
    """
    return (series - series.min()) / (series.max() - series.min())
