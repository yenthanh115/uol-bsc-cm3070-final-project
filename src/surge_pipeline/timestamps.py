"""Portable timestamp conversion utility.

Provides a single function to convert a pandas datetime Series to epoch
seconds as a numpy int64 array, regardless of the underlying datetime
resolution (nanoseconds, microseconds, etc.).

This avoids the pandas 2.x resolution bug where `datetime64[us]` columns
produce incorrect results with `.astype("int64") // 10**9`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_epoch_seconds(series: pd.Series) -> np.ndarray:
    """Convert a datetime Series to epoch seconds (int64 array).

    Handles any pandas datetime resolution (ns, us, ms, s) by converting
    through timedelta division rather than raw integer casting.

    Parameters
    ----------
    series : pd.Series
        A datetime64 or Timestamp Series. If not already datetime,
        will be parsed with pd.to_datetime(..., utc=True).

    Returns
    -------
    np.ndarray
        Array of int64 epoch seconds.
    """
    # Ensure we have a datetime series
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, utc=True)

    # Ensure UTC timezone
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")

    # Resolution-independent conversion via timedelta
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return ((series - epoch).dt.total_seconds()).values.astype(np.int64)
