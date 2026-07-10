"""Per-ticker temporal windowing with vectorised binary search.

Computes forward and backward posting counts within 24-hour windows
for each (record, ticker) pair, and derives posting_volume_growth.

Requirements: R2 (Temporal Windowing per Ticker), R3 (Posting Volume Growth)
Design Decision: D1 — searchsorted-based O(n log n) approach.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from surge_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# 24 hours in seconds
_WINDOW_SECONDS: int = 24 * 60 * 60


def compute_windowed_counts(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Compute per-ticker temporal window counts and posting volume growth.

    For each (record, ticker) pair:
      - backward_count: number of OTHER posts mentioning the same ticker
        in the prior 24h window, i.e. timestamps in (t - 24h, t) exclusive
        of self.
      - forward_count: number of posts mentioning the same ticker in the
        forward 24h window, i.e. timestamps in (t, t + 24h] exclusive of
        self.
      - posting_volume_growth: (forward_count / max(backward_count, 1)) - 1
      - excluded: True if forward_count < config.min_window_count

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from the loader, with columns including 'created_utc'
        (datetime64[ns, UTC]) and 'ticker'. Expected to be sorted
        chronologically.
    config : PipelineConfig
        Pipeline configuration; uses ``min_window_count``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: forward_count, backward_count,
        posting_volume_growth, excluded.
    """
    if df.empty:
        df = df.assign(
            forward_count=pd.array([], dtype="int64"),
            backward_count=pd.array([], dtype="int64"),
            posting_volume_growth=pd.array([], dtype="float64"),
            excluded=pd.array([], dtype="bool"),
        )
        return df

    # Pre-allocate output arrays
    n = len(df)
    forward_counts = np.zeros(n, dtype=np.int64)
    backward_counts = np.zeros(n, dtype=np.int64)

    # Convert timestamps to epoch seconds for numeric binary search
    # This avoids datetime comparison overhead and works with searchsorted.
    # Uses resolution-independent conversion (safe across pandas versions).
    from surge_pipeline.timestamps import to_epoch_seconds
    epoch_seconds = to_epoch_seconds(df["created_utc"])

    # Group by ticker and process each group with vectorised searchsorted
    for ticker, group in df.groupby("ticker", sort=False):
        idx = group.index.values  # Original DataFrame indices for this group
        times = epoch_seconds[idx]  # Already sorted (loader guarantees chrono order)

        # Vectorised binary search for window boundaries
        # backward window: (t - 24h, t) — posts strictly before t, within 24h
        #   left boundary: first index where time > (t - 24h)
        #     = searchsorted(times, t - window, side='right')
        #   right boundary: first index where time >= t
        #     = searchsorted(times, t, side='left')
        #   count = right - left  (excludes self since side='left' at t)
        backward_left = np.searchsorted(times, times - _WINDOW_SECONDS, side="right")
        backward_right = np.searchsorted(times, times, side="left")
        group_backward = backward_right - backward_left

        # forward window: (t, t + 24h] — posts strictly after t, within 24h
        #   left boundary: first index where time > t
        #     = searchsorted(times, t, side='right')
        #   right boundary: first index where time > (t + 24h)
        #     = searchsorted(times, t + window, side='right')
        #   count = right - left  (excludes self since side='right' at t)
        forward_left = np.searchsorted(times, times, side="right")
        forward_right = np.searchsorted(times, times + _WINDOW_SECONDS, side="right")
        group_forward = forward_right - forward_left

        # Write results back to the full-length arrays
        forward_counts[idx] = group_forward
        backward_counts[idx] = group_backward

    # Compute posting_volume_growth (R3)
    # growth = (forward / max(backward, 1)) - 1
    denominator = np.maximum(backward_counts, 1)
    posting_volume_growth = (forward_counts / denominator) - 1.0

    # Flag exclusions (R2-AC3): forward_count < min_window_count
    excluded = forward_counts < config.min_window_count

    # Assign new columns to DataFrame
    df = df.assign(
        forward_count=forward_counts,
        backward_count=backward_counts,
        posting_volume_growth=posting_volume_growth,
        excluded=excluded,
    )

    # Log exclusion rate (R2-AC4)
    exclusion_rate = excluded.sum() / n * 100
    logger.info(
        "Windowing complete — %d records | excluded: %d (%.1f%%) | "
        "min_window_count threshold: %d",
        n,
        excluded.sum(),
        exclusion_rate,
        config.min_window_count,
    )

    return df
