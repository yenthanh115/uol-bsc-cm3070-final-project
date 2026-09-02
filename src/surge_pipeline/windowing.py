"""Per-ticker temporal windowing with vectorised binary search.

Computes forward and backward posting counts within 24-hour windows
for each (record, ticker) pair, and derives surge metrics.

Supports two surge methods:
  - "forward_growth": posting_volume_growth = forward/backward - 1
    (requires future posts to exist).
  - "backward_only": backward_surge_ratio = backward_count / historical_mean - 1
    (no forward-looking data needed).

Requirements: R2 (Temporal Windowing per Ticker), R3 (Posting Volume Growth)
Design Decision: D1 — searchsorted-based O(n log n) approach.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from surge_pipeline.config import WINDOW_SECONDS, PipelineConfig
from surge_pipeline.timestamps import to_epoch_seconds

logger = logging.getLogger(__name__)

# Alias for internal use (preserves existing references without renaming)
_WINDOW_SECONDS = WINDOW_SECONDS


def compute_windowed_counts(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Compute per-ticker temporal window counts and surge metrics.

    For each (record, ticker) pair:
      - backward_count: number of OTHER posts mentioning the same ticker
        in the prior 24h window, i.e. timestamps in (t - 24h, t) exclusive
        of self.
      - forward_count: number of posts mentioning the same ticker in the
        forward 24h window, i.e. timestamps in (t, t + 24h] exclusive of
        self.

    Depending on config.surge_method:
      - "forward_growth":
          posting_volume_growth = (forward_count / max(backward_count, 1)) - 1
          excluded = forward_count < config.min_window_count
      - "backward_only":
          backward_surge_ratio = backward_count / max(ticker_historical_mean, 1) - 1
          excluded = backward_count < config.min_window_count

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from the loader, with columns including 'created_utc'
        (datetime64[ns, UTC]) and 'ticker'. Expected to be sorted
        chronologically.
    config : PipelineConfig
        Pipeline configuration; uses ``min_window_count`` and ``surge_method``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: forward_count, backward_count,
        posting_volume_growth (or backward_surge_ratio), excluded.
    """
    # Validate required columns before any processing so a missing column
    # surfaces as a clear message rather than an opaque pandas KeyError.
    missing_cols = [c for c in ("created_utc", "ticker") if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "compute_windowed_counts requires column(s) "
            f"{missing_cols}. Expected 'created_utc' and 'ticker'."
        )

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
    epoch_seconds = to_epoch_seconds(df["created_utc"])

    # The searchsorted-based counting relies on chronological ordering: the
    # loader guarantees this, but the invariant is asserted here so that any
    # caller passing unsorted data fails loudly rather than producing silently
    # incorrect window counts (a temporal-leakage risk).
    if np.any(np.diff(epoch_seconds) < 0):
        raise ValueError(
            "Input DataFrame must be sorted chronologically by 'created_utc' "
            "before windowing. Found records out of temporal order."
        )

    # Group by ticker and process each group with vectorised searchsorted
    ticker_groups = df.groupby("ticker", sort=False)
    for _ticker, group in tqdm(ticker_groups, desc="Windowing", unit="ticker", leave=True):
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

    # ------------------------------------------------------------------
    # Compute surge metric based on configured method
    # ------------------------------------------------------------------
    surge_method = getattr(config, "surge_method", "forward_growth")

    if surge_method == "backward_only":
        # Backward-only: compare each record's backward_count to the
        # ticker's expanding historical mean (all prior records for that
        # ticker). This requires no forward-looking data.
        #
        # NOTE: the resulting backward surge ratio is deliberately stored in
        # the `posting_volume_growth` column (rather than a separate
        # `backward_surge_ratio` column) so that downstream stages read a
        # single, method-agnostic surge-metric column. The value's meaning
        # therefore depends on config.surge_method.
        posting_volume_growth = _compute_backward_surge_ratio(df, backward_counts)

        # Exclusion: insufficient backward history
        excluded = backward_counts < config.min_window_count

        logger.info(
            "Windowing complete (backward_only) — %d records | "
            "excluded: %d (%.1f%%) | min_window_count threshold: %d",
            n,
            excluded.sum(),
            excluded.sum() / n * 100,
            config.min_window_count,
        )
    else:
        # Forward growth (original method)
        # growth = (forward / max(backward, 1)) - 1
        denominator = np.maximum(backward_counts, 1)
        posting_volume_growth = (forward_counts / denominator) - 1.0

        # Flag exclusions (R2-AC3): forward_count < min_window_count
        excluded = forward_counts < config.min_window_count

        logger.info(
            "Windowing complete (forward_growth) — %d records | "
            "excluded: %d (%.1f%%) | min_window_count threshold: %d",
            n,
            excluded.sum(),
            excluded.sum() / n * 100,
            config.min_window_count,
        )

    # Assign new columns to DataFrame
    df = df.assign(
        forward_count=forward_counts,
        backward_count=backward_counts,
        posting_volume_growth=posting_volume_growth,
        excluded=excluded,
    )

    return df


def _compute_backward_surge_ratio(
    df: pd.DataFrame, backward_counts: np.ndarray
) -> np.ndarray:
    """Compute backward surge ratio: how much current activity exceeds historical norm.

    For each record, the ticker's historical mean backward_count is the
    expanding mean of all *prior* records for the same ticker (excluding
    the current one). The ratio is:
        backward_count / max(historical_mean, 1) - 1

    A value of 0 means activity matches historical average.
    A value of 1.0 means activity is 2× the historical average.

    For the first record of a ticker (no history), historical_mean is
    set to the record's own backward_count, yielding ratio = 0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'ticker' column, sorted chronologically.
    backward_counts : np.ndarray
        Pre-computed backward_count for each record.

    Returns
    -------
    np.ndarray
        Array of float64 backward surge ratios.
    """
    n = len(df)
    result = np.zeros(n, dtype=np.float64)

    for _ticker, group in df.groupby("ticker", sort=False):
        idx = group.index.values
        counts = backward_counts[idx].astype(np.float64)

        # Expanding mean of all *prior* records for this ticker.
        # For record i, historical_mean = mean(counts[0:i])
        # For i=0 (first record), use the record's own count (ratio=0).
        group_size = len(idx)
        if group_size == 1:
            # Single record — no history, ratio = 0
            result[idx[0]] = 0.0
            continue

        # Cumulative sum for expanding mean computation
        cumsum = np.cumsum(counts)

        # historical_mean[i] = cumsum[i-1] / i for i >= 1
        # For i=0: use counts[0] itself (yields ratio=0)
        historical_means = np.empty(group_size, dtype=np.float64)
        historical_means[0] = max(counts[0], 1.0)  # self → ratio=0
        historical_means[1:] = cumsum[:-1] / np.arange(1, group_size)

        # Avoid division by zero
        denominators = np.maximum(historical_means, 1.0)
        ratios = (counts / denominators) - 1.0

        # First record gets ratio 0 (no prior history to compare against)
        ratios[0] = 0.0

        result[idx] = ratios

    return result
