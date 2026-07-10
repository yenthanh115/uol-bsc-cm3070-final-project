"""Feature engineering module — backward-only prediction features.

Computes 9 prediction features from the labelled dataset using only
backward-looking or creation-time information, preventing temporal leakage.

Features:
  1. sentiment_score — reuse sentiment_polarity from sentiment stage
  2. hour_of_day — extract from created_utc (0–23)
  3. day_of_week — extract from created_utc (0–6, Monday=0)
  4. time_since_previous — hours since last post mentioning same ticker (−1 if first)
  5. ticker_post_rate_24h — reuse backward_count from windowing
  6. ticker_post_acceleration — ratio of 12h/12h backward counts
  7. word_count — whitespace-separated tokens in title+selftext
  8. title_length — whitespace-separated tokens in title
  9. num_tickers_mentioned — count of distinct tickers per original record

Requirements: R11 (Prediction Feature Engineering), R12 (Feature Leakage Prevention)
Design Decision: D8 — Backward-only feature computation with vectorised windowing.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Window sizes in seconds
_12H_SECONDS: int = 12 * 60 * 60
_24H_SECONDS: int = 24 * 60 * 60

# The 9 feature columns produced by this module
FEATURE_COLUMNS: list[str] = [
    "sentiment_score",
    "hour_of_day",
    "day_of_week",
    "time_since_previous",
    "ticker_post_rate_24h",
    "ticker_post_acceleration",
    "word_count",
    "title_length",
    "num_tickers_mentioned",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 9 prediction features from the labelled dataset.

    All features use only information available at or before observation
    time t (R12-AC1). Engagement metrics (score, num_comments) are
    explicitly excluded (R11-AC9, R12-AC3).

    Parameters
    ----------
    df : pd.DataFrame
        Labelled dataset from pipeline.py with columns including:
        'id', 'ticker', 'created_utc', 'title', 'selftext',
        'backward_count', 'sentiment_polarity', 'excluded'.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 9 additional feature columns added.
    """
    if df.empty:
        for col in FEATURE_COLUMNS:
            df[col] = pd.array([], dtype="float64")
        return df

    n = len(df)

    # ------------------------------------------------------------------
    # Feature 1: sentiment_score (R11-AC1)
    # Reuse sentiment_polarity computed at observation time
    # ------------------------------------------------------------------
    df = df.assign(sentiment_score=df["sentiment_polarity"].values.copy())

    # ------------------------------------------------------------------
    # Features 2–3: hour_of_day, day_of_week (R11-AC2)
    # Extract from created_utc (creation-time information only)
    # ------------------------------------------------------------------
    created_utc = pd.to_datetime(df["created_utc"], utc=True)
    df = df.assign(
        hour_of_day=created_utc.dt.hour.values,
        day_of_week=created_utc.dt.dayofweek.values,  # Monday=0, Sunday=6
    )

    # ------------------------------------------------------------------
    # Feature 4: time_since_previous (R11-AC3, AC11)
    # Hours since most recent prior post mentioning same ticker.
    # Set to -1 if no prior post exists.
    # ------------------------------------------------------------------
    time_since_previous = _compute_time_since_previous(df, created_utc)
    df = df.assign(time_since_previous=time_since_previous)

    # ------------------------------------------------------------------
    # Feature 5: ticker_post_rate_24h (R11-AC4)
    # Reuse backward_count from windowing stage (already backward-only)
    # ------------------------------------------------------------------
    df = df.assign(ticker_post_rate_24h=df["backward_count"].values.copy())

    # ------------------------------------------------------------------
    # Feature 6: ticker_post_acceleration (R11-AC5, AC10)
    # Ratio of 12h/12h backward counts using searchsorted
    # ------------------------------------------------------------------
    ticker_post_acceleration = _compute_ticker_post_acceleration(df, created_utc)
    df = df.assign(ticker_post_acceleration=ticker_post_acceleration)

    # ------------------------------------------------------------------
    # Feature 7: word_count (R11-AC6)
    # Whitespace-separated token count of title + selftext
    # ------------------------------------------------------------------
    titles = df["title"].fillna("").astype(str)
    selftexts = df["selftext"].fillna("").astype(str)
    combined_text = titles + " " + selftexts
    word_count = combined_text.str.split().str.len().fillna(0).astype(int)
    df = df.assign(word_count=word_count.values)

    # ------------------------------------------------------------------
    # Feature 8: title_length (R11-AC7)
    # Whitespace-separated token count of title only
    # ------------------------------------------------------------------
    title_length = titles.str.split().str.len().fillna(0).astype(int)
    df = df.assign(title_length=title_length.values)

    # ------------------------------------------------------------------
    # Feature 9: num_tickers_mentioned (R11-AC8)
    # Count of distinct tickers per original record (using 'id' column)
    # ------------------------------------------------------------------
    num_tickers = _compute_num_tickers_mentioned(df)
    df = df.assign(num_tickers_mentioned=num_tickers)

    # ------------------------------------------------------------------
    # Log feature matrix shape and summary statistics (R12-AC5)
    # ------------------------------------------------------------------
    _log_feature_summary(df)

    return df


def _compute_time_since_previous(
    df: pd.DataFrame, created_utc: pd.Series
) -> np.ndarray:
    """Compute hours since the most recent prior post for the same ticker.

    Uses vectorised approach: for each ticker group (sorted chronologically),
    the previous timestamp is simply the preceding element in the group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'ticker' column.
    created_utc : pd.Series
        Parsed datetime series (UTC).

    Returns
    -------
    np.ndarray
        Array of float64 with hours since previous, or -1 if first.
    """
    n = len(df)
    result = np.full(n, -1.0, dtype=np.float64)

    from surge_pipeline.timestamps import to_epoch_seconds
    epoch_seconds = to_epoch_seconds(created_utc)

    for ticker, group in df.groupby("ticker", sort=False):
        idx = group.index.values
        times = epoch_seconds[idx]

        # For each record (except the first), time_since_previous is
        # the difference to the immediately preceding same-ticker record.
        # Since data is sorted chronologically, the previous element in
        # the group is the most recent prior post for that ticker.
        for j in range(1, len(idx)):
            hours_diff = (times[j] - times[j - 1]) / 3600.0
            result[idx[j]] = hours_diff

        # First occurrence for this ticker remains -1 (AC11)

    return result


def _compute_ticker_post_acceleration(
    df: pd.DataFrame, created_utc: pd.Series
) -> np.ndarray:
    """Compute ticker post acceleration using searchsorted.

    acceleration = count_in_(t-12h, t] / max(count_in_(t-24h, t-12h], 1)

    If denominator is zero (no posts in prior 12–24h window), set value
    to the numerator count — treating empty denominator as 1 (R11-AC10).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'ticker' column.
    created_utc : pd.Series
        Parsed datetime series (UTC).

    Returns
    -------
    np.ndarray
        Array of float64 with acceleration values.
    """
    n = len(df)
    result = np.zeros(n, dtype=np.float64)

    from surge_pipeline.timestamps import to_epoch_seconds
    epoch_seconds = to_epoch_seconds(created_utc)

    for ticker, group in df.groupby("ticker", sort=False):
        idx = group.index.values
        times = epoch_seconds[idx]

        # Count in (t-12h, t]: posts strictly after t-12h and at or before t
        # Using searchsorted on the sorted times array:
        #   left boundary: searchsorted(times, t - 12h, side='right')
        #     → first index where time > (t - 12h)
        #   right boundary: searchsorted(times, t, side='left')
        #     → first index where time >= t (excludes self)
        # count_recent = right - left
        recent_left = np.searchsorted(times, times - _12H_SECONDS, side="right")
        recent_right = np.searchsorted(times, times, side="left")
        count_recent = recent_right - recent_left

        # Count in (t-24h, t-12h]: posts strictly after t-24h and at or before t-12h
        #   left boundary: searchsorted(times, t - 24h, side='right')
        #     → first index where time > (t - 24h)
        #   right boundary: searchsorted(times, t - 12h, side='left')
        #     → first index where time >= (t - 12h)
        #   Wait — we want posts <= t-12h. Since side='right' gives first index
        #   where time > (t-12h), that's what we want as the right boundary.
        #   Actually: we want count of times in (t-24h, t-12h].
        #   (t-24h, t-12h] means time > t-24h AND time <= t-12h.
        #   left = searchsorted(times, t-24h, side='right') → first > t-24h
        #   right = searchsorted(times, t-12h, side='right') → first > t-12h
        #   count = right - left (all elements > t-24h and <= t-12h)
        older_left = np.searchsorted(times, times - _24H_SECONDS, side="right")
        older_right = np.searchsorted(times, times - _12H_SECONDS, side="right")
        count_older = older_right - older_left

        # Acceleration = count_recent / max(count_older, 1) (AC10)
        denominator = np.maximum(count_older, 1)
        acceleration = count_recent.astype(np.float64) / denominator.astype(np.float64)

        result[idx] = acceleration

    return result


def _compute_num_tickers_mentioned(df: pd.DataFrame) -> np.ndarray:
    """Count distinct tickers per original record.

    Since the dataset has been exploded (one row per ticker per post),
    we group by 'id' to count the number of distinct tickers mentioned
    in each original post.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'id' and 'ticker' columns.

    Returns
    -------
    np.ndarray
        Array of int with the count of distinct tickers per original record.
    """
    # Count distinct tickers per original post id
    ticker_counts = df.groupby("id")["ticker"].transform("nunique")
    return ticker_counts.values.astype(np.int64)


def _log_feature_summary(df: pd.DataFrame) -> None:
    """Log feature matrix shape and summary statistics (R12-AC5).

    Reports mean, std, min, max for each feature column on non-excluded records.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns and 'excluded' flag.
    """
    # Feature matrix shape
    feature_matrix = df[FEATURE_COLUMNS]
    logger.info(
        "Feature matrix shape: %d rows × %d features",
        len(feature_matrix),
        len(FEATURE_COLUMNS),
    )

    # Summary statistics on non-excluded records only
    if "excluded" in df.columns:
        included_mask = ~df["excluded"].astype(bool)
        included_features = feature_matrix.loc[included_mask]
    else:
        included_features = feature_matrix

    logger.info("Feature summary statistics (non-excluded records):")
    logger.info(
        "%-25s %10s %10s %10s %10s",
        "Feature", "Mean", "Std", "Min", "Max",
    )
    logger.info("-" * 70)

    for col in FEATURE_COLUMNS:
        values = included_features[col].astype(float)
        logger.info(
            "%-25s %10.4f %10.4f %10.4f %10.4f",
            col,
            values.mean(),
            values.std(),
            values.min(),
            values.max(),
        )


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the feature matrix from a DataFrame with computed features.

    Returns only the 9 feature columns, suitable for model training.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that has had compute_features() applied.

    Returns
    -------
    pd.DataFrame
        DataFrame with only the 9 feature columns.
    """
    return df[FEATURE_COLUMNS].copy()
