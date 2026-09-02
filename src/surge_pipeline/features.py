"""Feature engineering module — backward-only prediction features.

Computes 11 prediction features from the labelled dataset using only
backward-looking or creation-time information, preventing temporal leakage.

Base features (9):
  1. sentiment_score — reuse sentiment_polarity from sentiment stage
  2. hour_of_day — extract from created_utc (0–23)
  3. day_of_week — extract from created_utc (0–6, Monday=0)
  4. time_since_previous — hours since last post mentioning same ticker (−1 if first)
  5. ticker_post_rate_24h — reuse backward_count from windowing
  6. ticker_post_acceleration — ratio of 12h/12h backward counts
  7. word_count — whitespace-separated tokens in title+selftext
  8. title_length — whitespace-separated tokens in title
  9. num_tickers_mentioned — count of distinct tickers per original record

Interaction features (2, experiment B2):
  10. word_count_x_hour — word_count × hour_of_day (long posts at peak hours)
  11. accel_x_time_since_prev — ticker_post_acceleration × time_since_previous
      (rapid acceleration after silence)

Requirements: R11 (Prediction Feature Engineering), R12 (Feature Leakage Prevention)
Design Decision: D8 — Backward-only feature computation with vectorised windowing.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from surge_pipeline.timestamps import to_epoch_seconds

logger = logging.getLogger(__name__)

# Window sizes in seconds
_12H_SECONDS: int = 12 * 60 * 60
_24H_SECONDS: int = 24 * 60 * 60

# The 11 feature columns produced by this module (9 base + 2 interactions)
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
    # Interaction features (B2)
    "word_count_x_hour",
    "accel_x_time_since_prev",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 11 prediction features from the labelled dataset.

    All features use only information available at or before observation
    time t (R12-AC1). Engagement metrics (score, num_comments) are
    explicitly excluded (R11-AC9, R12-AC3). Includes 2 interaction
    features (B2 experiment).

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

    # ------------------------------------------------------------------
    # Validate required columns up front so a missing input surfaces as a
    # clear message rather than an opaque pandas KeyError (matches the
    # earlier pipeline stages).
    # ------------------------------------------------------------------
    required_cols = (
        "id",
        "ticker",
        "created_utc",
        "title",
        "selftext",
        "backward_count",
        "sentiment_polarity",
    )
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "compute_features requires column(s) "
            f"{missing_cols}. Expected {list(required_cols)} from the "
            "loader, windowing, and sentiment stages."
        )

    # Several features (time_since_previous, acceleration) rely on each
    # ticker group being chronologically ordered. The loader guarantees
    # global chronological order; assert it here so any caller passing
    # unsorted data fails loudly rather than producing silently incorrect,
    # potentially leaky, backward-looking features.
    _epoch_check = to_epoch_seconds(pd.to_datetime(df["created_utc"], utc=True))
    if np.any(np.diff(_epoch_check) < 0):
        raise ValueError(
            "Input DataFrame must be sorted chronologically by 'created_utc' "
            "before feature engineering. Found records out of temporal order."
        )

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
    # Feature 10: word_count_x_hour (B2 interaction)
    # Long posts at peak hours — gives models an explicit interaction
    # signal between content length and temporal posting pattern.
    # ------------------------------------------------------------------
    df = df.assign(
        word_count_x_hour=(df["word_count"] * df["hour_of_day"]).values
    )

    # ------------------------------------------------------------------
    # Feature 11: accel_x_time_since_prev (B2 interaction)
    # Rapid acceleration after silence — combines ticker momentum with
    # gap duration. For first-occurrence records (time_since_previous=-1),
    # use 0 to avoid spurious negative products.
    # ------------------------------------------------------------------
    tsp = df["time_since_previous"].values.copy()
    tsp_safe = np.where(tsp < 0, 0.0, tsp)
    df = df.assign(
        accel_x_time_since_prev=(df["ticker_post_acceleration"].values * tsp_safe)
    )

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

    epoch_seconds = to_epoch_seconds(created_utc)

    for _ticker, group in df.groupby("ticker", sort=False):
        idx = group.index.values
        if len(idx) < 2:
            # Single occurrence — first record stays -1 (AC11)
            continue

        times = epoch_seconds[idx]

        # Within each chronologically-sorted ticker group, the gap to the
        # immediately preceding same-ticker post is the successive time
        # difference. np.diff vectorises the previous per-record Python loop;
        # the first occurrence is left at -1 (AC11).
        result[idx[1:]] = np.diff(times) / 3600.0

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

    epoch_seconds = to_epoch_seconds(created_utc)

    for _ticker, group in df.groupby("ticker", sort=False):
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

    Returns only the 11 feature columns, suitable for model training.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that has had compute_features() applied.

    Returns
    -------
    pd.DataFrame
        DataFrame with only the 11 feature columns.
    """
    return df[FEATURE_COLUMNS].copy()
