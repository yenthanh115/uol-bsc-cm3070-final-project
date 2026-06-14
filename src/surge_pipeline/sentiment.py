"""Sentiment computation for the surge-labelling pipeline.

Computes TextBlob polarity per record (title + selftext with fallback),
then derives mean future sentiment within each record's forward ticker
window and sentiment change magnitude.

Requirements: R4 (Sentiment Computation)
Design Decision: D5 — TextBlob polarity on combined text with fallback logic.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from textblob import TextBlob

from surge_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# 24 hours in seconds (same window size as windowing stage)
_WINDOW_SECONDS: int = 24 * 60 * 60


def _compute_polarity(title: str, selftext: str) -> float:
    """Compute TextBlob polarity for a single record.

    Parameters
    ----------
    title : str
        Post title (required field, may still be empty in edge cases).
    selftext : str
        Post body text (may be empty/NaN).

    Returns
    -------
    float
        Polarity score in [-1.0, 1.0]. Returns 0.0 (neutral) if both
        title and selftext are empty.
    """
    # Combine title + selftext when selftext is available
    if selftext and str(selftext).strip():
        text = f"{title} {selftext}"
    elif title and str(title).strip():
        text = title
    else:
        return 0.0

    return TextBlob(text).sentiment.polarity


def compute_sentiment(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Compute sentiment polarity and mean future sentiment for each record.

    Workflow:
        1. Compute TextBlob polarity per record (AC1, AC4).
        2. For each record, compute mean sentiment of forward-window records
           mentioning the same ticker (AC2).
        3. Compute sentiment_change = |mean_future_sentiment - current| (AC3).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from the windowing stage, with columns including
        'created_utc' (datetime64[ns, UTC]), 'ticker', 'title', 'selftext'.
        May optionally have 'forward_count' and 'excluded' from windowing.
    config : PipelineConfig
        Pipeline configuration (used for consistency; no sentiment-specific
        params currently).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: sentiment_polarity,
        mean_future_sentiment, sentiment_change.
    """
    if df.empty:
        df = df.assign(
            sentiment_polarity=pd.array([], dtype="float64"),
            mean_future_sentiment=pd.array([], dtype="float64"),
            sentiment_change=pd.array([], dtype="float64"),
        )
        return df

    n = len(df)

    # ------------------------------------------------------------------
    # Step 1: Compute per-record polarity (AC1, AC4)
    # ------------------------------------------------------------------
    # Normalise text columns: replace NaN/None with empty string
    titles = df["title"].fillna("").astype(str)
    selftexts = df["selftext"].fillna("").astype(str)

    polarities = np.zeros(n, dtype=np.float64)
    fallback_count = 0
    neutral_count = 0

    for i in range(n):
        title = titles.iloc[i].strip()
        selftext = selftexts.iloc[i].strip()

        if selftext:
            # Full text: title + selftext
            text = f"{title} {selftext}"
            polarities[i] = TextBlob(text).sentiment.polarity
        elif title:
            # Fallback: title only (AC4 graceful handling)
            polarities[i] = TextBlob(title).sentiment.polarity
            fallback_count += 1
        else:
            # Both empty — assign neutral (D5 design decision)
            polarities[i] = 0.0
            neutral_count += 1

    logger.info(
        "Polarity computed — %d records | title-only fallback: %d | "
        "neutral (empty text): %d",
        n,
        fallback_count,
        neutral_count,
    )

    # ------------------------------------------------------------------
    # Step 2: Mean future sentiment per record (AC2)
    # ------------------------------------------------------------------
    mean_future = np.full(n, np.nan, dtype=np.float64)

    # Convert timestamps to epoch seconds for binary search
    epoch_seconds = (
        df["created_utc"].astype("int64") // 10**9
    ).values

    # Check if windowing columns are available
    has_excluded = "excluded" in df.columns

    # Group by ticker and compute mean forward-window sentiment
    for ticker, group in df.groupby("ticker", sort=False):
        idx = group.index.values
        times = epoch_seconds[idx]
        group_polarities = polarities[idx]

        # Forward window: (t, t + 24h] — same logic as windowing.py
        forward_left = np.searchsorted(times, times, side="right")
        forward_right = np.searchsorted(times, times + _WINDOW_SECONDS, side="right")

        for j, orig_idx in enumerate(idx):
            fl = forward_left[j]
            fr = forward_right[j]

            if fl < fr:
                # There are records in the forward window
                mean_future[orig_idx] = group_polarities[fl:fr].mean()
            else:
                # No forward records: default to current sentiment
                # (so sentiment_change = 0)
                mean_future[orig_idx] = polarities[orig_idx]

    # For excluded records (if windowing was applied), set mean_future
    # to current sentiment so sentiment_change = 0
    if has_excluded:
        excluded_mask = df["excluded"].values.astype(bool)
        mean_future[excluded_mask] = polarities[excluded_mask]

    # ------------------------------------------------------------------
    # Step 3: Sentiment change (AC3)
    # ------------------------------------------------------------------
    sentiment_change = np.abs(mean_future - polarities)

    # Assign columns
    df = df.assign(
        sentiment_polarity=polarities,
        mean_future_sentiment=mean_future,
        sentiment_change=sentiment_change,
    )

    # Log summary statistics
    logger.info(
        "Sentiment summary — mean polarity: %.4f | mean future: %.4f | "
        "mean |change|: %.4f",
        polarities.mean(),
        mean_future.mean(),
        sentiment_change.mean(),
    )

    return df
