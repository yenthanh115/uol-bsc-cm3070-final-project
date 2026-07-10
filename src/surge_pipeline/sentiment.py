"""Sentiment computation for the surge-labelling pipeline.

Computes sentiment polarity per record (title + selftext with fallback),
then derives mean future sentiment within each record's forward ticker
window and sentiment change magnitude.

Supports two sentiment backends:
  - "vader" (default): VADER SentimentIntensityAnalyzer compound score.
    Better for social media text (exclamation, capitalisation, slang).
  - "textblob": TextBlob polarity. Simpler, general-purpose.

Requirements: R4 (Sentiment Computation)
Design Decision: D5 — Sentiment on combined text with fallback logic.
                 D13 — VADER as default sentiment model for Reddit text.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from surge_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# 24 hours in seconds (same window size as windowing stage)
_WINDOW_SECONDS: int = 24 * 60 * 60


# ---------------------------------------------------------------------------
# Sentiment backend functions
# ---------------------------------------------------------------------------


def _compute_polarity_vader(title: str, selftext: str) -> float:
    """Compute VADER compound polarity for a single record.

    Parameters
    ----------
    title : str
        Post title.
    selftext : str
        Post body text (may be empty).

    Returns
    -------
    float
        Compound score in [-1.0, 1.0]. Returns 0.0 if text is empty.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    # Use module-level cache to avoid re-creating the analyzer per record
    if not hasattr(_compute_polarity_vader, "_analyzer"):
        _compute_polarity_vader._analyzer = SentimentIntensityAnalyzer()

    analyzer = _compute_polarity_vader._analyzer

    if selftext and str(selftext).strip():
        text = f"{title} {selftext}"
    elif title and str(title).strip():
        text = title
    else:
        return 0.0

    return analyzer.polarity_scores(text)["compound"]


def _compute_polarity_textblob(title: str, selftext: str) -> float:
    """Compute TextBlob polarity for a single record.

    Parameters
    ----------
    title : str
        Post title.
    selftext : str
        Post body text (may be empty).

    Returns
    -------
    float
        Polarity score in [-1.0, 1.0]. Returns 0.0 if text is empty.
    """
    from textblob import TextBlob

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
        1. Compute polarity per record using the configured backend (AC1, AC4).
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
        Pipeline configuration. Uses `sentiment_model` to select backend.

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

    # Select sentiment backend
    sentiment_model = getattr(config, "sentiment_model", "vader")
    if sentiment_model == "textblob":
        polarity_fn = _compute_polarity_textblob
    else:
        polarity_fn = _compute_polarity_vader

    logger.info("Sentiment model: %s", sentiment_model)

    # ------------------------------------------------------------------
    # Determine which records need polarity computation.
    # Excluded records (from windowing stage) will get sentiment_change=0
    # regardless, so we skip expensive VADER/TextBlob calls for them.
    # ------------------------------------------------------------------
    has_excluded = "excluded" in df.columns
    if has_excluded:
        included_mask = ~df["excluded"].values.astype(bool)
        n_included = int(included_mask.sum())
        n_excluded = n - n_included
        logger.info(
            "Skipping sentiment for %d excluded records; computing for %d included.",
            n_excluded,
            n_included,
        )
    else:
        included_mask = np.ones(n, dtype=bool)
        n_included = n
        n_excluded = 0

    # ------------------------------------------------------------------
    # Step 1: Compute per-record polarity (AC1, AC4)
    # ------------------------------------------------------------------
    titles = df["title"].fillna("").astype(str)
    selftexts = df["selftext"].fillna("").astype(str)

    polarities = np.zeros(n, dtype=np.float64)
    fallback_count = 0
    neutral_count = 0

    # Only compute polarity for included records
    included_indices = np.where(included_mask)[0]
    for i in included_indices:
        title = titles.iloc[i].strip()
        selftext = selftexts.iloc[i].strip()

        if selftext:
            polarities[i] = polarity_fn(title, selftext)
        elif title:
            polarities[i] = polarity_fn(title, "")
            fallback_count += 1
        else:
            polarities[i] = 0.0
            neutral_count += 1

    logger.info(
        "Polarity computed — %d records (of %d total) | title-only fallback: %d | "
        "neutral (empty text): %d",
        n_included,
        n,
        fallback_count,
        neutral_count,
    )

    # ------------------------------------------------------------------
    # Step 2: Mean future sentiment per record (AC2)
    # ------------------------------------------------------------------
    mean_future = np.full(n, np.nan, dtype=np.float64)

    # For excluded records, set mean_future = current polarity (=0.0)
    # so sentiment_change = 0. This avoids expensive groupby iteration.
    if has_excluded:
        excluded_indices = np.where(~included_mask)[0]
        mean_future[excluded_indices] = polarities[excluded_indices]

    # Convert timestamps to epoch seconds for binary search
    from surge_pipeline.timestamps import to_epoch_seconds
    epoch_seconds = to_epoch_seconds(df["created_utc"])

    # Only process included records in the forward-window computation.
    # Build a lookup from DataFrame index label → positional index to
    # correctly address the full-length arrays (epoch_seconds, polarities).
    if n_included > 0:
        index_to_pos = pd.Series(
            np.arange(n), index=df.index
        )
        included_df = df.loc[included_mask]

        for ticker, group in included_df.groupby("ticker", sort=False):
            idx = group.index.values
            pos = index_to_pos.loc[idx].values.astype(int)
            times = epoch_seconds[pos]
            group_polarities = polarities[pos]

            # Forward window: (t, t + 24h] — same logic as windowing.py
            forward_left = np.searchsorted(times, times, side="right")
            forward_right = np.searchsorted(
                times, times + _WINDOW_SECONDS, side="right"
            )

            for j, orig_pos in enumerate(pos):
                fl = forward_left[j]
                fr = forward_right[j]

                if fl < fr:
                    mean_future[orig_pos] = group_polarities[fl:fr].mean()
                else:
                    # No forward records: default to current sentiment
                    mean_future[orig_pos] = polarities[orig_pos]

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
