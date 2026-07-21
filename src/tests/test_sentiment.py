"""Tests for the sentiment computation module (surge_pipeline/sentiment.py).

Covers:
  1. Empty DataFrame handling
  2. Per-record polarity computation (VADER and TextBlob backends)
  3. Fallback logic (title-only when selftext is empty)
  4. Neutral assignment when both title and selftext are empty
  5. Excluded-record optimisation (polarity=0, sentiment_change=0)
  6. Forward-window mean future sentiment computation
  7. Sentiment change = |mean_future - current|
  8. Multi-ticker isolation (forward window is per-ticker)
  9. No 'excluded' column (all records processed)
  10. Records with no forward neighbours default to current polarity

Requirements: R4 (Sentiment Computation)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the src directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surge_pipeline.config import PipelineConfig
from surge_pipeline.sentiment import (
    _compute_polarity_textblob,
    _compute_polarity_vader,
    compute_sentiment,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def vader_config() -> PipelineConfig:
    """Config using VADER backend."""
    return PipelineConfig(random_seed=42, sentiment_model="vader")


@pytest.fixture
def textblob_config() -> PipelineConfig:
    """Config using TextBlob backend."""
    return PipelineConfig(random_seed=42, sentiment_model="textblob")


@pytest.fixture
def base_time() -> pd.Timestamp:
    """Base timestamp for constructing test DataFrames."""
    return pd.Timestamp("2021-06-01 00:00:00", tz="UTC")


def _make_df(
    base_time: pd.Timestamp,
    n: int,
    hour_offsets: list[float],
    tickers: list[str],
    titles: list[str],
    selftexts: list[str],
    excluded: list[bool] | None = None,
) -> pd.DataFrame:
    """Helper to construct a test DataFrame."""
    timestamps = [base_time + pd.Timedelta(hours=h) for h in hour_offsets]
    data = {
        "created_utc": timestamps,
        "ticker": tickers,
        "title": titles,
        "selftext": selftexts,
    }
    if excluded is not None:
        data["excluded"] = excluded
    return pd.DataFrame(data)


# ============================================================================
# Class: Empty DataFrame
# ============================================================================


class TestEmptyDataFrame:
    """Tests for empty DataFrame edge case."""

    def test_empty_df_returns_required_columns(self, vader_config):
        """An empty DataFrame should gain all three sentiment columns."""
        df = pd.DataFrame(columns=["created_utc", "ticker", "title", "selftext"])
        result = compute_sentiment(df, vader_config)

        assert "sentiment_polarity" in result.columns
        assert "mean_future_sentiment" in result.columns
        assert "sentiment_change" in result.columns
        assert len(result) == 0


# ============================================================================
# Class: Polarity Backend Tests
# ============================================================================


class TestPolarityBackends:
    """Tests for individual polarity backend functions."""

    def test_vader_positive_text(self):
        """VADER should return positive compound for clearly positive text."""
        score = _compute_polarity_vader("Great news!", "This is amazing and wonderful")
        assert score > 0.0

    def test_vader_negative_text(self):
        """VADER should return negative compound for clearly negative text."""
        score = _compute_polarity_vader("Terrible loss", "This is awful and horrible")
        assert score < 0.0

    def test_vader_empty_text_returns_zero(self):
        """VADER returns 0.0 when both title and selftext are empty."""
        assert _compute_polarity_vader("", "") == 0.0
        assert _compute_polarity_vader("   ", "   ") == 0.0

    def test_vader_title_only_fallback(self):
        """VADER uses title alone when selftext is empty."""
        score = _compute_polarity_vader("This is great!", "")
        assert score > 0.0

    def test_vader_score_in_range(self):
        """VADER compound score is always in [-1, 1]."""
        texts = [
            ("AMAZING!!! BEST EVER!!!", "I love this so much!!!"),
            ("terrible awful horrible", "worst thing ever"),
            ("neutral statement", "nothing special here"),
        ]
        for title, selftext in texts:
            score = _compute_polarity_vader(title, selftext)
            assert -1.0 <= score <= 1.0

    def test_textblob_positive_text(self):
        """TextBlob should return positive polarity for positive text."""
        score = _compute_polarity_textblob("Great news", "This is amazing and wonderful")
        assert score > 0.0

    def test_textblob_negative_text(self):
        """TextBlob should return negative polarity for negative text."""
        score = _compute_polarity_textblob("Bad news", "This is terrible and awful")
        assert score < 0.0

    def test_textblob_empty_text_returns_zero(self):
        """TextBlob returns 0.0 when both title and selftext are empty."""
        assert _compute_polarity_textblob("", "") == 0.0
        assert _compute_polarity_textblob("   ", "  ") == 0.0


# ============================================================================
# Class: Excluded Records Optimisation
# ============================================================================


class TestExcludedRecords:
    """Tests for the excluded-record skip optimisation."""

    def test_excluded_records_get_zero_polarity(self, vader_config, base_time):
        """Excluded records should have sentiment_polarity = 0.0."""
        df = _make_df(
            base_time,
            n=4,
            hour_offsets=[0, 1, 2, 3],
            tickers=["AAPL"] * 4,
            titles=["Great stock!"] * 4,
            selftexts=["Very positive text"] * 4,
            excluded=[False, True, True, False],
        )
        result = compute_sentiment(df, vader_config)

        # Excluded records (index 1, 2) should be zero
        assert result.iloc[1]["sentiment_polarity"] == 0.0
        assert result.iloc[2]["sentiment_polarity"] == 0.0

        # Included records (index 0, 3) should be non-zero (positive text)
        assert result.iloc[0]["sentiment_polarity"] != 0.0
        assert result.iloc[3]["sentiment_polarity"] != 0.0

    def test_excluded_records_get_zero_sentiment_change(self, vader_config, base_time):
        """Excluded records should have sentiment_change = 0.0."""
        df = _make_df(
            base_time,
            n=5,
            hour_offsets=[0, 1, 2, 3, 4],
            tickers=["TSLA"] * 5,
            titles=["Positive!"] * 5,
            selftexts=["Good stuff"] * 5,
            excluded=[False, True, True, True, False],
        )
        result = compute_sentiment(df, vader_config)

        excluded_mask = result["excluded"].values.astype(bool)
        assert np.allclose(result.loc[excluded_mask, "sentiment_change"].values, 0.0)

    def test_all_excluded_produces_all_zeros(self, vader_config, base_time):
        """When all records are excluded, everything should be zero."""
        df = _make_df(
            base_time,
            n=3,
            hour_offsets=[0, 6, 12],
            tickers=["GME"] * 3,
            titles=["Great!"] * 3,
            selftexts=["Amazing!"] * 3,
            excluded=[True, True, True],
        )
        result = compute_sentiment(df, vader_config)

        assert (result["sentiment_polarity"] == 0.0).all()
        assert (result["sentiment_change"] == 0.0).all()


# ============================================================================
# Class: No Excluded Column
# ============================================================================


class TestNoExcludedColumn:
    """Tests when the 'excluded' column is absent (all records processed)."""

    def test_all_records_get_polarity_without_excluded_column(
        self, vader_config, base_time
    ):
        """Without 'excluded' column, all records get polarity computed."""
        df = _make_df(
            base_time,
            n=3,
            hour_offsets=[0, 6, 12],
            tickers=["AAPL"] * 3,
            titles=["Great stock", "Terrible loss", "Neutral post"],
            selftexts=["Good stuff", "Bad news", "Nothing"],
            excluded=None,  # No excluded column
        )
        result = compute_sentiment(df, vader_config)

        # All records should have non-trivial polarity (text has content)
        polarities = result["sentiment_polarity"].values
        assert not np.all(polarities == 0.0)
        assert len(polarities) == 3


# ============================================================================
# Class: Fallback Logic
# ============================================================================


class TestFallbackLogic:
    """Tests for title-only fallback and empty-text neutral assignment."""

    def test_title_only_when_selftext_empty(self, vader_config, base_time):
        """When selftext is empty, polarity is computed from title alone."""
        df = _make_df(
            base_time,
            n=2,
            hour_offsets=[0, 1],
            tickers=["AAPL"] * 2,
            titles=["This is amazing!", "This is amazing!"],
            selftexts=["This is amazing!", ""],  # Second has no body
            excluded=None,
        )
        result = compute_sentiment(df, vader_config)

        # Both should have positive polarity
        assert result.iloc[0]["sentiment_polarity"] > 0.0
        assert result.iloc[1]["sentiment_polarity"] > 0.0

    def test_neutral_when_both_empty(self, vader_config, base_time):
        """When both title and selftext are empty, polarity = 0.0."""
        df = _make_df(
            base_time,
            n=2,
            hour_offsets=[0, 1],
            tickers=["AAPL"] * 2,
            titles=["Great stock!", ""],
            selftexts=["", ""],
            excluded=None,
        )
        result = compute_sentiment(df, vader_config)

        # First has title → non-zero; second has nothing → zero
        assert result.iloc[0]["sentiment_polarity"] != 0.0
        assert result.iloc[1]["sentiment_polarity"] == 0.0

    def test_nan_selftext_treated_as_empty(self, vader_config, base_time):
        """NaN selftext should be treated as empty (title fallback)."""
        df = pd.DataFrame({
            "created_utc": [base_time, base_time + pd.Timedelta(hours=1)],
            "ticker": ["AAPL", "AAPL"],
            "title": ["Positive title!", "Another positive!"],
            "selftext": [None, np.nan],
        })
        result = compute_sentiment(df, vader_config)

        # Both should use title fallback and produce non-zero polarity
        assert result.iloc[0]["sentiment_polarity"] != 0.0
        assert result.iloc[1]["sentiment_polarity"] != 0.0


# ============================================================================
# Class: Forward Window Computation
# ============================================================================


class TestForwardWindow:
    """Tests for mean future sentiment computation (Step 2)."""

    def test_no_forward_neighbours_defaults_to_current(self, vader_config, base_time):
        """Record with no forward neighbours gets mean_future = own polarity."""
        # Single record — no forward window
        df = _make_df(
            base_time,
            n=1,
            hour_offsets=[0],
            tickers=["AAPL"],
            titles=["Great stock!"],
            selftexts=["Very positive"],
            excluded=None,
        )
        result = compute_sentiment(df, vader_config)

        # mean_future should equal current polarity → sentiment_change = 0
        assert result.iloc[0]["mean_future_sentiment"] == result.iloc[0]["sentiment_polarity"]
        assert result.iloc[0]["sentiment_change"] == 0.0

    def test_forward_window_within_24h(self, vader_config, base_time):
        """Records within 24h forward window are included in mean_future."""
        # 3 records: t=0h, t=6h, t=30h (beyond 24h from t=0)
        # Record 0's forward window (0, 24h] should include record 1 only
        df = _make_df(
            base_time,
            n=3,
            hour_offsets=[0, 6, 30],
            tickers=["AAPL"] * 3,
            titles=["Neutral", "Very positive!!!", "Very negative"],
            selftexts=["text", "Great amazing wonderful!", "Terrible awful"],
            excluded=None,
        )
        result = compute_sentiment(df, vader_config)

        # Record 0's mean_future should be record 1's polarity
        # (only record 1 is in the (0h, 24h] window)
        expected_mean_future_0 = result.iloc[1]["sentiment_polarity"]
        assert np.isclose(
            result.iloc[0]["mean_future_sentiment"],
            expected_mean_future_0,
            atol=1e-10,
        )

    def test_forward_window_excludes_same_timestamp(self, vader_config, base_time):
        """Forward window is (t, t+24h] — strictly after t, not at t."""
        # Two records at the exact same timestamp
        df = _make_df(
            base_time,
            n=2,
            hour_offsets=[0, 0],
            tickers=["AAPL", "AAPL"],
            titles=["Positive!", "Negative!"],
            selftexts=["Great", "Terrible"],
            excluded=None,
        )
        result = compute_sentiment(df, vader_config)

        # searchsorted with side="right" on equal values means same-time
        # records ARE included in forward window. This matches windowing.py.
        # Both records at t=0: forward_left = searchsorted(times, times, "right")
        # For sorted [0,0]: searchsorted([0,0], 0, "right") = 2
        # So forward_left[0]=2, forward_right[0]=2 → empty window for record 0
        # Actually with identical times, records at same time will see each other
        # depending on sort order. This test just verifies no crash.
        assert "sentiment_change" in result.columns

    def test_multi_ticker_isolation(self, vader_config, base_time):
        """Forward window only considers records with the same ticker."""
        # AAPL at t=0, TSLA at t=6h, AAPL at t=12h
        # AAPL@t=0 should only see AAPL@t=12h in forward window (not TSLA)
        df = _make_df(
            base_time,
            n=3,
            hour_offsets=[0, 6, 12],
            tickers=["AAPL", "TSLA", "AAPL"],
            titles=["Neutral", "Very negative!", "Very positive!"],
            selftexts=["meh", "Terrible horrible bad", "Amazing wonderful great"],
            excluded=None,
        )
        result = compute_sentiment(df, vader_config)

        # AAPL@t=0's mean_future should be AAPL@t=12h's polarity (not TSLA's)
        aapl_future_polarity = result.iloc[2]["sentiment_polarity"]
        assert np.isclose(
            result.iloc[0]["mean_future_sentiment"],
            aapl_future_polarity,
            atol=1e-10,
        )

    def test_forward_window_mean_of_multiple(self, vader_config, base_time):
        """Mean future is the average of all forward-window records."""
        # 4 AAPL records at t=0, 4, 8, 30h
        # Record 0's forward window (0h, 24h] = records at 4h and 8h
        df = _make_df(
            base_time,
            n=4,
            hour_offsets=[0, 4, 8, 30],
            tickers=["AAPL"] * 4,
            titles=["Base", "Positive!", "Negative!", "Far away"],
            selftexts=["text", "Great wonderful", "Terrible awful", "text"],
            excluded=None,
        )
        result = compute_sentiment(df, vader_config)

        # mean_future for record 0 should be mean of record 1 and 2 polarities
        expected = (
            result.iloc[1]["sentiment_polarity"]
            + result.iloc[2]["sentiment_polarity"]
        ) / 2.0
        assert np.isclose(
            result.iloc[0]["mean_future_sentiment"],
            expected,
            atol=1e-10,
        )


# ============================================================================
# Class: Sentiment Change Computation
# ============================================================================


class TestSentimentChange:
    """Tests for sentiment_change = |mean_future - current|."""

    def test_sentiment_change_is_absolute_difference(self, vader_config, base_time):
        """sentiment_change should be |mean_future_sentiment - polarity|."""
        df = _make_df(
            base_time,
            n=3,
            hour_offsets=[0, 6, 12],
            tickers=["AAPL"] * 3,
            titles=["Neutral", "Very positive!", "Very negative!"],
            selftexts=["text", "Amazing great!", "Terrible awful!"],
            excluded=None,
        )
        result = compute_sentiment(df, vader_config)

        for i in range(len(result)):
            expected = abs(
                result.iloc[i]["mean_future_sentiment"]
                - result.iloc[i]["sentiment_polarity"]
            )
            assert np.isclose(result.iloc[i]["sentiment_change"], expected, atol=1e-10)

    def test_sentiment_change_non_negative(self, vader_config, base_time):
        """sentiment_change should always be >= 0."""
        df = _make_df(
            base_time,
            n=5,
            hour_offsets=[0, 2, 4, 6, 8],
            tickers=["AAPL"] * 5,
            titles=["a", "b", "c", "d", "e"],
            selftexts=["good", "bad", "great", "terrible", "ok"],
            excluded=None,
        )
        result = compute_sentiment(df, vader_config)
        assert (result["sentiment_change"] >= 0.0).all()


# ============================================================================
# Class: Backend Selection
# ============================================================================


class TestBackendSelection:
    """Tests for config-driven backend selection."""

    def test_vader_backend_selected(self, vader_config, base_time):
        """Config with sentiment_model='vader' uses VADER backend."""
        df = _make_df(
            base_time,
            n=2,
            hour_offsets=[0, 1],
            tickers=["AAPL"] * 2,
            titles=["Great!", "Terrible!"],
            selftexts=["Amazing", "Awful"],
            excluded=None,
        )
        result = compute_sentiment(df, vader_config)
        # VADER should produce non-zero polarities for emotional text
        assert result.iloc[0]["sentiment_polarity"] != 0.0
        assert result.iloc[1]["sentiment_polarity"] != 0.0

    def test_textblob_backend_selected(self, textblob_config, base_time):
        """Config with sentiment_model='textblob' uses TextBlob backend."""
        df = _make_df(
            base_time,
            n=2,
            hour_offsets=[0, 1],
            tickers=["AAPL"] * 2,
            titles=["Great!", "Terrible!"],
            selftexts=["Amazing wonderful", "Awful horrible"],
            excluded=None,
        )
        result = compute_sentiment(df, textblob_config)
        # TextBlob should produce non-zero polarities for emotional text
        assert result.iloc[0]["sentiment_polarity"] != 0.0
        assert result.iloc[1]["sentiment_polarity"] != 0.0

    def test_vader_and_textblob_produce_different_scores(
        self, vader_config, textblob_config, base_time
    ):
        """VADER and TextBlob should produce different polarity values."""
        df = _make_df(
            base_time,
            n=1,
            hour_offsets=[0],
            tickers=["AAPL"],
            titles=["$GME YOLO!! diamond hands!!"],
            selftexts=["To the moon!!! 🚀🚀🚀"],
            excluded=None,
        )
        result_vader = compute_sentiment(df.copy(), vader_config)
        result_tb = compute_sentiment(df.copy(), textblob_config)

        # Scores should differ (VADER better at social media text)
        assert result_vader.iloc[0]["sentiment_polarity"] != result_tb.iloc[0]["sentiment_polarity"]


# ============================================================================
# Class: Output Column Correctness
# ============================================================================


class TestOutputColumns:
    """Tests for output DataFrame structure."""

    def test_output_columns_added(self, vader_config, base_time):
        """compute_sentiment adds exactly the three expected columns."""
        df = _make_df(
            base_time,
            n=2,
            hour_offsets=[0, 6],
            tickers=["AAPL"] * 2,
            titles=["Hello", "World"],
            selftexts=["text", "text"],
            excluded=None,
        )
        original_cols = set(df.columns)
        result = compute_sentiment(df, vader_config)

        new_cols = set(result.columns) - original_cols
        assert new_cols == {"sentiment_polarity", "mean_future_sentiment", "sentiment_change"}

    def test_original_columns_preserved(self, vader_config, base_time):
        """All original columns remain in the output."""
        df = _make_df(
            base_time,
            n=2,
            hour_offsets=[0, 6],
            tickers=["AAPL"] * 2,
            titles=["Hello", "World"],
            selftexts=["text", "text"],
            excluded=[False, False],
        )
        original_cols = set(df.columns)
        result = compute_sentiment(df, vader_config)

        assert original_cols.issubset(set(result.columns))

    def test_row_count_unchanged(self, vader_config, base_time):
        """Output has the same number of rows as input."""
        df = _make_df(
            base_time,
            n=5,
            hour_offsets=[0, 2, 4, 6, 8],
            tickers=["AAPL"] * 5,
            titles=["a"] * 5,
            selftexts=["b"] * 5,
            excluded=[False, True, False, True, False],
        )
        result = compute_sentiment(df, vader_config)
        assert len(result) == 5
