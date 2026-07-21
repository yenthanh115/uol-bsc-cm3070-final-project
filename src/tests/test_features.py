"""Tests for the feature engineering module (features.py).

Tests:
  1. No feature uses future information (backward-only computation)
  2. Feature matrix has expected shape (n_records × 11 features)
  3. No NaN values exist in non-excluded records
  4. ticker_post_acceleration with zero denominator (AC10)
  5. time_since_previous first occurrence (AC11)
  6. Determinism (two runs produce identical output)

Requirements: R11, R12
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the src directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surge_pipeline.features import (
    FEATURE_COLUMNS,
    compute_features,
    get_feature_matrix,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def base_timestamp() -> int:
    """A base epoch timestamp: 2024-01-01 00:00:00 UTC (Monday)."""
    return 1704067200


@pytest.fixture
def labelled_df(base_timestamp: int) -> pd.DataFrame:
    """A synthetic labelled dataset matching pipeline output schema.

    10 records, 2 tickers (AAPL × 6, TSLA × 4), chronologically ordered.
    Some records share the same 'id' (simulating multi-ticker explosion).
    """
    # Offsets from base in hours
    offsets_hours = [0, 1, 6, 12, 13, 18, 24, 25, 36, 48]
    timestamps = [
        pd.Timestamp(base_timestamp + h * 3600, unit="s", tz="UTC")
        for h in offsets_hours
    ]

    return pd.DataFrame({
        "id": ["post1", "post1", "post2", "post3", "post3", "post4",
               "post5", "post5", "post6", "post7"],
        "ticker": ["AAPL", "TSLA", "AAPL", "AAPL", "TSLA", "TSLA",
                   "AAPL", "TSLA", "AAPL", "AAPL"],
        "created_utc": timestamps,
        "title": [
            "Buy AAPL now",
            "TSLA is great",
            "Another AAPL post",
            "AAPL and TSLA discussion",
            "TSLA continues",
            "TSLA update today",
            "AAPL earnings next week",
            "TSLA volume high",
            "AAPL short squeeze",
            "AAPL all time high",
        ],
        "selftext": [
            "Great stock to buy",
            "Tesla is doing well",
            "",
            "Both are great",
            "Tesla momentum",
            "Update on Tesla",
            "Earnings report soon",
            "Volume is increasing",
            "Short squeeze incoming",
            "New highs reached",
        ],
        "backward_count": [0, 0, 1, 2, 1, 2, 3, 3, 2, 1],
        "excluded": [False, False, False, False, False, False,
                     False, False, False, False],
        "sentiment_polarity": [0.5, 0.8, 0.0, 0.3, 0.6, 0.1, -0.2, 0.4, 0.7, 0.9],
        "score": [10, 20, 5, 15, 8, 3, 25, 12, 7, 30],
        "num_comments": [5, 10, 2, 8, 4, 1, 12, 6, 3, 15],
    })


@pytest.fixture
def labelled_df_with_exclusions(labelled_df: pd.DataFrame) -> pd.DataFrame:
    """Labelled dataset with some excluded records."""
    df = labelled_df.copy()
    df.loc[0, "excluded"] = True
    df.loc[4, "excluded"] = True
    df.loc[9, "excluded"] = True
    return df


# ============================================================================
# 1. No Feature Uses Future Information (R12)
# ============================================================================


class TestNoFutureLeakage:
    """Verify all features use only backward-looking or creation-time info."""

    def test_features_exclude_score_and_num_comments(
        self, labelled_df: pd.DataFrame
    ):
        """Feature columns must NOT include score or num_comments (R11-AC9)."""
        result = compute_features(labelled_df.copy())

        assert "score" not in FEATURE_COLUMNS
        assert "num_comments" not in FEATURE_COLUMNS

        # The feature matrix should not contain these columns
        feature_matrix = get_feature_matrix(result)
        assert "score" not in feature_matrix.columns
        assert "num_comments" not in feature_matrix.columns

    def test_sentiment_score_uses_observation_time_polarity(
        self, labelled_df: pd.DataFrame
    ):
        """sentiment_score should equal sentiment_polarity (computed at t)."""
        result = compute_features(labelled_df.copy())

        np.testing.assert_array_equal(
            result["sentiment_score"].values,
            result["sentiment_polarity"].values,
        )

    def test_temporal_features_use_creation_time_only(
        self, labelled_df: pd.DataFrame, base_timestamp: int
    ):
        """hour_of_day and day_of_week should match created_utc."""
        result = compute_features(labelled_df.copy())

        created = pd.to_datetime(result["created_utc"], utc=True)
        expected_hours = created.dt.hour.values
        expected_days = created.dt.dayofweek.values

        np.testing.assert_array_equal(result["hour_of_day"].values, expected_hours)
        np.testing.assert_array_equal(result["day_of_week"].values, expected_days)

    def test_ticker_post_rate_uses_backward_count(
        self, labelled_df: pd.DataFrame
    ):
        """ticker_post_rate_24h should equal backward_count (already backward-only)."""
        result = compute_features(labelled_df.copy())

        np.testing.assert_array_equal(
            result["ticker_post_rate_24h"].values,
            labelled_df["backward_count"].values,
        )

    def test_time_since_previous_is_backward_only(
        self, base_timestamp: int
    ):
        """time_since_previous should only look at records before t."""
        # Create records: AAPL at t=0h, t=6h, t=12h
        timestamps = [
            pd.Timestamp(base_timestamp + h * 3600, unit="s", tz="UTC")
            for h in [0, 6, 12]
        ]
        df = pd.DataFrame({
            "id": ["p1", "p2", "p3"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "created_utc": timestamps,
            "title": ["A", "B", "C"],
            "selftext": ["", "", ""],
            "backward_count": [0, 1, 2],
            "excluded": [False, False, False],
            "sentiment_polarity": [0.1, 0.2, 0.3],
        })

        result = compute_features(df.copy())

        # Record 0: first occurrence → -1
        assert result["time_since_previous"].iloc[0] == -1.0
        # Record 1: 6 hours since record 0
        assert result["time_since_previous"].iloc[1] == pytest.approx(6.0)
        # Record 2: 6 hours since record 1 (not 12 hours since record 0)
        assert result["time_since_previous"].iloc[2] == pytest.approx(6.0)


# ============================================================================
# 2. Feature Matrix Shape (R11)
# ============================================================================


class TestFeatureMatrixShape:
    """Verify feature matrix has expected shape."""

    def test_feature_matrix_has_11_columns(self, labelled_df: pd.DataFrame):
        """Feature matrix should have exactly 11 feature columns (9 base + 2 interaction)."""
        result = compute_features(labelled_df.copy())
        feature_matrix = get_feature_matrix(result)

        assert feature_matrix.shape[1] == 11
        assert list(feature_matrix.columns) == FEATURE_COLUMNS

    def test_feature_matrix_preserves_row_count(self, labelled_df: pd.DataFrame):
        """Feature computation should not add or remove rows."""
        result = compute_features(labelled_df.copy())

        assert len(result) == len(labelled_df)

    def test_empty_dataframe(self):
        """Empty DataFrame should produce empty feature matrix."""
        df = pd.DataFrame(columns=[
            "id", "ticker", "created_utc", "title", "selftext",
            "backward_count", "excluded", "sentiment_polarity",
        ])
        result = compute_features(df)

        assert len(result) == 0
        for col in FEATURE_COLUMNS:
            assert col in result.columns


# ============================================================================
# 3. No NaN Values in Non-Excluded Records
# ============================================================================


class TestNoNaNValues:
    """Verify no NaN values in feature columns for non-excluded records."""

    def test_no_nan_in_non_excluded_records(self, labelled_df: pd.DataFrame):
        """All feature columns should have no NaN for non-excluded records."""
        result = compute_features(labelled_df.copy())

        included_mask = ~result["excluded"].astype(bool)
        feature_matrix = get_feature_matrix(result)
        included_features = feature_matrix.loc[included_mask]

        nan_counts = included_features.isna().sum()
        for col in FEATURE_COLUMNS:
            assert nan_counts[col] == 0, (
                f"Feature '{col}' has {nan_counts[col]} NaN values in "
                f"non-excluded records"
            )

    def test_no_nan_with_empty_selftext(self, base_timestamp: int):
        """Features should handle empty selftext without producing NaN."""
        timestamps = [
            pd.Timestamp(base_timestamp + h * 3600, unit="s", tz="UTC")
            for h in [0, 6]
        ]
        df = pd.DataFrame({
            "id": ["p1", "p2"],
            "ticker": ["AAPL", "AAPL"],
            "created_utc": timestamps,
            "title": ["Good stock", "Buy now"],
            "selftext": [np.nan, ""],
            "backward_count": [0, 1],
            "excluded": [False, False],
            "sentiment_polarity": [0.5, 0.3],
        })

        result = compute_features(df.copy())
        feature_matrix = get_feature_matrix(result)

        assert feature_matrix.isna().sum().sum() == 0


# ============================================================================
# 4. Ticker Post Acceleration — Zero Denominator (R11-AC10)
# ============================================================================


class TestTickerPostAcceleration:
    """Tests for ticker_post_acceleration computation."""

    def test_zero_denominator_returns_numerator(self, base_timestamp: int):
        """When no posts in (t-24h, t-12h], acceleration = numerator count.

        Setup: AAPL records at t=0h (first), t=6h (recent window only).
        For record at t=6h:
          - count in (t-12h, t]: record at 0h is in (-6h, 6h] → count=1
          - count in (t-24h, t-12h]: (-18h, -6h] → count=0
          - acceleration = 1 / max(0, 1) = 1.0 (AC10)
        """
        timestamps = [
            pd.Timestamp(base_timestamp + h * 3600, unit="s", tz="UTC")
            for h in [0, 6]
        ]
        df = pd.DataFrame({
            "id": ["p1", "p2"],
            "ticker": ["AAPL", "AAPL"],
            "created_utc": timestamps,
            "title": ["A", "B"],
            "selftext": ["", ""],
            "backward_count": [0, 1],
            "excluded": [False, False],
            "sentiment_polarity": [0.1, 0.2],
        })

        result = compute_features(df.copy())

        # Record 0: first post, no posts in either window → 0/max(0,1) = 0
        assert result["ticker_post_acceleration"].iloc[0] == pytest.approx(0.0)
        # Record 1: 1 recent post, 0 older posts → 1/max(0,1) = 1.0
        assert result["ticker_post_acceleration"].iloc[1] == pytest.approx(1.0)

    def test_normal_acceleration_ratio(self, base_timestamp: int):
        """Test normal case where both windows have posts.

        Setup: AAPL records at t=0h, t=6h, t=13h, t=20h.
        For record at t=20h:
          - count in (t-12h, t]: (8h, 20h] → record at 13h → count=1
          - count in (t-24h, t-12h]: (-4h, 8h] → records at 0h, 6h → count=2
          - acceleration = 1 / 2 = 0.5
        """
        timestamps = [
            pd.Timestamp(base_timestamp + h * 3600, unit="s", tz="UTC")
            for h in [0, 6, 13, 20]
        ]
        df = pd.DataFrame({
            "id": ["p1", "p2", "p3", "p4"],
            "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "created_utc": timestamps,
            "title": ["A", "B", "C", "D"],
            "selftext": ["", "", "", ""],
            "backward_count": [0, 1, 2, 3],
            "excluded": [False, False, False, False],
            "sentiment_polarity": [0.1, 0.2, 0.3, 0.4],
        })

        result = compute_features(df.copy())

        # Record at t=20h
        assert result["ticker_post_acceleration"].iloc[3] == pytest.approx(0.5)


# ============================================================================
# 5. Time Since Previous — First Occurrence (R11-AC11)
# ============================================================================


class TestTimeSincePrevious:
    """Tests for time_since_previous computation."""

    def test_first_occurrence_is_negative_one(self, base_timestamp: int):
        """First post for a ticker should have time_since_previous = -1."""
        timestamps = [
            pd.Timestamp(base_timestamp + h * 3600, unit="s", tz="UTC")
            for h in [0, 6, 12]
        ]
        df = pd.DataFrame({
            "id": ["p1", "p2", "p3"],
            "ticker": ["AAPL", "TSLA", "AAPL"],
            "created_utc": timestamps,
            "title": ["A", "B", "C"],
            "selftext": ["", "", ""],
            "backward_count": [0, 0, 1],
            "excluded": [False, False, False],
            "sentiment_polarity": [0.1, 0.2, 0.3],
        })

        result = compute_features(df.copy())

        # Record 0: first AAPL → -1
        assert result["time_since_previous"].iloc[0] == -1.0
        # Record 1: first TSLA → -1
        assert result["time_since_previous"].iloc[1] == -1.0
        # Record 2: second AAPL, 12h since record 0
        assert result["time_since_previous"].iloc[2] == pytest.approx(12.0)

    def test_multiple_same_ticker_sequential(self, base_timestamp: int):
        """Time since previous should reference the immediately preceding post."""
        timestamps = [
            pd.Timestamp(base_timestamp + h * 3600, unit="s", tz="UTC")
            for h in [0, 3, 10, 15]
        ]
        df = pd.DataFrame({
            "id": ["p1", "p2", "p3", "p4"],
            "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "created_utc": timestamps,
            "title": ["A", "B", "C", "D"],
            "selftext": ["", "", "", ""],
            "backward_count": [0, 1, 2, 3],
            "excluded": [False, False, False, False],
            "sentiment_polarity": [0.0, 0.0, 0.0, 0.0],
        })

        result = compute_features(df.copy())

        assert result["time_since_previous"].iloc[0] == -1.0
        assert result["time_since_previous"].iloc[1] == pytest.approx(3.0)
        assert result["time_since_previous"].iloc[2] == pytest.approx(7.0)
        assert result["time_since_previous"].iloc[3] == pytest.approx(5.0)


# ============================================================================
# 6. Determinism
# ============================================================================


class TestDeterminism:
    """Verify that feature computation is deterministic."""

    def test_two_runs_produce_identical_output(self, labelled_df: pd.DataFrame):
        """Running compute_features twice produces identical results."""
        result1 = compute_features(labelled_df.copy())
        result2 = compute_features(labelled_df.copy())

        pd.testing.assert_frame_equal(result1, result2)

    def test_feature_values_are_deterministic(self, labelled_df: pd.DataFrame):
        """All 11 feature columns are bit-identical across runs."""
        result1 = compute_features(labelled_df.copy())
        result2 = compute_features(labelled_df.copy())

        for col in FEATURE_COLUMNS:
            np.testing.assert_array_equal(
                result1[col].values,
                result2[col].values,
                err_msg=f"Feature '{col}' is not deterministic",
            )
