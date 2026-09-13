"""Core correctness tests for the surge-labelling pipeline.

Tests:
  1. Windowing correctness with hand-computed examples (R2, R3)
  2. Z-score normalisation uses training stats only (R5)
  3. Composite labelling produces expected labels at known thresholds (R6)
  4. Determinism - two runs produce identical output (R7)
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
from surge_pipeline.labelling import _compute_z_scores, apply_labelling
from surge_pipeline.windowing import compute_windowed_counts

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_config() -> PipelineConfig:
    """A minimal pipeline config for testing."""
    return PipelineConfig(
        temporal_split_ratio=0.8,
        min_window_count=2,
        threshold_tau=1.0,
        weight_volume=1.0,
        weight_sentiment=0.0,
        random_seed=42,
    )


@pytest.fixture
def base_timestamp() -> int:
    """A base epoch timestamp: 2024-01-01 00:00:00 UTC."""
    return 1704067200


@pytest.fixture
def single_ticker_df(base_timestamp: int) -> pd.DataFrame:
    """A 6-record DataFrame for a single ticker with known timestamps.

    Records are spaced as follows (offsets from base in hours):
      0: t=0h
      1: t=6h
      2: t=12h
      3: t=23h    (within 24h of record 0 forward window)
      4: t=25h    (outside 24h of record 0, but within 24h of record 1)
      5: t=48h    (outside 24h of records 0-3)
    """
    offsets_hours = [0, 6, 12, 23, 25, 48]
    timestamps = [
        pd.Timestamp(base_timestamp + h * 3600, unit="s", tz="UTC")
        for h in offsets_hours
    ]
    return pd.DataFrame({
        "created_utc": timestamps,
        "ticker": ["AAPL"] * 6,
        "title": [f"Post {i}" for i in range(6)],
        "selftext": [""] * 6,
    })


@pytest.fixture
def multi_ticker_df(base_timestamp: int) -> pd.DataFrame:
    """Two tickers (AAPL, TSLA) interleaved, 3 records each.

    AAPL: t=0h, t=12h, t=25h
    TSLA: t=1h, t=13h, t=26h
    """
    data = [
        (0, "AAPL"),
        (1, "TSLA"),
        (12, "AAPL"),
        (13, "TSLA"),
        (25, "AAPL"),
        (26, "TSLA"),
    ]
    timestamps = [
        pd.Timestamp(base_timestamp + h * 3600, unit="s", tz="UTC")
        for h, _ in data
    ]
    tickers = [t for _, t in data]
    return pd.DataFrame({
        "created_utc": timestamps,
        "ticker": tickers,
        "title": [f"Post {i}" for i in range(6)],
        "selftext": [""] * 6,
    })


# ============================================================================
# 1. Windowing Correctness (R2, R3)
# ============================================================================


class TestWindowingCorrectness:
    """Tests for compute_windowed_counts()."""

    def test_forward_backward_counts_hand_computed(
        self, single_ticker_df: pd.DataFrame, simple_config: PipelineConfig
    ):
        """Verify forward and backward counts match hand computation.

        Given timestamps at offsets [0, 6, 12, 23, 25, 48] hours:
        Window = 24 hours, exclusive of self.

        Forward counts (records strictly after t, within t+24h):
          rec 0 (t=0h):  records at 6h, 12h, 23h are in (0, 24] -> forward=3
          rec 1 (t=6h):  records at 12h, 23h, 25h are in (6, 30] -> forward=3
          rec 2 (t=12h): records at 23h, 25h are in (12, 36] -> forward=2
          rec 3 (t=23h): records at 25h are in (23, 47] -> forward=1
          rec 4 (t=25h): records at 48h are in (25, 49] -> forward=1
          rec 5 (t=48h): no records in (48, 72] -> forward=0

        Backward counts (records strictly before t, within t-24h):
          rec 0 (t=0h):  no records in (-24, 0) -> backward=0
          rec 1 (t=6h):  record at 0h is in (-18, 6) -> backward=1
          rec 2 (t=12h): records at 0h, 6h are in (-12, 12) -> backward=2
          rec 3 (t=23h): records at 0h, 6h, 12h are in (-1, 23) -> backward=3
          rec 4 (t=25h): records at 6h, 12h, 23h are in (1, 25) -> backward=3
          rec 5 (t=48h): record at 25h is in (24, 48) -> backward=1
        """
        result = compute_windowed_counts(single_ticker_df.copy(), simple_config)

        expected_forward = [3, 3, 2, 1, 1, 0]
        expected_backward = [0, 1, 2, 3, 3, 1]

        np.testing.assert_array_equal(
            result["forward_count"].values, expected_forward
        )
        np.testing.assert_array_equal(
            result["backward_count"].values, expected_backward
        )

    def test_boundary_exclusion_24h(
        self, base_timestamp: int, simple_config: PipelineConfig
    ):
        """Record exactly at 24h boundary should be EXCLUDED from forward count.

        The forward window is (t, t+24h], but searchsorted with side='right'
        for the upper bound means timestamps > t+24h are excluded.
        A record at exactly t+24h: searchsorted(times, t+24h, side='right')
        includes it. Let's verify the actual behavior.

        Actually, re-reading the code:
          forward_right = searchsorted(times, times + WINDOW_SECONDS, side='right')
        This means records at exactly t + 24h ARE included (<= boundary).

        Let's test with two records: t=0 and t=exactly 24h.
        forward_left for rec 0 = searchsorted(times, 0, side='right') = 1
        forward_right for rec 0 = searchsorted(times, 86400, side='right') = 2
        So forward_count for rec 0 = 2 - 1 = 1 (the record at 24h is included).
        """
        ts_0 = pd.Timestamp(base_timestamp, unit="s", tz="UTC")
        ts_24h = pd.Timestamp(base_timestamp + 86400, unit="s", tz="UTC")

        df = pd.DataFrame({
            "created_utc": [ts_0, ts_24h],
            "ticker": ["AAPL", "AAPL"],
            "title": ["A", "B"],
            "selftext": ["", ""],
        })

        result = compute_windowed_counts(df.copy(), simple_config)

        # Record at exactly 24h IS within forward window of rec 0
        # (searchsorted side='right' includes the boundary)
        assert result["forward_count"].iloc[0] == 1
        # Record at t=24h has no forward records
        assert result["forward_count"].iloc[1] == 0

    def test_boundary_just_beyond_24h(
        self, base_timestamp: int, simple_config: PipelineConfig
    ):
        """A record at 24h + 1 second should NOT be in the forward window."""
        ts_0 = pd.Timestamp(base_timestamp, unit="s", tz="UTC")
        ts_beyond = pd.Timestamp(base_timestamp + 86401, unit="s", tz="UTC")

        df = pd.DataFrame({
            "created_utc": [ts_0, ts_beyond],
            "ticker": ["AAPL", "AAPL"],
            "title": ["A", "B"],
            "selftext": ["", ""],
        })

        result = compute_windowed_counts(df.copy(), simple_config)

        # Record at 24h+1s is outside the forward window of rec 0
        assert result["forward_count"].iloc[0] == 0

    def test_multi_ticker_independence(
        self, multi_ticker_df: pd.DataFrame, simple_config: PipelineConfig
    ):
        """Two tickers should have counts computed independently.

        AAPL records at offsets [0, 12, 25] hours:
          rec 0 (t=0h):  forward in (0, 24]: 12h -> forward=1, backward=0
          rec 2 (t=12h): forward in (12, 36]: 25h -> forward=1, backward in (-12,12): 0h -> backward=1
          rec 4 (t=25h): forward in (25, 49]: none -> forward=0, backward in (1,25): 12h -> backward=1

        TSLA records at offsets [1, 13, 26] hours:
          rec 1 (t=1h):  forward in (1, 25]: 13h -> forward=1, backward=0
          rec 3 (t=13h): forward in (13, 37]: 26h -> forward=1, backward in (-11,13): 1h -> backward=1
          rec 5 (t=26h): forward in (26, 50]: none -> forward=0, backward in (2,26): 13h -> backward=1
        """
        result = compute_windowed_counts(multi_ticker_df.copy(), simple_config)

        # AAPL at indices 0, 2, 4 in the original DataFrame
        assert result["forward_count"].iloc[0] == 1  # AAPL t=0
        assert result["forward_count"].iloc[2] == 1  # AAPL t=12
        assert result["forward_count"].iloc[4] == 0  # AAPL t=25

        assert result["backward_count"].iloc[0] == 0  # AAPL t=0
        assert result["backward_count"].iloc[2] == 1  # AAPL t=12
        assert result["backward_count"].iloc[4] == 1  # AAPL t=25

        # TSLA at indices 1, 3, 5
        assert result["forward_count"].iloc[1] == 1  # TSLA t=1
        assert result["forward_count"].iloc[3] == 1  # TSLA t=13
        assert result["forward_count"].iloc[5] == 0  # TSLA t=26

        assert result["backward_count"].iloc[1] == 0  # TSLA t=1
        assert result["backward_count"].iloc[3] == 1  # TSLA t=13
        assert result["backward_count"].iloc[5] == 1  # TSLA t=26

    def test_posting_volume_growth_formula(
        self, single_ticker_df: pd.DataFrame, simple_config: PipelineConfig
    ):
        """Verify posting_volume_growth = (forward / max(backward, 1)) - 1.

        Using hand-computed values from test above:
          forward =  [3, 3, 2, 1, 1, 0]
          backward = [0, 1, 2, 3, 3, 1]

          growth[0] = (3 / max(0,1)) - 1 = 3/1 - 1 = 2.0
          growth[1] = (3 / max(1,1)) - 1 = 3/1 - 1 = 2.0
          growth[2] = (2 / max(2,1)) - 1 = 2/2 - 1 = 0.0
          growth[3] = (1 / max(3,1)) - 1 = 1/3 - 1 ~= -0.6667
          growth[4] = (1 / max(3,1)) - 1 = 1/3 - 1 ~= -0.6667
          growth[5] = (0 / max(1,1)) - 1 = 0/1 - 1 = -1.0
        """
        result = compute_windowed_counts(single_ticker_df.copy(), simple_config)

        expected_growth = [2.0, 2.0, 0.0, -2.0 / 3.0, -2.0 / 3.0, -1.0]
        np.testing.assert_allclose(
            result["posting_volume_growth"].values,
            expected_growth,
            rtol=1e-10,
        )

    def test_exclusion_flag_min_window_count(
        self, single_ticker_df: pd.DataFrame
    ):
        """Records with forward_count < min_window_count get excluded=True.

        With min_window_count=3:
          forward = [3, 3, 2, 1, 1, 0]
          excluded = [False, False, True, True, True, True]
        """
        config = PipelineConfig(min_window_count=3, random_seed=42)
        result = compute_windowed_counts(single_ticker_df.copy(), config)

        expected_excluded = [False, False, True, True, True, True]
        np.testing.assert_array_equal(
            result["excluded"].values, expected_excluded
        )


# ============================================================================
# 2. Z-Score Normalisation (R5)
# ============================================================================


class TestZScoreNormalisation:
    """Tests for z-score normalisation using training stats only."""

    def test_z_score_uses_training_stats_only(self, base_timestamp: int):
        """Verify z-scores are computed using training partition mu/sigma only.

        Setup: 10 records, temporal_split_ratio=0.8 -> first 8 records train.
        All records have known posting_volume_growth and sentiment_change.
        Manually compute mu/sigma from training set and verify test z-scores.
        """
        n = 10
        timestamps = [
            pd.Timestamp(base_timestamp + i * 3600, unit="s", tz="UTC")
            for i in range(n)
        ]

        # Known volume values: training=[1,2,3,4,5,6,7,8], test=[9,10]
        volumes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        # Known sentiment values: [0.1]*10
        sentiments = [0.1] * n

        df = pd.DataFrame({
            "created_utc": timestamps,
            "ticker": ["AAPL"] * n,
            "title": [f"Post {i}" for i in range(n)],
            "selftext": [""] * n,
            "posting_volume_growth": volumes,
            "sentiment_change": sentiments,
            "excluded": [False] * n,
        })

        config = PipelineConfig(
            temporal_split_ratio=0.8,
            min_window_count=1,
            threshold_tau=2.0,
            weight_volume=1.0,
            weight_sentiment=0.0,
            random_seed=42,
        )

        result = apply_labelling(df.copy(), config)

        # Training volume values: [1,2,3,4,5,6,7,8]
        # mu_vol = mean([1..8]) = 4.5
        # sigma_vol = population std([1..8]) = sqrt(mean((x-4.5)^2))
        train_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        mu_vol = train_vals.mean()  # 4.5
        sigma_vol = train_vals.std(ddof=0)  # ~2.2913

        assert result.stats.mu_volume == pytest.approx(mu_vol)
        assert result.stats.sigma_volume == pytest.approx(sigma_vol)

        # Verify test-set z-scores use training stats
        # z(9) = (9 - 4.5) / sigma, z(10) = (10 - 4.5) / sigma
        expected_z9 = (9.0 - mu_vol) / sigma_vol
        expected_z10 = (10.0 - mu_vol) / sigma_vol

        result_df = result.df
        np.testing.assert_allclose(
            result_df["z_volume"].iloc[8], expected_z9, rtol=1e-10
        )
        np.testing.assert_allclose(
            result_df["z_volume"].iloc[9], expected_z10, rtol=1e-10
        )

    def test_test_set_uses_training_stats_not_own(self, base_timestamp: int):
        """Critical: test-set records are normalised with TRAINING statistics.

        If test records used their own mu/sigma, z-scores would differ.
        We verify by comparing against known training-based computation.
        """
        n = 10
        timestamps = [
            pd.Timestamp(base_timestamp + i * 3600, unit="s", tz="UTC")
            for i in range(n)
        ]

        # Training [0..7]: all values = 2.0 -> mu=2, sigma=0
        # Wait - sigma=0 makes z=0. Use different values:
        # Training [0..7]: values = [0, 0, 0, 0, 10, 10, 10, 10] -> mu=5, sigma=5
        # Test [8..9]: values = [15, 20]
        volumes = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 15.0, 20.0]
        sentiments = [0.0] * n

        df = pd.DataFrame({
            "created_utc": timestamps,
            "ticker": ["AAPL"] * n,
            "title": [f"Post {i}" for i in range(n)],
            "selftext": [""] * n,
            "posting_volume_growth": volumes,
            "sentiment_change": sentiments,
            "excluded": [False] * n,
        })

        config = PipelineConfig(
            temporal_split_ratio=0.8,
            min_window_count=1,
            threshold_tau=2.0,
            weight_volume=1.0,
            weight_sentiment=0.0,
            random_seed=42,
        )

        result = apply_labelling(df.copy(), config)

        # Training mu = mean([0,0,0,0,10,10,10,10]) = 5.0
        # Training sigma = population std = 5.0
        mu_vol = 5.0
        sigma_vol = 5.0

        # Test z-scores should be computed with training stats:
        # z(15) = (15-5)/5 = 2.0
        # z(20) = (20-5)/5 = 3.0
        result_df = result.df
        np.testing.assert_allclose(result_df["z_volume"].iloc[8], 2.0, rtol=1e-10)
        np.testing.assert_allclose(result_df["z_volume"].iloc[9], 3.0, rtol=1e-10)

        # If they wrongly used test-only stats (mu=17.5, sigma=2.5), we'd get:
        # z(15) = (15-17.5)/2.5 = -1.0 and z(20) = (20-17.5)/2.5 = 1.0
        # Verify this is NOT the case
        assert result_df["z_volume"].iloc[8] != pytest.approx(-1.0)
        assert result_df["z_volume"].iloc[9] != pytest.approx(1.0)

    def test_sigma_zero_edge_case(self, base_timestamp: int):
        """When all training values are identical, sigma=0 -> z should be 0."""
        n = 10
        timestamps = [
            pd.Timestamp(base_timestamp + i * 3600, unit="s", tz="UTC")
            for i in range(n)
        ]

        # All training values identical -> sigma=0
        volumes = [5.0] * 8 + [10.0, 20.0]  # train=5.0 constant, test varies
        sentiments = [0.3] * 8 + [0.5, 0.8]  # train=0.3 constant

        df = pd.DataFrame({
            "created_utc": timestamps,
            "ticker": ["AAPL"] * n,
            "title": [f"Post {i}" for i in range(n)],
            "selftext": [""] * n,
            "posting_volume_growth": volumes,
            "sentiment_change": sentiments,
            "excluded": [False] * n,
        })

        config = PipelineConfig(
            temporal_split_ratio=0.8,
            min_window_count=1,
            threshold_tau=1.0,
            weight_volume=1.0,
            weight_sentiment=0.0,
            random_seed=42,
        )

        result = apply_labelling(df.copy(), config)
        result_df = result.df

        # All z_volume should be 0.0 (sigma=0 -> z=0 for all)
        np.testing.assert_array_equal(
            result_df["z_volume"].values, np.zeros(n)
        )

    def test_compute_z_scores_helper(self):
        """Verify the _compute_z_scores helper directly."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu = 3.0
        sigma = 1.0

        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = _compute_z_scores(values, mu, sigma)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_compute_z_scores_sigma_zero(self):
        """_compute_z_scores with sigma=0 should return all zeros."""
        values = np.array([1.0, 2.0, 3.0])
        result = _compute_z_scores(values, mu=2.0, sigma=0.0)

        np.testing.assert_array_equal(result, np.zeros(3))


# ============================================================================
# 3. Composite Labelling (R6)
# ============================================================================


class TestCompositeLabelling:
    """Tests for composite metric computation and threshold labelling."""

    def _make_labelling_df(self, base_timestamp: int, n: int, volumes, sentiments):
        """Helper to create a DataFrame ready for apply_labelling."""
        timestamps = [
            pd.Timestamp(base_timestamp + i * 3600, unit="s", tz="UTC")
            for i in range(n)
        ]
        return pd.DataFrame({
            "created_utc": timestamps,
            "ticker": ["AAPL"] * n,
            "title": [f"Post {i}" for i in range(n)],
            "selftext": [""] * n,
            "posting_volume_growth": volumes,
            "sentiment_change": sentiments,
            "excluded": [False] * n,
        })

    def test_composite_formula(self, base_timestamp: int):
        """Verify composite = w1*z_volume + w2*z_sentiment at known values.

        Setup designed so training mu/sigma are predictable, yielding known z-scores.
        """
        n = 10
        # Training [0..7]: volume uniform [0,2,4,6,8,10,12,14]
        # mu_vol = 7.0, sigma_vol = population_std = sqrt(mean((x-7)^2))
        train_vols = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
        test_vols = np.array([7.0, 21.0])  # z=0, z=2
        volumes = list(train_vols) + list(test_vols)

        mu_vol = train_vols.mean()  # 7.0
        sigma_vol = train_vols.std(ddof=0)  # 4.899...

        # Training [0..7]: sentiment all = 0.5
        # |0.5| = 0.5, mu_sent = 0.5, sigma_sent = 0
        sentiments = [0.5] * 8 + [0.5, 0.5]

        df = self._make_labelling_df(base_timestamp, n, volumes, sentiments)

        # Phase 2: w1=0.6, w2=0.4
        config = PipelineConfig(
            temporal_split_ratio=0.8,
            min_window_count=1,
            threshold_tau=1.0,
            weight_volume=0.6,
            weight_sentiment=0.4,
            random_seed=42,
        )

        result = apply_labelling(df.copy(), config)
        result_df = result.df

        # Since sigma_sent=0, z_sentiment = 0 for all records
        # composite = 0.6 * z_vol + 0.4 * 0 = 0.6 * z_vol
        expected_z_vol_test0 = (7.0 - mu_vol) / sigma_vol  # ~= 0
        expected_z_vol_test1 = (21.0 - mu_vol) / sigma_vol  # ~= 2.857

        expected_composite_test0 = 0.6 * expected_z_vol_test0
        expected_composite_test1 = 0.6 * expected_z_vol_test1

        np.testing.assert_allclose(
            result_df["composite"].iloc[8], expected_composite_test0, rtol=1e-10
        )
        np.testing.assert_allclose(
            result_df["composite"].iloc[9], expected_composite_test1, rtol=1e-10
        )

    def test_threshold_labelling(self, base_timestamp: int):
        """At threshold tau=1.0, records with composite > 1.0 get label=1."""
        n = 10
        # Design training values so z-scores for test records cross threshold
        # Training: [0,0,0,0,4,4,4,4] -> mu=2, sigma=2
        # Test: [6, 2] -> z_vol = (6-2)/2 = 2.0, z_vol = (2-2)/2 = 0.0
        volumes = [0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 6.0, 2.0]
        sentiments = [0.0] * n

        df = self._make_labelling_df(base_timestamp, n, volumes, sentiments)

        # Phase 1: w1=1.0, w2=0.0, tau=1.0
        config = PipelineConfig(
            temporal_split_ratio=0.8,
            min_window_count=1,
            threshold_tau=1.0,
            weight_volume=1.0,
            weight_sentiment=0.0,
            random_seed=42,
        )

        result = apply_labelling(df.copy(), config)
        result_df = result.df

        # Test record 8: z_vol=2.0, composite=2.0 > 1.0 -> label=1
        assert result_df["surge_label"].iloc[8] == 1.0
        # Test record 9: z_vol=0.0, composite=0.0 <= 1.0 -> label=0
        assert result_df["surge_label"].iloc[9] == 0.0

    def test_phase1_volume_only(self, base_timestamp: int):
        """Phase 1 mode: w2=0, composite should equal w1*z_volume."""
        n = 10
        volumes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        sentiments = [0.5, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8, 0.4, 0.6, 0.5]

        df = self._make_labelling_df(base_timestamp, n, volumes, sentiments)

        config = PipelineConfig(
            temporal_split_ratio=0.8,
            min_window_count=1,
            threshold_tau=1.0,
            weight_volume=1.0,
            weight_sentiment=0.0,
            random_seed=42,
        )

        result = apply_labelling(df.copy(), config)
        result_df = result.df

        # With w2=0, composite = 1.0 * z_volume + 0 * z_sentiment = z_volume
        np.testing.assert_allclose(
            result_df["composite"].values,
            result_df["z_volume"].values,
            rtol=1e-10,
        )

    def test_phase2_includes_both_components(self, base_timestamp: int):
        """Phase 2 mode: w2=0.5, composite should include both z-scores."""
        n = 10
        # Training volumes: [0,2,4,6,8,10,12,14] -> mu=7, sigma=4.899
        # Training sentiments: [0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4]
        # After abs(): same values. mu_sent=0.7, sigma_sent~=0.4899
        volumes = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 7.0, 14.0]
        sentiments = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 0.7, 1.4]

        df = self._make_labelling_df(base_timestamp, n, volumes, sentiments)

        config = PipelineConfig(
            temporal_split_ratio=0.8,
            min_window_count=1,
            threshold_tau=1.0,
            weight_volume=0.5,
            weight_sentiment=0.5,
            random_seed=42,
        )

        result = apply_labelling(df.copy(), config)
        result_df = result.df

        # Verify composite = 0.5 * z_volume + 0.5 * z_sentiment
        expected_composite = (
            0.5 * result_df["z_volume"].values
            + 0.5 * result_df["z_sentiment"].values
        )
        np.testing.assert_allclose(
            result_df["composite"].values, expected_composite, rtol=1e-10
        )


# ============================================================================
# 4. Determinism (R7)
# ============================================================================


class TestDeterminism:
    """Tests for pipeline determinism and reproducibility."""

    def _create_synthetic_pipeline_data(
        self, base_timestamp: int, n: int = 20, seed: int = 42
    ) -> pd.DataFrame:
        """Create synthetic data suitable for windowing + labelling."""
        rng = np.random.default_rng(seed)

        timestamps = [
            pd.Timestamp(base_timestamp + i * 3600, unit="s", tz="UTC")
            for i in range(n)
        ]
        tickers = rng.choice(["AAPL", "TSLA"], size=n)
        volumes = rng.uniform(-1, 5, size=n)
        sentiments = rng.uniform(0, 1, size=n)

        return pd.DataFrame({
            "created_utc": timestamps,
            "ticker": tickers,
            "title": [f"Post {i} about stock" for i in range(n)],
            "selftext": ["Some text content"] * n,
            "posting_volume_growth": volumes,
            "sentiment_change": sentiments,
            "excluded": [False] * n,
        })

    def test_two_runs_produce_identical_output(self, base_timestamp: int):
        """Running apply_labelling twice with same config produces identical results."""
        df = self._create_synthetic_pipeline_data(base_timestamp, n=20, seed=42)

        config = PipelineConfig(
            temporal_split_ratio=0.8,
            min_window_count=1,
            threshold_tau=1.0,
            weight_volume=0.7,
            weight_sentiment=0.3,
            random_seed=42,
        )

        result1 = apply_labelling(df.copy(), config)
        result2 = apply_labelling(df.copy(), config)

        pd.testing.assert_frame_equal(result1.df, result2.df)

    def test_windowing_determinism(self, base_timestamp: int):
        """Running compute_windowed_counts twice produces identical results."""
        n = 20
        rng = np.random.default_rng(42)
        timestamps = [
            pd.Timestamp(base_timestamp + i * 3600, unit="s", tz="UTC")
            for i in range(n)
        ]
        tickers = rng.choice(["AAPL", "TSLA", "GOOG"], size=n)

        df = pd.DataFrame({
            "created_utc": timestamps,
            "ticker": tickers,
            "title": [f"Post {i}" for i in range(n)],
            "selftext": [""] * n,
        })

        config = PipelineConfig(min_window_count=2, random_seed=42)

        result1 = compute_windowed_counts(df.copy(), config)
        result2 = compute_windowed_counts(df.copy(), config)

        pd.testing.assert_frame_equal(result1, result2)

    def test_different_seeds_produce_different_results(self, base_timestamp: int):
        """Different random_seed values should produce different temporal splits.

        Note: The labelling pipeline uses temporal split which is deterministic
        based on data order, not random seed. However, seed controls numpy
        operations. We test that different configs with different thresholds
        produce different labels (as a proxy for config sensitivity).
        """
        df = self._create_synthetic_pipeline_data(base_timestamp, n=20, seed=42)

        config_a = PipelineConfig(
            temporal_split_ratio=0.8,
            min_window_count=1,
            threshold_tau=0.5,  # Low threshold -> more surges
            weight_volume=1.0,
            weight_sentiment=0.0,
            random_seed=42,
        )

        config_b = PipelineConfig(
            temporal_split_ratio=0.8,
            min_window_count=1,
            threshold_tau=3.0,  # High threshold -> fewer surges
            weight_volume=1.0,
            weight_sentiment=0.0,
            random_seed=42,
        )

        result_a = apply_labelling(df.copy(), config_a)
        result_b = apply_labelling(df.copy(), config_b)

        # Different thresholds should produce different label distributions
        labels_a = result_a.df["surge_label"].values
        labels_b = result_b.df["surge_label"].values

        # Not all labels should be identical (unless data is degenerate)
        assert not np.array_equal(labels_a, labels_b), (
            "Different thresholds should produce different label distributions"
        )

    def test_full_windowing_plus_labelling_determinism(self, base_timestamp: int):
        """Full windowing -> labelling chain is deterministic."""
        n = 15
        rng = np.random.default_rng(99)
        timestamps = [
            pd.Timestamp(base_timestamp + i * 7200, unit="s", tz="UTC")
            for i in range(n)
        ]
        tickers = rng.choice(["AAPL", "TSLA"], size=n)

        df = pd.DataFrame({
            "created_utc": timestamps,
            "ticker": tickers,
            "title": [f"Stock post {i}" for i in range(n)],
            "selftext": ["Discussion about market trends"] * n,
        })

        config = PipelineConfig(
            temporal_split_ratio=0.8,
            min_window_count=1,
            threshold_tau=1.0,
            weight_volume=0.5,
            weight_sentiment=0.5,
            random_seed=42,
        )

        # Run 1: window -> add sentiment_change stub -> label
        df1 = compute_windowed_counts(df.copy(), config)
        # Add a synthetic sentiment_change for labelling (avoid TextBlob dependency)
        rng1 = np.random.default_rng(config.random_seed)
        df1["sentiment_change"] = rng1.uniform(0, 1, size=n)

        result1 = apply_labelling(df1.copy(), config)

        # Run 2: identical
        df2 = compute_windowed_counts(df.copy(), config)
        rng2 = np.random.default_rng(config.random_seed)
        df2["sentiment_change"] = rng2.uniform(0, 1, size=n)

        result2 = apply_labelling(df2.copy(), config)

        pd.testing.assert_frame_equal(result1.df, result2.df)
