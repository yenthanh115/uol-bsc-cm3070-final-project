"""Tests for the temporal windowing module.

Validates:
  1. Forward/backward count correctness with hand-computed examples
  2. Boundary conditions (exact 24h edges)
  3. Self-exclusion (a record does not count itself)
  4. Multi-ticker isolation (tickers do not interfere)
  5. Empty DataFrame handling
  6. Posting volume growth calculation
  7. Exclusion flagging based on min_window_count
  8. backward_only surge method
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surge_pipeline.config import PipelineConfig
from surge_pipeline.windowing import _WINDOW_SECONDS, compute_windowed_counts

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def base_ts() -> int:
    """Base epoch timestamp: 2024-01-01 00:00:00 UTC."""
    return 1704067200


def _make_df(timestamps_epoch: list, tickers: list) -> pd.DataFrame:
    """Helper to create a DataFrame with UTC datetime and ticker columns."""
    df = pd.DataFrame({
        "created_utc": pd.to_datetime(timestamps_epoch, unit="s", utc=True),
        "ticker": tickers,
        "id": [f"rec_{i}" for i in range(len(timestamps_epoch))],
        "title": ["test"] * len(timestamps_epoch),
        "selftext": [""] * len(timestamps_epoch),
    })
    return df.sort_values("created_utc").reset_index(drop=True)


# ============================================================================
# Forward/backward count correctness
# ============================================================================


class TestForwardBackwardCounts:
    """Verify forward and backward counts with hand-computed examples."""

    def test_two_records_within_24h(self, base_ts):
        """Two records 12h apart: each sees the other."""
        ts = [base_ts, base_ts + 12 * 3600]  # 0h, 12h
        df = _make_df(ts, ["AAPL", "AAPL"])
        config = PipelineConfig(min_window_count=0, surge_method="forward_growth")

        result = compute_windowed_counts(df, config)

        # Record 0: backward=0 (nothing before), forward=1 (record 1)
        assert result.iloc[0]["backward_count"] == 0
        assert result.iloc[0]["forward_count"] == 1

        # Record 1: backward=1 (record 0), forward=0 (nothing after)
        assert result.iloc[1]["backward_count"] == 1
        assert result.iloc[1]["forward_count"] == 0

    def test_record_at_exact_24h_boundary_excluded(self, base_ts):
        """A record exactly 24h away should NOT be in the window.

        The backward window is (t - 24h, t) exclusive on both ends.
        A record at exactly t - 24h should not be counted.
        """
        # Record at t=0, record at t=24h exactly
        ts = [base_ts, base_ts + _WINDOW_SECONDS]
        df = _make_df(ts, ["AAPL", "AAPL"])
        config = PipelineConfig(min_window_count=0, surge_method="forward_growth")

        result = compute_windowed_counts(df, config)

        # Record 1 at t=24h: backward should NOT include record at t=0
        # because the window is (0h, 24h) and record 0 is at 0h (boundary)
        # searchsorted side='right' for backward_left means > (t-24h)
        # Record 0 is at exactly t-24h, so it IS included (side='right' means >)
        # Actually: backward_left = searchsorted(times, t-window, side='right')
        # t-window = 24h - 24h = 0h. side='right' gives first index > 0h.
        # Record 0 is at 0h, so searchsorted returns index 1 (past it).
        # backward_right = searchsorted(times, t=24h, side='left') = 1
        # So backward = 1 - 1 = 0. Record at exact boundary is EXCLUDED.
        assert result.iloc[1]["backward_count"] == 0

    def test_record_just_inside_24h_window(self, base_ts):
        """A record 23h59m apart should be within the window."""
        ts = [base_ts, base_ts + 23 * 3600 + 59 * 60]  # 0h, 23h59m
        df = _make_df(ts, ["AAPL", "AAPL"])
        config = PipelineConfig(min_window_count=0, surge_method="forward_growth")

        result = compute_windowed_counts(df, config)

        # Record 0: forward includes record 1 (23h59m < 24h)
        assert result.iloc[0]["forward_count"] == 1
        # Record 1: backward includes record 0
        assert result.iloc[1]["backward_count"] == 1

    def test_self_exclusion(self, base_ts):
        """A record should not count itself in forward or backward."""
        # Single record alone
        ts = [base_ts]
        df = _make_df(ts, ["AAPL"])
        config = PipelineConfig(min_window_count=0, surge_method="forward_growth")

        result = compute_windowed_counts(df, config)

        assert result.iloc[0]["backward_count"] == 0
        assert result.iloc[0]["forward_count"] == 0

    def test_multiple_records_in_window(self, base_ts):
        """Five records within 24h should all see each other."""
        # 5 records, each 4h apart (0, 4, 8, 12, 16h)
        ts = [base_ts + i * 4 * 3600 for i in range(5)]
        df = _make_df(ts, ["GME"] * 5)
        config = PipelineConfig(min_window_count=0, surge_method="forward_growth")

        result = compute_windowed_counts(df, config)

        # Record 0 (t=0h): forward should see records at 4,8,12,16h = 4
        assert result.iloc[0]["forward_count"] == 4
        # Record 0: backward = 0 (nothing before)
        assert result.iloc[0]["backward_count"] == 0

        # Record 4 (t=16h): backward should see records at 0,4,8,12h = 4
        assert result.iloc[4]["backward_count"] == 4
        # Record 4: forward = 0 (nothing after)
        assert result.iloc[4]["forward_count"] == 0


# ============================================================================
# Multi-ticker isolation
# ============================================================================


class TestMultiTickerIsolation:
    """Verify that different tickers don't interfere with each other."""

    def test_different_tickers_isolated(self, base_ts):
        """Records with different tickers should not affect each other."""
        ts = [base_ts, base_ts + 3600, base_ts + 2 * 3600]
        tickers = ["AAPL", "GME", "AAPL"]
        df = _make_df(ts, tickers)
        config = PipelineConfig(min_window_count=0, surge_method="forward_growth")

        result = compute_windowed_counts(df, config)

        # Record 0 (AAPL, t=0h): forward should only see record 2 (AAPL at 2h)
        assert result.iloc[0]["forward_count"] == 1
        # Record 1 (GME, t=1h): alone in its ticker group
        assert result.iloc[1]["forward_count"] == 0
        assert result.iloc[1]["backward_count"] == 0


# ============================================================================
# Empty DataFrame
# ============================================================================


class TestEmptyDataFrame:
    """Verify handling of empty input."""

    def test_empty_df_returns_empty_with_columns(self):
        df = pd.DataFrame(columns=["created_utc", "ticker", "id", "title", "selftext"])
        config = PipelineConfig(min_window_count=0, surge_method="forward_growth")

        result = compute_windowed_counts(df, config)

        assert len(result) == 0
        assert "forward_count" in result.columns
        assert "backward_count" in result.columns
        assert "posting_volume_growth" in result.columns
        assert "excluded" in result.columns


# ============================================================================
# Posting volume growth
# ============================================================================


class TestPostingVolumeGrowth:
    """Verify the surge metric calculation."""

    def test_growth_formula_forward_growth(self, base_ts):
        """growth = (forward / max(backward, 1)) - 1."""
        # 3 records: t=0, t=6h, t=12h (all within 24h of each other)
        ts = [base_ts, base_ts + 6 * 3600, base_ts + 12 * 3600]
        df = _make_df(ts, ["AAPL"] * 3)
        config = PipelineConfig(min_window_count=0, surge_method="forward_growth")

        result = compute_windowed_counts(df, config)

        # Record 0: forward=2, backward=0, growth = 2/max(0,1) - 1 = 1.0
        assert result.iloc[0]["posting_volume_growth"] == pytest.approx(1.0)

        # Record 1: forward=1, backward=1, growth = 1/1 - 1 = 0.0
        assert result.iloc[1]["posting_volume_growth"] == pytest.approx(0.0)

        # Record 2: forward=0, backward=2, growth = 0/2 - 1 = -1.0
        assert result.iloc[2]["posting_volume_growth"] == pytest.approx(-1.0)

    def test_zero_backward_uses_denominator_of_one(self, base_ts):
        """When backward=0, denominator should be 1 (not division by zero)."""
        ts = [base_ts, base_ts + 3600]
        df = _make_df(ts, ["AAPL"] * 2)
        config = PipelineConfig(min_window_count=0, surge_method="forward_growth")

        result = compute_windowed_counts(df, config)

        # Record 0: backward=0, forward=1, growth = 1/1 - 1 = 0.0
        growth = result.iloc[0]["posting_volume_growth"]
        assert np.isfinite(growth)


# ============================================================================
# Exclusion flagging
# ============================================================================


class TestExclusionFlagging:
    """Verify exclusion logic based on min_window_count."""

    def test_forward_growth_excludes_low_forward(self, base_ts):
        """Records with forward_count < min_window_count are excluded."""
        ts = [base_ts, base_ts + 3600, base_ts + 48 * 3600]
        df = _make_df(ts, ["AAPL"] * 3)
        config = PipelineConfig(min_window_count=2, surge_method="forward_growth")

        result = compute_windowed_counts(df, config)

        # Record 0: forward=1 (only record 1 within 24h), < 2 → excluded
        assert result.iloc[0]["excluded"] == True
        # Record 2: forward=0, < 2 → excluded
        assert result.iloc[2]["excluded"] == True

    def test_min_window_count_zero_excludes_nothing(self, base_ts):
        """min_window_count=0 should exclude nothing."""
        ts = [base_ts]
        df = _make_df(ts, ["AAPL"])
        config = PipelineConfig(min_window_count=0, surge_method="forward_growth")

        result = compute_windowed_counts(df, config)
        assert result.iloc[0]["excluded"] == False


# ============================================================================
# Backward-only surge method
# ============================================================================


class TestBackwardOnlyMethod:
    """Verify the backward_only surge method."""

    def test_backward_only_first_record_ratio_zero(self, base_ts):
        """First record for a ticker has no history → ratio = 0."""
        ts = [base_ts, base_ts + 3600, base_ts + 2 * 3600]
        df = _make_df(ts, ["AAPL"] * 3)
        config = PipelineConfig(min_window_count=0, surge_method="backward_only")

        result = compute_windowed_counts(df, config)

        assert result.iloc[0]["posting_volume_growth"] == pytest.approx(0.0)

    def test_backward_only_excludes_low_backward(self, base_ts):
        """Records with backward_count < min_window_count are excluded."""
        ts = [base_ts, base_ts + 3600, base_ts + 2 * 3600]
        df = _make_df(ts, ["AAPL"] * 3)
        config = PipelineConfig(min_window_count=2, surge_method="backward_only")

        result = compute_windowed_counts(df, config)

        # Record 0: backward=0, < 2 → excluded
        assert result.iloc[0]["excluded"] == True
        # Record 1: backward=1, < 2 → excluded
        assert result.iloc[1]["excluded"] == True
        # Record 2: backward=2, >= 2 → not excluded
        assert result.iloc[2]["excluded"] == False

    def test_backward_only_single_ticker_record(self, base_ts):
        """A ticker with only one record should have ratio 0."""
        ts = [base_ts]
        df = _make_df(ts, ["SOLO"])
        config = PipelineConfig(min_window_count=0, surge_method="backward_only")

        result = compute_windowed_counts(df, config)
        assert result.iloc[0]["posting_volume_growth"] == pytest.approx(0.0)
