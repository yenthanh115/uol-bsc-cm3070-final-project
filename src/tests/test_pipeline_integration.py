"""Integration tests for the pipeline orchestrator.

Validates:
  1. Full pipeline runs end-to-end without errors (synthetic data)
  2. Result structure contains expected keys
  3. Stage counts are consistent and monotonically tracked
  4. Determinism: two runs with same config produce identical results
  5. Threshold sweep produces valid output
  6. save_outputs creates expected files on disk
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surge_pipeline.config import PipelineConfig
from surge_pipeline.pipeline import run_pipeline, run_threshold_sweep, save_outputs

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_config(tmp_path) -> PipelineConfig:
    """Config using synthetic data with output to a temp directory."""
    return PipelineConfig(
        file_path="",  # triggers synthetic fallback
        output_dir=str(tmp_path / "output"),
        temporal_split_ratio=0.8,
        min_window_count=1,
        threshold_tau=1.5,
        weight_volume=0.5,
        weight_sentiment=0.5,
        random_seed=42,
        sentiment_model="vader",
        thresholds=[1.0, 1.5, 2.0],
    )


# ============================================================================
# Full pipeline execution
# ============================================================================


class TestFullPipeline:
    """End-to-end integration tests for run_pipeline."""

    def test_runs_without_error(self, synthetic_config):
        """Pipeline should complete without raising."""
        results = run_pipeline(synthetic_config)
        assert results is not None

    def test_result_keys(self, synthetic_config):
        """Results dict should contain all expected keys."""
        results = run_pipeline(synthetic_config)
        expected_keys = {
            "labelled_df",
            "stats",
            "class_distributions",
            "sweep_results",
            "stage_counts",
            "stage_durations",
            "total_duration_seconds",
        }
        assert expected_keys.issubset(set(results.keys()))

    def test_labelled_df_is_dataframe(self, synthetic_config):
        results = run_pipeline(synthetic_config)
        assert isinstance(results["labelled_df"], pd.DataFrame)
        assert len(results["labelled_df"]) > 0

    def test_labelled_df_has_required_columns(self, synthetic_config):
        results = run_pipeline(synthetic_config)
        df = results["labelled_df"]
        required = {"ticker", "created_utc", "forward_count", "backward_count",
                    "posting_volume_growth", "excluded", "surge_label",
                    "composite", "partition"}
        assert required.issubset(set(df.columns))

    def test_stage_counts_are_positive(self, synthetic_config):
        results = run_pipeline(synthetic_config)
        counts = results["stage_counts"]
        assert counts["after_load"] > 0
        assert counts["after_windowing"] > 0
        assert counts["after_sentiment"] > 0
        assert counts["after_labelling"] > 0

    def test_stage_counts_consistent(self, synthetic_config):
        """Record counts should not increase after windowing (no new rows)."""
        results = run_pipeline(synthetic_config)
        counts = results["stage_counts"]
        # After load = after windowing (windowing adds columns, not rows)
        assert counts["after_windowing"] == counts["after_load"]
        # After sentiment = after windowing (sentiment adds columns)
        assert counts["after_sentiment"] == counts["after_windowing"]

    def test_total_duration_positive(self, synthetic_config):
        results = run_pipeline(synthetic_config)
        assert results["total_duration_seconds"] > 0

    def test_class_distributions_present(self, synthetic_config):
        results = run_pipeline(synthetic_config)
        dist = results["class_distributions"]
        assert "all" in dist
        assert "surge_count" in dist["all"]
        assert "no_surge_count" in dist["all"]

    def test_sweep_results_present(self, synthetic_config):
        results = run_pipeline(synthetic_config)
        sweep = results["sweep_results"]
        assert isinstance(sweep, list)
        assert len(sweep) == len(synthetic_config.thresholds)


# ============================================================================
# Determinism
# ============================================================================


class TestDeterminism:
    """Two runs with the same seed should produce identical results."""

    def test_deterministic_labelled_df(self, synthetic_config):
        r1 = run_pipeline(synthetic_config)
        r2 = run_pipeline(synthetic_config)

        df1 = r1["labelled_df"].reset_index(drop=True)
        df2 = r2["labelled_df"].reset_index(drop=True)

        # Compare numeric columns
        numeric_cols = df1.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            np.testing.assert_array_almost_equal(
                df1[col].values, df2[col].values,
                err_msg=f"Column {col} differs between runs"
            )

    def test_deterministic_stage_counts(self, synthetic_config):
        r1 = run_pipeline(synthetic_config)
        r2 = run_pipeline(synthetic_config)
        assert r1["stage_counts"] == r2["stage_counts"]


# ============================================================================
# Threshold sweep
# ============================================================================


class TestThresholdSweep:
    """Tests for run_threshold_sweep."""

    def test_sweep_returns_dataframe(self, synthetic_config):
        result = run_threshold_sweep(synthetic_config)
        assert isinstance(result, pd.DataFrame)

    def test_sweep_has_expected_columns(self, synthetic_config):
        result = run_threshold_sweep(synthetic_config)
        expected = {"threshold", "surge_count", "no_surge_count",
                    "surge_rate", "imbalance_ratio", "viable"}
        assert expected.issubset(set(result.columns))

    def test_sweep_row_count_matches_thresholds(self, synthetic_config):
        result = run_threshold_sweep(synthetic_config)
        assert len(result) == len(synthetic_config.thresholds)

    def test_sweep_surge_rate_decreases_with_threshold(self, synthetic_config):
        """Higher thresholds should produce equal or fewer surges."""
        result = run_threshold_sweep(synthetic_config)
        surge_rates = result.sort_values("threshold")["surge_rate"].values
        # Each rate should be <= the previous (monotonically non-increasing)
        for i in range(1, len(surge_rates)):
            assert surge_rates[i] <= surge_rates[i - 1] + 0.01  # small tolerance


# ============================================================================
# save_outputs
# ============================================================================


class TestSaveOutputs:
    """Verify save_outputs creates expected files."""

    def test_creates_output_directory(self, synthetic_config):
        results = run_pipeline(synthetic_config)
        output_paths = save_outputs(results, synthetic_config)

        output_dir = Path(synthetic_config.output_dir)
        assert output_dir.exists()

    def test_creates_labelled_csv(self, synthetic_config):
        results = run_pipeline(synthetic_config)
        output_paths = save_outputs(results, synthetic_config)

        csv_path = Path(output_paths["labelled_dataset"])
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"

        # Should be loadable
        df = pd.read_csv(csv_path)
        assert len(df) > 0

    def test_creates_summary_json(self, synthetic_config):
        results = run_pipeline(synthetic_config)
        output_paths = save_outputs(results, synthetic_config)

        summary_path = Path(output_paths["pipeline_summary"])
        assert summary_path.exists()

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "normalisation_params" in data
        assert "class_distributions" in data
        assert "config" in data

    def test_creates_config_json(self, synthetic_config):
        results = run_pipeline(synthetic_config)
        output_paths = save_outputs(results, synthetic_config)

        config_path = Path(output_paths["pipeline_config"])
        assert config_path.exists()

        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["threshold_tau"] == synthetic_config.threshold_tau
        assert data["random_seed"] == synthetic_config.random_seed
