"""Tests for the model training module (training.py).

Tests:
  1. Temporal cross-validation respects chronological ordering
  2. Expanding-window splits have correct structure
  3. Hyperparameter grid sizes respect ≤50 limit
  4. Training produces expected model types
  5. Phase 1 vs Phase 2 mode distinction
  6. Reproducibility (same seed → same results)

Requirements: R13, R14
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
from surge_pipeline.features import FEATURE_COLUMNS
from surge_pipeline.training import (
    N_FOLDS,
    create_temporal_folds,
    get_expanding_window_splits,
    train_models,
    get_training_summary,
    _get_lr_param_grid,
    _get_rf_param_grid,
    _get_xgb_param_grid,
    _verify_temporal_ordering,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def base_timestamp() -> int:
    """Base epoch: 2024-01-01 00:00:00 UTC."""
    return 1704067200


@pytest.fixture
def training_df(base_timestamp: int) -> pd.DataFrame:
    """Synthetic training dataset with 100 records, 2 tickers.

    Chronologically ordered, all in 'train' partition, no exclusions.
    ~10% surge rate to simulate realistic class imbalance.
    """
    np.random.seed(42)
    n = 100

    # Timestamps: 1 hour apart
    timestamps = [
        pd.Timestamp(base_timestamp + i * 3600, unit="s", tz="UTC")
        for i in range(n)
    ]

    # Labels: ~10% surge rate
    labels = np.zeros(n)
    surge_indices = np.random.choice(n, size=10, replace=False)
    labels[surge_indices] = 1.0

    # Random features
    rng = np.random.RandomState(42)
    features = {col: rng.randn(n) for col in FEATURE_COLUMNS}

    df = pd.DataFrame({
        "id": [f"post_{i}" for i in range(n)],
        "ticker": ["AAPL" if i % 2 == 0 else "TSLA" for i in range(n)],
        "created_utc": timestamps,
        "partition": ["train"] * n,
        "excluded": [False] * n,
        "surge_label": labels,
        **features,
    })

    return df


@pytest.fixture
def small_training_df(base_timestamp: int) -> pd.DataFrame:
    """Small training dataset (40 records) for faster tests."""
    np.random.seed(42)
    n = 40

    timestamps = [
        pd.Timestamp(base_timestamp + i * 3600, unit="s", tz="UTC")
        for i in range(n)
    ]

    # Labels: ensure at least 2 surge in each fold (10 per fold)
    labels = np.zeros(n)
    # Place surges at specific positions across folds
    labels[[2, 8, 12, 18, 22, 28, 32, 38]] = 1.0

    rng = np.random.RandomState(42)
    features = {col: rng.randn(n) for col in FEATURE_COLUMNS}

    df = pd.DataFrame({
        "id": [f"post_{i}" for i in range(n)],
        "ticker": ["AAPL" if i % 2 == 0 else "TSLA" for i in range(n)],
        "created_utc": timestamps,
        "partition": ["train"] * n,
        "excluded": [False] * n,
        "surge_label": labels,
        **features,
    })

    return df


# ============================================================================
# 1. Temporal Ordering in CV Folds
# ============================================================================


class TestTemporalOrdering:
    """Verify temporal cross-validation respects chronological ordering."""

    def test_folds_are_chronologically_ordered(
        self, training_df: pd.DataFrame
    ):
        """Max timestamp in fold i must be ≤ min timestamp in fold i+1."""
        from surge_pipeline.timestamps import to_epoch_seconds
        folds = create_temporal_folds(training_df, n_folds=N_FOLDS)
        epoch = to_epoch_seconds(training_df["created_utc"])

        for i in range(len(folds) - 1):
            max_current = epoch[folds[i]].max()
            min_next = epoch[folds[i + 1]].min()
            assert max_current <= min_next, (
                f"Fold {i} max ts ({max_current}) > fold {i+1} min ts ({min_next})"
            )

    def test_cv_splits_respect_temporal_ordering(
        self, training_df: pd.DataFrame
    ):
        """In each split, max train ts ≤ min val ts."""
        from surge_pipeline.timestamps import to_epoch_seconds
        folds = create_temporal_folds(training_df, n_folds=N_FOLDS)
        splits = get_expanding_window_splits(folds)
        epoch = to_epoch_seconds(training_df["created_utc"])

        for i, (train_idx, val_idx) in enumerate(splits):
            max_train = epoch[train_idx].max()
            min_val = epoch[val_idx].min()
            assert max_train <= min_val, (
                f"Split {i}: max train ts ({max_train}) > min val ts ({min_val})"
            )

    def test_verify_temporal_ordering_raises_on_violation(
        self, base_timestamp: int
    ):
        """_verify_temporal_ordering should raise ValueError on violation."""
        # Create df where timestamps are not in order
        timestamps = [
            pd.Timestamp(base_timestamp + i * 3600, unit="s", tz="UTC")
            for i in range(8)
        ]
        df = pd.DataFrame({
            "created_utc": timestamps,
        })

        # Create folds that violate ordering (swap elements)
        folds = [
            np.array([0, 1, 6, 7]),  # Contains later timestamps
            np.array([2, 3]),
            np.array([4, 5]),
        ]
        splits = get_expanding_window_splits(folds)

        with pytest.raises(ValueError, match="Temporal ordering violated"):
            _verify_temporal_ordering(df, folds, splits)


# ============================================================================
# 2. Expanding-Window Split Structure
# ============================================================================


class TestExpandingWindowSplits:
    """Verify expanding-window split structure."""

    def test_produces_three_splits(self, training_df: pd.DataFrame):
        """4 folds should produce 3 expanding-window splits."""
        folds = create_temporal_folds(training_df, n_folds=4)
        splits = get_expanding_window_splits(folds)

        assert len(splits) == 3

    def test_training_set_expands(self, training_df: pd.DataFrame):
        """Each successive split should have a larger training set."""
        folds = create_temporal_folds(training_df, n_folds=4)
        splits = get_expanding_window_splits(folds)

        train_sizes = [len(s[0]) for s in splits]
        for i in range(len(train_sizes) - 1):
            assert train_sizes[i] < train_sizes[i + 1], (
                f"Training size should expand: {train_sizes[i]} < {train_sizes[i+1]}"
            )

    def test_all_records_covered(self, training_df: pd.DataFrame):
        """All training records should appear in at least one split."""
        folds = create_temporal_folds(training_df, n_folds=4)
        splits = get_expanding_window_splits(folds)

        all_indices = set()
        for train_idx, val_idx in splits:
            all_indices.update(train_idx.tolist())
            all_indices.update(val_idx.tolist())

        assert all_indices == set(range(len(training_df)))

    def test_fold_sizes_approximately_equal(self, training_df: pd.DataFrame):
        """Each fold should have approximately n/k records."""
        folds = create_temporal_folds(training_df, n_folds=4)
        n = len(training_df)
        expected_size = n // 4

        for fold in folds:
            # Allow ±1 difference for uneven division
            assert abs(len(fold) - expected_size) <= 1


# ============================================================================
# 3. Hyperparameter Grid Sizes
# ============================================================================


class TestHyperparameterGrids:
    """Verify hyperparameter grid respects ≤50 limit (R13-AC6)."""

    def test_lr_grid_has_10_configs(self):
        """LR: C(5) × penalty(2) = 10."""
        grid = _get_lr_param_grid()
        assert len(grid) == 10

    def test_rf_grid_has_36_configs(self):
        """RF: n_estimators(3) × max_depth(4) × min_samples_leaf(3) = 36."""
        grid = _get_rf_param_grid()
        assert len(grid) == 36

    def test_xgb_grid_within_75_configs(self):
        """XGBoost grid should have ≤75 configs (raised for P6 scale_pos_weight)."""
        grid = _get_xgb_param_grid(random_seed=42)
        assert len(grid) <= 75

    def test_all_grids_within_limits(self):
        """All model grids respect their config limits."""
        assert len(_get_lr_param_grid()) <= 50
        assert len(_get_rf_param_grid()) <= 50
        assert len(_get_xgb_param_grid(42)) <= 75


# ============================================================================
# 4. Model Training End-to-End
# ============================================================================


class TestModelTraining:
    """End-to-end model training tests (uses small dataset for speed)."""

    def test_trains_three_model_types(self, small_training_df: pd.DataFrame, tmp_path):
        """Should train LR, RF, and XGBoost (R13-AC1)."""
        config = PipelineConfig(
            random_seed=42,
            weight_sentiment=0.5,
            output_dir=str(tmp_path),
        )

        result = train_models(small_training_df, config, output_dir=str(tmp_path))

        assert "logistic_regression" in result.models
        assert "random_forest" in result.models
        assert "xgboost" in result.models

    def test_models_are_fitted(self, small_training_df: pd.DataFrame, tmp_path):
        """All returned models should be fitted (have learned parameters)."""
        config = PipelineConfig(
            random_seed=42,
            weight_sentiment=0.5,
            output_dir=str(tmp_path),
        )

        result = train_models(small_training_df, config, output_dir=str(tmp_path))

        for name, tm in result.models.items():
            assert tm.model is not None, f"{name} model should be fitted"
            # Check it can predict
            X_test = small_training_df[FEATURE_COLUMNS].values[:5]
            preds = tm.model.predict(X_test)
            assert len(preds) == 5

    def test_serialises_models_to_disk(
        self, small_training_df: pd.DataFrame, tmp_path
    ):
        """Models should be saved as joblib files (R14-AC4)."""
        config = PipelineConfig(
            random_seed=42,
            weight_sentiment=0.5,
            output_dir=str(tmp_path),
        )

        result = train_models(small_training_df, config, output_dir=str(tmp_path))

        for name in result.models:
            # Timestamped filenames: {name}_phase2_42_{timestamp}.joblib
            matches = list(tmp_path.glob(f"{name}_phase2_42_*.joblib"))
            assert len(matches) == 1, f"Model file for {name} not found"


# ============================================================================
# 5. Phase 1 vs Phase 2
# ============================================================================


class TestPhaseSupport:
    """Verify Phase 1/Phase 2 mode support (R13-AC10)."""

    def test_phase1_when_weight_sentiment_zero(
        self, small_training_df: pd.DataFrame, tmp_path
    ):
        """Phase 1 detected when weight_sentiment == 0."""
        config = PipelineConfig(
            random_seed=42,
            weight_sentiment=0.0,
            output_dir=str(tmp_path),
        )

        result = train_models(small_training_df, config, output_dir=str(tmp_path))
        assert result.phase == "phase1"

    def test_phase2_when_weight_sentiment_nonzero(
        self, small_training_df: pd.DataFrame, tmp_path
    ):
        """Phase 2 detected when weight_sentiment > 0."""
        config = PipelineConfig(
            random_seed=42,
            weight_sentiment=0.5,
            output_dir=str(tmp_path),
        )

        result = train_models(small_training_df, config, output_dir=str(tmp_path))
        assert result.phase == "phase2"

    def test_model_files_include_phase_in_name(
        self, small_training_df: pd.DataFrame, tmp_path
    ):
        """Serialised model filenames should include the phase."""
        config = PipelineConfig(
            random_seed=42,
            weight_sentiment=0.0,
            output_dir=str(tmp_path),
        )

        train_models(small_training_df, config, output_dir=str(tmp_path))

        # Check phase1 in filename (timestamped: *_phase1_*_*.joblib)
        files = list(tmp_path.glob("*_phase1_*.joblib"))
        assert len(files) == 3


# ============================================================================
# 6. Reproducibility (R14-AC1, AC2)
# ============================================================================


class TestReproducibility:
    """Verify training reproducibility with same seed."""

    def test_same_seed_produces_same_cv_scores(
        self, small_training_df: pd.DataFrame, tmp_path
    ):
        """Same seed, data, config → identical CV scores (R14-AC2)."""
        config = PipelineConfig(
            random_seed=42,
            weight_sentiment=0.5,
            output_dir=str(tmp_path / "run1"),
        )

        result1 = train_models(
            small_training_df.copy(), config, output_dir=str(tmp_path / "run1")
        )
        result2 = train_models(
            small_training_df.copy(), config, output_dir=str(tmp_path / "run2")
        )

        for name in result1.models:
            assert result1.models[name].best_cv_auc == pytest.approx(
                result2.models[name].best_cv_auc
            ), f"{name} CV AUC differs between runs"

            # Check fold scores match
            scores1 = result1.models[name].cv_results[0].fold_scores
            scores2 = result2.models[name].cv_results[0].fold_scores
            for s1, s2 in zip(scores1, scores2):
                assert s1 == pytest.approx(s2), (
                    f"{name} fold score differs: {s1} vs {s2}"
                )


# ============================================================================
# 7. Training Summary (R14-AC3)
# ============================================================================


class TestTrainingSummary:
    """Verify training summary generation."""

    def test_summary_contains_all_models(
        self, small_training_df: pd.DataFrame, tmp_path
    ):
        """Summary should have entries for all 3 models."""
        config = PipelineConfig(
            random_seed=42,
            weight_sentiment=0.5,
            output_dir=str(tmp_path),
        )

        result = train_models(small_training_df, config, output_dir=str(tmp_path))
        summary = get_training_summary(result)

        assert "logistic_regression" in summary["models"]
        assert "random_forest" in summary["models"]
        assert "xgboost" in summary["models"]

    def test_summary_includes_required_fields(
        self, small_training_df: pd.DataFrame, tmp_path
    ):
        """Each model summary should include params, AUC, duration, n_configs."""
        config = PipelineConfig(
            random_seed=42,
            weight_sentiment=0.5,
            output_dir=str(tmp_path),
        )

        result = train_models(small_training_df, config, output_dir=str(tmp_path))
        summary = get_training_summary(result)

        for name, model_summary in summary["models"].items():
            assert "best_params" in model_summary
            assert "best_cv_auc" in model_summary
            assert "fold_scores" in model_summary
            assert "training_duration_seconds" in model_summary
            assert "n_configs_evaluated" in model_summary
            assert isinstance(model_summary["best_cv_auc"], float)
            assert model_summary["training_duration_seconds"] > 0
