"""Tests for evaluation visualisation figures.

Tests:
  1. plot_confusion_matrix produces PNG file with correct naming
  2. plot_roc_curve produces PNG with AUC annotation
  3. plot_classification_threshold_sensitivity produces PNG
  4. Edge case: single-class y_true handled gracefully
  5. Edge case: perfectly separable predictions (AUC=1.0)
  6. Edge case: random predictions (AUC~0.5)
  7. Edge case: extreme class imbalance (matches real dataset)
  8. generate_evaluation_figures convenience function produces all 3 files
  9. plot_roc_curve_combined overlays multiple models

Requirements: R16 (AC1, AC2, AC4, AC5, AC6)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the src directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surge_pipeline.evaluation import (
    generate_evaluation_figures,
    plot_classification_threshold_sensitivity,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_roc_curve_combined,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def binary_predictions():
    """Standard binary prediction arrays (balanced)."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.4, 0.8, 0.85, 0.9, 0.95])
    return y_true, y_pred, y_prob


@pytest.fixture
def imbalanced_predictions():
    """Imbalanced predictions matching real dataset (10 pos, 509 neg)."""
    rng = np.random.default_rng(42)
    n_neg, n_pos = 509, 10
    y_true = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    # Model gives slightly higher probabilities to positives
    y_prob = np.concatenate([
        rng.beta(2, 5, size=n_neg),  # negatives: lower probs
        rng.beta(5, 2, size=n_pos),  # positives: higher probs
    ])
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


@pytest.fixture
def perfect_predictions():
    """Perfectly separable predictions (AUC=1.0)."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.0, 0.1, 0.15, 0.2, 0.25, 0.75, 0.8, 0.85, 0.9, 1.0])
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


@pytest.fixture
def random_predictions():
    """Random predictions (AUC ~ 0.5)."""
    rng = np.random.default_rng(123)
    y_true = np.concatenate([np.zeros(50), np.ones(50)])
    y_prob = rng.uniform(0, 1, size=100)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


@pytest.fixture
def single_class_data():
    """Single-class y_true (only negatives)."""
    y_true = np.zeros(20)
    y_pred = np.zeros(20)
    y_prob = np.full(20, 0.3)
    return y_true, y_pred, y_prob


# ============================================================================
# Tests: plot_confusion_matrix
# ============================================================================


class TestPlotConfusionMatrix:
    """Tests for the confusion matrix heatmap figure."""

    def test_produces_png_file(self, binary_predictions, tmp_path):
        """Confusion matrix figure is saved as PNG."""
        y_true, y_pred, _ = binary_predictions
        path = plot_confusion_matrix(
            y_true, y_pred, model_name="TestModel", figures_dir=tmp_path
        )
        assert path.exists()
        assert path.suffix == ".png"
        assert "10_confusion_matrix_TestModel" in path.stem

    def test_file_has_content(self, binary_predictions, tmp_path):
        """Saved figure file is non-empty."""
        y_true, y_pred, _ = binary_predictions
        path = plot_confusion_matrix(
            y_true, y_pred, model_name="LR", figures_dir=tmp_path
        )
        assert path.stat().st_size > 1000  # PNG should be > 1KB

    def test_imbalanced_data(self, imbalanced_predictions, tmp_path):
        """Works with highly imbalanced class distribution."""
        y_true, y_pred, _ = imbalanced_predictions
        path = plot_confusion_matrix(
            y_true, y_pred, model_name="Imbalanced", figures_dir=tmp_path
        )
        assert path.exists()

    def test_all_zeros_predictions(self, tmp_path):
        """Works when model predicts all-negative."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.zeros(5)
        path = plot_confusion_matrix(
            y_true, y_pred, model_name="AllZero", figures_dir=tmp_path
        )
        assert path.exists()

    def test_all_ones_predictions(self, tmp_path):
        """Works when model predicts all-positive."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.ones(5)
        path = plot_confusion_matrix(
            y_true, y_pred, model_name="AllOne", figures_dir=tmp_path
        )
        assert path.exists()


# ============================================================================
# Tests: plot_roc_curve
# ============================================================================


class TestPlotRocCurve:
    """Tests for the ROC curve figure."""

    def test_produces_png_file(self, binary_predictions, tmp_path):
        """ROC curve figure is saved as PNG."""
        y_true, _, y_prob = binary_predictions
        path = plot_roc_curve(
            y_true, y_prob, model_name="TestModel", figures_dir=tmp_path
        )
        assert path.exists()
        assert path.suffix == ".png"
        assert "11_roc_curve_TestModel" in path.stem

    def test_perfect_separation(self, perfect_predictions, tmp_path):
        """ROC curve for perfectly separable data (AUC=1.0)."""
        y_true, _, y_prob = perfect_predictions
        path = plot_roc_curve(
            y_true, y_prob, model_name="Perfect", figures_dir=tmp_path
        )
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_random_predictions(self, random_predictions, tmp_path):
        """ROC curve for random predictions (AUC~0.5)."""
        y_true, _, y_prob = random_predictions
        path = plot_roc_curve(
            y_true, y_prob, model_name="Random", figures_dir=tmp_path
        )
        assert path.exists()

    def test_imbalanced_data(self, imbalanced_predictions, tmp_path):
        """ROC curve works with highly imbalanced data."""
        y_true, _, y_prob = imbalanced_predictions
        path = plot_roc_curve(
            y_true, y_prob, model_name="Imbalanced", figures_dir=tmp_path
        )
        assert path.exists()


# ============================================================================
# Tests: plot_roc_curve_combined
# ============================================================================


class TestPlotRocCurveCombined:
    """Tests for the combined ROC curve figure."""

    def test_multiple_models(self, binary_predictions, tmp_path):
        """Combined ROC overlays multiple models."""
        y_true, _, y_prob = binary_predictions
        rng = np.random.default_rng(99)
        y_prob_2 = rng.uniform(0, 1, size=len(y_true))

        results = [
            ("ModelA", y_true, y_prob),
            ("ModelB", y_true, y_prob_2),
        ]
        path = plot_roc_curve_combined(results, figures_dir=tmp_path)
        assert path.exists()
        assert "combined" in path.stem


# ============================================================================
# Tests: plot_classification_threshold_sensitivity
# ============================================================================


class TestPlotClassificationThresholdSensitivity:
    """Tests for the classification threshold sensitivity figure."""

    def test_produces_png_file(self, binary_predictions, tmp_path):
        """Threshold sensitivity figure is saved as PNG."""
        y_true, _, y_prob = binary_predictions
        path = plot_classification_threshold_sensitivity(
            y_true, y_prob, model_name="TestModel", figures_dir=tmp_path
        )
        assert path.exists()
        assert path.suffix == ".png"
        assert "12_classification_threshold_sensitivity_TestModel" in path.stem

    def test_imbalanced_data(self, imbalanced_predictions, tmp_path):
        """Works with highly imbalanced class distribution."""
        y_true, _, y_prob = imbalanced_predictions
        path = plot_classification_threshold_sensitivity(
            y_true, y_prob, model_name="Imbalanced", figures_dir=tmp_path
        )
        assert path.exists()

    def test_perfect_predictions(self, perfect_predictions, tmp_path):
        """Threshold sensitivity with perfectly separable data."""
        y_true, _, y_prob = perfect_predictions
        path = plot_classification_threshold_sensitivity(
            y_true, y_prob, model_name="Perfect", figures_dir=tmp_path
        )
        assert path.exists()

    def test_threshold_boundaries(self, binary_predictions, tmp_path):
        """At threshold=0 recall should be 1.0, at threshold=1 recall=0."""
        y_true, _, y_prob = binary_predictions
        from sklearn.metrics import recall_score

        # threshold near 0 -> all predicted positive -> recall = 1.0
        y_pred_low = (y_prob >= 0.01).astype(int)
        assert recall_score(y_true, y_pred_low) == 1.0

        # threshold near 1 -> all predicted negative -> recall = 0.0
        y_pred_high = (y_prob >= 0.99).astype(int)
        assert recall_score(y_true, y_pred_high, zero_division=0.0) == 0.0

        # Figure still produces correctly
        path = plot_classification_threshold_sensitivity(
            y_true, y_prob, model_name="Boundary", figures_dir=tmp_path
        )
        assert path.exists()


# ============================================================================
# Tests: generate_evaluation_figures (convenience function)
# ============================================================================


class TestGenerateEvaluationFigures:
    """Tests for the convenience function that produces all figures."""

    def test_produces_three_figures(self, binary_predictions, tmp_path):
        """All three figures are generated for two-class data."""
        y_true, y_pred, y_prob = binary_predictions
        paths = generate_evaluation_figures(
            y_true, y_pred, y_prob, model_name="AllFigs", figures_dir=tmp_path
        )
        assert len(paths) == 3
        assert all(p.exists() for p in paths)
        names = [p.stem for p in paths]
        assert any("confusion_matrix" in n for n in names)
        assert any("roc_curve" in n for n in names)
        assert any("threshold_sensitivity" in n for n in names)

    def test_single_class_skips_roc_and_threshold(self, single_class_data, tmp_path):
        """With single-class data, only confusion matrix is produced."""
        y_true, y_pred, y_prob = single_class_data
        paths = generate_evaluation_figures(
            y_true, y_pred, y_prob, model_name="SingleClass", figures_dir=tmp_path
        )
        assert len(paths) == 1
        assert "confusion_matrix" in paths[0].stem
