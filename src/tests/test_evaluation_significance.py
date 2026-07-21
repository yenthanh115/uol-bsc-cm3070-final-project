"""Tests for statistical significance testing and baseline evaluation (R19, R21).

Validates the mcnemar_pairwise_test, evaluate_baselines, validate_success_tiers,
and produce_final_summary functions added in Task 15.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from surge_pipeline.config import PipelineConfig
from surge_pipeline.evaluation import (
    SUCCESS_TIER_MINIMUM,
    SUCCESS_TIER_STRETCH,
    SUCCESS_TIER_TARGET,
    BaselineComparison,
    FinalSummary,
    McNemarResult,
    ModelMetrics,
    SuccessTierResult,
    evaluate_baselines,
    mcnemar_pairwise_test,
    produce_final_summary,
    validate_success_tiers,
)
from surge_pipeline.features import FEATURE_COLUMNS

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_y_test():
    """Sample test labels with known class distribution."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 2, size=200).astype(np.float64)


@pytest.fixture
def sample_predictions(sample_y_test):
    """Sample model predictions with known differences."""
    rng = np.random.RandomState(42)
    n = len(sample_y_test)

    # Model A: reasonably good (80% accurate)
    pred_a = sample_y_test.copy()
    flip_idx = rng.choice(n, size=int(n * 0.2), replace=False)
    pred_a[flip_idx] = 1 - pred_a[flip_idx]

    # Model B: slightly different (75% accurate)
    pred_b = sample_y_test.copy()
    flip_idx = rng.choice(n, size=int(n * 0.25), replace=False)
    pred_b[flip_idx] = 1 - pred_b[flip_idx]

    # Model C: identical to A (to test no-difference case)
    pred_c = pred_a.copy()

    return {
        "model_a": pred_a,
        "model_b": pred_b,
        "model_c": pred_c,
    }


@pytest.fixture
def sample_model_metrics():
    """Sample ModelMetrics for testing success tiers."""
    return {
        "logistic_regression": ModelMetrics(
            model_name="logistic_regression",
            accuracy=0.75,
            precision=0.70,
            recall=0.65,
            f1=0.67,
            auc_roc=0.72,
            confusion_matrix=[[80, 20], [35, 65]],
            n_test_samples=200,
        ),
        "random_forest": ModelMetrics(
            model_name="random_forest",
            accuracy=0.82,
            precision=0.78,
            recall=0.75,
            f1=0.76,
            auc_roc=0.85,
            confusion_matrix=[[85, 15], [25, 75]],
            n_test_samples=200,
        ),
        "xgboost": ModelMetrics(
            model_name="xgboost",
            accuracy=0.55,
            precision=0.50,
            recall=0.45,
            f1=0.47,
            auc_roc=0.58,
            confusion_matrix=[[60, 40], [55, 45]],
            n_test_samples=200,
        ),
    }


@pytest.fixture
def sample_df_for_baselines():
    """Create a minimal DataFrame for baseline evaluation."""
    rng = np.random.RandomState(42)
    n_train = 500
    n_test = 200
    n_total = n_train + n_test

    # Generate feature columns
    data = {}
    for col in FEATURE_COLUMNS:
        data[col] = rng.randn(n_total)

    data["surge_label"] = rng.randint(0, 2, size=n_total).astype(float)
    data["partition"] = ["train"] * n_train + ["test"] * n_test
    data["excluded"] = [False] * n_total

    return pd.DataFrame(data)


# =============================================================================
# Tests for mcnemar_pairwise_test
# =============================================================================


class TestMcNemarPairwiseTest:
    """Tests for McNemar's test with Bonferroni correction."""

    def test_returns_list_of_mcnemar_results(self, sample_y_test, sample_predictions):
        """Should return a list of McNemarResult objects."""
        results = mcnemar_pairwise_test(sample_y_test, sample_predictions)
        assert isinstance(results, list)
        assert all(isinstance(r, McNemarResult) for r in results)

    def test_correct_number_of_comparisons(self, sample_y_test, sample_predictions):
        """With 3 models, should produce 3 pairwise comparisons."""
        results = mcnemar_pairwise_test(sample_y_test, sample_predictions)
        # C(3, 2) = 3 pairs
        assert len(results) == 3

    def test_bonferroni_correction_applied(self, sample_y_test, sample_predictions):
        """Adjusted alpha should be 0.05 / n_comparisons."""
        results = mcnemar_pairwise_test(sample_y_test, sample_predictions, alpha=0.05)
        expected_adj_alpha = 0.05 / 3
        for r in results:
            assert abs(r.adjusted_alpha - expected_adj_alpha) < 1e-10

    def test_identical_predictions_not_significant(self, sample_y_test, sample_predictions):
        """Models with identical predictions should not differ significantly."""
        # model_a and model_c are identical
        results = mcnemar_pairwise_test(sample_y_test, sample_predictions)
        ac_result = next(
            r for r in results
            if (r.model_a == "model_a" and r.model_b == "model_c")
            or (r.model_a == "model_c" and r.model_b == "model_a")
        )
        assert ac_result.is_significant is False
        assert ac_result.p_value == 1.0

    def test_significance_decision_uses_adjusted_alpha(self, sample_y_test):
        """Significance should be determined using adjusted alpha, not raw alpha."""
        # Create two very different predictors
        pred_good = sample_y_test.copy()
        pred_bad = 1 - sample_y_test  # Always wrong

        predictions = {"good": pred_good, "bad": pred_bad}
        results = mcnemar_pairwise_test(sample_y_test, predictions, alpha=0.05)

        assert len(results) == 1
        # These are maximally different; should be significant
        assert results[0].is_significant is True

    def test_reports_test_statistic_and_p_value(self, sample_y_test, sample_predictions):
        """Each result should have valid test_statistic and p_value fields."""
        results = mcnemar_pairwise_test(sample_y_test, sample_predictions)
        for r in results:
            assert r.test_statistic >= 0
            assert 0 <= r.p_value <= 1


# =============================================================================
# Tests for evaluate_baselines
# =============================================================================


class TestEvaluateBaselines:
    """Tests for baseline comparison evaluation."""

    def test_returns_dict_of_baseline_comparisons(
        self, sample_df_for_baselines, sample_model_metrics
    ):
        """Should return a dict mapping model_name -> BaselineComparison."""
        results = evaluate_baselines(sample_df_for_baselines, sample_model_metrics)
        assert isinstance(results, dict)
        assert all(isinstance(v, BaselineComparison) for v in results.values())

    def test_all_models_compared(self, sample_df_for_baselines, sample_model_metrics):
        """Should produce comparisons for all trained models."""
        results = evaluate_baselines(sample_df_for_baselines, sample_model_metrics)
        assert set(results.keys()) == set(sample_model_metrics.keys())

    def test_random_baseline_is_05(self, sample_df_for_baselines, sample_model_metrics):
        """Random baseline AUC should always be 0.5."""
        results = evaluate_baselines(sample_df_for_baselines, sample_model_metrics)
        for comp in results.values():
            assert comp.random_baseline_auc == 0.5

    def test_single_feature_aucs_computed(
        self, sample_df_for_baselines, sample_model_metrics
    ):
        """Should compute AUC for each individual feature."""
        results = evaluate_baselines(sample_df_for_baselines, sample_model_metrics)
        for comp in results.values():
            assert len(comp.single_feature_aucs) == len(FEATURE_COLUMNS)
            for feature in FEATURE_COLUMNS:
                assert feature in comp.single_feature_aucs

    def test_beats_random_flag_correct(
        self, sample_df_for_baselines, sample_model_metrics
    ):
        """beats_random should be True iff model AUC > 0.5."""
        results = evaluate_baselines(sample_df_for_baselines, sample_model_metrics)
        for model_name, comp in results.items():
            expected = sample_model_metrics[model_name].auc_roc > 0.5
            assert comp.beats_random == expected


# =============================================================================
# Tests for validate_success_tiers
# =============================================================================


class TestValidateSuccessTiers:
    """Tests for success tier validation."""

    def test_returns_dict_of_tier_results(self, sample_model_metrics):
        """Should return dict mapping model_name -> SuccessTierResult."""
        results = validate_success_tiers(sample_model_metrics)
        assert isinstance(results, dict)
        assert all(isinstance(v, SuccessTierResult) for v in results.values())

    def test_stretch_tier_classification(self, sample_model_metrics):
        """Model with AUC > 0.80 should achieve stretch tier."""
        results = validate_success_tiers(sample_model_metrics)
        # random_forest has AUC=0.85
        rf_result = results["random_forest"]
        assert rf_result.achieves_stretch is True
        assert rf_result.tier_achieved == "stretch"

    def test_target_tier_classification(self, sample_model_metrics):
        """Model with 0.70 < AUC <= 0.80 should achieve target tier."""
        results = validate_success_tiers(sample_model_metrics)
        # logistic_regression has AUC=0.72
        lr_result = results["logistic_regression"]
        assert lr_result.achieves_target is True
        assert lr_result.achieves_stretch is False
        assert lr_result.tier_achieved == "target"

    def test_below_minimum_classification(self, sample_model_metrics):
        """Model with AUC <= 0.60 should be below minimum."""
        results = validate_success_tiers(sample_model_metrics)
        # xgboost has AUC=0.58
        xgb_result = results["xgboost"]
        assert xgb_result.achieves_minimum is False
        assert xgb_result.tier_achieved == "below_minimum"

    def test_tier_thresholds_correct(self):
        """Verify the tier threshold constants."""
        assert SUCCESS_TIER_MINIMUM == 0.60
        assert SUCCESS_TIER_TARGET == 0.70
        assert SUCCESS_TIER_STRETCH == 0.80


# =============================================================================
# Tests for produce_final_summary
# =============================================================================


class TestProduceFinalSummary:
    """Tests for final summary report generation."""

    def test_produces_final_summary_object(self, sample_model_metrics):
        """Should return a FinalSummary dataclass."""
        tier_results = validate_success_tiers(sample_model_metrics)
        config = PipelineConfig(threshold_tau=1.0, weight_sentiment=0.5)

        summary = produce_final_summary(
            model_metrics=sample_model_metrics,
            mcnemar_results=[],
            baseline_comparisons={},
            tier_results=tier_results,
            config=config,
            output_dir=tempfile.mkdtemp(),
        )

        assert isinstance(summary, FinalSummary)

    def test_identifies_best_model(self, sample_model_metrics):
        """Should identify the model with highest AUC-ROC as best."""
        tier_results = validate_success_tiers(sample_model_metrics)
        config = PipelineConfig(threshold_tau=1.0, weight_sentiment=0.5)

        summary = produce_final_summary(
            model_metrics=sample_model_metrics,
            mcnemar_results=[],
            baseline_comparisons={},
            tier_results=tier_results,
            config=config,
            output_dir=tempfile.mkdtemp(),
        )

        assert summary.best_model == "random_forest"
        assert summary.best_auc_roc == 0.85

    def test_overall_pass_when_minimum_achieved(self, sample_model_metrics):
        """Should report overall_pass=True when any model exceeds 0.60."""
        tier_results = validate_success_tiers(sample_model_metrics)
        config = PipelineConfig(threshold_tau=1.0, weight_sentiment=0.5)

        summary = produce_final_summary(
            model_metrics=sample_model_metrics,
            mcnemar_results=[],
            baseline_comparisons={},
            tier_results=tier_results,
            config=config,
            output_dir=tempfile.mkdtemp(),
        )

        assert summary.overall_pass is True

    def test_overall_fail_when_no_minimum(self):
        """Should report overall_pass=False when no model exceeds 0.60."""
        weak_metrics = {
            "model_a": ModelMetrics(
                model_name="model_a",
                accuracy=0.5,
                precision=0.4,
                recall=0.4,
                f1=0.4,
                auc_roc=0.55,
                confusion_matrix=[[50, 50], [50, 50]],
                n_test_samples=200,
            ),
        }
        tier_results = validate_success_tiers(weak_metrics)
        config = PipelineConfig(threshold_tau=1.0, weight_sentiment=0.5)

        summary = produce_final_summary(
            model_metrics=weak_metrics,
            mcnemar_results=[],
            baseline_comparisons={},
            tier_results=tier_results,
            config=config,
            output_dir=tempfile.mkdtemp(),
        )

        assert summary.overall_pass is False

    def test_saves_json_file(self, sample_model_metrics):
        """Should save a valid JSON file to the output directory."""
        tier_results = validate_success_tiers(sample_model_metrics)
        config = PipelineConfig(threshold_tau=1.0, weight_sentiment=0.5)
        out_dir = tempfile.mkdtemp()

        produce_final_summary(
            model_metrics=sample_model_metrics,
            mcnemar_results=[],
            baseline_comparisons={},
            tier_results=tier_results,
            config=config,
            output_dir=out_dir,
        )

        filepath = Path(out_dir) / "final_summary.json"
        assert filepath.exists()

        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert data["best_model"] == "random_forest"
        assert data["overall_pass"] is True
        assert data["success_tier_achieved"] == "stretch"

    def test_recommended_config_included(self, sample_model_metrics):
        """Should include recommended configuration in summary."""
        tier_results = validate_success_tiers(sample_model_metrics)
        config = PipelineConfig(threshold_tau=1.0, weight_sentiment=0.5)

        summary = produce_final_summary(
            model_metrics=sample_model_metrics,
            mcnemar_results=[],
            baseline_comparisons={},
            tier_results=tier_results,
            config=config,
            output_dir=tempfile.mkdtemp(),
        )

        assert summary.recommended_config["threshold_tau"] == 1.0
        assert summary.recommended_config["weight_w2"] == 0.5

    def test_phase1_vs_phase2_optional(self, sample_model_metrics):
        """Phase 1 vs Phase 2 comparison should be optional (None allowed)."""
        tier_results = validate_success_tiers(sample_model_metrics)
        config = PipelineConfig(threshold_tau=1.0, weight_sentiment=0.5)

        summary = produce_final_summary(
            model_metrics=sample_model_metrics,
            mcnemar_results=[],
            baseline_comparisons={},
            tier_results=tier_results,
            config=config,
            phase1_vs_phase2=None,
            output_dir=tempfile.mkdtemp(),
        )

        assert summary.phase1_vs_phase2 is None
