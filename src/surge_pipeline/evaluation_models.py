"""Dataclasses and constants for evaluation results.

Extracted from evaluation.py to keep data contracts lightweight and
importable without pulling in heavy dependencies (matplotlib, sklearn).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Success tier constants (R21)
# ---------------------------------------------------------------------------

SUCCESS_TIER_MINIMUM: float = 0.60
"""Minimum acceptable AUC-ROC threshold."""

SUCCESS_TIER_TARGET: float = 0.70
"""Target AUC-ROC threshold."""

SUCCESS_TIER_STRETCH: float = 0.80
"""Stretch goal AUC-ROC threshold."""


# ---------------------------------------------------------------------------
# Evaluation dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelMetrics:
    """Comprehensive metrics for a single trained model on the test set."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    confusion_matrix: list[list[int]]
    n_test_samples: int


@dataclass
class McNemarResult:
    """Result from a McNemar's pairwise comparison between two models."""

    model_a: str
    model_b: str
    test_statistic: float
    p_value: float
    adjusted_alpha: float
    is_significant: bool


@dataclass
class MetricCI:
    """Confidence interval for a single metric."""

    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95


@dataclass
class BootstrapCI:
    """Bootstrap confidence intervals for all metrics of one model."""

    model_name: str
    n_bootstrap: int
    metric_cis: list[MetricCI]


@dataclass
class BaselineComparison:
    """Comparison of a model against random and single-feature baselines."""

    model_name: str
    model_auc: float
    random_baseline_auc: float
    beats_random: bool
    single_feature_aucs: dict[str, float]
    best_single_feature: str
    best_single_feature_auc: float
    improvement_over_best_single_feature: float


@dataclass
class SuccessTierResult:
    """Success tier classification for a single model."""

    model_name: str
    auc_roc: float
    achieves_minimum: bool
    achieves_target: bool
    achieves_stretch: bool
    tier_achieved: str  # "below_minimum", "minimum", "target", "stretch"


@dataclass
class ThresholdResult:
    """Result from optimal classification threshold selection."""

    model_name: str
    optimal_threshold: float
    strategy: str  # "max_f1" or "precision_floor"
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
    precision_at_default: float
    recall_at_default: float
    f1_at_default: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialisation."""
        return asdict(self)


@dataclass
class FinalSummary:
    """Final evaluation summary encompassing all analysis results."""

    best_model: str
    best_auc_roc: float
    overall_pass: bool
    success_tier_achieved: str
    model_metrics: dict[str, Any]
    mcnemar_results: list[dict[str, Any]]
    baseline_comparisons: dict[str, Any]
    tier_results: dict[str, Any]
    recommended_config: dict[str, Any]
    phase1_vs_phase2: dict[str, Any] | None = None


@dataclass
class EvaluationMetrics:
    """Container for basic evaluation metrics on a single partition."""

    model_name: str
    partition: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    support_positive: int
    support_negative: int
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialisation."""
        return asdict(self)


@dataclass
class FeatureImportanceResult:
    """Result from feature importance computation for a single model."""

    model_name: str
    method: str  # "permutation" or "builtin_gain"
    feature_names: list[str]
    importances: list[float]
    importances_std: list[float]
    scoring: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialisation."""
        return asdict(self)

    def ranked(self) -> list[tuple[str, float, float]]:
        """Return features sorted by importance (descending).

        Returns
        -------
        List[Tuple[str, float, float]]
            List of (feature_name, importance, std) sorted descending.
        """
        indices = np.argsort(self.importances)[::-1]
        return [
            (self.feature_names[i], self.importances[i], self.importances_std[i])
            for i in indices
        ]
