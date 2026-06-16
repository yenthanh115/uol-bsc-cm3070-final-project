"""Evaluation metrics and visualisations module.

Computes classification metrics on the held-out test set, produces
confusion matrices, ROC curves, and feature importance plots. Outputs
structured metrics JSON and publication-ready figures.

Requirements: R15 (Classification Metrics Computation),
              R16 (Evaluation Visualisations),
              R17 (Bootstrap Confidence Intervals),
              R18 (Multi-Seed Evaluation)
Design Decision: D12 — Evaluation output structure.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for figure generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from surge_pipeline.config import PipelineConfig
from surge_pipeline.features import FEATURE_COLUMNS
from surge_pipeline.training import TrainingResult, train_models

logger = logging.getLogger(__name__)

# Figure naming convention (D12)
_CONFUSION_MATRIX_FIG = "10_confusion_matrix_{model}.png"
_ROC_CURVES_FIG = "11_roc_curves.png"
_FEATURE_IMPORTANCE_FIG = "12_feature_importance_{model}.png"

# DPI settings: draft for fast iteration, publication for final output
_DPI_DRAFT = 100
_DPI_PUBLICATION = 300


# =============================================================================
# Data classes for evaluation results
# =============================================================================


@dataclass
class ModelMetrics:
    """Classification metrics for a single model."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    n_test_samples: int


@dataclass
class EvaluationResult:
    """Complete evaluation result for all models."""

    model_metrics: Dict[str, ModelMetrics]
    train_surge_rate: float
    test_surge_rate: float
    n_train: int
    n_test: int
    figures_dir: str
    metrics_dir: str


# =============================================================================
# Bootstrap Confidence Interval data classes (R17)
# =============================================================================

# Default seeds for multi-seed evaluation (R18-AC1)
MULTI_SEED_LIST: List[int] = [42, 123, 256, 512, 1024]

# Instability threshold for std(AUC-ROC) (R18-AC3)
INSTABILITY_THRESHOLD: float = 0.05


@dataclass
class MetricCI:
    """95% bootstrap confidence interval for a single metric (R17-AC2)."""

    lower: float
    point_estimate: float
    upper: float


@dataclass
class BootstrapCI:
    """Bootstrap confidence intervals for all classification metrics (R17).

    Contains lower bound, point estimate, and upper bound for each metric.
    """

    accuracy: MetricCI
    precision: MetricCI
    recall: MetricCI
    f1: MetricCI
    auc_roc: MetricCI
    n_iterations: int = 1000
    bootstrap_seed: int = 42


@dataclass
class ModelSeedMetrics:
    """Metrics for a single model across all seeds (R18-AC2)."""

    model_name: str
    seed_metrics: Dict[int, ModelMetrics]  # seed -> ModelMetrics
    mean_accuracy: float
    std_accuracy: float
    mean_precision: float
    std_precision: float
    mean_recall: float
    std_recall: float
    mean_f1: float
    std_f1: float
    mean_auc_roc: float
    std_auc_roc: float
    is_unstable: bool  # True if std(AUC-ROC) > 0.05 (R18-AC3)


@dataclass
class MultiSeedResult:
    """Complete multi-seed evaluation result (R18).

    Contains per-seed results, mean/std summary, and instability flags.
    """

    model_seed_metrics: Dict[str, ModelSeedMetrics]
    seeds: List[int]
    unstable_models: List[str]  # Models with std(AUC-ROC) > 0.05
    per_seed_results: Dict[int, EvaluationResult]  # seed -> EvaluationResult


# =============================================================================
# Core evaluation function
# =============================================================================


def evaluate_models(
    df: pd.DataFrame,
    training_result: TrainingResult,
    config: PipelineConfig,
    figures_dir: Optional[str] = None,
    metrics_dir: Optional[str] = None,
    skip_figures: bool = False,
    publication_quality: bool = False,
) -> EvaluationResult:
    """Evaluate trained models on the held-out test set (R15, R16).

    Workflow:
      1. Extract test partition features and labels
      2. Compute accuracy, precision, recall, F1, AUC-ROC per model (R15-AC1)
      3. Report train/test surge rates for concept drift analysis (R15-AC5)
      4. Generate confusion matrix figures (R16-AC1)
      5. Generate combined ROC curve figure (R16-AC2, AC5)
      6. Generate feature importance figures for tree models (R16-AC3)
      7. Save metrics JSON (R15-AC3)

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with FEATURE_COLUMNS, `surge_label`,
        `partition`, `excluded`, and `created_utc`.
    training_result : TrainingResult
        Result from training.py containing all trained models.
    config : PipelineConfig
        Pipeline configuration.
    figures_dir : str, optional
        Directory for saving figures. Defaults to "figures/".
    metrics_dir : str, optional
        Directory for saving metrics JSON. Defaults to
        "data/processed/evaluation/".
    skip_figures : bool, optional
        If True, skip all figure generation for faster iteration.
        Defaults to False.
    publication_quality : bool, optional
        If True, render figures at 300 DPI for publication. If False,
        use 100 DPI draft mode for faster rendering. Defaults to False.

    Returns
    -------
    EvaluationResult
        Structured evaluation results for all models.
    """
    t_total_start = time.perf_counter()

    logger.info("=" * 70)
    logger.info("MODEL EVALUATION — %s", training_result.phase.upper())
    logger.info("=" * 70)
    if skip_figures:
        logger.info("  (skip_figures=True — figure generation disabled)")
    elif not publication_quality:
        logger.info("  (draft mode — figures at %d DPI)", _DPI_DRAFT)

    # Determine DPI for this run
    dpi = _DPI_PUBLICATION if publication_quality else _DPI_DRAFT

    # Set up output directories
    fig_dir = Path(figures_dir) if figures_dir else Path("figures")
    met_dir = Path(metrics_dir) if metrics_dir else Path("data/processed/evaluation")
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Extract test and train partitions (R15-AC2)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    train_mask = (
        (df["partition"] == "train")
        & (~df["excluded"].astype(bool))
        & (df["surge_label"].notna())
    )
    test_mask = (
        (df["partition"] == "test")
        & (~df["excluded"].astype(bool))
        & (df["surge_label"].notna())
    )

    df_train = df.loc[train_mask]
    df_test = df.loc[test_mask]

    X_test = df_test[FEATURE_COLUMNS].values.astype(np.float64)
    y_test = df_test["surge_label"].values.astype(np.float64)

    y_train = df_train["surge_label"].values.astype(np.float64)

    n_train = len(df_train)
    n_test = len(df_test)

    logger.info("  [timing] Data extraction: %.2fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Step 2: Compute surge rates (R15-AC5)
    # ------------------------------------------------------------------
    train_surge_rate = float(y_train.mean()) if n_train > 0 else 0.0
    test_surge_rate = float(y_test.mean()) if n_test > 0 else 0.0

    logger.info("Train partition: %d records, surge rate = %.4f", n_train, train_surge_rate)
    logger.info("Test partition:  %d records, surge rate = %.4f", n_test, test_surge_rate)
    logger.info(
        "Concept drift indicator: train_surge=%.4f, test_surge=%.4f, diff=%.4f",
        train_surge_rate,
        test_surge_rate,
        abs(test_surge_rate - train_surge_rate),
    )

    # ------------------------------------------------------------------
    # Step 3: Compute metrics per model (R15-AC1, AC4)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    model_metrics: Dict[str, ModelMetrics] = {}
    roc_data: Dict[str, Dict[str, Any]] = {}  # For combined ROC plot

    for model_name, trained_model in training_result.models.items():
        logger.info("Evaluating %s...", model_name)

        metrics, y_prob = _compute_model_metrics(
            trained_model.model, model_name, X_test, y_test
        )
        model_metrics[model_name] = metrics

        # Store ROC data for combined plot
        if y_prob is not None and len(np.unique(y_test)) >= 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[model_name] = {"fpr": fpr, "tpr": tpr, "auc": metrics.auc_roc}

        logger.info(
            "  %s — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
            model_name,
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.auc_roc,
        )

    logger.info("  [timing] Model inference + metrics: %.2fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Steps 4-6: Generate figures (R16) — skippable for fast iteration
    # ------------------------------------------------------------------
    if not skip_figures:
        t0 = time.perf_counter()

        # Step 4: Confusion matrix figures (R16-AC1, AC4)
        for model_name, metrics in model_metrics.items():
            _plot_confusion_matrix(metrics, fig_dir, dpi=dpi)

        logger.info(
            "  [timing] Confusion matrices: %.2fs", time.perf_counter() - t0
        )

        # Step 5: Combined ROC curve (R16-AC2, AC5)
        t0 = time.perf_counter()
        _plot_roc_curves(roc_data, fig_dir, dpi=dpi)
        logger.info("  [timing] ROC curves: %.2fs", time.perf_counter() - t0)

        # Step 6: Feature importance figures (R16-AC3)
        t0 = time.perf_counter()
        for model_name in ["random_forest", "xgboost"]:
            if model_name in training_result.models:
                _plot_feature_importance(
                    training_result.models[model_name].model,
                    model_name,
                    fig_dir,
                    dpi=dpi,
                )
        logger.info(
            "  [timing] Feature importance: %.2fs", time.perf_counter() - t0
        )
    else:
        logger.info("  [timing] Figures: SKIPPED")

    # ------------------------------------------------------------------
    # Step 7: Save structured metrics JSON (R15-AC3)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    result = EvaluationResult(
        model_metrics=model_metrics,
        train_surge_rate=train_surge_rate,
        test_surge_rate=test_surge_rate,
        n_train=n_train,
        n_test=n_test,
        figures_dir=str(fig_dir),
        metrics_dir=str(met_dir),
    )

    _save_metrics_json(result, met_dir)
    _print_metrics_table(result)

    logger.info("  [timing] Metrics JSON save: %.2fs", time.perf_counter() - t0)

    t_total = time.perf_counter() - t_total_start
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE (total: %.2fs)", t_total)
    if not skip_figures:
        logger.info("  Figures saved to: %s", fig_dir)
    logger.info("  Metrics saved to: %s", met_dir)
    logger.info("=" * 70)

    return result


# =============================================================================
# Metrics computation
# =============================================================================


def _compute_model_metrics(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[ModelMetrics, Optional[np.ndarray]]:
    """Compute classification metrics for one model (R15-AC1, AC4).

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model with predict() and optionally predict_proba().
    model_name : str
        Name of the model for reporting.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test labels.

    Returns
    -------
    tuple of (ModelMetrics, y_prob or None)
        Metrics object and predicted probabilities (for ROC curve).
    """
    n_test = len(y_test)

    # Handle edge case: empty test set
    if n_test == 0:
        return ModelMetrics(
            model_name=model_name,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            auc_roc=0.5,
            confusion_matrix=[[0, 0], [0, 0]],
            n_test_samples=0,
        ), None

    # Generate predictions
    y_pred = model.predict(X_test)

    # Get predicted probabilities for AUC-ROC (R15-AC4)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0.0)
    recall = recall_score(y_test, y_pred, zero_division=0.0)
    f1 = f1_score(y_test, y_pred, zero_division=0.0)

    # AUC-ROC using probabilities (R15-AC4)
    if y_prob is not None and len(np.unique(y_test)) >= 2:
        auc_roc = roc_auc_score(y_test, y_prob)
    else:
        # Fallback: single-class test set or no probability output
        auc_roc = 0.5
        logger.warning(
            "%s: AUC-ROC set to 0.5 (single class in test or no probabilities)",
            model_name,
        )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_list = cm.tolist()

    return ModelMetrics(
        model_name=model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
        confusion_matrix=cm_list,
        n_test_samples=n_test,
    ), y_prob


# =============================================================================
# Visualisations (simplified for speed)
# =============================================================================


def _plot_confusion_matrix(
    metrics: ModelMetrics, fig_dir: Path, *, dpi: int = _DPI_DRAFT
) -> None:
    """Produce confusion matrix figure for one model (R16-AC1, AC4).

    Uses matshow for faster rendering than imshow + colorbar.

    Parameters
    ----------
    metrics : ModelMetrics
        Model metrics containing confusion matrix data.
    fig_dir : Path
        Directory to save the figure.
    dpi : int
        Output resolution (default: draft quality).
    """
    cm = np.array(metrics.confusion_matrix)
    model_name = metrics.model_name

    fig, ax = plt.subplots(figsize=(5, 4))

    # Simple matshow — faster than imshow + colorbar
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)

    # Labels
    classes = ["Non-Surge (0)", "Surge (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(f"Confusion Matrix — {_format_model_name(model_name)}", pad=10)

    # Annotate cells with counts
    labels = [["TN", "FP"], ["FN", "TP"]]
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{labels[i][j]}\n{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11, fontweight="bold",
            )

    plt.tight_layout()

    filename = _CONFUSION_MATRIX_FIG.format(model=model_name)
    filepath = fig_dir / filename
    fig.savefig(filepath, dpi=dpi)
    plt.close(fig)

    logger.info("  Saved confusion matrix: %s", filepath)


def _plot_roc_curves(
    roc_data: Dict[str, Dict[str, Any]], fig_dir: Path, *, dpi: int = _DPI_DRAFT
) -> None:
    """Produce combined ROC curve plot with all models (R16-AC2, AC5).

    Parameters
    ----------
    roc_data : dict
        Mapping of model_name -> {"fpr", "tpr", "auc"}.
    fig_dir : Path
        Directory to save the figure.
    dpi : int
        Output resolution (default: draft quality).
    """
    if not roc_data:
        logger.warning("No ROC data available — skipping ROC curve figure")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    # Color palette for models
    colors = {"logistic_regression": "#1f77b4", "random_forest": "#2ca02c", "xgboost": "#ff7f0e"}

    for model_name, data in roc_data.items():
        color = colors.get(model_name, None)
        label = f"{_format_model_name(model_name)} (AUC = {data['auc']:.4f})"
        ax.plot(data["fpr"], data["tpr"], label=label, color=color, linewidth=1.5)

    # Diagonal reference line (R16-AC5)
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", color="grey", linewidth=1,
        label="Random Baseline (AUC = 0.5)",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = fig_dir / _ROC_CURVES_FIG
    fig.savefig(filepath, dpi=dpi)
    plt.close(fig)

    logger.info("  Saved ROC curves: %s", filepath)


def _plot_feature_importance(
    model: Any,
    model_name: str,
    fig_dir: Path,
    *,
    dpi: int = _DPI_DRAFT,
) -> None:
    """Produce feature importance plot for tree-based models (R16-AC3, AC4).

    Parameters
    ----------
    model : sklearn/xgboost estimator
        Fitted tree-based model with `feature_importances_` attribute.
    model_name : str
        Model name for title and filename.
    fig_dir : Path
        Directory to save the figure.
    dpi : int
        Output resolution (default: draft quality).
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning(
            "%s does not have feature_importances_ — skipping importance plot",
            model_name,
        )
        return

    importances = model.feature_importances_
    feature_names = FEATURE_COLUMNS

    # Sort by importance (descending)
    sorted_idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 4))

    # Horizontal bar chart
    y_pos = np.arange(len(sorted_names))
    ax.barh(
        y_pos, sorted_importances, align="center",
        color="#2ca02c" if "forest" in model_name else "#ff7f0e",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance — {_format_model_name(model_name)}")

    plt.tight_layout()

    filename = _FEATURE_IMPORTANCE_FIG.format(model=model_name)
    filepath = fig_dir / filename
    fig.savefig(filepath, dpi=dpi)
    plt.close(fig)

    logger.info("  Saved feature importance: %s", filepath)


# =============================================================================
# Output helpers
# =============================================================================


def _save_metrics_json(result: EvaluationResult, met_dir: Path) -> None:
    """Save structured metrics to JSON files (R15-AC3).

    Produces:
      - Per-model JSON: {model_name}_metrics.json
      - Summary JSON: evaluation_summary.json

    Parameters
    ----------
    result : EvaluationResult
        Complete evaluation result.
    met_dir : Path
        Output directory for metrics files.
    """
    # Per-model metrics files
    for model_name, metrics in result.model_metrics.items():
        model_data = {
            "model_name": metrics.model_name,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "auc_roc": metrics.auc_roc,
            "confusion_matrix": metrics.confusion_matrix,
            "n_test_samples": metrics.n_test_samples,
        }
        filepath = met_dir / f"{model_name}_metrics.json"
        filepath.write_text(json.dumps(model_data, indent=2), encoding="utf-8")
        logger.info("  Saved metrics: %s", filepath)

    # Summary JSON with all models and surge rates
    summary = {
        "train_surge_rate": result.train_surge_rate,
        "test_surge_rate": result.test_surge_rate,
        "concept_drift_delta": abs(result.test_surge_rate - result.train_surge_rate),
        "n_train": result.n_train,
        "n_test": result.n_test,
        "models": {},
    }
    for model_name, metrics in result.model_metrics.items():
        summary["models"][model_name] = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "auc_roc": metrics.auc_roc,
        }

    filepath = met_dir / "evaluation_summary.json"
    filepath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("  Saved summary: %s", filepath)


def _print_metrics_table(result: EvaluationResult) -> None:
    """Print a formatted metrics table to the logger.

    Parameters
    ----------
    result : EvaluationResult
        Complete evaluation result.
    """
    logger.info("-" * 70)
    logger.info("EVALUATION METRICS SUMMARY")
    logger.info("-" * 70)
    logger.info(
        "%-22s %8s %8s %8s %8s %8s",
        "Model", "Acc", "Prec", "Rec", "F1", "AUC-ROC",
    )
    logger.info("-" * 70)

    for model_name, metrics in result.model_metrics.items():
        logger.info(
            "%-22s %8.4f %8.4f %8.4f %8.4f %8.4f",
            _format_model_name(model_name),
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.auc_roc,
        )

    logger.info("-" * 70)
    logger.info(
        "Train surge rate: %.4f | Test surge rate: %.4f | Delta: %.4f",
        result.train_surge_rate,
        result.test_surge_rate,
        abs(result.test_surge_rate - result.train_surge_rate),
    )
    logger.info("-" * 70)


# =============================================================================
# Utility functions
# =============================================================================


def _format_model_name(name: str) -> str:
    """Format a snake_case model name for display.

    Parameters
    ----------
    name : str
        Model name in snake_case (e.g., "logistic_regression").

    Returns
    -------
    str
        Formatted name (e.g., "Logistic Regression").
    """
    return name.replace("_", " ").title()


def get_evaluation_summary(result: EvaluationResult) -> Dict[str, Any]:
    """Generate a JSON-serialisable summary of evaluation results.

    Parameters
    ----------
    result : EvaluationResult
        Complete evaluation result.

    Returns
    -------
    dict
        Summary suitable for JSON serialisation.
    """
    summary = {
        "train_surge_rate": result.train_surge_rate,
        "test_surge_rate": result.test_surge_rate,
        "concept_drift_delta": abs(result.test_surge_rate - result.train_surge_rate),
        "n_train": result.n_train,
        "n_test": result.n_test,
        "models": {},
    }

    for model_name, metrics in result.model_metrics.items():
        summary["models"][model_name] = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "auc_roc": metrics.auc_roc,
            "confusion_matrix": metrics.confusion_matrix,
            "n_test_samples": metrics.n_test_samples,
        }

    return summary


# =============================================================================
# Bootstrap Confidence Intervals (R17)
# =============================================================================


def compute_bootstrap_ci(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_iterations: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> BootstrapCI:
    """Compute 95% bootstrap confidence intervals for all classification metrics (R17).

    Resamples the test set with replacement for `n_iterations` iterations,
    computing accuracy, precision, recall, F1, and AUC-ROC on each bootstrap
    sample. Reports the lower bound, point estimate, and upper bound for
    each metric based on the specified confidence level.

    Parameters
    ----------
    y_test : np.ndarray
        True labels for the test set.
    y_pred : np.ndarray
        Predicted binary labels for the test set.
    y_prob : np.ndarray
        Predicted probabilities (positive class) for the test set.
    n_iterations : int, optional
        Number of bootstrap resampling iterations (default: 1000, R17-AC1).
    seed : int, optional
        Fixed random seed for reproducible bootstrap sampling (R17-AC3).
    confidence_level : float, optional
        Confidence level for intervals (default: 0.95 for 95% CIs).

    Returns
    -------
    BootstrapCI
        Bootstrap confidence intervals for all metrics (R17-AC2).
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_test)

    # Storage for bootstrap metric distributions
    boot_accuracy = np.zeros(n_iterations)
    boot_precision = np.zeros(n_iterations)
    boot_recall = np.zeros(n_iterations)
    boot_f1 = np.zeros(n_iterations)
    boot_auc_roc = np.zeros(n_iterations)

    for i in range(n_iterations):
        # Resample indices with replacement
        indices = rng.randint(0, n_samples, size=n_samples)

        y_test_boot = y_test[indices]
        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices]

        # Compute metrics on bootstrap sample
        boot_accuracy[i] = accuracy_score(y_test_boot, y_pred_boot)
        boot_precision[i] = precision_score(y_test_boot, y_pred_boot, zero_division=0.0)
        boot_recall[i] = recall_score(y_test_boot, y_pred_boot, zero_division=0.0)
        boot_f1[i] = f1_score(y_test_boot, y_pred_boot, zero_division=0.0)

        # AUC-ROC requires both classes present
        if len(np.unique(y_test_boot)) >= 2:
            boot_auc_roc[i] = roc_auc_score(y_test_boot, y_prob_boot)
        else:
            boot_auc_roc[i] = 0.5

    # Compute confidence interval bounds
    alpha = 1 - confidence_level
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100

    # Point estimates from original data
    point_accuracy = accuracy_score(y_test, y_pred)
    point_precision = precision_score(y_test, y_pred, zero_division=0.0)
    point_recall = recall_score(y_test, y_pred, zero_division=0.0)
    point_f1 = f1_score(y_test, y_pred, zero_division=0.0)
    if len(np.unique(y_test)) >= 2:
        point_auc_roc = roc_auc_score(y_test, y_prob)
    else:
        point_auc_roc = 0.5

    ci = BootstrapCI(
        accuracy=MetricCI(
            lower=float(np.percentile(boot_accuracy, lower_pct)),
            point_estimate=point_accuracy,
            upper=float(np.percentile(boot_accuracy, upper_pct)),
        ),
        precision=MetricCI(
            lower=float(np.percentile(boot_precision, lower_pct)),
            point_estimate=point_precision,
            upper=float(np.percentile(boot_precision, upper_pct)),
        ),
        recall=MetricCI(
            lower=float(np.percentile(boot_recall, lower_pct)),
            point_estimate=point_recall,
            upper=float(np.percentile(boot_recall, upper_pct)),
        ),
        f1=MetricCI(
            lower=float(np.percentile(boot_f1, lower_pct)),
            point_estimate=point_f1,
            upper=float(np.percentile(boot_f1, upper_pct)),
        ),
        auc_roc=MetricCI(
            lower=float(np.percentile(boot_auc_roc, lower_pct)),
            point_estimate=point_auc_roc,
            upper=float(np.percentile(boot_auc_roc, upper_pct)),
        ),
        n_iterations=n_iterations,
        bootstrap_seed=seed,
    )

    logger.info(
        "Bootstrap CI (n=%d, seed=%d): AUC-ROC = [%.4f, %.4f, %.4f]",
        n_iterations,
        seed,
        ci.auc_roc.lower,
        ci.auc_roc.point_estimate,
        ci.auc_roc.upper,
    )

    return ci


# =============================================================================
# Multi-Seed Evaluation (R18)
# =============================================================================


def run_multi_seed_evaluation(
    df: pd.DataFrame,
    config: PipelineConfig,
    figures_dir: Optional[str] = None,
    metrics_dir: Optional[str] = None,
    skip_figures: bool = True,
) -> MultiSeedResult:
    """Train and evaluate each model across multiple seeds (R18).

    Iterates over MULTI_SEED_LIST, training and evaluating all models at
    each seed. The temporal split is deterministic (order-based) and remains
    constant across seeds — seed variation affects only model initialisation
    (R18-AC4).

    Reports mean ± std for each metric across seeds (R18-AC2) and flags
    models with std(AUC-ROC) > 0.05 as exhibiting initialisation instability
    (R18-AC3).

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with FEATURE_COLUMNS, `surge_label`,
        `partition`, `excluded`, and `created_utc`.
    config : PipelineConfig
        Pipeline configuration. The `random_seed` field will be overridden
        for each seed iteration.
    figures_dir : str, optional
        Directory for saving figures. Defaults to "figures/".
    metrics_dir : str, optional
        Directory for saving metrics JSON. Defaults to
        "data/processed/evaluation/".
    skip_figures : bool, optional
        If True, skip figure generation for all but the last seed
        (default: True for efficiency).

    Returns
    -------
    MultiSeedResult
        Complete multi-seed evaluation results with mean/std and
        instability flags.
    """
    t_start = time.perf_counter()

    logger.info("=" * 70)
    logger.info("MULTI-SEED EVALUATION — Seeds: %s", MULTI_SEED_LIST)
    logger.info("=" * 70)

    # Set up output directories
    fig_dir = Path(figures_dir) if figures_dir else Path("figures")
    met_dir = Path(metrics_dir) if metrics_dir else Path("data/processed/evaluation")
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    # Storage for per-seed results
    per_seed_results: Dict[int, EvaluationResult] = {}
    per_seed_training: Dict[int, TrainingResult] = {}

    for i, seed in enumerate(MULTI_SEED_LIST):
        logger.info("-" * 70)
        logger.info("Seed %d/%d: %d", i + 1, len(MULTI_SEED_LIST), seed)
        logger.info("-" * 70)

        # Create a config copy with the current seed (R18-AC4)
        seed_config = copy.deepcopy(config)
        seed_config.random_seed = seed

        # Train models with this seed
        training_result = train_models(df, seed_config)
        per_seed_training[seed] = training_result

        # Evaluate — generate figures only for the last seed (efficiency)
        is_last_seed = (i == len(MULTI_SEED_LIST) - 1)
        seed_skip_figures = skip_figures if not is_last_seed else skip_figures

        eval_result = evaluate_models(
            df=df,
            training_result=training_result,
            config=seed_config,
            figures_dir=str(fig_dir),
            metrics_dir=str(met_dir),
            skip_figures=seed_skip_figures,
        )
        per_seed_results[seed] = eval_result

    # ------------------------------------------------------------------
    # Aggregate metrics across seeds (R18-AC2)
    # ------------------------------------------------------------------
    model_names = list(per_seed_results[MULTI_SEED_LIST[0]].model_metrics.keys())
    model_seed_metrics: Dict[str, ModelSeedMetrics] = {}
    unstable_models: List[str] = []

    for model_name in model_names:
        # Collect per-seed metrics for this model
        seed_metrics_dict: Dict[int, ModelMetrics] = {}
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        auc_rocs = []

        for seed in MULTI_SEED_LIST:
            metrics = per_seed_results[seed].model_metrics[model_name]
            seed_metrics_dict[seed] = metrics
            accuracies.append(metrics.accuracy)
            precisions.append(metrics.precision)
            recalls.append(metrics.recall)
            f1s.append(metrics.f1)
            auc_rocs.append(metrics.auc_roc)

        # Compute mean and std
        mean_accuracy = float(np.mean(accuracies))
        std_accuracy = float(np.std(accuracies, ddof=1))
        mean_precision = float(np.mean(precisions))
        std_precision = float(np.std(precisions, ddof=1))
        mean_recall = float(np.mean(recalls))
        std_recall = float(np.std(recalls, ddof=1))
        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s, ddof=1))
        mean_auc_roc = float(np.mean(auc_rocs))
        std_auc_roc = float(np.std(auc_rocs, ddof=1))

        # Flag instability (R18-AC3)
        is_unstable = std_auc_roc > INSTABILITY_THRESHOLD

        if is_unstable:
            unstable_models.append(model_name)
            logger.warning(
                "⚠ %s: UNSTABLE — std(AUC-ROC) = %.4f > %.2f threshold",
                model_name,
                std_auc_roc,
                INSTABILITY_THRESHOLD,
            )

        model_seed_metrics[model_name] = ModelSeedMetrics(
            model_name=model_name,
            seed_metrics=seed_metrics_dict,
            mean_accuracy=mean_accuracy,
            std_accuracy=std_accuracy,
            mean_precision=mean_precision,
            std_precision=std_precision,
            mean_recall=mean_recall,
            std_recall=std_recall,
            mean_f1=mean_f1,
            std_f1=std_f1,
            mean_auc_roc=mean_auc_roc,
            std_auc_roc=std_auc_roc,
            is_unstable=is_unstable,
        )

    result = MultiSeedResult(
        model_seed_metrics=model_seed_metrics,
        seeds=MULTI_SEED_LIST,
        unstable_models=unstable_models,
        per_seed_results=per_seed_results,
    )

    # ------------------------------------------------------------------
    # Save multi-seed results and print summary
    # ------------------------------------------------------------------
    _save_multi_seed_json(result, met_dir)
    _print_multi_seed_summary(result)

    t_elapsed = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info("MULTI-SEED EVALUATION COMPLETE (total: %.1fs)", t_elapsed)
    logger.info("  Seeds evaluated: %s", MULTI_SEED_LIST)
    logger.info("  Unstable models: %s", unstable_models if unstable_models else "None")
    logger.info("  Results saved to: %s", met_dir)
    logger.info("=" * 70)

    return result


def _save_multi_seed_json(result: MultiSeedResult, met_dir: Path) -> None:
    """Save multi-seed evaluation results to JSON.

    Parameters
    ----------
    result : MultiSeedResult
        Complete multi-seed evaluation result.
    met_dir : Path
        Output directory for metrics files.
    """
    output = {
        "seeds": result.seeds,
        "unstable_models": result.unstable_models,
        "instability_threshold": INSTABILITY_THRESHOLD,
        "models": {},
    }

    for model_name, msm in result.model_seed_metrics.items():
        per_seed_data = {}
        for seed, metrics in msm.seed_metrics.items():
            per_seed_data[str(seed)] = {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "auc_roc": metrics.auc_roc,
            }

        output["models"][model_name] = {
            "mean_accuracy": msm.mean_accuracy,
            "std_accuracy": msm.std_accuracy,
            "mean_precision": msm.mean_precision,
            "std_precision": msm.std_precision,
            "mean_recall": msm.mean_recall,
            "std_recall": msm.std_recall,
            "mean_f1": msm.mean_f1,
            "std_f1": msm.std_f1,
            "mean_auc_roc": msm.mean_auc_roc,
            "std_auc_roc": msm.std_auc_roc,
            "is_unstable": msm.is_unstable,
            "per_seed": per_seed_data,
        }

    filepath = met_dir / "multi_seed_evaluation.json"
    filepath.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("  Saved multi-seed results: %s", filepath)


def _print_multi_seed_summary(result: MultiSeedResult) -> None:
    """Print a formatted multi-seed evaluation summary table.

    Parameters
    ----------
    result : MultiSeedResult
        Complete multi-seed evaluation result.
    """
    logger.info("-" * 70)
    logger.info("MULTI-SEED EVALUATION SUMMARY (mean ± std across %d seeds)", len(result.seeds))
    logger.info("-" * 70)
    logger.info(
        "%-22s %14s %14s %14s %14s %14s %s",
        "Model", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "Stable?",
    )
    logger.info("-" * 70)

    for model_name, msm in result.model_seed_metrics.items():
        stability = "✗ UNSTABLE" if msm.is_unstable else "✓"
        logger.info(
            "%-22s %6.4f±%.4f %6.4f±%.4f %6.4f±%.4f %6.4f±%.4f %6.4f±%.4f %s",
            _format_model_name(model_name),
            msm.mean_accuracy, msm.std_accuracy,
            msm.mean_precision, msm.std_precision,
            msm.mean_recall, msm.std_recall,
            msm.mean_f1, msm.std_f1,
            msm.mean_auc_roc, msm.std_auc_roc,
            stability,
        )

    logger.info("-" * 70)
    if result.unstable_models:
        logger.info(
            "⚠ Unstable models (std AUC-ROC > %.2f): %s",
            INSTABILITY_THRESHOLD,
            ", ".join(result.unstable_models),
        )
    else:
        logger.info("✓ All models are stable (std AUC-ROC ≤ %.2f)", INSTABILITY_THRESHOLD)
    logger.info("-" * 70)


# =============================================================================
# Statistical Significance Testing (R19)
# =============================================================================


@dataclass
class McNemarResult:
    """Result of a McNemar's test pairwise comparison (R19-AC1, AC4)."""

    model_a: str
    model_b: str
    test_statistic: float
    p_value: float
    adjusted_alpha: float  # Bonferroni-corrected α
    is_significant: bool


@dataclass
class BaselineComparison:
    """Comparison of a trained model against baselines (R19-AC5)."""

    model_name: str
    model_auc_roc: float
    random_baseline_auc: float  # Always 0.5
    majority_class_accuracy: float
    majority_class_auc: float
    single_feature_aucs: Dict[str, float]  # feature_name -> AUC-ROC
    beats_random: bool
    beats_majority: bool
    best_single_feature: str
    best_single_feature_auc: float
    beats_all_single_features: bool


@dataclass
class SuccessTierResult:
    """Success tier validation result for a model (R21-AC1, AC2)."""

    model_name: str
    auc_roc: float
    achieves_minimum: bool  # > 0.60
    achieves_target: bool  # > 0.70
    achieves_stretch: bool  # > 0.80
    tier_achieved: str  # "stretch", "target", "minimum", or "below_minimum"


@dataclass
class FinalSummary:
    """Final summary report with pass/fail determination (R21-AC5)."""

    best_model: str
    best_auc_roc: float
    best_auc_roc_ci: Optional[Dict[str, float]]  # {lower, upper} if available
    success_tier_achieved: str
    overall_pass: bool  # At least one model > 0.60
    model_tiers: Dict[str, SuccessTierResult]
    mcnemar_results: List[McNemarResult]
    baseline_comparisons: Dict[str, BaselineComparison]
    phase1_vs_phase2: Optional[Dict[str, Any]]  # Optional comparison
    recommended_config: Dict[str, Any]  # {threshold_tau, weight_w2}


# Success tier thresholds (R21-AC1)
SUCCESS_TIER_MINIMUM: float = 0.60
SUCCESS_TIER_TARGET: float = 0.70
SUCCESS_TIER_STRETCH: float = 0.80


def mcnemar_pairwise_test(
    y_test: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> List[McNemarResult]:
    """Apply McNemar's test for pairwise model comparison (R19-AC1, AC3, AC4).

    Compares all pairs of models using McNemar's test on paired predictions
    from the same test set. Applies Bonferroni correction for multiple
    comparisons to control the family-wise error rate.

    Parameters
    ----------
    y_test : np.ndarray
        True labels for the test set.
    model_predictions : dict
        Mapping of model_name -> predicted binary labels (np.ndarray).
    alpha : float, optional
        Significance level before correction (default: 0.05, R19-AC2).

    Returns
    -------
    list of McNemarResult
        McNemar's test results for each pair of models.
    """
    from itertools import combinations
    from scipy.stats import chi2

    model_names = list(model_predictions.keys())
    n_comparisons = len(list(combinations(model_names, 2)))

    # Bonferroni correction (R19-AC3)
    adjusted_alpha = alpha / max(n_comparisons, 1)

    results: List[McNemarResult] = []

    for model_a, model_b in combinations(model_names, 2):
        pred_a = model_predictions[model_a]
        pred_b = model_predictions[model_b]

        # Compute correct/incorrect for each model
        correct_a = (pred_a == y_test).astype(int)
        correct_b = (pred_b == y_test).astype(int)

        # McNemar's contingency table
        # b = A correct, B incorrect
        # c = A incorrect, B correct
        b = np.sum((correct_a == 1) & (correct_b == 0))
        c = np.sum((correct_a == 0) & (correct_b == 1))

        # McNemar's test statistic with continuity correction
        if b + c == 0:
            # No discordant pairs — models make identical predictions
            statistic = 0.0
            p_value = 1.0
        else:
            # Chi-squared statistic with continuity correction
            statistic = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1.0 - chi2.cdf(statistic, df=1)

        is_significant = bool(p_value < adjusted_alpha)

        result = McNemarResult(
            model_a=model_a,
            model_b=model_b,
            test_statistic=float(statistic),
            p_value=float(p_value),
            adjusted_alpha=adjusted_alpha,
            is_significant=is_significant,
        )
        results.append(result)

        logger.info(
            "  McNemar %s vs %s: χ²=%.4f, p=%.6f, α_adj=%.6f → %s",
            model_a,
            model_b,
            statistic,
            p_value,
            adjusted_alpha,
            "SIGNIFICANT" if is_significant else "not significant",
        )

    return results


def evaluate_baselines(
    df: pd.DataFrame,
    model_metrics: Dict[str, ModelMetrics],
) -> Dict[str, BaselineComparison]:
    """Compare trained models against baselines (R19-AC5).

    Baselines:
      1. Random baseline: AUC = 0.5
      2. Majority-class: always predict no-surge (label=0)
      3. Single-feature predictors: logistic regression with each feature alone

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with FEATURE_COLUMNS, `surge_label`,
        `partition`, `excluded`.
    model_metrics : dict
        Mapping of model_name -> ModelMetrics from evaluation.

    Returns
    -------
    dict
        Mapping of model_name -> BaselineComparison for each trained model.
    """
    from sklearn.linear_model import LogisticRegression as LR

    logger.info("=" * 70)
    logger.info("BASELINE COMPARISONS (R19-AC5)")
    logger.info("=" * 70)

    # Extract test partition
    test_mask = (
        (df["partition"] == "test")
        & (~df["excluded"].astype(bool))
        & (df["surge_label"].notna())
    )
    train_mask = (
        (df["partition"] == "train")
        & (~df["excluded"].astype(bool))
        & (df["surge_label"].notna())
    )

    df_test = df.loc[test_mask]
    df_train = df.loc[train_mask]

    X_test = df_test[FEATURE_COLUMNS].values.astype(np.float64)
    y_test = df_test["surge_label"].values.astype(np.float64)
    X_train = df_train[FEATURE_COLUMNS].values.astype(np.float64)
    y_train = df_train["surge_label"].values.astype(np.float64)

    n_test = len(y_test)

    # Majority-class baseline: always predict 0 (no-surge)
    majority_pred = np.zeros(n_test)
    majority_accuracy = accuracy_score(y_test, majority_pred)
    # AUC-ROC for constant predictor is 0.5
    majority_auc = 0.5

    logger.info("  Majority-class baseline: accuracy=%.4f, AUC=%.4f", majority_accuracy, majority_auc)

    # Single-feature predictors (R19-AC5)
    single_feature_aucs: Dict[str, float] = {}

    for i, feature_name in enumerate(FEATURE_COLUMNS):
        try:
            X_train_single = X_train[:, i].reshape(-1, 1)
            X_test_single = X_test[:, i].reshape(-1, 1)

            # Simple logistic regression with one feature
            lr = LR(max_iter=1000, random_state=42, solver="lbfgs")
            lr.fit(X_train_single, y_train)

            if hasattr(lr, "predict_proba"):
                y_prob_single = lr.predict_proba(X_test_single)[:, 1]
            else:
                y_prob_single = lr.decision_function(X_test_single)

            if len(np.unique(y_test)) >= 2:
                auc_single = roc_auc_score(y_test, y_prob_single)
            else:
                auc_single = 0.5

            single_feature_aucs[feature_name] = auc_single
        except Exception as e:
            logger.warning("  Single-feature %s failed: %s", feature_name, e)
            single_feature_aucs[feature_name] = 0.5

    # Find best single-feature predictor
    best_single_feature = max(single_feature_aucs, key=single_feature_aucs.get)
    best_single_feature_auc = single_feature_aucs[best_single_feature]

    logger.info(
        "  Best single-feature predictor: %s (AUC=%.4f)",
        best_single_feature,
        best_single_feature_auc,
    )

    # Compare each trained model against baselines
    comparisons: Dict[str, BaselineComparison] = {}

    for model_name, metrics in model_metrics.items():
        beats_random = metrics.auc_roc > 0.5
        beats_majority = metrics.auc_roc > majority_auc
        beats_all_single = metrics.auc_roc > best_single_feature_auc

        comparison = BaselineComparison(
            model_name=model_name,
            model_auc_roc=metrics.auc_roc,
            random_baseline_auc=0.5,
            majority_class_accuracy=majority_accuracy,
            majority_class_auc=majority_auc,
            single_feature_aucs=single_feature_aucs,
            beats_random=beats_random,
            beats_majority=beats_majority,
            best_single_feature=best_single_feature,
            best_single_feature_auc=best_single_feature_auc,
            beats_all_single_features=beats_all_single,
        )
        comparisons[model_name] = comparison

        logger.info(
            "  %s (AUC=%.4f): beats_random=%s, beats_majority=%s, beats_single_feature=%s",
            model_name,
            metrics.auc_roc,
            beats_random,
            beats_majority,
            beats_all_single,
        )

    return comparisons


def validate_success_tiers(
    model_metrics: Dict[str, ModelMetrics],
) -> Dict[str, SuccessTierResult]:
    """Validate model AUC-ROC against success tiers (R21-AC1, AC2, AC3).

    Success tiers:
      - Minimum: AUC-ROC > 0.60
      - Target: AUC-ROC > 0.70
      - Stretch: AUC-ROC > 0.80

    Parameters
    ----------
    model_metrics : dict
        Mapping of model_name -> ModelMetrics.

    Returns
    -------
    dict
        Mapping of model_name -> SuccessTierResult.
    """
    logger.info("=" * 70)
    logger.info("SUCCESS TIER VALIDATION (R21)")
    logger.info("=" * 70)

    results: Dict[str, SuccessTierResult] = {}

    for model_name, metrics in model_metrics.items():
        auc = metrics.auc_roc
        achieves_minimum = auc > SUCCESS_TIER_MINIMUM
        achieves_target = auc > SUCCESS_TIER_TARGET
        achieves_stretch = auc > SUCCESS_TIER_STRETCH

        if achieves_stretch:
            tier = "stretch"
        elif achieves_target:
            tier = "target"
        elif achieves_minimum:
            tier = "minimum"
        else:
            tier = "below_minimum"

        result = SuccessTierResult(
            model_name=model_name,
            auc_roc=auc,
            achieves_minimum=achieves_minimum,
            achieves_target=achieves_target,
            achieves_stretch=achieves_stretch,
            tier_achieved=tier,
        )
        results[model_name] = result

        logger.info(
            "  %s: AUC=%.4f → tier=%s (min=%s, target=%s, stretch=%s)",
            model_name,
            auc,
            tier,
            "✓" if achieves_minimum else "✗",
            "✓" if achieves_target else "✗",
            "✓" if achieves_stretch else "✗",
        )

    # Overall pass/fail (R21-AC3)
    any_achieves_minimum = any(r.achieves_minimum for r in results.values())
    if any_achieves_minimum:
        logger.info("  ✓ PASS: At least one model achieves minimum success (AUC > 0.60)")
    else:
        logger.info("  ✗ FAIL: No model achieves minimum success (AUC > 0.60)")

    return results


def produce_final_summary(
    model_metrics: Dict[str, ModelMetrics],
    mcnemar_results: List[McNemarResult],
    baseline_comparisons: Dict[str, BaselineComparison],
    tier_results: Dict[str, SuccessTierResult],
    config: PipelineConfig,
    bootstrap_ci: Optional[Dict[str, BootstrapCI]] = None,
    phase1_vs_phase2: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
) -> FinalSummary:
    """Produce the final summary report JSON with pass/fail determination (R21-AC5).

    Generates a comprehensive summary containing:
      - Best model identity and AUC-ROC with CI
      - Success tier achieved
      - Phase 1 vs Phase 2 comparison result (if available)
      - Recommended operating configuration (threshold τ, weight w₂)
      - Statistical significance test results
      - Baseline comparison results

    Parameters
    ----------
    model_metrics : dict
        Mapping of model_name -> ModelMetrics.
    mcnemar_results : list of McNemarResult
        Pairwise McNemar's test results.
    baseline_comparisons : dict
        Mapping of model_name -> BaselineComparison.
    tier_results : dict
        Mapping of model_name -> SuccessTierResult.
    config : PipelineConfig
        Pipeline configuration for recommended operating point.
    bootstrap_ci : dict, optional
        Mapping of model_name -> BootstrapCI for confidence intervals.
    phase1_vs_phase2 : dict, optional
        Phase 1 vs Phase 2 comparison result (R21-AC4).
    output_dir : str, optional
        Directory for saving final_summary.json.
        Defaults to "data/processed/evaluation/".

    Returns
    -------
    FinalSummary
        Complete final summary with pass/fail determination.
    """
    logger.info("=" * 70)
    logger.info("PRODUCING FINAL SUMMARY REPORT (R21-AC5)")
    logger.info("=" * 70)

    # Identify best model by AUC-ROC (R21-AC1)
    best_model = max(model_metrics, key=lambda m: model_metrics[m].auc_roc)
    best_auc = model_metrics[best_model].auc_roc

    # Get CI for best model if available
    best_ci = None
    if bootstrap_ci and best_model in bootstrap_ci:
        ci = bootstrap_ci[best_model]
        best_ci = {
            "lower": ci.auc_roc.lower,
            "point_estimate": ci.auc_roc.point_estimate,
            "upper": ci.auc_roc.upper,
        }

    # Determine success tier (R21-AC1)
    best_tier = tier_results[best_model].tier_achieved

    # Overall pass/fail (R21-AC3)
    overall_pass = any(r.achieves_minimum for r in tier_results.values())

    # Recommended configuration
    recommended_config = {
        "threshold_tau": config.threshold_tau,
        "weight_w2": config.weight_sentiment,
    }

    summary = FinalSummary(
        best_model=best_model,
        best_auc_roc=best_auc,
        best_auc_roc_ci=best_ci,
        success_tier_achieved=best_tier,
        overall_pass=overall_pass,
        model_tiers=tier_results,
        mcnemar_results=mcnemar_results,
        baseline_comparisons=baseline_comparisons,
        phase1_vs_phase2=phase1_vs_phase2,
        recommended_config=recommended_config,
    )

    # Save to JSON (R21-AC5)
    out_dir = Path(output_dir) if output_dir else Path("data/processed/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json = _build_final_summary_json(summary)
    filepath = out_dir / "final_summary.json"
    filepath.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    logger.info("  Best model: %s (AUC=%.4f)", best_model, best_auc)
    if best_ci:
        logger.info(
            "  Best AUC-ROC CI: [%.4f, %.4f]", best_ci["lower"], best_ci["upper"]
        )
    logger.info("  Success tier achieved: %s", best_tier)
    logger.info("  Overall pass: %s", "PASS" if overall_pass else "FAIL")
    logger.info("  Final summary saved to: %s", filepath)
    logger.info("=" * 70)

    return summary


def _build_final_summary_json(summary: FinalSummary) -> Dict[str, Any]:
    """Build JSON-serialisable dict from FinalSummary.

    Parameters
    ----------
    summary : FinalSummary
        Complete final summary object.

    Returns
    -------
    dict
        JSON-serialisable dictionary.
    """
    # Model tiers
    model_tiers_json = {}
    for model_name, tier in summary.model_tiers.items():
        model_tiers_json[model_name] = {
            "auc_roc": tier.auc_roc,
            "achieves_minimum": tier.achieves_minimum,
            "achieves_target": tier.achieves_target,
            "achieves_stretch": tier.achieves_stretch,
            "tier_achieved": tier.tier_achieved,
        }

    # McNemar results
    mcnemar_json = []
    for r in summary.mcnemar_results:
        mcnemar_json.append({
            "model_a": r.model_a,
            "model_b": r.model_b,
            "test_statistic": r.test_statistic,
            "p_value": r.p_value,
            "adjusted_alpha": r.adjusted_alpha,
            "is_significant": r.is_significant,
        })

    # Baseline comparisons
    baselines_json = {}
    for model_name, comp in summary.baseline_comparisons.items():
        baselines_json[model_name] = {
            "model_auc_roc": comp.model_auc_roc,
            "random_baseline_auc": comp.random_baseline_auc,
            "majority_class_accuracy": comp.majority_class_accuracy,
            "majority_class_auc": comp.majority_class_auc,
            "best_single_feature": comp.best_single_feature,
            "best_single_feature_auc": comp.best_single_feature_auc,
            "beats_random": comp.beats_random,
            "beats_majority": comp.beats_majority,
            "beats_all_single_features": comp.beats_all_single_features,
            "single_feature_aucs": comp.single_feature_aucs,
        }

    output = {
        "best_model": summary.best_model,
        "best_auc_roc": summary.best_auc_roc,
        "best_auc_roc_ci": summary.best_auc_roc_ci,
        "success_tier_achieved": summary.success_tier_achieved,
        "overall_pass": summary.overall_pass,
        "model_tiers": model_tiers_json,
        "mcnemar_pairwise_tests": mcnemar_json,
        "baseline_comparisons": baselines_json,
        "phase1_vs_phase2": summary.phase1_vs_phase2,
        "recommended_config": summary.recommended_config,
    }

    return output
