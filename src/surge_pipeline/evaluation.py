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
