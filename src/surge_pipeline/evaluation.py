"""Evaluation metrics and visualisation figures for surge prediction models.

Computes Precision, Recall, F1-score, and ROC-AUC on the held-out test
set. Produces publication-ready evaluation figures: confusion matrix heatmap,
ROC curve, and classification threshold sensitivity plot.

Requirements: R15 (Evaluation Metrics), R16 (Evaluation Visualisations)
Design Decision: D12 — Evaluation output structure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from surge_pipeline.training import TrainingResult, predict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Figure configuration — matches eda_pipeline.py style
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = _PROJECT_ROOT / "output" / "figures" / "evaluation"
DPI = 300
FIG_FORMAT = "png"


def _setup_plot_style() -> None:
    """Apply publication-ready plot style (consistent with EDA figures)."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.family": "serif",
    })


def _save_figure(fig: plt.Figure, name: str, figures_dir: Path | None = None) -> Path:
    """Save a figure to the figures directory.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure to save.
    name : str
        Filename stem (without extension).
    figures_dir : Path, optional
        Override output directory (defaults to project figures/).

    Returns
    -------
    Path
        Path to the saved figure.
    """
    out_dir = figures_dir if figures_dir is not None else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{FIG_FORMAT}"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


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
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialisation."""
        return asdict(self)


def evaluate_model(
    result: TrainingResult,
    df: "pd.DataFrame",
    partition: str = "test",
) -> EvaluationMetrics:
    """Evaluate a trained model on a data partition.

    Computes precision, recall, F1, and ROC-AUC (R15-AC1).

    Parameters
    ----------
    result : TrainingResult
        Trained model result from training.
    df : pd.DataFrame
        Full dataset with features computed.
    partition : str
        Which partition to evaluate ('test' or 'train').

    Returns
    -------
    EvaluationMetrics
        Computed metrics for the model on the given partition.
    """
    import pandas as pd

    y_true, y_pred, y_prob = predict(result, df, partition=partition)

    # Handle edge cases (single-class test set)
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        logger.warning(
            "Only one class present in %s partition for %s. "
            "ROC-AUC is undefined; setting to 0.5.",
            partition,
            result.model_name,
        )
        auc = 0.5
    else:
        auc = roc_auc_score(y_true, y_prob)

    prec = precision_score(y_true, y_pred, zero_division=0.0)
    rec = recall_score(y_true, y_pred, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, zero_division=0.0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    support_pos = int(y_true.sum())
    support_neg = int(len(y_true) - support_pos)

    metrics = EvaluationMetrics(
        model_name=result.model_name,
        partition=partition,
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        roc_auc=float(auc),
        support_positive=support_pos,
        support_negative=support_neg,
        confusion_matrix=cm,
    )

    # Log metrics
    logger.info("=" * 60)
    logger.info("EVALUATION: %s on %s partition", result.model_name, partition)
    logger.info("=" * 60)
    logger.info("  Precision : %.4f", metrics.precision)
    logger.info("  Recall    : %.4f", metrics.recall)
    logger.info("  F1-score  : %.4f", metrics.f1)
    logger.info("  ROC-AUC   : %.4f", metrics.roc_auc)
    logger.info(
        "  Support   : positive=%d, negative=%d",
        metrics.support_positive,
        metrics.support_negative,
    )
    logger.info("  Confusion matrix:")
    logger.info("    TN=%d  FP=%d", cm[0][0], cm[0][1])
    logger.info("    FN=%d  TP=%d", cm[1][0], cm[1][1])

    return metrics


def save_evaluation_results(
    metrics_list: List[EvaluationMetrics],
    output_dir: str = "output/evaluation",
) -> Path:
    """Save evaluation metrics to a JSON file.

    Parameters
    ----------
    metrics_list : List[EvaluationMetrics]
        List of evaluation metric objects.
    output_dir : str
        Output directory for the results file.

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "models": [m.to_dict() for m in metrics_list],
        "summary": {
            "num_models": len(metrics_list),
            "best_model_by_auc": max(metrics_list, key=lambda m: m.roc_auc).model_name
            if metrics_list
            else None,
            "best_auc": max(m.roc_auc for m in metrics_list) if metrics_list else None,
        },
    }

    out_path = out_dir / "evaluation_metrics.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Evaluation results saved to %s", out_path)

    return out_path


# ---------------------------------------------------------------------------
# Evaluation Visualisation Figures (R16)
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    figures_dir: Path | None = None,
) -> Path:
    """Produce a confusion matrix heatmap figure (R16-AC1).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    model_name : str
        Model name for title and filename.
    figures_dir : Path, optional
        Override output directory.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    _setup_plot_style()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # Compute percentages for annotation
    cm_pct = cm.astype(float) / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 5))

    # Create heatmap with counts
    sns.heatmap(
        cm,
        annot=False,  # We'll add custom annotations
        fmt="d",
        cmap="Blues",
        xticklabels=["No Surge", "Surge"],
        yticklabels=["No Surge", "Surge"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    # Add custom annotations with count + percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.5,
                f"{cm[i, j]:d}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")

    return _save_figure(fig, f"10_confusion_matrix_{model_name}", figures_dir)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    figures_dir: Path | None = None,
) -> Path:
    """Produce an ROC curve figure with AUC annotation (R16-AC2, AC5).

    Supports overlaying multiple models when called with lists.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    model_name : str
        Model name for legend and filename.
    figures_dir : Path, optional
        Override output directory.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    _setup_plot_style()

    fig, ax = plt.subplots(figsize=(7, 6))

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    # Plot ROC curve
    ax.plot(
        fpr, tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {auc_value:.3f})",
        color="steelblue",
    )

    # Diagonal reference line (R16-AC5)
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", linewidth=1.2,
        color="gray", label="Random Baseline (AUC = 0.500)",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    return _save_figure(fig, f"11_roc_curve_{model_name}", figures_dir)


def plot_roc_curve_combined(
    results: List[Tuple[str, np.ndarray, np.ndarray]],
    figures_dir: Path | None = None,
) -> Path:
    """Produce a combined ROC curve with multiple models overlaid (R16-AC2).

    Parameters
    ----------
    results : list of (model_name, y_true, y_prob)
        Each tuple contains model name, true labels, and predicted probs.
    figures_dir : Path, optional
        Override output directory.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    _setup_plot_style()

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["steelblue", "darkorange", "forestgreen", "crimson", "purple"]

    for i, (name, y_true, y_prob) in enumerate(results):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        ax.plot(
            fpr, tpr,
            linewidth=2,
            label=f"{name} (AUC = {auc_value:.3f})",
            color=colors[i % len(colors)],
        )

    # Diagonal reference line
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", linewidth=1.2,
        color="gray", label="Random Baseline (AUC = 0.500)",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    return _save_figure(fig, "11_roc_curves_combined", figures_dir)


def plot_classification_threshold_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    figures_dir: Path | None = None,
) -> Path:
    """Produce a classification threshold sensitivity figure (R16-AC6).

    Shows precision, recall, and F1-score as functions of the decision
    probability threshold (0.01–0.99).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    model_name : str
        Model name for title and filename.
    figures_dir : Path, optional
        Override output directory.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    _setup_plot_style()

    thresholds = np.arange(0.01, 1.00, 0.01)
    precisions = []
    recalls = []
    f1_scores = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred_t, zero_division=0.0))
        recalls.append(recall_score(y_true, y_pred_t, zero_division=0.0))
        f1_scores.append(f1_score(y_true, y_pred_t, zero_division=0.0))

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)

    # Find optimal F1 threshold
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, precisions, linewidth=1.8, label="Precision", color="steelblue")
    ax.plot(thresholds, recalls, linewidth=1.8, label="Recall", color="darkorange")
    ax.plot(thresholds, f1_scores, linewidth=2.0, label="F1-score", color="forestgreen")

    # Mark default threshold (0.5)
    ax.axvline(
        x=0.5, linestyle="--", linewidth=1.0,
        color="gray", label="Default (0.5)",
    )

    # Mark optimal F1 threshold
    ax.axvline(
        x=best_threshold, linestyle=":", linewidth=1.2,
        color="red", label=f"Best F1 = {best_f1:.3f} @ t={best_threshold:.2f}",
    )
    ax.scatter(
        [best_threshold], [best_f1],
        color="red", s=80, zorder=5, marker="*",
    )

    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Classification Threshold Sensitivity — {model_name}")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.45))
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([-0.02, 1.05])

    return _save_figure(
        fig, f"12_classification_threshold_sensitivity_{model_name}", figures_dir
    )


def generate_evaluation_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    figures_dir: Path | None = None,
) -> List[Path]:
    """Generate all evaluation figures for a single model.

    Convenience function that calls all three plot functions.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    model_name : str
        Model name for titles and filenames.
    figures_dir : Path, optional
        Override output directory.

    Returns
    -------
    List[Path]
        Paths to the three saved figures.
    """
    n_classes = len(np.unique(y_true))
    paths = []

    # Confusion matrix (always possible)
    paths.append(plot_confusion_matrix(y_true, y_pred, model_name, figures_dir))

    # ROC curve and threshold sensitivity require two classes
    if n_classes < 2:
        logger.warning(
            "Only one class in y_true — skipping ROC curve and threshold "
            "sensitivity figures for %s.",
            model_name,
        )
        return paths

    paths.append(plot_roc_curve(y_true, y_prob, model_name, figures_dir))
    paths.append(
        plot_classification_threshold_sensitivity(y_true, y_prob, model_name, figures_dir)
    )

    return paths
