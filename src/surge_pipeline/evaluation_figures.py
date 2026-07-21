"""Evaluation visualisation figures for surge prediction models.

Produces publication-ready evaluation figures: confusion matrix heatmap,
ROC curve, classification threshold sensitivity plot, and feature
importance comparison chart.

Requirements: R16 (Evaluation Visualisations)
Design Decision: D12 — Evaluation output structure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from surge_pipeline.evaluation_models import FeatureImportanceResult

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
    ax.set_xlim((-0.02, 1.02))
    ax.set_ylim((-0.02, 1.02))

    return _save_figure(fig, f"11_roc_curve_{model_name}", figures_dir)


def plot_roc_curve_combined(
    results: list[tuple[str, np.ndarray, np.ndarray]],
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
    ax.set_xlim((-0.02, 1.02))
    ax.set_ylim((-0.02, 1.02))

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
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((-0.02, 1.05))

    return _save_figure(
        fig, f"12_classification_threshold_sensitivity_{model_name}", figures_dir
    )


# ---------------------------------------------------------------------------
# Feature importance figure (P3)
# ---------------------------------------------------------------------------


def plot_feature_importance(
    results: list[FeatureImportanceResult],
    figures_dir: Path | None = None,
) -> Path:
    """Generate a grouped horizontal bar chart comparing feature importance.

    Shows permutation importance (mean decrease in AUC) for all models
    side-by-side, features sorted by the best model's importance.

    Parameters
    ----------
    results : List[FeatureImportanceResult]
        Permutation importance results for each model.
    figures_dir : Path, optional
        Override output directory.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    _setup_plot_style()

    if not results:
        raise ValueError("No results to plot.")

    # Sort features by the first model's importance (descending)
    primary = results[0]
    sort_idx = np.argsort(primary.importances)  # ascending for barh
    feature_names = [primary.feature_names[i] for i in sort_idx]

    n_features = len(feature_names)
    n_models = len(results)
    bar_height = 0.8 / n_models
    colors = ["#2196F3", "#4CAF50", "#FF9800"]  # blue, green, orange

    fig, ax = plt.subplots(figsize=(9, 6))

    for j, res in enumerate(results):
        importances_sorted = [res.importances[i] for i in sort_idx]
        stds_sorted = [res.importances_std[i] for i in sort_idx]
        y_positions = np.arange(n_features) + j * bar_height

        ax.barh(
            y_positions,
            importances_sorted,
            height=bar_height,
            xerr=stds_sorted,
            label=res.model_name.replace("_", " ").title(),
            color=colors[j % len(colors)],
            alpha=0.85,
            capsize=2,
        )

    ax.set_yticks(np.arange(n_features) + bar_height * (n_models - 1) / 2)
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_xlabel("Mean Decrease in AUC-ROC (Permutation Importance)")
    ax.set_title("Feature Importance Comparison (Permutation, test set)")
    ax.legend(loc="lower right")
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="-")

    plt.tight_layout()

    return _save_figure(fig, "13_feature_importance_comparison", figures_dir)


# ---------------------------------------------------------------------------
# Convenience: generate all standard figures for a single model
# ---------------------------------------------------------------------------


def generate_evaluation_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    figures_dir: Path | None = None,
) -> list[Path]:
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
