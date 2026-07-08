"""Evaluation metrics and visualisation figures for surge prediction models.

Computes Precision, Recall, F1-score, and ROC-AUC on the held-out test
set. Produces publication-ready evaluation figures: confusion matrix heatmap,
ROC curve, and classification threshold sensitivity plot.

Also provides advanced statistical evaluation: McNemar's pairwise test,
baseline comparisons, bootstrap confidence intervals, success tier
validation, and final summary generation.

Requirements: R15 (Evaluation Metrics), R16 (Evaluation Visualisations),
              R19 (Statistical Significance), R21 (Success Tiers)
Design Decision: D12 — Evaluation output structure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from surge_pipeline.training import TrainingResult, predict

logger = logging.getLogger(__name__)


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
# Advanced evaluation dataclasses
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
    confusion_matrix: List[List[int]]
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
class BaselineComparison:
    """Comparison of a model against random and single-feature baselines."""

    model_name: str
    model_auc: float
    random_baseline_auc: float
    beats_random: bool
    single_feature_aucs: Dict[str, float]
    best_single_feature: str
    best_single_feature_auc: float
    improvement_over_best_single_feature: float


@dataclass
class MetricCI:
    """Bootstrap confidence interval for a single metric."""

    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95


@dataclass
class BootstrapCI:
    """Bootstrap confidence intervals for all metrics of a model."""

    model_name: str
    n_resamples: int
    metrics: List[MetricCI]


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
class FinalSummary:
    """Final evaluation summary encompassing all analysis results."""

    best_model: str
    best_auc_roc: float
    overall_pass: bool
    success_tier_achieved: str
    model_metrics: Dict[str, Any]
    mcnemar_results: List[Dict[str, Any]]
    baseline_comparisons: Dict[str, Any]
    tier_results: Dict[str, Any]
    recommended_config: Dict[str, Any]
    phase1_vs_phase2: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# McNemar's pairwise test (R19)
# ---------------------------------------------------------------------------


def mcnemar_pairwise_test(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> List[McNemarResult]:
    """Perform McNemar's pairwise test between all model pairs.

    Builds a 2×2 contingency table for each pair of models based on
    record-level correct/incorrect predictions, then applies McNemar's
    exact test with Bonferroni correction.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    predictions : Dict[str, np.ndarray]
        Mapping of model_name -> predicted binary labels.
    alpha : float
        Significance level before Bonferroni correction.

    Returns
    -------
    List[McNemarResult]
        One result per unique pair of models.
    """
    model_names = sorted(predictions.keys())
    pairs = list(combinations(model_names, 2))
    n_comparisons = len(pairs)
    adjusted_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha

    results: List[McNemarResult] = []

    for name_a, name_b in pairs:
        pred_a = predictions[name_a]
        pred_b = predictions[name_b]

        # Record-level correct/incorrect
        correct_a = (pred_a == y_true).astype(int)
        correct_b = (pred_b == y_true).astype(int)

        # 2×2 contingency: b (A correct, B wrong), c (A wrong, B correct)
        b = int(np.sum((correct_a == 1) & (correct_b == 0)))  # A right, B wrong
        c = int(np.sum((correct_a == 0) & (correct_b == 1)))  # A wrong, B right

        # McNemar's test: if b + c == 0, models are identical
        if b + c == 0:
            test_stat = 0.0
            p_value = 1.0
        else:
            # Use exact binomial test (more appropriate for small b+c)
            # Under H0, b ~ Binomial(b+c, 0.5)
            # Two-sided p-value
            n_discordant = b + c
            test_stat = float((b - c) ** 2) / (b + c)
            # Use chi-squared approximation with continuity correction
            # or exact binomial for small counts
            if n_discordant < 25:
                # Exact binomial test
                p_value = float(
                    stats.binom_test(b, n_discordant, 0.5)
                    if hasattr(stats, "binom_test")
                    else stats.binomtest(b, n_discordant, 0.5).pvalue
                )
            else:
                # Chi-squared approximation (McNemar's chi-squared)
                p_value = float(1.0 - stats.chi2.cdf(test_stat, df=1))

        is_significant = p_value < adjusted_alpha

        results.append(McNemarResult(
            model_a=name_a,
            model_b=name_b,
            test_statistic=test_stat,
            p_value=p_value,
            adjusted_alpha=adjusted_alpha,
            is_significant=is_significant,
        ))

    return results


# ---------------------------------------------------------------------------
# Baseline comparisons
# ---------------------------------------------------------------------------


def evaluate_baselines(
    df: pd.DataFrame,
    model_metrics: Dict[str, ModelMetrics],
) -> Dict[str, BaselineComparison]:
    """Compare each model against random and single-feature baselines.

    For each of the 9 features, trains a single-feature Logistic Regression
    on the training partition and evaluates AUC on the test partition.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with features, partition, excluded, and surge_label columns.
    model_metrics : Dict[str, ModelMetrics]
        Metrics for each trained model.

    Returns
    -------
    Dict[str, BaselineComparison]
        Mapping model_name -> BaselineComparison.
    """
    from surge_pipeline.features import FEATURE_COLUMNS

    # Split into train/test
    train_mask = (df["partition"] == "train") & (~df["excluded"].astype(bool))
    test_mask = (df["partition"] == "test") & (~df["excluded"].astype(bool))

    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]

    y_train = train_df["surge_label"].values.astype(np.int64)
    y_test = test_df["surge_label"].values.astype(np.int64)

    # Compute single-feature AUCs
    single_feature_aucs: Dict[str, float] = {}
    for feature in FEATURE_COLUMNS:
        X_train_f = train_df[[feature]].values.astype(np.float64)
        X_test_f = test_df[[feature]].values.astype(np.float64)

        # Standardise
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_f)
        X_test_scaled = scaler.transform(X_test_f)

        # Train single-feature LR
        lr = LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced"
        )
        lr.fit(X_train_scaled, y_train)

        # Predict probabilities
        y_prob = lr.predict_proba(X_test_scaled)[:, 1]

        # Compute AUC (handle single-class edge case)
        if len(np.unique(y_test)) < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y_test, y_prob))

        single_feature_aucs[feature] = auc

    # Find best single-feature baseline
    best_feature = max(single_feature_aucs, key=single_feature_aucs.get)
    best_feature_auc = single_feature_aucs[best_feature]

    # Build comparison for each model
    comparisons: Dict[str, BaselineComparison] = {}
    for model_name, metrics in model_metrics.items():
        model_auc = metrics.auc_roc
        comparisons[model_name] = BaselineComparison(
            model_name=model_name,
            model_auc=model_auc,
            random_baseline_auc=0.5,
            beats_random=model_auc > 0.5,
            single_feature_aucs=single_feature_aucs.copy(),
            best_single_feature=best_feature,
            best_single_feature_auc=best_feature_auc,
            improvement_over_best_single_feature=model_auc - best_feature_auc,
        )

    return comparisons


# ---------------------------------------------------------------------------
# Success tier validation (R21)
# ---------------------------------------------------------------------------


def validate_success_tiers(
    model_metrics: Dict[str, ModelMetrics],
) -> Dict[str, SuccessTierResult]:
    """Classify each model into a success tier based on AUC-ROC.

    Tiers:
      - stretch: AUC > 0.80
      - target:  AUC > 0.70
      - minimum: AUC > 0.60
      - below_minimum: AUC <= 0.60

    Parameters
    ----------
    model_metrics : Dict[str, ModelMetrics]
        Metrics for each trained model.

    Returns
    -------
    Dict[str, SuccessTierResult]
        Mapping model_name -> SuccessTierResult.
    """
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

        results[model_name] = SuccessTierResult(
            model_name=model_name,
            auc_roc=auc,
            achieves_minimum=achieves_minimum,
            achieves_target=achieves_target,
            achieves_stretch=achieves_stretch,
            tier_achieved=tier,
        )

    return results


# ---------------------------------------------------------------------------
# Final summary generation
# ---------------------------------------------------------------------------


def produce_final_summary(
    model_metrics: Dict[str, ModelMetrics],
    mcnemar_results: List[McNemarResult],
    baseline_comparisons: Dict[str, BaselineComparison],
    tier_results: Dict[str, SuccessTierResult],
    config: "PipelineConfig",
    output_dir: str = "output/evaluation",
    phase1_vs_phase2: Optional[Dict[str, Any]] = None,
) -> FinalSummary:
    """Produce the final evaluation summary and save to JSON.

    Parameters
    ----------
    model_metrics : Dict[str, ModelMetrics]
        Metrics for each trained model.
    mcnemar_results : List[McNemarResult]
        McNemar's pairwise test results.
    baseline_comparisons : Dict[str, BaselineComparison]
        Baseline comparison results.
    tier_results : Dict[str, SuccessTierResult]
        Success tier classification per model.
    config : PipelineConfig
        Pipeline configuration used for this run.
    output_dir : str
        Output directory for the summary JSON.
    phase1_vs_phase2 : Dict, optional
        Phase 1 vs Phase 2 comparison data.

    Returns
    -------
    FinalSummary
        The assembled final summary.
    """
    from surge_pipeline.config import PipelineConfig

    # Identify best model by AUC-ROC
    best_model_name = max(model_metrics, key=lambda k: model_metrics[k].auc_roc)
    best_auc = model_metrics[best_model_name].auc_roc

    # Determine overall tier achieved (best among all models)
    best_tier_result = tier_results[best_model_name]
    overall_tier = best_tier_result.tier_achieved

    # Overall pass: any model exceeds minimum
    overall_pass = any(tr.achieves_minimum for tr in tier_results.values())

    # Build recommended config
    recommended_config = {
        "threshold_tau": config.threshold_tau,
        "weight_w2": config.weight_sentiment,
    }

    # Serialise sub-results
    metrics_dict = {
        name: asdict(m) for name, m in model_metrics.items()
    }
    mcnemar_list = [asdict(r) for r in mcnemar_results]
    baselines_dict = {
        name: asdict(b) for name, b in baseline_comparisons.items()
    }
    tiers_dict = {
        name: asdict(t) for name, t in tier_results.items()
    }

    summary = FinalSummary(
        best_model=best_model_name,
        best_auc_roc=best_auc,
        overall_pass=overall_pass,
        success_tier_achieved=overall_tier,
        model_metrics=metrics_dict,
        mcnemar_results=mcnemar_list,
        baseline_comparisons=baselines_dict,
        tier_results=tiers_dict,
        recommended_config=recommended_config,
        phase1_vs_phase2=phase1_vs_phase2,
    )

    # Save to JSON
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "final_summary.json"

    json_data = {
        "best_model": summary.best_model,
        "best_auc_roc": summary.best_auc_roc,
        "overall_pass": summary.overall_pass,
        "success_tier_achieved": summary.success_tier_achieved,
        "model_metrics": summary.model_metrics,
        "mcnemar_results": summary.mcnemar_results,
        "baseline_comparisons": summary.baseline_comparisons,
        "tier_results": summary.tier_results,
        "recommended_config": summary.recommended_config,
        "phase1_vs_phase2": summary.phase1_vs_phase2,
    }

    out_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    logger.info("Final summary saved to %s", out_path)

    return summary


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (Phase 2.3)
# ---------------------------------------------------------------------------


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> BootstrapCI:
    """Compute bootstrap confidence intervals for evaluation metrics.

    Resamples the test set with replacement and computes precision, recall,
    F1, and AUC-ROC on each resample to estimate 95% confidence intervals.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    model_name : str
        Model identifier.
    n_resamples : int
        Number of bootstrap resamples (default 1000).
    ci_level : float
        Confidence interval level (default 0.95).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    BootstrapCI
        Bootstrap confidence intervals for all metrics.
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)

    alpha = 1.0 - ci_level
    lower_pct = (alpha / 2) * 100
    upper_pct = (1.0 - alpha / 2) * 100

    # Storage for bootstrap metric distributions
    precisions = np.zeros(n_resamples)
    recalls = np.zeros(n_resamples)
    f1_scores = np.zeros(n_resamples)
    aucs = np.zeros(n_resamples)

    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]
        y_prob_boot = y_prob[idx]

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            precisions[i] = 0.0
            recalls[i] = 0.0
            f1_scores[i] = 0.0
            aucs[i] = 0.5
            continue

        precisions[i] = precision_score(y_true_boot, y_pred_boot, zero_division=0.0)
        recalls[i] = recall_score(y_true_boot, y_pred_boot, zero_division=0.0)
        f1_scores[i] = f1_score(y_true_boot, y_pred_boot, zero_division=0.0)
        aucs[i] = roc_auc_score(y_true_boot, y_prob_boot)

    # Point estimates from original data
    point_prec = float(precision_score(y_true, y_pred, zero_division=0.0))
    point_rec = float(recall_score(y_true, y_pred, zero_division=0.0))
    point_f1 = float(f1_score(y_true, y_pred, zero_division=0.0))
    if len(np.unique(y_true)) < 2:
        point_auc = 0.5
    else:
        point_auc = float(roc_auc_score(y_true, y_prob))

    metrics = [
        MetricCI(
            metric_name="precision",
            point_estimate=point_prec,
            ci_lower=float(np.percentile(precisions, lower_pct)),
            ci_upper=float(np.percentile(precisions, upper_pct)),
            ci_level=ci_level,
        ),
        MetricCI(
            metric_name="recall",
            point_estimate=point_rec,
            ci_lower=float(np.percentile(recalls, lower_pct)),
            ci_upper=float(np.percentile(recalls, upper_pct)),
            ci_level=ci_level,
        ),
        MetricCI(
            metric_name="f1",
            point_estimate=point_f1,
            ci_lower=float(np.percentile(f1_scores, lower_pct)),
            ci_upper=float(np.percentile(f1_scores, upper_pct)),
            ci_level=ci_level,
        ),
        MetricCI(
            metric_name="auc_roc",
            point_estimate=point_auc,
            ci_lower=float(np.percentile(aucs, lower_pct)),
            ci_upper=float(np.percentile(aucs, upper_pct)),
            ci_level=ci_level,
        ),
    ]

    logger.info(
        "%s — Bootstrap CIs (%d resamples, %.0f%% level):",
        model_name, n_resamples, ci_level * 100,
    )
    for m in metrics:
        logger.info(
            "  %s: %.4f [%.4f, %.4f]",
            m.metric_name, m.point_estimate, m.ci_lower, m.ci_upper,
        )

    return BootstrapCI(
        model_name=model_name,
        n_resamples=n_resamples,
        metrics=metrics,
    )


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
