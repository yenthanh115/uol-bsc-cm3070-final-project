"""Evaluation metrics for surge prediction models.

Computes Precision, Recall, F1-score, and ROC-AUC on the held-out test set.
Provides advanced statistical evaluation: McNemar's pairwise test, baseline
comparisons, bootstrap confidence intervals, success tier validation, and
final summary generation.

Requirements: R15 (Evaluation Metrics), R19 (Statistical Significance),
              R21 (Success Tiers)
Design Decision: D12 — Evaluation output structure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from surge_pipeline.config import PipelineConfig
from surge_pipeline.evaluation_models import (
    SUCCESS_TIER_MINIMUM,
    SUCCESS_TIER_STRETCH,
    SUCCESS_TIER_TARGET,
    BaselineComparison,
    BootstrapCI,
    EvaluationMetrics,
    FeatureImportanceResult,
    FinalSummary,
    McNemarResult,
    MetricCI,
    ModelMetrics,
    SuccessTierResult,
    ThresholdResult,
)
from surge_pipeline.training import predict
from surge_pipeline.training_models import TrainedModel, TrainingResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification threshold tuning (P1)
# ---------------------------------------------------------------------------


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    strategy: str = "max_f1",
    precision_floor: float = 0.10,
) -> ThresholdResult:
    """Find the optimal classification probability threshold.

    Sweeps thresholds from 0.01 to 0.99 and selects the best operating
    point based on the chosen strategy. This should be called on a
    **validation set** (not the test set) to avoid optimistic bias.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    model_name : str
        Model identifier for the result.
    strategy : str
        Threshold selection strategy:
        - "max_f1": Maximise F1-score (default).
        - "precision_floor": Find the lowest threshold where
          precision >= `precision_floor`, then maximise F1 among
          those candidates.
    precision_floor : float
        Minimum acceptable precision (only used with "precision_floor"
        strategy). Default is 0.10 (10%).

    Returns
    -------
    ThresholdResult
        Contains the optimal threshold and metrics at both the optimal
        and default (0.5) thresholds for comparison.
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    precisions = np.empty(len(thresholds))
    recalls = np.empty(len(thresholds))
    f1_scores = np.empty(len(thresholds))

    for i, t in enumerate(thresholds):
        y_pred_t = (y_prob >= t).astype(int)
        precisions[i] = precision_score(y_true, y_pred_t, zero_division=0.0)
        recalls[i] = recall_score(y_true, y_pred_t, zero_division=0.0)
        f1_scores[i] = f1_score(y_true, y_pred_t, zero_division=0.0)

    if strategy == "precision_floor":
        # Find candidates where precision >= floor
        valid_mask = precisions >= precision_floor
        if valid_mask.any():
            # Among valid candidates, maximise F1
            valid_f1 = np.where(valid_mask, f1_scores, -1.0)
            best_idx = int(np.argmax(valid_f1))
        else:
            # No threshold achieves the precision floor; fall back to max F1
            logger.warning(
                "%s: No threshold achieves precision >= %.2f. "
                "Falling back to max F1 strategy.",
                model_name, precision_floor,
            )
            best_idx = int(np.argmax(f1_scores))
    else:
        # Default: max F1
        best_idx = int(np.argmax(f1_scores))

    optimal_threshold = float(thresholds[best_idx])

    # Metrics at default threshold (0.5)
    y_pred_default = (y_prob >= 0.5).astype(int)
    prec_default = float(precision_score(y_true, y_pred_default, zero_division=0.0))
    rec_default = float(recall_score(y_true, y_pred_default, zero_division=0.0))
    f1_default = float(f1_score(y_true, y_pred_default, zero_division=0.0))

    return ThresholdResult(
        model_name=model_name,
        optimal_threshold=optimal_threshold,
        strategy=strategy,
        precision_at_threshold=float(precisions[best_idx]),
        recall_at_threshold=float(recalls[best_idx]),
        f1_at_threshold=float(f1_scores[best_idx]),
        precision_at_default=prec_default,
        recall_at_default=rec_default,
        f1_at_default=f1_default,
    )


# ---------------------------------------------------------------------------
# McNemar's pairwise test (R19)
# ---------------------------------------------------------------------------


def mcnemar_pairwise_test(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> list[McNemarResult]:
    """Perform McNemar's pairwise test between all model pairs.

    Builds a 2x contingency table for each pair of models based on
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

    results: list[McNemarResult] = []

    for name_a, name_b in pairs:
        pred_a = predictions[name_a]
        pred_b = predictions[name_b]

        # Record-level correct/incorrect
        correct_a = (pred_a == y_true).astype(int)
        correct_b = (pred_b == y_true).astype(int)

        # 2x contingency: b (A correct, B wrong), c (A wrong, B correct)
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
                # Exact binomial test (SciPy >= 1.7 binomtest, removes deprecated binom_test)
                p_value = float(stats.binomtest(b, n_discordant, 0.5).pvalue)
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
    model_metrics: dict[str, ModelMetrics],
) -> dict[str, BaselineComparison]:
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
    single_feature_aucs: dict[str, float] = {}
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
    comparisons: dict[str, BaselineComparison] = {}
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
    model_metrics: dict[str, ModelMetrics],
) -> dict[str, SuccessTierResult]:
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
    results: dict[str, SuccessTierResult] = {}

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
    model_metrics: dict[str, ModelMetrics],
    mcnemar_results: list[McNemarResult],
    baseline_comparisons: dict[str, BaselineComparison],
    tier_results: dict[str, SuccessTierResult],
    config: PipelineConfig,
    output_dir: str = "output/evaluation",
    phase1_vs_phase2: dict[str, Any] | None = None,
    timestamp_prefix: str | None = None,
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
    timestamp_prefix : str, optional
        YYYYMMDDHHMM prefix for the output filename. If provided, the file
        is saved as ``<prefix>_final_summary.json``.

    Returns
    -------
    FinalSummary
        The assembled final summary.
    """

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
    filename = f"{timestamp_prefix}_final_summary.json" if timestamp_prefix else "final_summary.json"
    out_path = out_dir / filename

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
            confidence_level=ci_level,
        ),
        MetricCI(
            metric_name="recall",
            point_estimate=point_rec,
            ci_lower=float(np.percentile(recalls, lower_pct)),
            ci_upper=float(np.percentile(recalls, upper_pct)),
            confidence_level=ci_level,
        ),
        MetricCI(
            metric_name="f1",
            point_estimate=point_f1,
            ci_lower=float(np.percentile(f1_scores, lower_pct)),
            ci_upper=float(np.percentile(f1_scores, upper_pct)),
            confidence_level=ci_level,
        ),
        MetricCI(
            metric_name="auc_roc",
            point_estimate=point_auc,
            ci_lower=float(np.percentile(aucs, lower_pct)),
            ci_upper=float(np.percentile(aucs, upper_pct)),
            confidence_level=ci_level,
        ),
    ]

    logger.info(
        "%s - Bootstrap CIs (%d resamples, %.0f%% level):",
        model_name, n_resamples, ci_level * 100,
    )
    for m in metrics:
        logger.info(
            "  %s: %.4f [%.4f, %.4f]",
            m.metric_name, m.point_estimate, m.ci_lower, m.ci_upper,
        )

    return BootstrapCI(
        model_name=model_name,
        n_bootstrap=n_resamples,
        metric_cis=metrics,
    )


def evaluate_model(
    result: TrainingResult,
    df: pd.DataFrame,
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


def evaluate_trained_model(
    trained_model: TrainedModel,
    df: pd.DataFrame,
    partition: str = "test",
) -> EvaluationMetrics:
    """Evaluate a TrainedModel (from train_models) on a data partition.

    This is the multi-model counterpart of evaluate_model, which accepts
    the legacy TrainingResult. Use this when working with results from
    train_models / TrainingPipelineResult.

    Parameters
    ----------
    trained_model : TrainedModel
        Trained model container from _train_single_model.
    df : pd.DataFrame
        Full dataset with features computed.
    partition : str
        Which partition to evaluate ('test' or 'train').

    Returns
    -------
    EvaluationMetrics
        Computed metrics for the model on the given partition.
    """
    from surge_pipeline.features import FEATURE_COLUMNS

    mask = (df["partition"] == partition) & (~df["excluded"].astype(bool))
    subset = df.loc[mask]

    X = subset[FEATURE_COLUMNS].values.astype(np.float64)
    y_true = subset["surge_label"].values.astype(np.int64)

    X_scaled = trained_model.scaler.transform(X)
    y_pred = trained_model.model.predict(X_scaled)
    y_prob = trained_model.model.predict_proba(X_scaled)[:, 1]

    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        logger.warning(
            "Only one class present in %s partition for %s. "
            "ROC-AUC is undefined; setting to 0.5.",
            partition,
            trained_model.name,
        )
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, y_prob))

    prec = float(precision_score(y_true, y_pred, zero_division=0.0))
    rec = float(recall_score(y_true, y_pred, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    support_pos = int(y_true.sum())
    support_neg = int(len(y_true) - support_pos)

    metrics = EvaluationMetrics(
        model_name=trained_model.name,
        partition=partition,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=auc,
        support_positive=support_pos,
        support_negative=support_neg,
        confusion_matrix=cm,
    )

    logger.info("=" * 60)
    logger.info("EVALUATION: %s on %s partition", trained_model.name, partition)
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
    metrics_list: list[EvaluationMetrics],
    output_dir: str = "output/evaluation",
    timestamp_prefix: str | None = None,
) -> Path:
    """Save evaluation metrics to a JSON file.

    Parameters
    ----------
    metrics_list : List[EvaluationMetrics]
        List of evaluation metric objects.
    output_dir : str
        Output directory for the results file.
    timestamp_prefix : str, optional
        YYYYMMDDHHMM prefix for the output filename. If provided, the file
        is saved as ``<prefix>_evaluation_metrics.json``.

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

    filename = f"{timestamp_prefix}_evaluation_metrics.json" if timestamp_prefix else "evaluation_metrics.json"
    out_path = out_dir / filename
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Evaluation results saved to %s", out_path)

    return out_path


# ---------------------------------------------------------------------------
# Feature importance (P3)
# ---------------------------------------------------------------------------


def compute_feature_importance(
    model,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    feature_names: list[str],
    n_repeats: int = 10,
    random_seed: int = 42,
) -> FeatureImportanceResult:
    """Compute permutation importance on the test set.

    Measures the decrease in AUC-ROC when each feature is randomly
    shuffled, repeated n_repeats times for stability.

    Parameters
    ----------
    model : estimator
        Trained sklearn-compatible model with predict_proba.
    scaler : StandardScaler
        Scaler fitted on training data (applied to X_test before scoring).
    X_test : np.ndarray
        Raw (unscaled) test feature matrix.
    y_test : np.ndarray
        True binary labels for the test set.
    model_name : str
        Model identifier for the result.
    feature_names : List[str]
        Names of the features (matching columns of X_test).
    n_repeats : int
        Number of shuffles per feature (default 10).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    FeatureImportanceResult
        Permutation importance values per feature.
    """
    from sklearn.inspection import permutation_importance

    # Scale test data using the already-fitted scaler.
    # permutation_importance shuffles columns of X then scores -
    # we pass scaled data so shuffling happens in scaled space.
    # This is standard practice: the model expects scaled input.
    X_test_scaled = scaler.transform(X_test)

    result = permutation_importance(
        model,
        X_test_scaled,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_seed,
        n_jobs=-1,
    )

    return FeatureImportanceResult(
        model_name=model_name,
        method="permutation",
        feature_names=list(feature_names),
        importances=[float(x) for x in result.importances_mean],
        importances_std=[float(x) for x in result.importances_std],
        scoring="roc_auc",
    )


def compute_builtin_importance(
    model,
    model_name: str,
    feature_names: list[str],
) -> FeatureImportanceResult | None:
    """Extract built-in feature importances (gain-based) if available.

    Works for tree-based models (RF, XGBoost) that expose
    `feature_importances_`. Returns None for models without this attribute.

    Parameters
    ----------
    model : estimator
        Trained model.
    model_name : str
        Model identifier.
    feature_names : List[str]
        Feature names matching the model's input.

    Returns
    -------
    Optional[FeatureImportanceResult]
        Built-in importances, or None if not available.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    return FeatureImportanceResult(
        model_name=model_name,
        method="builtin_gain",
        feature_names=list(feature_names),
        importances=[float(x) for x in importances],
        importances_std=[0.0] * len(importances),  # no std for built-in
        scoring="gain",
    )


# ---------------------------------------------------------------------------
# Re-export figure functions for backward compatibility
# ---------------------------------------------------------------------------

from surge_pipeline.evaluation_figures import (  # noqa: E402, F401
    generate_evaluation_figures,
    plot_classification_threshold_sensitivity,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    plot_roc_curve_combined,
)
