"""Basic evaluation metrics for surge prediction models.

Computes Precision, Recall, F1-score, and ROC-AUC on the held-out test
set. Outputs structured metrics for reporting.

Requirements: R15 (Evaluation Metrics)
Design Decision: D12 — Evaluation output structure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from surge_pipeline.training import TrainingResult, predict

logger = logging.getLogger(__name__)


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
    cm = confusion_matrix(y_true, y_pred).tolist()

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
    output_dir: str = "data/processed/evaluation",
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
