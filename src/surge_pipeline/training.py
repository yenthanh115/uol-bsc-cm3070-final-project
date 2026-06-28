"""Model training with temporal cross-validation.

Implements expanding-window temporal cross-validation for a Logistic
Regression baseline model. Divides the training partition into k=4
sequential folds and selects the best hyperparameters by mean
validation AUC-ROC.

Requirements: R13 (Model Training with Temporal Cross-Validation),
              R14 (Model Training Reproducibility)
Design Decision: D9 - Expanding-window temporal cross-validation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from surge_pipeline.config import PipelineConfig
from surge_pipeline.features import FEATURE_COLUMNS, compute_features, get_feature_matrix

logger = logging.getLogger(__name__)

# Hyperparameter search space for Logistic Regression (R13-AC7)
# C ∈ {0.01, 0.1, 1, 10, 100} × l1_ratio ∈ {0.0 (L2), 1.0 (L1)} = 10 configs (≤50, R13-AC6)
# Uses l1_ratio API (sklearn ≥1.8): l1_ratio=0 → L2, l1_ratio=1 → L1.
LR_PARAM_GRID: List[Dict[str, Any]] = [
    {"C": c, "l1_ratio": l1_ratio, "solver": "saga"}
    for c in [0.01, 0.1, 1.0, 10.0, 100.0]
    for l1_ratio in [0.0, 1.0]
]


@dataclass
class CVFold:
    """Metadata for a single cross-validation fold."""

    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_max_ts: float
    val_min_ts: float


@dataclass
class TrainingResult:
    """Result from model training including best params and scores."""

    model_name: str
    best_params: Dict[str, Any]
    cv_scores: List[float]  # AUC-ROC per fold
    mean_cv_score: float
    std_cv_score: float
    training_duration_seconds: float
    model: Any  # trained sklearn model
    scaler: StandardScaler  # fitted scaler for feature normalisation
    feature_columns: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))


def create_temporal_cv_folds(
    df: pd.DataFrame, k: int = 4
) -> List[CVFold]:
    """Create k sequential folds from the training partition.

    Implements expanding-window temporal cross-validation (R13-AC2, AC3):
    - Divide training data into k sequential folds by timestamp order
    - For fold i (i=2,3,4): train on folds 1..i-1, validate on fold i
    - This produces k-1=3 train/validation splits

    Parameters
    ----------
    df : pd.DataFrame
        Training partition only (partition == 'train', excluded == False).
        Must have 'created_utc' column for temporal ordering.
    k : int
        Number of sequential folds (default 4, producing 3 splits).

    Returns
    -------
    List[CVFold]
        List of k-1 CVFold objects with train/val indices.
    """
    # Sort by timestamp to ensure temporal ordering
    epoch_seconds = pd.to_datetime(df["created_utc"], utc=True).astype("int64") // 10**9
    sorted_indices = epoch_seconds.values.argsort()
    n = len(sorted_indices)

    # Divide into k approximately equal folds
    fold_size = n // k
    folds_indices: List[np.ndarray] = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n
        folds_indices.append(sorted_indices[start:end])

    # Create expanding-window splits: train on 1..i-1, validate on fold i
    cv_folds: List[CVFold] = []
    for i in range(1, k):  # i = 1, 2, 3 (validate on folds 1, 2, 3)
        train_idx = np.concatenate(folds_indices[:i])
        val_idx = folds_indices[i]

        # Get timestamps for validation
        train_times = epoch_seconds.iloc[train_idx].values
        val_times = epoch_seconds.iloc[val_idx].values

        cv_folds.append(
            CVFold(
                fold_idx=i,
                train_indices=train_idx,
                val_indices=val_idx,
                train_max_ts=float(train_times.max()),
                val_min_ts=float(val_times.min()),
            )
        )

    # Log fold info
    for fold in cv_folds:
        logger.info(
            "CV Fold %d: train=%d samples, val=%d samples | "
            "max(train_ts)=%.0f <= min(val_ts)=%.0f ✓",
            fold.fold_idx,
            len(fold.train_indices),
            len(fold.val_indices),
            fold.train_max_ts,
            fold.val_min_ts,
        )
        # Verify temporal ordering (R13-AC3)
        assert fold.train_max_ts <= fold.val_min_ts, (
            f"Temporal violation in fold {fold.fold_idx}: "
            f"max(train_ts)={fold.train_max_ts} > min(val_ts)={fold.val_min_ts}"
        )

    return cv_folds


def train_logistic_regression(
    df: pd.DataFrame,
    config: PipelineConfig,
    seed: int | None = None,
) -> TrainingResult:
    """Train Logistic Regression with temporal cross-validation.

    Implements R13-AC1 (Logistic Regression), R13-AC4 (best by AUC-ROC),
    R13-AC5 (retrain on full training partition), R14-AC1 (seeded).

    Parameters
    ----------
    df : pd.DataFrame
        Full labelled dataset (with features computed). Must contain
        'partition', 'excluded', 'surge_label', and all FEATURE_COLUMNS.
    config : PipelineConfig
        Pipeline configuration.
    seed : int | None
        Random seed override. Uses config.random_seed if None.

    Returns
    -------
    TrainingResult
        Training result with the best model, params, and CV scores.
    """
    start_time = time.time()
    random_seed = seed if seed is not None else config.random_seed

    logger.info("=" * 60)
    logger.info("TRAINING: Logistic Regression (seed=%d)", random_seed)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Prepare training data (non-excluded, training partition only)
    # ------------------------------------------------------------------
    train_mask = (df["partition"] == "train") & (~df["excluded"].astype(bool))
    train_df = df.loc[train_mask].copy()

    X_train_full = train_df[FEATURE_COLUMNS].values.astype(np.float64)
    y_train_full = train_df["surge_label"].values.astype(np.int64)

    logger.info(
        "Training data: %d samples, %d features",
        X_train_full.shape[0],
        X_train_full.shape[1],
    )
    logger.info(
        "Class distribution: surge=%d (%.1f%%), no-surge=%d (%.1f%%)",
        y_train_full.sum(),
        y_train_full.mean() * 100,
        (1 - y_train_full).sum(),
        (1 - y_train_full).mean() * 100,
    )

    # ------------------------------------------------------------------
    # Create temporal CV folds (R13-AC2, AC3)
    # ------------------------------------------------------------------
    cv_folds = create_temporal_cv_folds(train_df, k=4)

    # ------------------------------------------------------------------
    # Hyperparameter search (R13-AC4, AC6, AC7)
    # ------------------------------------------------------------------
    logger.info(
        "Evaluating %d hyperparameter configurations...", len(LR_PARAM_GRID)
    )

    best_mean_auc: float = -1.0
    best_params: Dict[str, Any] = {}
    best_fold_scores: List[float] = []

    for params in LR_PARAM_GRID:
        fold_aucs: List[float] = []

        for fold in cv_folds:
            # Scale features (fit on fold training data only)
            scaler = StandardScaler()
            X_fold_train = scaler.fit_transform(X_train_full[fold.train_indices])
            X_fold_val = scaler.transform(X_train_full[fold.val_indices])
            y_fold_train = y_train_full[fold.train_indices]
            y_fold_val = y_train_full[fold.val_indices]

            # Skip if validation set has only one class
            if len(np.unique(y_fold_val)) < 2:
                fold_aucs.append(0.5)
                continue

            # Train model (using l1_ratio API for sklearn ≥1.8)
            model = LogisticRegression(
                C=params["C"],
                penalty="elasticnet",
                l1_ratio=params["l1_ratio"],
                solver=params["solver"],
                random_state=random_seed,
                max_iter=2000,
                class_weight="balanced",  # Handle imbalanced classes
            )
            model.fit(X_fold_train, y_fold_train)

            # Predict probabilities for AUC-ROC
            y_prob = model.predict_proba(X_fold_val)[:, 1]
            auc = roc_auc_score(y_fold_val, y_prob)
            fold_aucs.append(auc)

        mean_auc = np.mean(fold_aucs)
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_params = params.copy()
            best_fold_scores = fold_aucs.copy()

    logger.info(
        "Best hyperparameters: C=%.4f, l1_ratio=%.1f, solver=%s",
        best_params["C"],
        best_params["l1_ratio"],
        best_params["solver"],
    )
    logger.info(
        "Best mean CV AUC-ROC: %.4f (± %.4f) | Folds: %s",
        best_mean_auc,
        np.std(best_fold_scores),
        [f"{s:.4f}" for s in best_fold_scores],
    )

    # ------------------------------------------------------------------
    # Retrain on full training partition with best params (R13-AC5)
    # ------------------------------------------------------------------
    logger.info("Retraining on full training partition with best configuration...")

    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train_full)

    final_model = LogisticRegression(
        C=best_params["C"],
        penalty="elasticnet",
        l1_ratio=best_params["l1_ratio"],
        solver=best_params["solver"],
        random_state=random_seed,
        max_iter=2000,
        class_weight="balanced",
    )
    final_model.fit(X_train_scaled, y_train_full)

    duration = time.time() - start_time
    logger.info("Training complete in %.2f seconds.", duration)

    # R14-AC3: Log selected hyperparams and training duration
    result = TrainingResult(
        model_name="LogisticRegression",
        best_params=best_params,
        cv_scores=best_fold_scores,
        mean_cv_score=best_mean_auc,
        std_cv_score=float(np.std(best_fold_scores)),
        training_duration_seconds=duration,
        model=final_model,
        scaler=final_scaler,
    )

    return result


def predict(
    result: TrainingResult,
    df: pd.DataFrame,
    partition: str = "test",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate predictions on a data partition.

    Parameters
    ----------
    result : TrainingResult
        Trained model result from train_logistic_regression.
    df : pd.DataFrame
        Full dataset with features computed.
    partition : str
        Which partition to predict on ('test' or 'train').

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (y_true, y_pred, y_prob) — true labels, predicted classes,
        and predicted probabilities for the positive class.
    """
    mask = (df["partition"] == partition) & (~df["excluded"].astype(bool))
    subset = df.loc[mask]

    X = subset[FEATURE_COLUMNS].values.astype(np.float64)
    y_true = subset["surge_label"].values.astype(np.int64)

    X_scaled = result.scaler.transform(X)
    y_pred = result.model.predict(X_scaled)
    y_prob = result.model.predict_proba(X_scaled)[:, 1]

    return y_true, y_pred, y_prob
