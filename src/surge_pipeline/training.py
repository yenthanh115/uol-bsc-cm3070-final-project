"""Model training with temporal cross-validation (multi-model).

Implements expanding-window temporal cross-validation for Logistic
Regression, Random Forest, and XGBoost models. Divides the training
partition into k=4 sequential folds and selects the best hyperparameters
by mean validation AUC-ROC.

Requirements: R13 (Model Training with Temporal Cross-Validation),
              R14 (Model Training Reproducibility)
Design Decision: D9 - Expanding-window temporal cross-validation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from surge_pipeline.config import PipelineConfig
from surge_pipeline.features import FEATURE_COLUMNS
from surge_pipeline.training_models import (
    CVResult,
    N_FOLDS,
    TrainedModel,
    TrainingPipelineResult,
    TrainingResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------


def _get_lr_param_grid() -> List[Dict[str, Any]]:
    """Logistic Regression grid: C(5) x l1_ratio(2) = 10 configs."""
    return [
        {"C": c, "l1_ratio": l1_ratio, "solver": "saga"}
        for c in [0.01, 0.1, 1.0, 10.0, 100.0]
        for l1_ratio in [0.0, 1.0]
    ]


def _get_rf_param_grid() -> List[Dict[str, Any]]:
    """Random Forest grid: n_estimators(3) x max_depth(4) x min_samples_leaf(3) = 36 configs."""
    return [
        {
            "n_estimators": n,
            "max_depth": d,
            "min_samples_leaf": m,
        }
        for n in [50, 100, 200]
        for d in [3, 5, 10, None]
        for m in [1, 2, 5]
    ]


def _get_xgb_param_grid(
    random_seed: int = 42, imbalance_ratio: float = 1.0
) -> List[Dict[str, Any]]:
    """XGBoost grid with scale_pos_weight for class imbalance handling (P6).

    n_estimators(3) x max_depth(3) x learning_rate(3) x scale_pos_weight(3) = 81 → capped at 75.

    The scale_pos_weight values are:
      - 1.0: no reweighting (baseline)
      - imbalance_ratio / 2: moderate reweighting
      - imbalance_ratio: full reweighting (equivalent to sklearn's 'balanced')

    Subsample fixed at 1.0 to make room for the weight dimension while
    keeping the grid manageable.
    """
    # Deduplicate weight values in case imbalance_ratio ≈ 1.0
    weight_values = sorted(set([1.0, imbalance_ratio / 2, imbalance_ratio]))

    grid = [
        {
            "n_estimators": n,
            "max_depth": d,
            "learning_rate": lr,
            "subsample": 1.0,
            "scale_pos_weight": w,
            "random_state": random_seed,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }
        for n in [50, 100, 200]
        for d in [3, 5, 7]
        for lr in [0.01, 0.1, 0.3]
        for w in weight_values
    ]
    return grid[:75]


# ---------------------------------------------------------------------------
# Temporal fold creation
# ---------------------------------------------------------------------------


def create_temporal_folds(
    df: pd.DataFrame, n_folds: int = N_FOLDS
) -> List[np.ndarray]:
    """Create n_folds sequential groups from the dataframe by timestamp order.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'created_utc' column for temporal ordering.
    n_folds : int
        Number of sequential folds (default N_FOLDS=4).

    Returns
    -------
    List[np.ndarray]
        List of n_folds arrays, each containing positional indices for that fold.
        Folds are in chronological order.
    """
    from surge_pipeline.timestamps import to_epoch_seconds
    epoch_seconds = pd.Series(to_epoch_seconds(df["created_utc"]), index=df.index)
    sorted_indices = epoch_seconds.values.argsort()
    n = len(sorted_indices)

    fold_size = n // n_folds
    folds: List[np.ndarray] = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n
        folds.append(sorted_indices[start:end])

    return folds


def get_expanding_window_splits(
    folds: List[np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Convert sequential folds into expanding-window train/val splits.

    For k folds, produces k-1 splits where split i uses folds 0..i as
    training and fold i+1 as validation.

    Parameters
    ----------
    folds : List[np.ndarray]
        Sequential fold indices from create_temporal_folds.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) tuples.
    """
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, len(folds)):
        train_idx = np.concatenate(folds[:i])
        val_idx = folds[i]
        splits.append((train_idx, val_idx))
    return splits


def _verify_temporal_ordering(
    df: pd.DataFrame,
    folds: List[np.ndarray],
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> None:
    """Verify that all splits respect chronological ordering.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'created_utc' column.
    folds : List[np.ndarray]
        Sequential fold indices.
    splits : List[Tuple[np.ndarray, np.ndarray]]
        Expanding-window splits.

    Raises
    ------
    ValueError
        If any split has max(train_ts) > min(val_ts).
    """
    from surge_pipeline.timestamps import to_epoch_seconds
    epoch = to_epoch_seconds(df["created_utc"])

    for i, (train_idx, val_idx) in enumerate(splits):
        max_train = epoch[train_idx].max()
        min_val = epoch[val_idx].min()
        if max_train > min_val:
            raise ValueError(
                f"Temporal ordering violated in split {i}: "
                f"max(train_ts)={max_train} > min(val_ts)={min_val}"
            )


# ---------------------------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------------------------


def _make_lr(params: Dict[str, Any], random_seed: int) -> LogisticRegression:
    """Create a LogisticRegression instance from params."""
    return LogisticRegression(
        C=params["C"],
        l1_ratio=params["l1_ratio"],
        solver=params["solver"],
        random_state=random_seed,
        max_iter=5000,
        class_weight="balanced",
    )


def _make_rf(params: Dict[str, Any], random_seed: int) -> RandomForestClassifier:
    """Create a RandomForestClassifier instance from params."""
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=random_seed,
        class_weight="balanced",
        n_jobs=-1,
    )


def _make_xgb(params: Dict[str, Any], random_seed: int):
    """Create an XGBClassifier instance from params.

    Passes scale_pos_weight for class imbalance handling (P6).
    """
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        scale_pos_weight=params.get("scale_pos_weight", 1.0),
        random_state=random_seed,
        eval_metric=params.get("eval_metric", "logloss"),
        use_label_encoder=False,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Single-model training logic
# ---------------------------------------------------------------------------


def _train_single_model(
    model_name: str,
    param_grid: List[Dict[str, Any]],
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    random_seed: int,
    make_model_fn,
) -> TrainedModel:
    """Train a single model type with temporal CV and grid search.

    Parameters
    ----------
    model_name : str
        Identifier for this model type.
    param_grid : List[Dict]
        Hyperparameter configurations to search.
    X_train_full : np.ndarray
        Full training feature matrix.
    y_train_full : np.ndarray
        Full training labels.
    splits : List[Tuple[np.ndarray, np.ndarray]]
        Expanding-window CV splits.
    random_seed : int
        Random seed for reproducibility.
    make_model_fn : callable
        Factory function (params, seed) -> sklearn estimator.

    Returns
    -------
    TrainedModel
        Trained model with metadata.
    """
    start_time = time.time()

    best_mean_auc: float = -1.0
    best_params: Dict[str, Any] = {}
    best_fold_scores: List[float] = []
    all_cv_results: List[CVResult] = []

    for params in tqdm(param_grid, desc=f"  {model_name}", unit="cfg", leave=True):
        fold_aucs: List[float] = []

        for train_idx, val_idx in splits:
            scaler = StandardScaler()
            X_fold_train = scaler.fit_transform(X_train_full[train_idx])
            X_fold_val = scaler.transform(X_train_full[val_idx])
            y_fold_train = y_train_full[train_idx]
            y_fold_val = y_train_full[val_idx]

            # Skip if validation set has only one class
            if len(np.unique(y_fold_val)) < 2:
                fold_aucs.append(0.5)
                continue

            model = make_model_fn(params, random_seed)
            model.fit(X_fold_train, y_fold_train)

            y_prob = model.predict_proba(X_fold_val)[:, 1]
            auc = roc_auc_score(y_fold_val, y_prob)
            fold_aucs.append(auc)

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))

        cv_result = CVResult(
            params=params.copy(),
            fold_scores=fold_aucs,
            mean_score=mean_auc,
            std_score=std_auc,
        )
        all_cv_results.append(cv_result)

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_params = params.copy()
            best_fold_scores = fold_aucs.copy()

    # Retrain on full training data with best params
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train_full)
    final_model = make_model_fn(best_params, random_seed)
    final_model.fit(X_train_scaled, y_train_full)

    # Generate validation predictions on the last CV fold for threshold tuning.
    # Use the final model's scaler (fitted on full training data) for consistency
    # with how test-set predictions will be generated.
    last_train_idx, last_val_idx = splits[-1]
    X_val_last = final_scaler.transform(X_train_full[last_val_idx])
    val_y_true = y_train_full[last_val_idx]
    val_y_prob = final_model.predict_proba(X_val_last)[:, 1]

    duration = time.time() - start_time

    logger.info(
        "%s — best CV AUC: %.4f (± %.4f) | params: %s | %.2fs",
        model_name, best_mean_auc, float(np.std(best_fold_scores)),
        best_params, duration,
    )

    # Store only the best CV result for the summary
    best_cv_result = CVResult(
        params=best_params,
        fold_scores=best_fold_scores,
        mean_score=best_mean_auc,
        std_score=float(np.std(best_fold_scores)),
    )

    return TrainedModel(
        name=model_name,
        model=final_model,
        scaler=final_scaler,
        best_params=best_params,
        best_cv_auc=best_mean_auc,
        cv_results=[best_cv_result],
        training_duration_seconds=duration,
        n_configs_evaluated=len(param_grid),
        val_y_true=val_y_true,
        val_y_prob=val_y_prob,
    )


# ---------------------------------------------------------------------------
# Multi-model training orchestrator
# ---------------------------------------------------------------------------


def train_models(
    df: pd.DataFrame,
    config: PipelineConfig,
    output_dir: str | None = None,
) -> TrainingPipelineResult:
    """Train all models (LR, RF, XGBoost) with temporal cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Full labelled dataset with features computed. Must contain
        'partition', 'excluded', 'surge_label', and all FEATURE_COLUMNS.
    config : PipelineConfig
        Pipeline configuration (random_seed, weight_sentiment used).
    output_dir : str, optional
        Directory to serialise trained models. Defaults to config.output_dir.

    Returns
    -------
    TrainingPipelineResult
        Contains all trained models, phase info, and metadata.
    """
    random_seed = config.random_seed
    phase = "phase1" if config.weight_sentiment == 0.0 else "phase2"
    out_dir = Path(output_dir) if output_dir else Path(config.output_dir)

    logger.info("=" * 60)
    logger.info("MULTI-MODEL TRAINING (phase=%s, seed=%d)", phase, random_seed)
    logger.info("=" * 60)

    # Prepare training data
    train_mask = (df["partition"] == "train") & (~df["excluded"].astype(bool))
    train_df = df.loc[train_mask].copy()

    X_train_full = train_df[FEATURE_COLUMNS].values.astype(np.float64)
    y_train_full = train_df["surge_label"].values.astype(np.int64)

    # Compute class imbalance ratio for scale_pos_weight (P6)
    n_positive = int(np.sum(y_train_full == 1))
    n_negative = int(np.sum(y_train_full == 0))
    imbalance_ratio = float(n_negative) / max(n_positive, 1)

    logger.info(
        "Training data: %d samples, %d features | surge=%.1f%% | "
        "imbalance_ratio=%.1f:1",
        X_train_full.shape[0], X_train_full.shape[1],
        y_train_full.mean() * 100, imbalance_ratio,
    )

    # Create temporal CV folds and splits
    folds = create_temporal_folds(train_df, n_folds=N_FOLDS)
    splits = get_expanding_window_splits(folds)
    _verify_temporal_ordering(train_df, folds, splits)

    # Train each model type
    models: Dict[str, TrainedModel] = {}

    # --- Logistic Regression ---
    logger.info("Training Logistic Regression...")
    models["logistic_regression"] = _train_single_model(
        "logistic_regression", _get_lr_param_grid(),
        X_train_full, y_train_full, splits, random_seed, _make_lr,
    )

    # --- Random Forest ---
    logger.info("Training Random Forest...")
    models["random_forest"] = _train_single_model(
        "random_forest", _get_rf_param_grid(),
        X_train_full, y_train_full, splits, random_seed, _make_rf,
    )

    # --- XGBoost (with scale_pos_weight grid for class imbalance — P6) ---
    logger.info("Training XGBoost...")
    models["xgboost"] = _train_single_model(
        "xgboost", _get_xgb_param_grid(random_seed, imbalance_ratio=imbalance_ratio),
        X_train_full, y_train_full, splits, random_seed, _make_xgb,
    )

    # Serialise models to disk (including optimal threshold from validation fold)
    from surge_pipeline.evaluation import find_optimal_threshold

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tm in models.items():
        # Select optimal threshold on last CV validation fold
        optimal_threshold = 0.5  # default fallback
        if tm.val_y_true is not None and tm.val_y_prob is not None:
            threshold_result = find_optimal_threshold(
                tm.val_y_true, tm.val_y_prob, model_name=name
            )
            optimal_threshold = threshold_result.optimal_threshold
            logger.info(
                "%s — optimal threshold: %.2f (F1=%.3f on validation fold)",
                name, optimal_threshold, threshold_result.f1_at_threshold,
            )

        model_path = out_dir / f"{name}_{phase}_{random_seed}.joblib"
        joblib.dump(
            {
                "model": tm.model,
                "scaler": tm.scaler,
                "params": tm.best_params,
                "optimal_threshold": optimal_threshold,
            },
            model_path,
        )
        logger.info("Saved model: %s", model_path)

    result = TrainingPipelineResult(
        models=models,
        phase=phase,
        random_seed=random_seed,
    )

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE — %d models trained", len(models))
    logger.info("=" * 60)

    return result


def get_training_summary(result: TrainingPipelineResult) -> Dict[str, Any]:
    """Generate a summary dict of the training pipeline result.

    Parameters
    ----------
    result : TrainingPipelineResult
        Result from train_models.

    Returns
    -------
    Dict[str, Any]
        Summary with per-model metrics and metadata.
    """
    summary: Dict[str, Any] = {
        "phase": result.phase,
        "random_seed": result.random_seed,
        "n_folds": result.n_folds,
        "models": {},
    }

    for name, tm in result.models.items():
        best_cv = tm.cv_results[0] if tm.cv_results else None
        summary["models"][name] = {
            "best_params": tm.best_params,
            "best_cv_auc": tm.best_cv_auc,
            "fold_scores": best_cv.fold_scores if best_cv else [],
            "training_duration_seconds": tm.training_duration_seconds,
            "n_configs_evaluated": tm.n_configs_evaluated,
        }

    return summary


# ---------------------------------------------------------------------------
# Backward-compatible single-model training (used by run_training.py)
# ---------------------------------------------------------------------------


def train_logistic_regression(
    df: pd.DataFrame,
    config: PipelineConfig,
    seed: int | None = None,
) -> TrainingResult:
    """Train Logistic Regression with temporal cross-validation.

    Backward-compatible wrapper that returns a TrainingResult.

    Parameters
    ----------
    df : pd.DataFrame
        Full labelled dataset with features computed.
    config : PipelineConfig
        Pipeline configuration.
    seed : int | None
        Random seed override.

    Returns
    -------
    TrainingResult
        Training result with the best model, params, and CV scores.
    """
    random_seed = seed if seed is not None else config.random_seed

    # Prepare training data
    train_mask = (df["partition"] == "train") & (~df["excluded"].astype(bool))
    train_df = df.loc[train_mask].copy()

    X_train_full = train_df[FEATURE_COLUMNS].values.astype(np.float64)
    y_train_full = train_df["surge_label"].values.astype(np.int64)

    # Create temporal CV folds and splits
    folds = create_temporal_folds(train_df, n_folds=N_FOLDS)
    splits = get_expanding_window_splits(folds)

    # Train LR
    trained = _train_single_model(
        "LogisticRegression", _get_lr_param_grid(),
        X_train_full, y_train_full, splits, random_seed, _make_lr,
    )

    best_cv = trained.cv_results[0]

    return TrainingResult(
        model_name="LogisticRegression",
        best_params=trained.best_params,
        cv_scores=best_cv.fold_scores,
        mean_cv_score=trained.best_cv_auc,
        std_cv_score=best_cv.std_score,
        training_duration_seconds=trained.training_duration_seconds,
        model=trained.model,
        scaler=trained.scaler,
    )


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------


def predict(
    result: TrainingResult,
    df: pd.DataFrame,
    partition: str = "test",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate predictions on a data partition.

    Parameters
    ----------
    result : TrainingResult
        Trained model result.
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


def predict_with_threshold(
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply a custom classification threshold to predicted probabilities.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    threshold : float
        Classification threshold (default 0.5). A sample is predicted
        positive if y_prob >= threshold.

    Returns
    -------
    np.ndarray
        Binary predictions (0 or 1).
    """
    return (y_prob >= threshold).astype(np.int64)
