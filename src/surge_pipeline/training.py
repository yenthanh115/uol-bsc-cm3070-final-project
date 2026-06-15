"""Model training with expanding-window temporal cross-validation.

Trains Logistic Regression, Random Forest, and XGBoost classifiers using
temporal cross-validation that respects chronological ordering. Supports
Phase 1 (w₂=0, volume-only labels) and Phase 2 (w₂=0.5, composite labels).

Requirements: R13 (Model Training with Temporal Cross-Validation),
              R14 (Model Training Reproducibility)
Design Decision: D9 — Expanding-window temporal cross-validation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from surge_pipeline.config import PipelineConfig
from surge_pipeline.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Number of temporal CV folds (k=4 sequential folds, 3 train/val splits)
N_FOLDS: int = 4


# =============================================================================
# Data classes for results
# =============================================================================


@dataclass
class CVResult:
    """Cross-validation result for one hyperparameter configuration."""

    params: Dict[str, Any]
    fold_scores: List[float]  # AUC-ROC per fold (3 values)
    mean_auc: float


@dataclass
class TrainedModel:
    """A trained model with associated metadata."""

    model_name: str
    model: Any  # The fitted sklearn/xgboost estimator
    best_params: Dict[str, Any]
    cv_results: List[CVResult]
    best_cv_auc: float
    training_duration_seconds: float
    n_configs_evaluated: int


@dataclass
class TrainingResult:
    """Complete training result for all models."""

    models: Dict[str, TrainedModel]
    phase: str  # "phase1" or "phase2"
    random_seed: int
    output_dir: str


# =============================================================================
# Hyperparameter search spaces (D9)
# =============================================================================


def _get_lr_param_grid() -> List[Dict[str, Any]]:
    """Logistic Regression: C × penalty = 10 configs (R13-AC7).

    C ∈ {0.01, 0.1, 1, 10, 100} × penalty ∈ {L1, L2}
    Uses l1_ratio: 1.0 for L1, 0.0 for L2 (scikit-learn ≥1.8 compatible).
    """
    configs = []
    for c_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
        for l1_ratio in [1.0, 0.0]:  # 1.0 = L1, 0.0 = L2
            configs.append({"C": c_val, "l1_ratio": l1_ratio})
    return configs


def _get_rf_param_grid() -> List[Dict[str, Any]]:
    """Random Forest: n_estimators × max_depth × min_samples_leaf = 36 configs (R13-AC8).

    n_estimators ∈ {100, 200, 500} × max_depth ∈ {5, 10, 20, None}
    × min_samples_leaf ∈ {1, 5, 10}
    """
    configs = []
    for n_est in [100, 200, 500]:
        for depth in [5, 10, 20, None]:
            for min_leaf in [1, 5, 10]:
                configs.append({
                    "n_estimators": n_est,
                    "max_depth": depth,
                    "min_samples_leaf": min_leaf,
                })
    return configs


def _get_xgb_param_grid(random_seed: int) -> List[Dict[str, Any]]:
    """XGBoost: n_estimators × max_depth × learning_rate × subsample = 36 configs (R13-AC9).

    n_estimators ∈ {100, 200, 500} × max_depth ∈ {3, 5, 7}
    × learning_rate ∈ {0.01, 0.1} × subsample ∈ {0.8, 1.0}

    Full grid = 3×3×2×2 = 36. If exceeds 50, subsample via random search.
    Design includes colsample_bytree but actual grid is 36 configs which is ≤50.
    """
    configs = []
    for n_est in [100, 200, 500]:
        for depth in [3, 5, 7]:
            for lr in [0.01, 0.1]:
                for subsample in [0.8, 1.0]:
                    configs.append({
                        "n_estimators": n_est,
                        "max_depth": depth,
                        "learning_rate": lr,
                        "subsample": subsample,
                    })

    # If exceeds 50, subsample (design note: 36 ≤ 50, so no subsampling needed)
    if len(configs) > 50:
        rng = np.random.RandomState(random_seed)
        indices = rng.choice(len(configs), size=50, replace=False)
        configs = [configs[i] for i in sorted(indices)]

    return configs


# =============================================================================
# Temporal cross-validation fold construction
# =============================================================================


def create_temporal_folds(
    df_train: pd.DataFrame, n_folds: int = N_FOLDS
) -> List[np.ndarray]:
    """Divide training partition into k sequential folds by timestamp.

    Records are sorted by `created_utc` and divided into k approximately
    equal-sized folds. This ensures temporal ordering within and between folds.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training partition records (excluded=False, partition='train').
        Must have `created_utc` column.
    n_folds : int
        Number of sequential folds (default 4).

    Returns
    -------
    list of np.ndarray
        Each element contains the integer index positions for that fold.
        Folds are in chronological order.
    """
    # Sort by timestamp to ensure temporal ordering
    sorted_indices = df_train["created_utc"].argsort().values
    n = len(sorted_indices)

    # Divide into approximately equal folds
    fold_sizes = np.array_split(sorted_indices, n_folds)
    folds = [fold for fold in fold_sizes]

    logger.info(
        "Created %d temporal folds: sizes = %s",
        n_folds,
        [len(f) for f in folds],
    )

    return folds


def get_expanding_window_splits(
    folds: List[np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window train/validation splits (R13-AC3).

    For fold i (i=2,3,4 in 1-indexed; i=1,2,3 in 0-indexed):
      train on folds 0..i-1, validate on fold i.

    This produces 3 train/validation splits with expanding training window.

    Parameters
    ----------
    folds : list of np.ndarray
        Sequential folds from create_temporal_folds.

    Returns
    -------
    list of (train_indices, val_indices)
        3 splits: [(fold0, fold1), (fold0+1, fold2), (fold0+1+2, fold3)]
    """
    splits = []
    for i in range(1, len(folds)):
        train_idx = np.concatenate(folds[:i])
        val_idx = folds[i]
        splits.append((train_idx, val_idx))

    logger.info(
        "Expanding-window splits: %d splits, train sizes = %s, val sizes = %s",
        len(splits),
        [len(s[0]) for s in splits],
        [len(s[1]) for s in splits],
    )

    return splits


# =============================================================================
# Model construction helpers
# =============================================================================


def _build_lr_model(params: Dict[str, Any], seed: int) -> LogisticRegression:
    """Build a Logistic Regression model with given hyperparameters."""
    return LogisticRegression(
        C=params["C"],
        l1_ratio=params["l1_ratio"],
        solver="saga",
        max_iter=5000,
        random_state=seed,
    )


def _build_rf_model(params: Dict[str, Any], seed: int) -> RandomForestClassifier:
    """Build a Random Forest model with given hyperparameters."""
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=seed,
        n_jobs=-1,
    )


def _build_xgb_model(params: Dict[str, Any], seed: int) -> Any:
    """Build an XGBoost model with given hyperparameters."""
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=1.0,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )


# =============================================================================
# Cross-validation evaluation
# =============================================================================


def _evaluate_config(
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    build_fn,
    params: Dict[str, Any],
    seed: int,
) -> CVResult:
    """Evaluate one hyperparameter config across all CV splits.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (training partition).
    y : np.ndarray
        Labels (training partition).
    splits : list of (train_idx, val_idx)
        Expanding-window splits.
    build_fn : callable
        Function to build a model: (params, seed) -> estimator.
    params : dict
        Hyperparameter configuration.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    CVResult
        Cross-validation result with per-fold AUC-ROC scores.
    """
    fold_scores = []

    for train_idx, val_idx in splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Skip if only one class in train or val
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            fold_scores.append(0.5)  # Chance-level if degenerate
            continue

        model = build_fn(params, seed)
        model.fit(X_train, y_train)

        # Predict probabilities for AUC-ROC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            y_prob = model.decision_function(X_val)

        try:
            auc = roc_auc_score(y_val, y_prob)
        except ValueError:
            auc = 0.5  # Fallback for edge cases

        fold_scores.append(auc)

    mean_auc = float(np.mean(fold_scores))

    return CVResult(params=params, fold_scores=fold_scores, mean_auc=mean_auc)


def _run_hyperparameter_search(
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    param_grid: List[Dict[str, Any]],
    build_fn,
    model_name: str,
    seed: int,
) -> List[CVResult]:
    """Run hyperparameter search over all configurations (R13-AC4, AC6).

    Evaluates at most 50 configurations per model type.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (training partition).
    y : np.ndarray
        Labels (training partition).
    splits : list
        Expanding-window CV splits.
    param_grid : list of dict
        Hyperparameter configurations to evaluate.
    build_fn : callable
        Model builder function.
    model_name : str
        Name for logging.
    seed : int
        Random seed.

    Returns
    -------
    list of CVResult
        Results sorted by mean AUC-ROC (descending).
    """
    assert len(param_grid) <= 50, (
        f"Max 50 configs per model, got {len(param_grid)} for {model_name}"
    )

    logger.info(
        "Hyperparameter search for %s: %d configurations × %d folds",
        model_name,
        len(param_grid),
        len(splits),
    )

    results = []
    for i, params in enumerate(param_grid):
        result = _evaluate_config(X, y, splits, build_fn, params, seed)
        results.append(result)

        if (i + 1) % 10 == 0 or (i + 1) == len(param_grid):
            logger.info(
                "  %s: evaluated %d/%d configs (best so far: %.4f)",
                model_name,
                i + 1,
                len(param_grid),
                max(r.mean_auc for r in results),
            )

    # Sort by mean AUC-ROC descending
    results.sort(key=lambda r: r.mean_auc, reverse=True)

    return results


# =============================================================================
# Main training function
# =============================================================================


def train_models(
    df: pd.DataFrame,
    config: PipelineConfig,
    output_dir: Optional[str] = None,
) -> TrainingResult:
    """Train all models with temporal cross-validation (R13, R14).

    Workflow:
      1. Filter to training partition (non-excluded records)
      2. Create k=4 temporal folds
      3. For each model: search hyperparameters via expanding-window CV
      4. Select best config by mean validation AUC-ROC
      5. Retrain best config on full training partition
      6. Serialise trained models to disk

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with FEATURE_COLUMNS, `surge_label`,
        `partition`, `excluded`, and `created_utc`.
    config : PipelineConfig
        Pipeline configuration (uses random_seed, weight_sentiment, output_dir).
    output_dir : str, optional
        Override output directory for model files. If None, uses
        config.output_dir / "models".

    Returns
    -------
    TrainingResult
        Contains all trained models with metadata.
    """
    total_start = time.time()
    seed = config.random_seed

    # Determine phase
    phase = "phase1" if config.weight_sentiment == 0.0 else "phase2"
    logger.info("=" * 70)
    logger.info("MODEL TRAINING — %s (seed=%d)", phase.upper(), seed)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Filter to training partition (R13-AC2)
    # ------------------------------------------------------------------
    train_mask = (
        (df["partition"] == "train")
        & (~df["excluded"].astype(bool))
        & (df["surge_label"].notna())
    )
    df_train = df.loc[train_mask].copy().reset_index(drop=True)

    logger.info(
        "Training partition: %d records (surge=%.1f%%)",
        len(df_train),
        df_train["surge_label"].mean() * 100,
    )

    # Extract feature matrix and labels
    X = df_train[FEATURE_COLUMNS].values.astype(np.float64)
    y = df_train["surge_label"].values.astype(np.float64)

    # ------------------------------------------------------------------
    # Step 2: Create temporal folds (R13-AC2)
    # ------------------------------------------------------------------
    folds = create_temporal_folds(df_train, n_folds=N_FOLDS)
    splits = get_expanding_window_splits(folds)

    # Verify temporal ordering (Validation requirement)
    _verify_temporal_ordering(df_train, folds, splits)

    # ------------------------------------------------------------------
    # Step 3–4: Hyperparameter search for each model (R13-AC1)
    # ------------------------------------------------------------------
    trained_models: Dict[str, TrainedModel] = {}

    # --- Logistic Regression (R13-AC7) ---
    trained_models["logistic_regression"] = _train_single_model(
        X=X,
        y=y,
        splits=splits,
        param_grid=_get_lr_param_grid(),
        build_fn=_build_lr_model,
        model_name="Logistic Regression",
        seed=seed,
    )

    # --- Random Forest (R13-AC8) ---
    trained_models["random_forest"] = _train_single_model(
        X=X,
        y=y,
        splits=splits,
        param_grid=_get_rf_param_grid(),
        build_fn=_build_rf_model,
        model_name="Random Forest",
        seed=seed,
    )

    # --- XGBoost (R13-AC9) ---
    trained_models["xgboost"] = _train_single_model(
        X=X,
        y=y,
        splits=splits,
        param_grid=_get_xgb_param_grid(seed),
        build_fn=_build_xgb_model,
        model_name="XGBoost",
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Step 5: Retrain best configs on full training partition (R13-AC5)
    # ------------------------------------------------------------------
    for name, tm in trained_models.items():
        logger.info(
            "Retraining %s on full training partition (%d records)...",
            name,
            len(X),
        )
        if name == "logistic_regression":
            final_model = _build_lr_model(tm.best_params, seed)
        elif name == "random_forest":
            final_model = _build_rf_model(tm.best_params, seed)
        else:
            final_model = _build_xgb_model(tm.best_params, seed)

        final_model.fit(X, y)
        tm.model = final_model

    # ------------------------------------------------------------------
    # Step 6: Serialise models (R14-AC4)
    # ------------------------------------------------------------------
    out_dir = Path(output_dir) if output_dir else Path(config.output_dir) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, tm in trained_models.items():
        model_path = out_dir / f"{name}_{phase}_{seed}.joblib"
        joblib.dump(tm.model, model_path)
        logger.info("Serialised %s → %s", name, model_path)

    # ------------------------------------------------------------------
    # Log summary (R14-AC3)
    # ------------------------------------------------------------------
    total_duration = time.time() - total_start
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE — Total duration: %.1fs", total_duration)
    logger.info("-" * 70)
    for name, tm in trained_models.items():
        logger.info(
            "  %-22s | Best CV AUC: %.4f | Params: %s | Duration: %.1fs | Configs: %d",
            name,
            tm.best_cv_auc,
            tm.best_params,
            tm.training_duration_seconds,
            tm.n_configs_evaluated,
        )
    logger.info("=" * 70)

    return TrainingResult(
        models=trained_models,
        phase=phase,
        random_seed=seed,
        output_dir=str(out_dir),
    )


def _train_single_model(
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    param_grid: List[Dict[str, Any]],
    build_fn,
    model_name: str,
    seed: int,
) -> TrainedModel:
    """Train a single model type: search hyperparams, select best (R13-AC4).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    splits : list
        CV splits.
    param_grid : list of dict
        Hyperparameter configs.
    build_fn : callable
        Model builder.
    model_name : str
        Name for logging.
    seed : int
        Random seed.

    Returns
    -------
    TrainedModel
        Model metadata (model field is None until final retraining).
    """
    start = time.time()

    cv_results = _run_hyperparameter_search(
        X, y, splits, param_grid, build_fn, model_name, seed
    )

    best = cv_results[0]
    duration = time.time() - start

    logger.info(
        "%s — Best config: %s (mean AUC: %.4f, folds: %s) [%.1fs]",
        model_name,
        best.params,
        best.mean_auc,
        [f"{s:.4f}" for s in best.fold_scores],
        duration,
    )

    return TrainedModel(
        model_name=model_name,
        model=None,  # Will be set after retraining on full partition
        best_params=best.params,
        cv_results=cv_results,
        best_cv_auc=best.mean_auc,
        training_duration_seconds=duration,
        n_configs_evaluated=len(param_grid),
    )


# =============================================================================
# Temporal ordering verification
# =============================================================================


def _verify_temporal_ordering(
    df_train: pd.DataFrame,
    folds: List[np.ndarray],
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> None:
    """Verify that CV splits respect temporal ordering (Validation).

    For each split, max timestamp in training set must be ≤ min timestamp
    in validation set.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training partition DataFrame with `created_utc`.
    folds : list of np.ndarray
        The k temporal folds.
    splits : list of (train_idx, val_idx)
        Expanding-window splits.

    Raises
    ------
    ValueError
        If temporal ordering is violated.
    """
    timestamps = pd.to_datetime(df_train["created_utc"], utc=True)
    epoch_seconds = (timestamps.astype("int64") // 10**9).values

    for i, (train_idx, val_idx) in enumerate(splits):
        max_train_ts = epoch_seconds[train_idx].max()
        min_val_ts = epoch_seconds[val_idx].min()

        if max_train_ts > min_val_ts:
            raise ValueError(
                f"Temporal ordering violated in split {i}: "
                f"max train timestamp ({max_train_ts}) > "
                f"min validation timestamp ({min_val_ts})"
            )

        logger.info(
            "Split %d: temporal ordering OK — max_train_ts=%d ≤ min_val_ts=%d "
            "(gap: %d seconds)",
            i,
            max_train_ts,
            min_val_ts,
            min_val_ts - max_train_ts,
        )


# =============================================================================
# Utility functions
# =============================================================================


def get_training_summary(result: TrainingResult) -> Dict[str, Any]:
    """Generate a JSON-serialisable summary of training results (R14-AC3).

    Parameters
    ----------
    result : TrainingResult
        Complete training result.

    Returns
    -------
    dict
        Summary with model names, best params, CV scores, durations.
    """
    summary = {
        "phase": result.phase,
        "random_seed": result.random_seed,
        "output_dir": result.output_dir,
        "models": {},
    }

    for name, tm in result.models.items():
        summary["models"][name] = {
            "best_params": tm.best_params,
            "best_cv_auc": tm.best_cv_auc,
            "fold_scores": tm.cv_results[0].fold_scores if tm.cv_results else [],
            "training_duration_seconds": tm.training_duration_seconds,
            "n_configs_evaluated": tm.n_configs_evaluated,
        }

    return summary
