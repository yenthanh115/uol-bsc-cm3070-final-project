<<<<<<< HEAD
"""Model training with expanding-window temporal cross-validation.

Trains Logistic Regression, Random Forest, and XGBoost classifiers using
temporal cross-validation that respects chronological ordering. Supports
Phase 1 (w₂=0, volume-only labels) and Phase 2 (w₂=0.5, composite labels).
=======
"""Model training with temporal cross-validation.

Implements expanding-window temporal cross-validation for a Logistic
Regression baseline model. Divides the training partition into k=4
sequential folds and selects the best hyperparameters by mean
validation AUC-ROC.
>>>>>>> 7ea149c

Requirements: R13 (Model Training with Temporal Cross-Validation),
              R14 (Model Training Reproducibility)
Design Decision: D9 — Expanding-window temporal cross-validation.
"""

from __future__ import annotations

import logging
import time
<<<<<<< HEAD
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
=======
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
>>>>>>> 7ea149c


@dataclass
class TrainingResult:
<<<<<<< HEAD
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
=======
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
>>>>>>> 7ea149c

    Parameters
    ----------
    df : pd.DataFrame
<<<<<<< HEAD
        Feature-engineered DataFrame with FEATURE_COLUMNS, `surge_label`,
        `partition`, `excluded`, and `created_utc`.
    config : PipelineConfig
        Pipeline configuration (uses random_seed, weight_sentiment, output_dir).
    output_dir : str, optional
        Override output directory for model files. If None, uses
        config.output_dir / "models".
=======
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
>>>>>>> 7ea149c

    Returns
    -------
    TrainingResult
<<<<<<< HEAD
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
=======
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
>>>>>>> 7ea149c

    Parameters
    ----------
    result : TrainingResult
<<<<<<< HEAD
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
=======
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
>>>>>>> 7ea149c
