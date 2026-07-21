"""Dataclasses for the training pipeline.

Holds the result and model container types used across training.py,
evaluation.py, and run_training.py.

Requirements: R13 (Model Training with Temporal Cross-Validation),
              R14 (Model Training Reproducibility)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from surge_pipeline.features import FEATURE_COLUMNS

# Re-exported so callers can import N_FOLDS from here alongside the dataclasses.
N_FOLDS: int = 4
"""Number of temporal folds (produces N_FOLDS - 1 expanding-window splits)."""


@dataclass
class CVResult:
    """Result from a single hyperparameter configuration evaluated over CV folds."""

    params: dict[str, Any]
    fold_scores: list[float]
    mean_score: float
    std_score: float


@dataclass
class TrainedModel:
    """A single trained model with its metadata."""

    name: str
    model: Any
    scaler: StandardScaler
    best_params: dict[str, Any]
    best_cv_auc: float
    cv_results: list[CVResult]
    training_duration_seconds: float
    n_configs_evaluated: int
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    # Validation predictions from the last CV fold (for threshold tuning)
    val_y_true: np.ndarray | None = field(default=None, repr=False)
    val_y_prob: np.ndarray | None = field(default=None, repr=False)


@dataclass
class TrainingPipelineResult:
    """Result from the full multi-model training pipeline."""

    models: dict[str, TrainedModel]
    phase: str
    random_seed: int
    n_folds: int = N_FOLDS


@dataclass
class TrainingResult:
    """Result from single-model training (backward-compatible interface).

    Used by evaluation.py and the legacy train_logistic_regression wrapper
    in training.py.
    """

    model_name: str
    best_params: dict[str, Any]
    cv_scores: list[float]
    mean_cv_score: float
    std_cv_score: float
    training_duration_seconds: float
    model: Any
    scaler: StandardScaler
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
