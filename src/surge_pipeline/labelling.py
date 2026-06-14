"""Z-score normalisation and composite surge labelling.

Performs temporal train/test split, z-score normalisation using training
statistics only, composite metric computation, and binary surge labelling.

Requirements: R5 (Z-Score Normalisation), R6 (Composite Surge Labelling)
Design Decision: D2 — Training/test split before normalisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from surge_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class NormalisationStats:
    """Stores normalisation parameters for reproducibility audit."""

    mu_volume: float
    sigma_volume: float
    mu_sentiment: float
    sigma_sentiment: float
    train_size: int
    test_size: int
    split_timestamp: float  # epoch seconds of the split point


@dataclass
class LabellingResult:
    """Result of the labelling stage containing DataFrame and metadata."""

    df: pd.DataFrame
    stats: NormalisationStats
    class_distributions: Dict[str, Dict[str, float]]


def _temporal_split(
    df: pd.DataFrame, ratio: float
) -> Tuple[np.ndarray, float]:
    """Assign train/test partition labels based on temporal split.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'created_utc' column (datetime64[ns, UTC]).
    ratio : float
        Fraction of records assigned to training (by timestamp percentile).

    Returns
    -------
    tuple of (ndarray[str], float)
        Array of partition labels ('train'/'test') and the split timestamp
        (epoch seconds).
    """
    epoch_seconds = df["created_utc"].astype("int64") // 10**9
    split_ts = np.percentile(epoch_seconds.values, ratio * 100)

    partitions = np.where(epoch_seconds.values <= split_ts, "train", "test")
    return partitions, float(split_ts)


def _compute_z_scores(
    values: np.ndarray, mu: float, sigma: float
) -> np.ndarray:
    """Compute z-scores, handling σ=0 edge case.

    Parameters
    ----------
    values : np.ndarray
        Raw values to normalise.
    mu : float
        Mean computed from training partition.
    sigma : float
        Standard deviation computed from training partition.

    Returns
    -------
    np.ndarray
        Z-score normalised values. Returns zeros if σ=0.
    """
    if sigma == 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mu) / sigma


def _compute_class_distribution(
    labels: np.ndarray, partition_name: str
) -> Dict[str, float]:
    """Compute and log class distribution statistics.

    Parameters
    ----------
    labels : np.ndarray
        Binary labels (0 or 1). NaN values are excluded.
    partition_name : str
        Name for logging (e.g., 'train', 'test', 'all').

    Returns
    -------
    dict
        Dictionary with surge_count, no_surge_count, total, surge_rate,
        imbalance_ratio.
    """
    valid = labels[~np.isnan(labels)]
    if len(valid) == 0:
        return {
            "surge_count": 0,
            "no_surge_count": 0,
            "total": 0,
            "surge_rate": 0.0,
            "imbalance_ratio": 0.0,
        }

    surge_count = int((valid == 1).sum())
    no_surge_count = int((valid == 0).sum())
    total = len(valid)
    surge_rate = surge_count / total * 100 if total > 0 else 0.0
    imbalance_ratio = (
        no_surge_count / surge_count if surge_count > 0 else float("inf")
    )

    logger.info(
        "Class distribution [%s] — surge: %d | no-surge: %d | "
        "total: %d | surge rate: %.2f%% | imbalance ratio: %.2f:1",
        partition_name,
        surge_count,
        no_surge_count,
        total,
        surge_rate,
        imbalance_ratio,
    )

    return {
        "surge_count": surge_count,
        "no_surge_count": no_surge_count,
        "total": total,
        "surge_rate": surge_rate,
        "imbalance_ratio": imbalance_ratio,
    }


def apply_labelling(
    df: pd.DataFrame, config: PipelineConfig
) -> LabellingResult:
    """Apply z-score normalisation and composite surge labelling.

    Workflow:
        1. Temporal train/test split at 80th percentile timestamp (R5-AC1).
        2. Compute μ/σ from training partition only (R5-AC2, AC3, AC6).
        3. Z-score normalise ALL records using training stats (R5-AC4).
        4. Handle σ=0 by setting z=0 (R5-AC5).
        5. Compute composite metric with configurable weights (R6-AC1).
        6. Apply threshold for binary label (R6-AC2).
        7. Report class distribution (R6-AC5).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from the sentiment stage with columns including:
        'created_utc', 'posting_volume_growth', 'sentiment_change',
        'excluded'.
    config : PipelineConfig
        Pipeline configuration with temporal_split_ratio, weight_volume,
        weight_sentiment, threshold_tau.

    Returns
    -------
    LabellingResult
        Named result containing the labelled DataFrame, normalisation
        statistics, and class distribution metrics.
    """
    # ------------------------------------------------------------------
    # Edge case: empty DataFrame
    # ------------------------------------------------------------------
    if df.empty:
        df = df.assign(
            partition=pd.array([], dtype="object"),
            z_volume=pd.array([], dtype="float64"),
            z_sentiment=pd.array([], dtype="float64"),
            composite=pd.array([], dtype="float64"),
            surge_label=pd.array([], dtype="float64"),
        )
        stats = NormalisationStats(
            mu_volume=0.0,
            sigma_volume=0.0,
            mu_sentiment=0.0,
            sigma_sentiment=0.0,
            train_size=0,
            test_size=0,
            split_timestamp=0.0,
        )
        return LabellingResult(df=df, stats=stats, class_distributions={})

    n = len(df)

    # ------------------------------------------------------------------
    # Step 1: Temporal train/test split (R5-AC1)
    # ------------------------------------------------------------------
    partitions, split_ts = _temporal_split(df, config.temporal_split_ratio)
    df = df.assign(partition=partitions)

    # Identify included (non-excluded) records for each partition
    excluded_mask = df["excluded"].values.astype(bool)
    included_mask = ~excluded_mask
    train_mask = (partitions == "train") & included_mask
    test_mask = (partitions == "test") & included_mask

    train_size = int(train_mask.sum())
    test_size = int(test_mask.sum())

    logger.info(
        "Temporal split — train: %d | test: %d | excluded: %d | "
        "split timestamp: %.0f",
        train_size,
        test_size,
        int(excluded_mask.sum()),
        split_ts,
    )

    # ------------------------------------------------------------------
    # Edge case: all records excluded
    # ------------------------------------------------------------------
    if train_size == 0:
        logger.warning(
            "No included training records — cannot compute normalisation. "
            "All labels set to NaN."
        )
        df = df.assign(
            z_volume=np.nan,
            z_sentiment=np.nan,
            composite=np.nan,
            surge_label=np.nan,
        )
        stats = NormalisationStats(
            mu_volume=0.0,
            sigma_volume=0.0,
            mu_sentiment=0.0,
            sigma_sentiment=0.0,
            train_size=0,
            test_size=test_size,
            split_timestamp=split_ts,
        )
        return LabellingResult(df=df, stats=stats, class_distributions={})

    # ------------------------------------------------------------------
    # Step 2: Compute μ/σ from training partition ONLY (R5-AC2, AC3, AC6)
    # ------------------------------------------------------------------
    train_volume = df.loc[train_mask, "posting_volume_growth"].values.astype(
        np.float64
    )
    train_sentiment = np.abs(
        df.loc[train_mask, "sentiment_change"].values.astype(np.float64)
    )

    mu_vol = float(np.mean(train_volume))
    sigma_vol = float(np.std(train_volume, ddof=0))  # population std
    mu_sent = float(np.mean(train_sentiment))
    sigma_sent = float(np.std(train_sentiment, ddof=0))  # population std

    logger.info(
        "Normalisation params (training only) — "
        "μ_vol: %.6f | σ_vol: %.6f | μ_sent: %.6f | σ_sent: %.6f",
        mu_vol,
        sigma_vol,
        mu_sent,
        sigma_sent,
    )

    if sigma_vol == 0.0:
        logger.warning("σ_vol = 0 — z_volume will be set to 0 for all records.")
    if sigma_sent == 0.0:
        logger.warning("σ_sent = 0 — z_sentiment will be set to 0 for all records.")

    # ------------------------------------------------------------------
    # Step 3: Z-score normalise ALL included records (R5-AC4, AC5)
    # ------------------------------------------------------------------
    all_volume = df["posting_volume_growth"].values.astype(np.float64)
    all_sentiment = np.abs(
        df["sentiment_change"].values.astype(np.float64)
    )

    z_volume = _compute_z_scores(all_volume, mu_vol, sigma_vol)
    z_sentiment = _compute_z_scores(all_sentiment, mu_sent, sigma_sent)

    # Excluded records: set z-scores to NaN
    z_volume[excluded_mask] = np.nan
    z_sentiment[excluded_mask] = np.nan

    # ------------------------------------------------------------------
    # Step 4: Composite metric (R6-AC1, AC4)
    # ------------------------------------------------------------------
    w1 = config.weight_volume
    w2 = config.weight_sentiment
    composite = (w1 * z_volume) + (w2 * z_sentiment)

    logger.info(
        "Composite metric — w_volume: %.2f | w_sentiment: %.2f | "
        "mode: %s",
        w1,
        w2,
        "Phase 1 (volume-only)" if w2 == 0 else "Phase 2 (composite)",
    )

    # ------------------------------------------------------------------
    # Step 5: Binary labelling (R6-AC2)
    # ------------------------------------------------------------------
    tau = config.threshold_tau
    surge_label = np.where(composite > tau, 1.0, 0.0)

    # Excluded records: set label to NaN
    surge_label[excluded_mask] = np.nan

    # Assign columns to DataFrame
    df = df.assign(
        z_volume=z_volume,
        z_sentiment=z_sentiment,
        composite=composite,
        surge_label=surge_label,
    )

    # ------------------------------------------------------------------
    # Step 6: Report class distribution (R6-AC5)
    # ------------------------------------------------------------------
    class_distributions: Dict[str, Dict[str, float]] = {}

    # Overall (included only)
    class_distributions["all"] = _compute_class_distribution(
        surge_label[included_mask], "all (included)"
    )

    # Per partition
    class_distributions["train"] = _compute_class_distribution(
        surge_label[train_mask], "train"
    )
    class_distributions["test"] = _compute_class_distribution(
        surge_label[test_mask], "test"
    )

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    stats = NormalisationStats(
        mu_volume=mu_vol,
        sigma_volume=sigma_vol,
        mu_sentiment=mu_sent,
        sigma_sentiment=sigma_sent,
        train_size=train_size,
        test_size=test_size,
        split_timestamp=split_ts,
    )

    logger.info(
        "Labelling complete — %d records labelled | threshold τ=%.2f",
        int(included_mask.sum()),
        tau,
    )

    return LabellingResult(df=df, stats=stats, class_distributions=class_distributions)


def sweep_thresholds(
    df: pd.DataFrame, config: PipelineConfig
) -> List[Dict[str, object]]:
    """Run labelling across multiple thresholds for empirical selection.

    This supports R6-AC3: threshold sweep across configurable values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from the sentiment stage (pre-labelling).
    config : PipelineConfig
        Pipeline configuration with thresholds list.

    Returns
    -------
    list of dict
        Each entry contains threshold value and class distribution metrics.
    """
    results = []

    for tau in config.thresholds:
        # Create a temporary config with the sweep threshold
        sweep_config = PipelineConfig(
            temporal_split_ratio=config.temporal_split_ratio,
            weight_volume=config.weight_volume,
            weight_sentiment=config.weight_sentiment,
            threshold_tau=tau,
            min_window_count=config.min_window_count,
            random_seed=config.random_seed,
        )

        result = apply_labelling(df.copy(), sweep_config)
        results.append(
            {
                "threshold": tau,
                "class_distributions": result.class_distributions,
                "stats": result.stats,
            }
        )

        logger.info(
            "Threshold sweep τ=%.2f — surge rate: %.2f%%",
            tau,
            result.class_distributions.get("all", {}).get("surge_rate", 0.0),
        )

    return results
