"""Pipeline orchestrator — chains all stages and manages outputs.

Chains: load → window → sentiment → label.
Supports threshold sweep mode for sensitivity analysis.
Ensures deterministic execution via seeded randomness.

Requirements: R7 (Pipeline Determinism and Reproducibility),
              R8 (Threshold Sensitivity Output)
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from surge_pipeline.config import PipelineConfig
from surge_pipeline.labelling import apply_labelling, sweep_thresholds
from surge_pipeline.loader import load_data
from surge_pipeline.sentiment import compute_sentiment
from surge_pipeline.windowing import compute_windowed_counts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config: PipelineConfig) -> dict:
    """Execute the full surge-labelling pipeline.

    Chains all stages in order: load → window → sentiment → label.
    Seeds numpy/random at start for reproducibility (R7-AC1, AC2).
    Logs record counts after each stage (R7-AC4).

    Parameters
    ----------
    config : PipelineConfig
        Complete pipeline configuration.

    Returns
    -------
    dict
        Results dictionary with keys:
        - labelled_df: pd.DataFrame with all computed columns
        - stats: NormalisationStats from the labelling stage
        - class_distributions: dict of class distribution per partition
        - sweep_results: list of dicts from threshold sweep
        - stage_counts: dict of record counts after each stage
    """
    # ------------------------------------------------------------------
    # Seed randomness for determinism (R7-AC1, AC2)
    # ------------------------------------------------------------------
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    logger.info("Random seeds set to %d for deterministic execution.", config.random_seed)

    stage_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Stage 1: Load data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1: Loading data")
    logger.info("=" * 60)

    df = load_data(config)
    stage_counts["after_load"] = len(df)
    logger.info("After load: %d records", len(df))

    # ------------------------------------------------------------------
    # Stage 2: Windowing
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2: Computing windowed counts")
    logger.info("=" * 60)

    df = compute_windowed_counts(df, config)
    stage_counts["after_windowing"] = len(df)
    logger.info("After windowing: %d records", len(df))

    # Log exclusion summary
    if "excluded" in df.columns:
        excluded_count = df["excluded"].sum()
        exclusion_rate = excluded_count / len(df) * 100 if len(df) > 0 else 0
        logger.info(
            "Exclusion summary — excluded: %d (%.1f%%)",
            excluded_count,
            exclusion_rate,
        )
        stage_counts["excluded_count"] = int(excluded_count)

    # ------------------------------------------------------------------
    # Stage 3: Sentiment
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 3: Computing sentiment")
    logger.info("=" * 60)

    df = compute_sentiment(df, config)
    stage_counts["after_sentiment"] = len(df)
    logger.info("After sentiment: %d records", len(df))

    # ------------------------------------------------------------------
    # Stage 4: Labelling
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 4: Applying labelling (τ=%.2f)", config.threshold_tau)
    logger.info("=" * 60)

    labelling_result = apply_labelling(df, config)
    df = labelling_result.df
    stage_counts["after_labelling"] = len(df)
    logger.info("After labelling: %d records", len(df))

    # ------------------------------------------------------------------
    # Stage 5: Threshold sweep (R8)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 5: Threshold sweep")
    logger.info("=" * 60)

    sweep_results = sweep_thresholds(df, config)
    logger.info("Threshold sweep complete — %d thresholds evaluated.", len(sweep_results))

    # ------------------------------------------------------------------
    # Pipeline complete
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Final record count: %d", len(df))

    return {
        "labelled_df": df,
        "stats": labelling_result.stats,
        "class_distributions": labelling_result.class_distributions,
        "sweep_results": sweep_results,
        "stage_counts": stage_counts,
    }


# ---------------------------------------------------------------------------
# Threshold sweep (R8)
# ---------------------------------------------------------------------------


def run_threshold_sweep(config: PipelineConfig) -> pd.DataFrame:
    """Run pipeline through sentiment, then sweep thresholds.

    Executes load → window → sentiment stages, then applies labelling
    at each threshold in config.thresholds. Returns a summary table
    with class distribution metrics and viability flags.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration with thresholds list.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: threshold, surge_count, no_surge_count,
        surge_rate, imbalance_ratio, viable (surge rate 5-10%).
    """
    # Seed randomness
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Run through sentiment stage
    df = load_data(config)
    df = compute_windowed_counts(df, config)
    df = compute_sentiment(df, config)

    # Sweep thresholds
    sweep_results = sweep_thresholds(df, config)

    # Build summary table (R8-AC1, AC2)
    rows: List[Dict] = []
    for result in sweep_results:
        dist = result["class_distributions"].get("all", {})
        tau = result["threshold"]
        surge_count = dist.get("surge_count", 0)
        no_surge_count = dist.get("no_surge_count", 0)
        surge_rate = dist.get("surge_rate", 0.0)
        imbalance_ratio = dist.get("imbalance_ratio", 0.0)

        # R8-AC4: Viable if surge rate between 5% and 10%
        viable = 5.0 <= surge_rate <= 10.0

        rows.append(
            {
                "threshold": tau,
                "surge_count": surge_count,
                "no_surge_count": no_surge_count,
                "surge_rate": surge_rate,
                "imbalance_ratio": imbalance_ratio,
                "viable": viable,
            }
        )

    sweep_df = pd.DataFrame(rows)

    # Log the summary table (R8-AC3)
    logger.info("Threshold sensitivity table:")
    logger.info("-" * 70)
    logger.info(
        "%-10s %-12s %-14s %-12s %-16s %-8s",
        "τ", "Surge", "No-Surge", "Rate(%)", "Imbalance", "Viable",
    )
    logger.info("-" * 70)
    for _, row in sweep_df.iterrows():
        logger.info(
            "%-10.2f %-12d %-14d %-12.2f %-16.2f %-8s",
            row["threshold"],
            row["surge_count"],
            row["no_surge_count"],
            row["surge_rate"],
            row["imbalance_ratio"],
            "✓" if row["viable"] else "✗",
        )
    logger.info("-" * 70)

    viable_count = sweep_df["viable"].sum()
    if viable_count > 0:
        logger.info(
            "Viable thresholds (surge rate 5-10%%): %d found.",
            viable_count,
        )
    else:
        logger.warning(
            "No viable thresholds found — no threshold produced surge rate 5-10%%."
        )

    return sweep_df


# ---------------------------------------------------------------------------
# Output management
# ---------------------------------------------------------------------------


def save_outputs(results: dict, config: PipelineConfig) -> dict:
    """Save pipeline outputs to disk.

    Saves:
    - Labelled dataset as CSV
    - Summary statistics as JSON
    - Threshold sensitivity table as CSV
    - Pipeline config for audit trail (R7-AC3)

    Parameters
    ----------
    results : dict
        Results dictionary from run_pipeline().
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    dict
        Dictionary of output file paths.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp prefix for experiment comparison (YYYYMMDDHHMM)
    prefix = datetime.now().strftime("%Y%m%d%H%M")

    output_paths: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. Labelled dataset (CSV)
    # ------------------------------------------------------------------
    labelled_path = output_dir / f"{prefix}_labelled_dataset.csv"
    results["labelled_df"].to_csv(labelled_path, index=False)
    output_paths["labelled_dataset"] = str(labelled_path)
    logger.info("Labelled dataset saved: %s (%d records)", labelled_path, len(results["labelled_df"]))

    # ------------------------------------------------------------------
    # 2. Pipeline summary (JSON)
    # ------------------------------------------------------------------
    stats = results["stats"]
    summary = {
        "normalisation_params": {
            "mu_volume": stats.mu_volume,
            "sigma_volume": stats.sigma_volume,
            "mu_sentiment": stats.mu_sentiment,
            "sigma_sentiment": stats.sigma_sentiment,
            "train_size": stats.train_size,
            "test_size": stats.test_size,
            "split_timestamp": stats.split_timestamp,
        },
        "class_distributions": results["class_distributions"],
        "stage_counts": results["stage_counts"],
        "config": {
            "threshold_tau": config.threshold_tau,
            "weight_volume": config.weight_volume,
            "weight_sentiment": config.weight_sentiment,
            "temporal_split_ratio": config.temporal_split_ratio,
            "min_window_count": config.min_window_count,
            "random_seed": config.random_seed,
            "sentiment_model": config.sentiment_model,
        },
    }

    # Compute exclusion rate for summary
    labelled_df = results["labelled_df"]
    if "excluded" in labelled_df.columns:
        total = len(labelled_df)
        excluded = int(labelled_df["excluded"].sum())
        summary["exclusion_rate"] = excluded / total * 100 if total > 0 else 0.0

    # Include dataset fingerprint if available
    if "dataset_fingerprint" in labelled_df.attrs:
        summary["dataset_fingerprint"] = labelled_df.attrs["dataset_fingerprint"]

    summary_path = output_dir / f"{prefix}_pipeline_summary.json"

    def _json_serialise(obj):
        """Handle non-serialisable values like inf/nan."""
        if isinstance(obj, float):
            if np.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            if np.isnan(obj):
                return "NaN"
        return str(obj)

    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_serialise), encoding="utf-8"
    )
    output_paths["pipeline_summary"] = str(summary_path)
    logger.info("Pipeline summary saved: %s", summary_path)

    # ------------------------------------------------------------------
    # 3. Threshold sensitivity table (CSV) (R8-AC3)
    # ------------------------------------------------------------------
    if results.get("sweep_results"):
        sweep_rows: List[Dict] = []
        for result in results["sweep_results"]:
            dist = result["class_distributions"].get("all", {})
            tau = result["threshold"]
            surge_rate = dist.get("surge_rate", 0.0)
            sweep_rows.append(
                {
                    "threshold": tau,
                    "surge_count": dist.get("surge_count", 0),
                    "no_surge_count": dist.get("no_surge_count", 0),
                    "surge_rate": surge_rate,
                    "imbalance_ratio": dist.get("imbalance_ratio", 0.0),
                    "viable": 5.0 <= surge_rate <= 10.0,
                }
            )

        sweep_df = pd.DataFrame(sweep_rows)
        sweep_path = output_dir / f"{prefix}_threshold_sensitivity.csv"
        sweep_df.to_csv(sweep_path, index=False)
        output_paths["threshold_sensitivity"] = str(sweep_path)
        logger.info("Threshold sensitivity table saved: %s", sweep_path)

    # ------------------------------------------------------------------
    # 4. Pipeline config for audit trail (R7-AC3)
    # ------------------------------------------------------------------
    config_path = output_dir / f"{prefix}_pipeline_config.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    output_paths["pipeline_config"] = str(config_path)
    logger.info("Pipeline config saved: %s", config_path)

    # ------------------------------------------------------------------
    # 5. Latest outputs manifest (for downstream tool discovery)
    # ------------------------------------------------------------------
    manifest_path = output_dir / "latest_outputs.json"
    manifest_path.write_text(
        json.dumps({"prefix": prefix, "outputs": output_paths}, indent=2),
        encoding="utf-8",
    )
    logger.info("Latest outputs manifest: %s", manifest_path)

    logger.info("All outputs saved to: %s", output_dir)
    return output_paths
