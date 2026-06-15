"""Load and apply normalisation parameters from pipeline summary.

Provides a single entry point for the modelling stage to retrieve
training-set normalisation statistics (μ, σ) without recomputing them.
This prevents data leakage by ensuring inference-time normalisation uses
the same parameters fitted during pipeline execution.

Usage:
    from surge_pipeline.normalisation import load_normalisation_params

    params = load_normalisation_params("data/processed/pipeline_summary.json")
    z_vol = params.normalise_volume(raw_volume_growth)
    z_sent = params.normalise_sentiment(raw_sentiment_change)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NormalisationParams:
    """Immutable normalisation parameters loaded from pipeline summary.

    Attributes
    ----------
    mu_volume : float
        Mean posting volume growth from training partition.
    sigma_volume : float
        Std dev of posting volume growth from training partition.
    mu_sentiment : float
        Mean |sentiment_change| from training partition.
    sigma_sentiment : float
        Std dev of |sentiment_change| from training partition.
    train_size : int
        Number of included records in the training partition.
    test_size : int
        Number of included records in the test partition.
    split_timestamp : float
        Epoch seconds of the temporal train/test split point.
    """

    mu_volume: float
    sigma_volume: float
    mu_sentiment: float
    sigma_sentiment: float
    train_size: int
    test_size: int
    split_timestamp: float

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def normalise_volume(self, values: np.ndarray) -> np.ndarray:
        """Z-score normalise posting volume growth values.

        Parameters
        ----------
        values : np.ndarray
            Raw posting_volume_growth values.

        Returns
        -------
        np.ndarray
            Z-score normalised values. Returns zeros if σ_volume == 0.
        """
        if self.sigma_volume == 0.0:
            return np.zeros_like(values, dtype=np.float64)
        return (values - self.mu_volume) / self.sigma_volume

    def normalise_sentiment(self, values: np.ndarray) -> np.ndarray:
        """Z-score normalise absolute sentiment change values.

        Parameters
        ----------
        values : np.ndarray
            Raw |sentiment_change| values (should already be absolute).

        Returns
        -------
        np.ndarray
            Z-score normalised values. Returns zeros if σ_sentiment == 0.
        """
        if self.sigma_sentiment == 0.0:
            return np.zeros_like(values, dtype=np.float64)
        return (values - self.mu_sentiment) / self.sigma_sentiment

    def compute_composite(
        self,
        volume_values: np.ndarray,
        sentiment_values: np.ndarray,
        weight_volume: float = 0.5,
        weight_sentiment: float = 0.5,
    ) -> np.ndarray:
        """Compute composite surge metric from raw values.

        Normalises both components using stored training statistics,
        then combines them with the specified weights.

        Parameters
        ----------
        volume_values : np.ndarray
            Raw posting_volume_growth values.
        sentiment_values : np.ndarray
            Raw |sentiment_change| values.
        weight_volume : float
            Weight for volume component (default 0.5).
        weight_sentiment : float
            Weight for sentiment component (default 0.5).

        Returns
        -------
        np.ndarray
            Composite surge metric values.
        """
        z_vol = self.normalise_volume(volume_values)
        z_sent = self.normalise_sentiment(sentiment_values)
        return (weight_volume * z_vol) + (weight_sentiment * z_sent)


def load_normalisation_params(
    summary_path: str | Path = "data/processed/pipeline_summary.json",
) -> NormalisationParams:
    """Load normalisation parameters from the pipeline summary JSON.

    Parameters
    ----------
    summary_path : str or Path
        Path to pipeline_summary.json produced by save_outputs().

    Returns
    -------
    NormalisationParams
        Frozen dataclass with training-set normalisation statistics
        and convenience methods for applying normalisation.

    Raises
    ------
    FileNotFoundError
        If the summary file does not exist.
    KeyError
        If required fields are missing from the JSON.
    """
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline summary not found at '{path}'. "
            "Run the pipeline first to generate it."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    norm = data["normalisation_params"]

    return NormalisationParams(
        mu_volume=float(norm["mu_volume"]),
        sigma_volume=float(norm["sigma_volume"]),
        mu_sentiment=float(norm["mu_sentiment"]),
        sigma_sentiment=float(norm["sigma_sentiment"]),
        train_size=int(norm["train_size"]),
        test_size=int(norm["test_size"]),
        split_timestamp=float(norm["split_timestamp"]),
    )
