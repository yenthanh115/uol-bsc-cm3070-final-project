"""Data loader with ticker explosion for the surge-labelling pipeline.

Loads the pennystocks CSV, parses timestamps, handles missing tickers,
and explodes multi-ticker records into one row per (record_id, ticker) pair.

Requirements: R1 (Data Loading and Ticker Extraction)
"""

from __future__ import annotations

import logging
import random
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from surge_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column schema expected in the raw CSV
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "id",
    "created_utc",
    "title",
    "selftext",
    "score",
    "num_comments",
    "subreddit",
    "tickers",
]

# Sample tickers used by the synthetic data generator
_SAMPLE_TICKERS = [
    "AMC", "GME", "BBBY", "AAPL", "TSLA", "PLTR", "BB", "NOK",
    "WISH", "CLOV", "SOFI", "LCID", "RIVN", "MVIS", "SNDL",
    "TLRY", "DKNG", "NIO", "XPEV", "LI", "SPCE", "WKHS",
]


# ---------------------------------------------------------------------------
# Synthetic data generator (R1 development fallback)
# ---------------------------------------------------------------------------


def generate_synthetic_data(
    n_records: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic Reddit-style records mimicking the pennystocks schema.

    Used when the raw CSV is not available locally, enabling development
    and testing without the real dataset.

    Parameters
    ----------
    n_records : int
        Number of synthetic records to generate (default 500).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame matching the expected CSV schema.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    base_time = datetime(2021, 1, 1)
    records = []

    for i in range(n_records):
        record_id = "".join(rng.choices(string.ascii_lowercase + string.digits, k=7))

        # Timestamps spread over ~2 years, roughly sorted with some jitter
        offset_seconds = int(i * 3600 * 4 + rng.randint(-3600, 3600))
        created_utc = base_time + timedelta(seconds=offset_seconds)

        title = f"Synthetic post about stocks #{i}"
        selftext = f"Discussion body for record {record_id}." if rng.random() > 0.3 else ""

        score = int(np_rng.exponential(scale=50))
        num_comments = int(np_rng.exponential(scale=20))
        subreddit = rng.choice(["pennystocks", "wallstreetbets", "stocks"])

        # Ticker assignment: ~10% empty, ~50% single, ~40% multi
        roll = rng.random()
        if roll < 0.10:
            tickers = ""
        elif roll < 0.60:
            tickers = rng.choice(_SAMPLE_TICKERS)
        else:
            n_tickers = rng.randint(2, 4)
            tickers = ",".join(rng.sample(_SAMPLE_TICKERS, n_tickers))

        records.append(
            {
                "id": record_id,
                "created_utc": created_utc.timestamp(),
                "title": title,
                "selftext": selftext,
                "score": score,
                "num_comments": num_comments,
                "subreddit": subreddit,
                "tickers": tickers,
            }
        )

    df = pd.DataFrame(records)
    logger.info("Generated %d synthetic records for development.", n_records)
    return df


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------


def load_data(config: PipelineConfig) -> pd.DataFrame:
    """Load and preprocess the pennystocks dataset.

    Workflow:
        1. Load from CSV (or generate synthetic data if file unavailable).
        2. Parse ``created_utc`` into datetime and sort chronologically (AC2).
        3. Exclude records with no identifiable ticker (AC4).
        4. Explode multi-ticker records into one row per (id, ticker) (AC3).
        5. Log record counts at each step (AC5).

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration containing ``file_path`` and ``random_seed``.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with one row per (record, ticker) pair,
        sorted chronologically.
    """
    # ------------------------------------------------------------------
    # Step 1: Load raw data (AC1)
    # ------------------------------------------------------------------
    file_path = Path(config.file_path) if config.file_path else None

    if file_path and file_path.exists():
        logger.info("Loading data from %s", file_path)
        df = pd.read_csv(file_path)
    else:
        if file_path:
            logger.warning(
                "File not found: %s — falling back to synthetic data.", file_path
            )
        else:
            logger.info("No file_path configured — using synthetic data.")
        df = generate_synthetic_data(n_records=500, seed=config.random_seed)

    total_loaded = len(df)
    logger.info("Records loaded: %d", total_loaded)

    # ------------------------------------------------------------------
    # Step 2: Parse created_utc into datetime, sort chronologically (AC2)
    # ------------------------------------------------------------------
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    df = df.sort_values("created_utc").reset_index(drop=True)
    logger.info("Timestamps parsed and sorted chronologically.")

    # ------------------------------------------------------------------
    # Step 3: Exclude records with missing/empty tickers (AC4)
    # ------------------------------------------------------------------
    # Convert to string, strip whitespace, replace empty strings with NaN
    df["tickers"] = df["tickers"].astype(str).str.strip()
    df["tickers"] = df["tickers"].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    mask_has_tickers = df["tickers"].notna()
    excluded_count = (~mask_has_tickers).sum()
    df = df[mask_has_tickers].reset_index(drop=True)

    retained_before_explode = len(df)
    logger.info(
        "Records excluded (no tickers): %d | Retained: %d",
        excluded_count,
        retained_before_explode,
    )

    # ------------------------------------------------------------------
    # Step 4: Explode multi-ticker records (AC3)
    # ------------------------------------------------------------------
    # Split comma-separated tickers into lists, then explode
    df["tickers"] = df["tickers"].str.split(",")
    df = df.explode("tickers", ignore_index=True)

    # Clean individual ticker values
    df["tickers"] = df["tickers"].str.strip().str.upper()

    # Remove any rows where an individual ticker became empty after splitting
    df = df[df["tickers"].str.len() > 0].reset_index(drop=True)

    # Rename for clarity: each row now represents a single ticker
    df = df.rename(columns={"tickers": "ticker"})

    total_after_explode = len(df)
    logger.info(
        "After ticker explosion: %d rows (one per record-ticker pair).",
        total_after_explode,
    )

    # ------------------------------------------------------------------
    # Summary log (AC5)
    # ------------------------------------------------------------------
    logger.info(
        "Loader summary — Loaded: %d | Excluded: %d | Retained records: %d | "
        "Exploded rows: %d",
        total_loaded,
        excluded_count,
        retained_before_explode,
        total_after_explode,
    )

    return df
