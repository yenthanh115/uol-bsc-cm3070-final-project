"""Data loader with ticker extraction and explosion for the surge-labelling pipeline.

Loads the Reddit submissions CSV, extracts tickers from title/selftext using
regex patterns, handles missing/deleted text, and explodes multi-ticker records
into one row per (record_id, ticker) pair.

Requirements: R1 (Data Loading and Ticker Extraction)
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from surge_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords: common English words and Reddit/finance terms that look like tickers
#
# Split into named sub-sets by category for easier auditing and extension.
# The public TICKER_STOPWORDS is their union.
# ---------------------------------------------------------------------------

_COMMON_ENGLISH_WORDS: Set[str] = {
    # 1-2 char words
    "TO", "IS", "IT", "IF", "IN", "OR", "SO", "UP",
    "AT", "AN", "AS", "BE", "BY", "DO", "GO", "HE", "ME", "MY", "NO",
    "OF", "OH", "ON", "WE",
    # 3-5 char words
    "THE", "FOR", "AND", "BUT", "NOT", "ARE", "WAS", "HAS", "HAD", "CAN",
    "ALL", "NEW", "OLD", "BIG", "LOW", "HIGH", "BUY", "PUT", "GET", "GOT",
    "SET", "RUN", "SAW", "MAY", "OUR", "HIS", "HER", "WHO", "HOW", "WHY",
    "ITS", "OWN", "NOW", "ANY", "FEW", "ONE", "TWO", "TEN", "TOP", "RED",
    "HOT", "DAY", "DID",
    "DONT", "INFO", "JUST", "LIKE", "MAKE", "MUCH", "SOME", "THAN", "THEM",
    "THEN", "THEY", "THIS", "THAT", "VERY", "WHAT", "WHEN", "WILL", "WITH",
    "BEEN", "FROM", "HAVE", "HERE", "INTO", "KEEP", "KNOW", "LETS", "LOOK",
    "MADE", "MORE", "MOST", "MUST", "NEED", "ONLY", "OVER", "SAME", "SUCH",
    "TAKE", "TELL", "WANT", "WERE", "YOUR", "ALSO", "BACK", "BEST", "BOTH",
    "COME", "DOES", "DONE", "EACH", "EVEN", "FEEL", "FIND", "GIVE", "GOOD",
    "HELP", "HUGE", "LEFT", "LESS", "LIFE", "LINE", "LIST", "LOTS", "MANY",
    "ONCE", "OPEN", "PART", "PAST", "PLAN", "REAL", "REST", "SAID", "SHOW",
    "SIDE", "SURE", "TECH", "TERM", "TIME", "TYPE", "USED", "WAIT", "WELL",
    "WENT", "WORK",
}

_REDDIT_SLANG: Set[str] = {
    # Reddit community terms
    "DD", "APE", "YOLO", "FOMO", "FUD", "WSB", "IMO", "IMHO", "EDIT",
    "TLDR", "NSFW", "LOL", "WTF", "OMG", "SMH", "TBH", "LMAO", "ROFL",
    "DM", "OP", "TL", "DR", "ICYMI",
    # Trading-post verbs and nouns that are not tickers
    "UPDATE", "HOLD", "SELL", "SHORT", "LONG", "CALL", "PUTS", "MOON",
    "PUMP", "DUMP", "GAIN", "LOSS", "PLAY", "PICK", "MOVE", "DROP", "RISE",
    "FALL", "FREE", "DOWN", "NEXT", "LAST", "WEEK", "YEAR", "LINK", "POST",
    "STOCK", "SHARE", "SHARES", "PRICE", "TRADE", "PENNY",
}

_FINANCE_ABBREVIATIONS: Set[str] = {
    # Corporate/financial abbreviations
    "CEO", "CFO", "IPO", "ETF", "OTC", "SEC", "FDA", "EPS", "ATH", "ATL",
    "NYSE", "EOD", "EOW", "EOM", "GDP", "CPI", "ROI", "ITM", "OTM",
    "RSI", "MACD", "EMA", "SMA", "AVG", "MAX", "MIN", "VS",
    # Entity suffixes
    "LLC", "INC", "LTD", "CORP", "CO",
}

_MARKET_VENUES: Set[str] = {
    # Exchange names and market identifiers (never valid tickers in context)
    "OTCQB", "OTCQX", "TSX", "TSXV", "CSE", "NASDAQ", "AMEX", "SP",
}

_CURRENCIES_AND_ASSETS: Set[str] = {
    "USD", "CAD", "GBP", "EUR", "BTC", "CRYPTO",
}

_DATETIME_AND_UNITS: Set[str] = {
    # Time zones and units
    "PM", "AM", "EST", "PST", "UTC",
    "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
    "JAN", "FEB", "MAR", "APR", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    "FT", "LB", "OZ",
}

_GEOGRAPHY: Set[str] = {
    "USA", "US", "UK", "EU", "CA", "NY", "TX", "FL",
    "PT",  # Portugal (common in EU context)
}

_TECHNOLOGY_BUZZWORDS: Set[str] = {
    # Tech acronyms frequently false-positive matched in WSB/pennystocks posts
    "EV", "AI", "AR", "VR", "NFT", "CBD", "COVID",
}

_REDDIT_FP: Set[str] = {
    # Remaining high-frequency false positives specific to penny stock posts
    "ZERO", "GLOBE", "PINK", "XXXX", "PR",
}

#: Union of all stopword sub-sets. Use this for ticker filtering.
TICKER_STOPWORDS: Set[str] = (
    _COMMON_ENGLISH_WORDS
    | _REDDIT_SLANG
    | _FINANCE_ABBREVIATIONS
    | _MARKET_VENUES
    | _CURRENCIES_AND_ASSETS
    | _DATETIME_AND_UNITS
    | _GEOGRAPHY
    | _TECHNOLOGY_BUZZWORDS
    | _REDDIT_FP
)

# Regex patterns for ticker extraction
_DOLLAR_SIGN_PATTERN = re.compile(r"\$([A-Z]{1,5})")
_UPPERCASE_WORD_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")


# ---------------------------------------------------------------------------
# Ticker extraction
# ---------------------------------------------------------------------------


def extract_tickers(title: str, selftext: str) -> List[str]:
    """Extract stock tickers from post title and selftext.

    Strategy:
        1. Dollar-sign pattern (highest priority): $BNGO, $AMC, etc.
        2. Uppercase word pattern: standalone 2-5 char uppercase words.
        3. Filter against stopword set to exclude common English/Reddit terms.
        4. Deduplicate across title and selftext.

    Parameters
    ----------
    title : str
        The post title text.
    selftext : str
        The post body text.

    Returns
    -------
    List[str]
        Sorted list of unique extracted tickers (may be empty).
    """
    tickers: Set[str] = set()

    combined_text = f"{title} {selftext}"

    # 1. Dollar-sign pattern (highest priority — always included)
    dollar_matches = _DOLLAR_SIGN_PATTERN.findall(combined_text)
    for match in dollar_matches:
        ticker = match.upper()
        if ticker not in TICKER_STOPWORDS:
            tickers.add(ticker)

    # 2. Uppercase word pattern (filtered against stopwords)
    word_matches = _UPPERCASE_WORD_PATTERN.findall(combined_text)
    for match in word_matches:
        ticker = match.upper()
        if ticker not in TICKER_STOPWORDS:
            tickers.add(ticker)

    return sorted(tickers)


# ---------------------------------------------------------------------------
# Sample tickers used by the synthetic data generator
# ---------------------------------------------------------------------------
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
# Dataset fingerprinting for cross-machine reproducibility
# ---------------------------------------------------------------------------


def compute_dataset_fingerprint(file_path: Path) -> Dict[str, object]:
    """Compute a reproducibility fingerprint for the raw CSV.

    Produces a SHA-256 hash of the file content plus structural metadata
    (row count, column names, timestamp range) so that two machines can
    verify they are working from identical source data.

    Parameters
    ----------
    file_path : Path
        Path to the raw CSV file.

    Returns
    -------
    dict
        Fingerprint dictionary with keys: sha256, file_size_bytes, columns,
        num_rows, timestamp_column, timestamp_min, timestamp_max,
        has_tickers_column.
    """
    # File-level hash (detects any byte-level difference)
    sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
    file_size = file_path.stat().st_size

    # Structural metadata (fast — only reads headers + first/last rows)
    df_head = pd.read_csv(file_path, nrows=5)
    columns = sorted(df_head.columns.tolist())
    with open(file_path, encoding="utf-8") as _f:
        num_rows = sum(1 for _ in _f) - 1  # minus header

    # Timestamp column detection
    ts_col = None
    ts_min = None
    ts_max = None
    if "created_utc" in df_head.columns:
        ts_col = "created_utc"
        # Read just the timestamp column for min/max
        ts_series = pd.read_csv(file_path, usecols=[ts_col])
        ts_min = float(ts_series[ts_col].min())
        ts_max = float(ts_series[ts_col].max())
    elif "created" in df_head.columns:
        ts_col = "created"
        ts_series = pd.read_csv(file_path, usecols=[ts_col])
        ts_min = str(ts_series[ts_col].min())
        ts_max = str(ts_series[ts_col].max())

    fingerprint = {
        "sha256": sha256,
        "file_size_bytes": file_size,
        "columns": columns,
        "num_rows": num_rows,
        "timestamp_column": ts_col,
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
        "has_tickers_column": "tickers" in columns,
    }

    logger.info(
        "Dataset fingerprint — SHA256: %.16s... | rows: %d | ts_col: %s | "
        "has_tickers: %s",
        sha256,
        num_rows,
        ts_col,
        "tickers" in columns,
    )

    return fingerprint


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------


def load_data(config: PipelineConfig) -> pd.DataFrame:
    """Load and preprocess the Reddit submissions dataset.

    Workflow:
        1. Load from CSV (or generate synthetic data if file unavailable).
        2. Handle real dataset: extract tickers from title/selftext.
        3. Parse timestamps (epoch seconds or datetime strings) and sort (AC2).
        4. Exclude records with no identifiable ticker (AC4).
        5. Explode multi-ticker records into one row per (id, ticker) (AC3).
        6. Log record counts at each step (AC5).

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
        is_synthetic = False
        fingerprint = compute_dataset_fingerprint(file_path)
    else:
        if file_path:
            logger.warning(
                "File not found: %s — falling back to synthetic data.", file_path
            )
        else:
            logger.info("No file_path configured — using synthetic data.")
        df = generate_synthetic_data(n_records=500, seed=config.random_seed)
        is_synthetic = True
        fingerprint = {"sha256": "synthetic", "num_rows": len(df), "columns": sorted(df.columns.tolist())}

    total_loaded = len(df)
    logger.info("Records loaded: %d", total_loaded)

    # ------------------------------------------------------------------
    # Step 2: Handle ticker extraction for real data (no 'tickers' column)
    # ------------------------------------------------------------------
    if "tickers" not in df.columns and not is_synthetic:
        logger.info("No 'tickers' column found — extracting tickers from title/selftext.")

        # Clean selftext: replace [deleted], [removed], NaN with empty string
        df["selftext"] = df["selftext"].fillna("")
        df["selftext"] = df["selftext"].replace(
            {"[deleted]": "", "[removed]": ""}
        )

        # Clean title: fill NaN with empty string
        df["title"] = df["title"].fillna("")

        # Extract tickers for each record
        df["tickers"] = df.apply(
            lambda row: ",".join(extract_tickers(row["title"], row["selftext"])),
            axis=1,
        )

        # Log extraction statistics
        ticker_counts = df["tickers"].apply(
            lambda x: len(x.split(",")) if x else 0
        )
        avg_tickers = ticker_counts.mean()
        zero_ticker_pct = (ticker_counts == 0).sum() / len(df) * 100

        logger.info(
            "Ticker extraction stats — Avg tickers per record: %.2f | "
            "Records with 0 tickers: %.1f%%",
            avg_tickers,
            zero_ticker_pct,
        )

    # ------------------------------------------------------------------
    # Step 3: Parse timestamps and sort chronologically (AC2)
    # ------------------------------------------------------------------
    if "created_utc" in df.columns:
        # Synthetic data or datasets with epoch-second timestamps
        df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    elif "created" in df.columns:
        # Real dataset with datetime string format (e.g., "2021-01-01 00:13:41")
        df["created_utc"] = pd.to_datetime(df["created"], utc=True)
        df = df.drop(columns=["created"])
    else:
        raise ValueError(
            "Dataset must contain either 'created_utc' (epoch) or 'created' (datetime string) column."
        )

    df = df.sort_values("created_utc").reset_index(drop=True)
    logger.info("Timestamps parsed and sorted chronologically.")

    # ------------------------------------------------------------------
    # Step 4: Exclude records with missing/empty tickers (AC4)
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
    # Step 5: Explode multi-ticker records (AC3)
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

    # Attach fingerprint as DataFrame attribute for pipeline summary
    df.attrs["dataset_fingerprint"] = fingerprint

    return df
