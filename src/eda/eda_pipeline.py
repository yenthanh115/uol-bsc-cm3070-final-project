"""Exploratory Data Analysis pipeline for the surge-labelling dataset.

Produces publication-ready figures (PNG, 300 DPI) saved to `figures/`.
Five focus areas:
    1. Dataset overview — record counts, temporal coverage, ticker frequency
    2. Sparsity analysis — exclusion rate by ticker, min_window_count impact
    3. Feature distributions — posting_volume_growth and |sentiment_change|
       histograms for surge vs non-surge
    4. Phase 1 vs Phase 2 comparison — label agreement
    5. Threshold sensitivity — surge rate curve across τ values

Requirements: R2 (Exploratory Data Analysis)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Resolve paths relative to the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

LABELLED_DATASET_PATH = _PROJECT_ROOT / "data" / "processed" / "labelled_dataset.csv"
THRESHOLD_SENSITIVITY_PATH = _PROJECT_ROOT / "data" / "processed" / "threshold_sensitivity.csv"
FIGURES_DIR = _PROJECT_ROOT / "figures"

# Publication defaults
DPI = 300
FIG_FORMAT = "png"
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.family": "serif",
})


def _save_figure(fig: plt.Figure, name: str) -> Path:
    """Save a figure to the figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.{FIG_FORMAT}"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


# ---------------------------------------------------------------------------
# Focus Area 1: Dataset Overview (R2-AC1, R2-AC2)
# ---------------------------------------------------------------------------


def plot_dataset_overview(df: pd.DataFrame) -> None:
    """Produce dataset overview statistics and figures.

    - Record counts (total, included, excluded)
    - Temporal coverage (date range)
    - Posting frequency over time
    - Ticker frequency distribution (top-N and long tail)
    """
    logger.info("=" * 60)
    logger.info("FOCUS AREA 1: Dataset Overview")
    logger.info("=" * 60)

    total_records = len(df)
    excluded_count = df["excluded"].sum()
    included_count = total_records - excluded_count
    logger.info(
        "Records — total: %d | included: %d | excluded: %d (%.1f%%)",
        total_records, included_count, excluded_count,
        excluded_count / total_records * 100 if total_records > 0 else 0,
    )

    # Temporal coverage
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    date_min = df["created_utc"].min()
    date_max = df["created_utc"].max()
    logger.info("Temporal coverage: %s to %s", date_min.date(), date_max.date())

    # --- Figure 1a: Posting frequency over time ---
    fig, ax = plt.subplots(figsize=(10, 4))
    df.set_index("created_utc").resample("W")["id"].count().plot(
        ax=ax, color="steelblue", linewidth=1.2
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Posts per week")
    ax.set_title("Posting Frequency Over Time")
    ax.axhline(y=df.set_index("created_utc").resample("W")["id"].count().mean(),
               color="red", linestyle="--", linewidth=0.8, label="Mean")
    ax.legend()
    _save_figure(fig, "01_posting_frequency_over_time")

    # --- Figure 1b: Ticker frequency distribution (top-20) ---
    ticker_counts = df["ticker"].value_counts()
    top_n = 20

    fig, ax = plt.subplots(figsize=(10, 5))
    ticker_counts.head(top_n).plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Record Count")
    ax.set_title(f"Top-{top_n} Ticker Frequency Distribution")
    ax.tick_params(axis="x", rotation=45)
    _save_figure(fig, "02_ticker_frequency_top20")

    # --- Figure 1c: Long-tail ticker distribution (log scale) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    rank = np.arange(1, len(ticker_counts) + 1)
    ax.plot(rank, ticker_counts.values, color="steelblue", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ticker Rank (log)")
    ax.set_ylabel("Frequency (log)")
    ax.set_title("Ticker Frequency — Long-Tail Distribution")
    _save_figure(fig, "03_ticker_frequency_longtail")

    n_unique_tickers = df["ticker"].nunique()
    logger.info("Unique tickers: %d", n_unique_tickers)
    logger.info("Top-5 tickers: %s", ticker_counts.head(5).to_dict())


# ---------------------------------------------------------------------------
# Focus Area 2: Sparsity Analysis (R2-AC3)
# ---------------------------------------------------------------------------


def plot_sparsity_analysis(df: pd.DataFrame) -> None:
    """Analyse sparsity: exclusion rate by ticker, min_window_count impact.

    - Per-ticker exclusion rate distribution
    - Impact of varying min_window_count on coverage
    """
    logger.info("=" * 60)
    logger.info("FOCUS AREA 2: Sparsity Analysis")
    logger.info("=" * 60)

    # Per-ticker exclusion rate
    ticker_stats = df.groupby("ticker").agg(
        total=("excluded", "count"),
        excluded=("excluded", "sum"),
    )
    ticker_stats["exclusion_rate"] = ticker_stats["excluded"] / ticker_stats["total"] * 100

    logger.info(
        "Ticker exclusion rate — mean: %.1f%% | median: %.1f%%",
        ticker_stats["exclusion_rate"].mean(),
        ticker_stats["exclusion_rate"].median(),
    )

    # --- Figure 2a: Exclusion rate distribution by ticker ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ticker_stats["exclusion_rate"], bins=30, color="coral", edgecolor="white")
    ax.axvline(
        ticker_stats["exclusion_rate"].mean(), color="red",
        linestyle="--", linewidth=1.2, label=f"Mean ({ticker_stats['exclusion_rate'].mean():.1f}%)"
    )
    ax.set_xlabel("Exclusion Rate (%)")
    ax.set_ylabel("Number of Tickers")
    ax.set_title("Per-Ticker Exclusion Rate Distribution")
    ax.legend()
    _save_figure(fig, "04_exclusion_rate_by_ticker")

    # --- Figure 2b: Impact of min_window_count on coverage ---
    # Simulate different min_window_count thresholds
    min_counts = range(1, 11)
    coverage_rates = []
    for n_min in min_counts:
        included = (df["forward_count"] >= n_min).sum()
        coverage_rates.append(included / len(df) * 100)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(min_counts), coverage_rates, marker="o", color="steelblue", linewidth=1.5)
    ax.axvline(x=3, color="red", linestyle="--", linewidth=0.8, label="Current N_min=3")
    ax.set_xlabel("min_window_count (N_min)")
    ax.set_ylabel("Coverage Rate (%)")
    ax.set_title("Impact of min_window_count on Dataset Coverage")
    ax.legend()
    ax.set_xticks(list(min_counts))
    _save_figure(fig, "05_min_window_count_coverage")

    logger.info(
        "Coverage at N_min=3: %.1f%% | N_min=1: %.1f%% | N_min=5: %.1f%%",
        coverage_rates[2], coverage_rates[0], coverage_rates[4],
    )


# ---------------------------------------------------------------------------
# Focus Area 3: Feature Distributions (R2-AC4)
# ---------------------------------------------------------------------------


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Visualise feature distributions split by surge vs non-surge.

    - posting_volume_growth histograms
    - |sentiment_change| histograms
    """
    logger.info("=" * 60)
    logger.info("FOCUS AREA 3: Feature Distributions")
    logger.info("=" * 60)

    # Filter to included records with valid labels
    included = df[(~df["excluded"]) & (df["surge_label"].notna())].copy()
    included["abs_sentiment_change"] = included["sentiment_change"].abs()
    included["label"] = included["surge_label"].map({1.0: "Surge", 0.0: "No Surge"})

    logger.info(
        "Included records for distribution analysis: %d (surge: %d, no-surge: %d)",
        len(included),
        (included["surge_label"] == 1.0).sum(),
        (included["surge_label"] == 0.0).sum(),
    )

    # --- Figure 3a: posting_volume_growth by surge label ---
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, group in included.groupby("label"):
        ax.hist(
            group["posting_volume_growth"].clip(-5, 20),
            bins=50, alpha=0.6, label=label, density=True,
        )
    ax.set_xlabel("Posting Volume Growth")
    ax.set_ylabel("Density")
    ax.set_title("Posting Volume Growth Distribution (Surge vs Non-Surge)")
    ax.legend()
    _save_figure(fig, "06_volume_growth_by_label")

    # --- Figure 3b: |sentiment_change| by surge label ---
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, group in included.groupby("label"):
        ax.hist(
            group["abs_sentiment_change"].clip(0, 1),
            bins=50, alpha=0.6, label=label, density=True,
        )
    ax.set_xlabel("|Sentiment Change|")
    ax.set_ylabel("Density")
    ax.set_title("|Sentiment Change| Distribution (Surge vs Non-Surge)")
    ax.legend()
    _save_figure(fig, "07_sentiment_change_by_label")

    # Log basic stats
    surge = included[included["surge_label"] == 1.0]
    no_surge = included[included["surge_label"] == 0.0]
    logger.info(
        "Volume growth — surge mean: %.3f | no-surge mean: %.3f",
        surge["posting_volume_growth"].mean(),
        no_surge["posting_volume_growth"].mean(),
    )
    logger.info(
        "|Sentiment change| — surge mean: %.4f | no-surge mean: %.4f",
        surge["abs_sentiment_change"].mean(),
        no_surge["abs_sentiment_change"].mean(),
    )


# ---------------------------------------------------------------------------
# Focus Area 4: Phase 1 vs Phase 2 Comparison (R2-AC5)
# ---------------------------------------------------------------------------


def plot_phase_comparison(df: pd.DataFrame) -> None:
    """Compare Phase 1 (volume-only) vs Phase 2 (composite) label agreement.

    Quantifies sentiment's marginal contribution by relabelling with w2=0
    and comparing against the composite labels.
    """
    logger.info("=" * 60)
    logger.info("FOCUS AREA 4: Phase 1 vs Phase 2 Comparison")
    logger.info("=" * 60)

    # Filter to included records with valid labels
    included = df[(~df["excluded"]) & (df["surge_label"].notna())].copy()

    # Phase 2 labels are the existing surge_label (w1=0.5, w2=0.5, tau=1.5)
    phase2_labels = included["surge_label"].values

    # Phase 1 labels: volume-only (composite = 0.5 * z_volume + 0 * z_sentiment)
    # Equivalent to: surge if z_volume > tau / w1 = 1.5 / 0.5 = 3.0
    # But more accurately: composite_phase1 = 0.5 * z_volume, label if > 1.5
    composite_phase1 = 0.5 * included["z_volume"].values
    phase1_labels = np.where(composite_phase1 > 1.5, 1.0, 0.0)

    # Agreement analysis
    agree = (phase1_labels == phase2_labels).sum()
    disagree = len(included) - agree
    agreement_rate = agree / len(included) * 100 if len(included) > 0 else 0

    # Confusion-style breakdown
    both_surge = ((phase1_labels == 1.0) & (phase2_labels == 1.0)).sum()
    both_no_surge = ((phase1_labels == 0.0) & (phase2_labels == 0.0)).sum()
    phase1_only = ((phase1_labels == 1.0) & (phase2_labels == 0.0)).sum()
    phase2_only = ((phase1_labels == 0.0) & (phase2_labels == 1.0)).sum()

    logger.info("Phase comparison — agreement: %.1f%%", agreement_rate)
    logger.info(
        "Both surge: %d | Both no-surge: %d | Phase 1 only: %d | Phase 2 only: %d",
        both_surge, both_no_surge, phase1_only, phase2_only,
    )

    # --- Figure 4: Label agreement heatmap ---
    confusion = np.array([[both_no_surge, phase2_only],
                          [phase1_only, both_surge]])
    labels_matrix = [["No Surge", "Surge"], ["No Surge", "Surge"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Surge", "Surge"],
        yticklabels=["No Surge", "Surge"],
        ax=ax,
    )
    ax.set_xlabel("Phase 2 (Composite) Label")
    ax.set_ylabel("Phase 1 (Volume-Only) Label")
    ax.set_title(
        f"Phase 1 vs Phase 2 Label Agreement\n"
        f"(Agreement: {agreement_rate:.1f}%, Sentiment marginal: {disagree} records)"
    )
    _save_figure(fig, "08_phase1_vs_phase2_agreement")


# ---------------------------------------------------------------------------
# Focus Area 5: Threshold Sensitivity (R2-AC6)
# ---------------------------------------------------------------------------


def plot_threshold_sensitivity(threshold_df: pd.DataFrame) -> None:
    """Visualise threshold sensitivity: surge rate curve with viable region.

    Uses pre-computed threshold sensitivity data.
    """
    logger.info("=" * 60)
    logger.info("FOCUS AREA 5: Threshold Sensitivity")
    logger.info("=" * 60)

    logger.info("Threshold sensitivity data:\n%s", threshold_df.to_string(index=False))

    # --- Figure 5: Surge rate curve across τ values ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        threshold_df["threshold"], threshold_df["surge_rate"],
        marker="o", color="steelblue", linewidth=2, markersize=8,
        label="Surge Rate (%)",
    )

    # Highlight viable region (5-10% surge rate)
    ax.axhspan(5, 10, alpha=0.15, color="green", label="Viable Region (5–10%)")
    ax.axhline(y=5, color="green", linestyle=":", linewidth=0.8)
    ax.axhline(y=10, color="green", linestyle=":", linewidth=0.8)

    # Mark viable thresholds
    viable = threshold_df[threshold_df["viable"]]
    if not viable.empty:
        ax.scatter(
            viable["threshold"], viable["surge_rate"],
            color="green", s=120, zorder=5, marker="*", label="Viable τ",
        )

    ax.set_xlabel("Threshold (τ)")
    ax.set_ylabel("Surge Rate (%)")
    ax.set_title("Threshold Sensitivity: Surge Rate vs τ")
    ax.legend(loc="upper right")
    ax.set_xticks(threshold_df["threshold"].values)
    _save_figure(fig, "09_threshold_sensitivity_curve")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_eda(
    labelled_path: str | Path | None = None,
    threshold_path: str | Path | None = None,
) -> None:
    """Run the full EDA pipeline and produce all figures.

    Parameters
    ----------
    labelled_path : str or Path, optional
        Path to the labelled dataset CSV. Defaults to LABELLED_DATASET_PATH.
    threshold_path : str or Path, optional
        Path to the threshold sensitivity CSV. Defaults to THRESHOLD_SENSITIVITY_PATH.
    """
    labelled_path = Path(labelled_path) if labelled_path else LABELLED_DATASET_PATH
    threshold_path = Path(threshold_path) if threshold_path else THRESHOLD_SENSITIVITY_PATH

    logger.info("=" * 60)
    logger.info("EDA PIPELINE — Starting")
    logger.info("=" * 60)
    logger.info("Labelled dataset: %s", labelled_path)
    logger.info("Threshold sensitivity: %s", threshold_path)

    # Load data
    logger.info("Loading labelled dataset...")
    df = pd.read_csv(labelled_path)
    logger.info("Loaded %d records with %d columns", len(df), len(df.columns))

    threshold_df = pd.read_csv(threshold_path)
    logger.info("Loaded threshold sensitivity: %d rows", len(threshold_df))

    # Ensure figures directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Run all focus areas
    plot_dataset_overview(df)
    plot_sparsity_analysis(df)
    plot_feature_distributions(df)
    plot_phase_comparison(df)
    plot_threshold_sensitivity(threshold_df)

    logger.info("=" * 60)
    logger.info("EDA PIPELINE — Complete. Figures saved to %s/", FIGURES_DIR)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_eda()
