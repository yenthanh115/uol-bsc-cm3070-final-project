"""CLI entry point for the surge-labelling pipeline.

Usage:
    python run_labeling.py --file-path ../input/raw/dataset.csv
    python run_labeling.py --config pipeline_config.json
    python run_labeling.py --sweep-only  # Run threshold sweep only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the src directory is on the path so surge_pipeline is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surge_pipeline.config import PipelineConfig  # noqa: E402
from surge_pipeline.experiment_log import append_experiment  # noqa: E402
from surge_pipeline.pipeline import run_pipeline, run_threshold_sweep, save_outputs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Surge-labelling pipeline for trend prediction."
    )

    # --- Config file shortcut ---
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file. Overrides all other arguments.",
    )

    # --- Individual parameters ---
    parser.add_argument("--file-path", type=str, default="", help="Input data file path.")
    parser.add_argument("--output-dir", type=str, default="output/processed", help="Output directory.")
    parser.add_argument(
        "--temporal-split-ratio", type=float, default=0.8, help="Train/test temporal split ratio."
    )
    parser.add_argument(
        "--min-window-count", type=int, default=1, help="Minimum window count for filtering."
    )
    parser.add_argument(
        "--threshold-tau", type=float, default=1.5, help="Surge threshold τ (tau)."
    )
    parser.add_argument(
        "--weight-volume", type=float, default=0.5, help="Weight for volume in composite score."
    )
    parser.add_argument(
        "--weight-sentiment", type=float, default=0.5, help="Weight for sentiment in composite score."
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 2.5],
        help="List of threshold values for sweep (batch mode).",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--sentiment-model",
        type=str,
        default="vader",
        choices=["vader", "textblob"],
        help="Sentiment model to use (default: vader).",
    )

    # --- Mode flags ---
    parser.add_argument(
        "--sweep-only",
        action="store_true",
        default=False,
        help="Run threshold sweep only (skip full labelling output).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-text annotation for the experiment log.",
    )

    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build a PipelineConfig from parsed CLI arguments or config file."""
    if args.config:
        return PipelineConfig.load_json(args.config)

    return PipelineConfig(
        file_path=args.file_path,
        output_dir=args.output_dir,
        temporal_split_ratio=args.temporal_split_ratio,
        min_window_count=args.min_window_count,
        threshold_tau=args.threshold_tau,
        weight_volume=args.weight_volume,
        weight_sentiment=args.weight_sentiment,
        thresholds=args.thresholds,
        random_seed=args.random_seed,
        sentiment_model=args.sentiment_model,
    )


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the pipeline CLI."""
    args = parse_args(argv)
    _setup_logging(verbose=args.verbose)

    config = build_config(args)

    print("=" * 60)
    print("SURGE-LABELLING PIPELINE")
    print("=" * 60)
    print(f"  Random seed:       {config.random_seed}")
    print(f"  Sentiment model:   {config.sentiment_model}")
    print(f"  Threshold τ:       {config.threshold_tau}")
    print(f"  Sweep thresholds:  {config.thresholds}")
    print(f"  Output directory:  {config.output_dir}")
    print(f"  Input file:        {config.file_path or '(synthetic data)'}")
    print("=" * 60)

    if args.sweep_only:
        # Threshold sweep mode only
        print("\n[MODE] Threshold sweep only\n")
        sweep_df = run_threshold_sweep(config)

        # Print table to console (R8-AC3)
        print("\nThreshold Sensitivity Table:")
        print("-" * 70)
        print(f"{'τ':<10} {'Surge':<12} {'No-Surge':<14} {'Rate(%)':<12} {'Imbalance':<16} {'Viable':<8}")
        print("-" * 70)
        for _, row in sweep_df.iterrows():
            viable_flag = "✓" if row["viable"] else "✗"
            print(
                f"{row['threshold']:<10.2f} {int(row['surge_count']):<12d} "
                f"{int(row['no_surge_count']):<14d} {row['surge_rate']:<12.2f} "
                f"{row['imbalance_ratio']:<16.2f} {viable_flag:<8}"
            )
        print("-" * 70)

        # Save sweep table
        from datetime import datetime
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now().strftime("%Y-%m-%d_%H-%M")
        sweep_path = output_dir / f"{prefix}_threshold_sensitivity.csv"
        sweep_df.to_csv(sweep_path, index=False)
        print(f"\nSweep table saved to: {sweep_path}")

        # Log experiment
        viable_count = int(sweep_df["viable"].sum())
        append_experiment(
            run_id=prefix,
            pipeline="sweep",
            config={
                "thresholds": config.thresholds,
                "sentiment_model": config.sentiment_model,
                "weight_volume": config.weight_volume,
                "weight_sentiment": config.weight_sentiment,
                "random_seed": config.random_seed,
            },
            outputs=[str(sweep_path)],
            summary={
                "n_thresholds": len(config.thresholds),
                "viable_thresholds": viable_count,
            },
            notes=args.notes,
        )

    else:
        # Full pipeline mode
        print("\n[MODE] Full pipeline execution\n")

        # Run full pipeline
        results = run_pipeline(config)

        # Save all outputs
        output_paths = save_outputs(results, config)

        # Print summary to console
        print("\n" + "=" * 60)
        print("PIPELINE RESULTS SUMMARY")
        print("=" * 60)

        stage_counts = results["stage_counts"]
        print(f"  Records loaded:      {stage_counts.get('after_load', 'N/A')}")
        print(f"  After windowing:     {stage_counts.get('after_windowing', 'N/A')}")
        print(f"  After sentiment:     {stage_counts.get('after_sentiment', 'N/A')}")
        print(f"  After labelling:     {stage_counts.get('after_labelling', 'N/A')}")

        if "excluded_count" in stage_counts:
            total = stage_counts["after_windowing"]
            excluded = stage_counts["excluded_count"]
            rate = excluded / total * 100 if total > 0 else 0
            print(f"  Excluded:            {excluded} ({rate:.1f}%)")

        # Class distribution
        class_dist = results.get("class_distributions", {}).get("all", {})
        if class_dist:
            print(f"\n  Class distribution (all included):")
            print(f"    Surge:        {class_dist.get('surge_count', 0)}")
            print(f"    No-Surge:     {class_dist.get('no_surge_count', 0)}")
            print(f"    Surge rate:   {class_dist.get('surge_rate', 0.0):.2f}%")
            print(f"    Imbalance:    {class_dist.get('imbalance_ratio', 0.0):.2f}:1")

        # Output paths
        print(f"\n  Output files:")
        for key, path in output_paths.items():
            print(f"    {key}: {path}")

        # Log experiment
        from datetime import datetime
        prefix = datetime.now().strftime("%Y-%m-%d_%H-%M")
        append_experiment(
            run_id=prefix,
            pipeline="labelling",
            config={
                "threshold_tau": config.threshold_tau,
                "weight_volume": config.weight_volume,
                "weight_sentiment": config.weight_sentiment,
                "sentiment_model": config.sentiment_model,
                "min_window_count": config.min_window_count,
                "surge_method": config.surge_method,
                "temporal_split_ratio": config.temporal_split_ratio,
                "random_seed": config.random_seed,
            },
            outputs=list(output_paths.values()),
            summary={
                "train_size": results.get("class_distributions", {}).get("train", {}).get("total", 0),
                "test_size": results.get("class_distributions", {}).get("test", {}).get("total", 0),
                "surge_rate": results.get("class_distributions", {}).get("all", {}).get("surge_rate", 0.0),
                "exclusion_rate": results.get("exclusion_rate", 0.0),
            },
            notes=args.notes,
        )

        print("\n" + "=" * 60)
        print("DONE")
        print("=" * 60)


if __name__ == "__main__":
    main()
