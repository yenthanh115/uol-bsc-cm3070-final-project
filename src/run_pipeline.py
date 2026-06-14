"""CLI entry point for the surge-labelling pipeline.

Usage:
    python run_pipeline.py --file-path data/raw/dataset.csv
    python run_pipeline.py --config pipeline_config.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the src directory is on the path so surge_pipeline is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surge_pipeline.config import PipelineConfig  # noqa: E402


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
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory.")
    parser.add_argument(
        "--temporal-split-ratio", type=float, default=0.8, help="Train/test temporal split ratio."
    )
    parser.add_argument(
        "--min-window-count", type=int, default=3, help="Minimum window count for filtering."
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
    )


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the pipeline CLI."""
    args = parse_args(argv)
    config = build_config(args)

    # Save config for audit trail (R7-AC3)
    config_path = config.save_json()
    print(f"[surge_pipeline] Configuration saved to {config_path}")
    print(f"[surge_pipeline] Random seed: {config.random_seed}")
    print(f"[surge_pipeline] Threshold τ: {config.threshold_tau}")
    print(f"[surge_pipeline] Sweep thresholds: {config.thresholds}")

    # TODO: Pipeline stages will be added in subsequent tasks.
    print("[surge_pipeline] Pipeline execution placeholder — no stages implemented yet.")


if __name__ == "__main__":
    main()
