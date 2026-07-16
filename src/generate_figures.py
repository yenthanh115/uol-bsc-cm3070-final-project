"""Standalone figure generation from saved models.

Generates evaluation figures (confusion matrices, ROC curves, threshold
sensitivity) without re-running the full training pipeline.

Usage:
    # Generate Phase 2 figures (default)
    python generate_figures.py

    # Generate Phase 1 figures
    python generate_figures.py --phase phase1

    # Custom data path and output directory
    python generate_figures.py --data-path output/processed/my_dataset.csv \
        --models-dir ../output/models --figures-dir ../output/figures/evaluation

    # Tag figures with a prefix to avoid overwriting
    python generate_figures.py --phase phase1 --prefix "phase1_"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Ensure surge_pipeline is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surge_pipeline.evaluation import (
    generate_evaluation_figures,
    plot_roc_curve_combined,
)
from surge_pipeline.features import FEATURE_COLUMNS


MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluation figures from saved models (no retraining)."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the labelled dataset CSV. If not specified, resolves from latest_outputs.json.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="../output/models",
        help="Directory containing saved .joblib models.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="../output/figures/evaluation",
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="phase2",
        choices=["phase1", "phase2"],
        help="Which phase models to load (affects filename pattern).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in model filename.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output figure filenames (e.g., 'phase1_' to avoid overwriting).",
    )
    return parser.parse_args(argv)


def resolve_data_path() -> str:
    """Resolve data path from output/processed/latest_outputs.json."""
    import json

    latest_file = Path(__file__).resolve().parent.parent / "output" / "processed" / "latest_outputs.json"
    if latest_file.exists():
        data = json.loads(latest_file.read_text(encoding="utf-8"))
        rel_path = data.get("outputs", {}).get("labelled_dataset", "")
        if rel_path:
            candidate = (latest_file.parent / rel_path).resolve()
            if candidate.exists():
                return str(candidate)
            candidate = (Path(__file__).resolve().parent / rel_path).resolve()
            if candidate.exists():
                return str(candidate)
    # Fallback
    return str(Path(__file__).resolve().parent.parent / "output" / "processed" / "labelled_dataset.csv")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve data path
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = Path(resolve_data_path())

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    print("=" * 60)
    print("STANDALONE FIGURE GENERATION")
    print("=" * 60)
    print(f"  Data path  : {data_path}")
    print(f"  Models dir : {args.models_dir}")
    print(f"  Figures dir: {args.figures_dir}")
    print(f"  Phase      : {args.phase}")
    print(f"  Seed       : {args.seed}")
    print(f"  Prefix     : {args.prefix or '(none)'}")

    # Load dataset and prepare test set
    print("\n  Loading dataset...")
    df = pd.read_csv(data_path)
    test_mask = (df["partition"] == "test") & (~df["excluded"].astype(bool))
    test_df = df.loc[test_mask]

    X_test = test_df[FEATURE_COLUMNS].values.astype(np.float64)
    y_test = test_df["surge_label"].values.astype(np.int64)

    n_positive = int(y_test.sum())
    n_negative = int(len(y_test) - n_positive)
    print(f"  Test set: {len(y_test)} samples ({n_positive} surges, {n_negative} non-surges)")

    # Load models and generate figures
    models_dir = Path(args.models_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    roc_data = []
    generated_paths = []

    for name in MODEL_NAMES:
        model_filename = f"{name}_{args.phase}_{args.seed}.joblib"
        model_path = models_dir / model_filename

        if not model_path.exists():
            print(f"\n  WARNING: Model not found: {model_path} — skipping.")
            continue

        print(f"\n  Loading {name}...")
        model_dict = joblib.load(model_path)
        model = model_dict["model"]
        scaler = model_dict["scaler"]
        threshold = model_dict.get("optimal_threshold", 0.5)

        # Generate predictions
        X_scaled = scaler.transform(X_test)
        y_prob = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        print(f"    Threshold: {threshold:.3f}")
        print(f"    Predictions: {int(y_pred.sum())} positive, {int(len(y_pred) - y_pred.sum())} negative")

        # Generate per-model figures
        display_name = f"{args.prefix}{name}" if args.prefix else name
        paths = generate_evaluation_figures(
            y_test, y_pred, y_prob,
            model_name=display_name,
            figures_dir=figures_dir,
        )
        generated_paths.extend(paths)

        roc_data.append((display_name, y_test, y_prob))

    # Combined ROC curve
    if roc_data:
        combined_path = plot_roc_curve_combined(roc_data, figures_dir=figures_dir)
        generated_paths.append(combined_path)

    # Summary
    print("\n" + "=" * 60)
    print(f"  Generated {len(generated_paths)} figures:")
    for p in generated_paths:
        print(f"    {p}")
    print("=" * 60)
    print("DONE")


if __name__ == "__main__":
    main()
