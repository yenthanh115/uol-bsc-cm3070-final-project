"""Generate prediction examples (TP, FP, FN) from saved models + test set.

Loads a serialised model from disk, runs predictions on the test partition
of a labelled dataset, and outputs sample records for each classification
outcome (True Positive, False Positive, False Negative).

Usage:
    python src/generate_prediction_examples.py \
        --data-path output/processed/2026-07-19_07-49_labelled_dataset.csv \
        --models-dir output/models/A2 \
        --phase phase2 \
        --seed 42 \
        --model-name xgboost \
        --n-examples 5 \
        --output-path output/evaluation/prediction_examples_A2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from surge_pipeline.features import FEATURE_COLUMNS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prediction examples (TP/FP/FN) from saved models."
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to the labelled dataset CSV."
    )
    parser.add_argument(
        "--models-dir", type=str, required=True,
        help="Directory containing serialised .joblib model files."
    )
    parser.add_argument(
        "--phase", type=str, default="phase2",
        help="Phase label used in model filename (default: phase2)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed used in model filename (default: 42)."
    )
    parser.add_argument(
        "--model-name", type=str, default="xgboost",
        choices=["logistic_regression", "random_forest", "xgboost"],
        help="Which model to generate examples for (default: xgboost)."
    )
    parser.add_argument(
        "--n-examples", type=int, default=5,
        help="Number of examples per category (TP, FP, FN). Default: 5."
    )
    parser.add_argument(
        "--use-tuned-threshold", action="store_true", default=True,
        help="Use the validation-fold tuned threshold instead of 0.5."
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Output JSON path. Defaults to output/evaluation/prediction_examples_{model}.json"
    )
    return parser.parse_args(argv)


def load_model(models_dir: str, model_name: str, phase: str, seed: int) -> dict:
    """Load a serialised model dict from disk."""
    model_path = Path(models_dir) / f"{model_name}_{phase}_{seed}.joblib"
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    return joblib.load(model_path)


def extract_examples(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_examples: int,
) -> dict:
    """Extract TP, FP, FN examples with context columns."""

    # Classification outcome masks
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tn_mask = (y_true == 0) & (y_pred == 0)

    # Columns to include in output (human-readable context + features)
    context_cols = ["id", "ticker", "created_utc", "title"]
    feature_cols = FEATURE_COLUMNS
    display_cols = [c for c in context_cols + feature_cols if c in df.columns]

    def _sample(mask: np.ndarray, category: str, sort_descending: bool = True):
        """Sample n examples, sorted by predicted probability."""
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []

        # Sort by probability (highest first for TP/FP, lowest first for FN)
        probs_subset = y_prob[indices]
        if sort_descending:
            order = np.argsort(-probs_subset)
        else:
            order = np.argsort(probs_subset)

        selected = indices[order[:n_examples]]
        examples = []
        for idx in selected:
            row = df.iloc[idx]
            example = {
                "category": category,
                "predicted_probability": round(float(y_prob[idx]), 4),
                "actual_label": int(y_true[idx]),
                "predicted_label": int(y_pred[idx]),
            }
            for col in display_cols:
                val = row[col]
                if pd.isna(val):
                    example[col] = None
                elif isinstance(val, (np.integer, np.int64)):
                    example[col] = int(val)
                elif isinstance(val, (np.floating, np.float64)):
                    example[col] = round(float(val), 4)
                else:
                    example[col] = str(val)
            examples.append(example)
        return examples

    results = {
        "true_positives": _sample(tp_mask, "TP", sort_descending=True),
        "false_positives": _sample(fp_mask, "FP", sort_descending=True),
        "false_negatives": _sample(fn_mask, "FN", sort_descending=False),
        "true_negatives_sample": _sample(tn_mask, "TN", sort_descending=False),
        "counts": {
            "TP": int(tp_mask.sum()),
            "FP": int(fp_mask.sum()),
            "FN": int(fn_mask.sum()),
            "TN": int(tn_mask.sum()),
        },
    }
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load dataset
    print(f"Loading dataset: {args.data_path}")
    df = pd.read_csv(args.data_path, low_memory=False)
    print(f"  Total records: {len(df)}")

    # Compute features if missing (some older labelled CSVs don't include them)
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        print(f"  Computing {len(missing_features)} missing feature columns...")
        from surge_pipeline.features import compute_features
        df = compute_features(df)

    # Filter to test partition (non-excluded)
    test_mask = (df["partition"] == "test") & (~df["excluded"].astype(bool))
    test_df = df.loc[test_mask].reset_index(drop=True)
    print(f"Test partition: {len(test_df)} records")

    X_test = test_df[FEATURE_COLUMNS].values.astype(np.float64)
    y_test = test_df["surge_label"].values.astype(np.int64)

    # Load model
    print(f"Loading model: {args.model_name} ({args.phase}, seed={args.seed})")
    model_dict = load_model(args.models_dir, args.model_name, args.phase, args.seed)
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    optimal_threshold = model_dict.get("optimal_threshold", 0.5)

    # Generate predictions
    X_scaled = scaler.transform(X_test)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Apply threshold
    threshold = optimal_threshold if args.use_tuned_threshold else 0.5
    y_pred = (y_prob >= threshold).astype(np.int64)

    print(f"Threshold used: {threshold:.4f} ({'tuned' if args.use_tuned_threshold else 'default'})")
    print(f"Predictions: {int(y_pred.sum())} positives out of {len(y_pred)} records")

    # Extract examples
    examples = extract_examples(test_df, y_test, y_pred, y_prob, args.n_examples)

    # Add metadata
    output = {
        "metadata": {
            "model_name": args.model_name,
            "models_dir": args.models_dir,
            "data_path": args.data_path,
            "phase": args.phase,
            "seed": args.seed,
            "threshold": threshold,
            "threshold_type": "tuned" if args.use_tuned_threshold else "default",
            "n_test_records": len(test_df),
            "n_surges_actual": int(y_test.sum()),
        },
        **examples,
    }

    # Save output
    if args.output_path:
        out_path = Path(args.output_path)
    else:
        out_path = Path("output/evaluation") / f"prediction_examples_{args.model_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_path}")
    print(f"  TP examples: {len(examples['true_positives'])} (of {examples['counts']['TP']} total)")
    print(f"  FP examples: {len(examples['false_positives'])} (of {examples['counts']['FP']} total)")
    print(f"  FN examples: {len(examples['false_negatives'])} (of {examples['counts']['FN']} total)")
    print(f"  TN sample:   {len(examples['true_negatives_sample'])} (of {examples['counts']['TN']} total)")


if __name__ == "__main__":
    main()
