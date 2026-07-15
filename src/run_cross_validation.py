"""Cross-dataset validation: evaluate trained models on a different subreddit.

Tests whether models trained on one subreddit (e.g., r/wallstreetbets)
generalise to another (e.g., r/pennystocks), measuring transfer of
learned surge dynamics across different communities.

Usage:
    # Evaluate WSB-trained models on pennystocks:
    python run_cross_validation.py \
        --model-dir ../output/models \
        --eval-data ../output/processed/2026-07-13_06-05_labelled_dataset.csv \
        --notes "C1 cross-val: WSB-trained models on r/pennystocks"

    # Evaluate on a specific partition (default: all non-excluded):
    python run_cross_validation.py \
        --model-dir ../output/models \
        --eval-data ../output/processed/labelled_dataset.csv \
        --partition test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from surge_pipeline.features import FEATURE_COLUMNS, compute_features
from surge_pipeline.experiment_log import append_experiment

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-dataset validation: evaluate trained models on a different dataset."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="../output/models",
        help="Directory containing saved .joblib model files.",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        required=True,
        help="Path to the labelled dataset CSV to evaluate on.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="Partition to evaluate ('train', 'test', or None for all non-excluded).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../output/evaluation",
        help="Output directory for cross-validation results.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-text annotation for the experiment log.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging.",
    )
    return parser.parse_args(argv)


def load_models(model_dir: str) -> dict[str, dict]:
    """Load all .joblib model files from the given directory.

    Returns
    -------
    dict[str, dict]
        Mapping of model name -> {"model": ..., "scaler": ..., "params": ...}
    """
    model_path = Path(model_dir)
    models = {}

    for joblib_file in sorted(model_path.glob("*.joblib")):
        # Extract model name from filename pattern: {name}_{phase}_{seed}.joblib
        parts = joblib_file.stem.rsplit("_", 2)
        if len(parts) >= 3:
            name = parts[0]
        else:
            name = joblib_file.stem

        data = joblib.load(joblib_file)
        models[name] = data
        logger.info("Loaded model: %s from %s", name, joblib_file.name)

    return models


def evaluate_cross_dataset(
    models: dict[str, dict],
    df: pd.DataFrame,
    partition: str | None = None,
) -> list[dict]:
    """Evaluate loaded models on a dataset they were NOT trained on.

    Parameters
    ----------
    models : dict[str, dict]
        Loaded model dictionaries with 'model', 'scaler', and optionally
        'optimal_threshold' keys.
    df : pd.DataFrame
        Labelled dataset with features computed.
    partition : str, optional
        If specified, evaluate only on this partition ('train' or 'test').
        If None, evaluate on all non-excluded records.

    Returns
    -------
    list[dict]
        List of per-model result dictionaries (includes both default
        and tuned-threshold metrics).
    """
    # Filter to non-excluded records
    mask = ~df["excluded"].astype(bool)
    if partition:
        mask = mask & (df["partition"] == partition)

    eval_df = df.loc[mask]
    X = eval_df[FEATURE_COLUMNS].values.astype(np.float64)
    y = eval_df["surge_label"].values.astype(np.int64)

    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)

    logger.info(
        "Evaluation set: %d samples (positive=%d, negative=%d, surge_rate=%.2f%%)",
        len(y), n_positive, n_negative, n_positive / len(y) * 100,
    )

    results = []

    for name, model_data in models.items():
        model = model_data["model"]
        scaler = model_data["scaler"]
        optimal_threshold = model_data.get("optimal_threshold", 0.5)

        # Scale features using the model's original scaler
        X_scaled = scaler.transform(X)

        # Predict at default threshold (0.5)
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        # Predict at tuned threshold
        y_pred_tuned = (y_prob >= optimal_threshold).astype(int)

        # Compute metrics at default threshold
        if len(np.unique(y)) < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y, y_prob))

        prec = float(precision_score(y, y_pred, zero_division=0.0))
        rec = float(recall_score(y, y_pred, zero_division=0.0))
        f1_val = float(f1_score(y, y_pred, zero_division=0.0))
        acc = float(accuracy_score(y, y_pred))
        cm = confusion_matrix(y, y_pred, labels=[0, 1]).tolist()

        # Compute metrics at tuned threshold
        prec_tuned = float(precision_score(y, y_pred_tuned, zero_division=0.0))
        rec_tuned = float(recall_score(y, y_pred_tuned, zero_division=0.0))
        f1_tuned = float(f1_score(y, y_pred_tuned, zero_division=0.0))
        acc_tuned = float(accuracy_score(y, y_pred_tuned))
        cm_tuned = confusion_matrix(y, y_pred_tuned, labels=[0, 1]).tolist()

        result = {
            "model_name": name,
            "roc_auc": auc,
            "precision": prec,
            "recall": rec,
            "f1": f1_val,
            "accuracy": acc,
            "support_positive": n_positive,
            "support_negative": n_negative,
            "confusion_matrix": cm,
            "optimal_threshold": optimal_threshold,
            "tuned_threshold_metrics": {
                "precision": prec_tuned,
                "recall": rec_tuned,
                "f1": f1_tuned,
                "accuracy": acc_tuned,
                "confusion_matrix": cm_tuned,
            },
        }
        results.append(result)

        logger.info(
            "  %s: AUC=%.4f | Default(0.5): P=%.4f R=%.4f F1=%.4f | "
            "Tuned(%.2f): P=%.4f R=%.4f F1=%.4f",
            name, auc, prec, rec, f1_val,
            optimal_threshold, prec_tuned, rec_tuned, f1_tuned,
        )

    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 60)
    print("CROSS-DATASET VALIDATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load trained models
    # ------------------------------------------------------------------
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error("Model directory not found: %s", model_dir)
        sys.exit(1)

    print(f"\n  Model directory: {model_dir}")
    models = load_models(str(model_dir))
    print(f"  Models loaded: {list(models.keys())}")

    # ------------------------------------------------------------------
    # 2. Load evaluation dataset
    # ------------------------------------------------------------------
    eval_path = Path(args.eval_data)
    if not eval_path.exists():
        logger.error("Evaluation data not found: %s", eval_path)
        sys.exit(1)

    print(f"\n  Evaluation dataset: {eval_path.name}")
    df = pd.read_csv(eval_path)
    print(f"  Dataset shape: {df.shape}")

    # ------------------------------------------------------------------
    # 3. Compute features if not present
    # ------------------------------------------------------------------
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        print(f"  Computing features (missing: {missing_features})...")
        df = compute_features(df)
    else:
        print("  All features already present.")

    # ------------------------------------------------------------------
    # 4. Evaluate models on the cross-dataset
    # ------------------------------------------------------------------
    partition_label = args.partition or "all (non-excluded)"
    print(f"\n  Partition: {partition_label}")
    print("-" * 60)

    results = evaluate_cross_dataset(models, df, partition=args.partition)

    # ------------------------------------------------------------------
    # 5. Print results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    print(f"\n  {'Model':<25} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    best_result = None
    for r in results:
        print(
            f"  {r['model_name']:<25} {r['roc_auc']:>8.4f} "
            f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}"
        )
        if best_result is None or r["roc_auc"] > best_result["roc_auc"]:
            best_result = r

    print(f"\n  Best model: {best_result['model_name']} (AUC={best_result['roc_auc']:.4f})")

    # Show tuned threshold results
    print(f"\n  {'Model':<25} {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    for r in results:
        t = r["optimal_threshold"]
        tm = r["tuned_threshold_metrics"]
        f1_gain = tm["f1"] - r["f1"]
        print(
            f"  {r['model_name']:<25} {t:>10.2f} "
            f"{tm['precision']:>10.4f} {tm['recall']:>8.4f} {tm['f1']:>8.4f}"
            f"  ({f1_gain:+.3f})"
        )

    # Support info
    print(f"\n  Test support: {best_result['support_positive']} surges, "
          f"{best_result['support_negative']} non-surges")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    prefix = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "cross_dataset_validation",
        "model_source": str(model_dir.resolve()),
        "eval_dataset": str(eval_path.resolve()),
        "partition": args.partition,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": {
            "best_model": best_result["model_name"],
            "best_auc": best_result["roc_auc"],
            "num_models": len(results),
        },
    }

    out_path = out_dir / f"{prefix}_cross_validation.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n  Results saved: {out_path}")

    # ------------------------------------------------------------------
    # 7. Append to experiment log
    # ------------------------------------------------------------------
    append_experiment(
        run_id=prefix,
        pipeline="cross_validation",
        config={
            "model_dir": str(model_dir),
            "eval_data": eval_path.name,
            "partition": args.partition,
        },
        outputs=[str(out_path)],
        summary={
            "best_model": best_result["model_name"],
            "best_auc": best_result["roc_auc"],
        },
        notes=args.notes,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
