"""CLI entry point for model training and evaluation.

Runs the baseline Logistic Regression model with temporal cross-validation,
then evaluates on the held-out test set with Precision, Recall, F1, ROC-AUC.

Usage:
    python run_training.py
    python run_training.py --data-path ../output/processed/labelled_dataset.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the src directory is on the path so surge_pipeline is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from surge_pipeline.config import PipelineConfig  # noqa: E402
from surge_pipeline.features import compute_features, FEATURE_COLUMNS  # noqa: E402
from surge_pipeline.training import train_logistic_regression  # noqa: E402
from surge_pipeline.evaluation import evaluate_model, save_evaluation_results  # noqa: E402
from surge_pipeline.evaluation import generate_evaluation_figures  # noqa: E402
from surge_pipeline.training import predict  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline model and evaluate with Precision, Recall, F1, ROC-AUC."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../output/processed/labelled_dataset.csv",
        help="Path to the labelled dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../output/evaluation",
        help="Output directory for evaluation results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        default=False,
        help="Skip evaluation figure generation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("BASELINE MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load labelled dataset
    # ------------------------------------------------------------------
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    logger.info("Loading labelled dataset from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Dataset shape: %s", df.shape)

    # ------------------------------------------------------------------
    # 2. Compute features (if not already present)
    # ------------------------------------------------------------------
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        logger.info("Computing features (missing: %s)...", missing_features)
        df = compute_features(df)
    else:
        logger.info("All features already present in dataset.")

    # ------------------------------------------------------------------
    # 3. Train Logistic Regression with temporal CV
    # ------------------------------------------------------------------
    config = PipelineConfig(random_seed=args.seed)
    result = train_logistic_regression(df, config, seed=args.seed)

    # ------------------------------------------------------------------
    # 4. Evaluate on test set
    # ------------------------------------------------------------------
    metrics = evaluate_model(result, df, partition="test")

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS: Logistic Regression Baseline")
    print("=" * 60)
    print(f"  Best hyperparameters: C={result.best_params['C']}, "
          f"l1_ratio={result.best_params['l1_ratio']}")
    print(f"  CV AUC-ROC (mean±std): {result.mean_cv_score:.4f} ± {result.std_cv_score:.4f}")
    print(f"  CV fold scores: {[f'{s:.4f}' for s in result.cv_scores]}")
    print(f"  Training duration: {result.training_duration_seconds:.2f}s")
    print()
    print("  Test Set Metrics:")
    print(f"    Precision : {metrics.precision:.4f}")
    print(f"    Recall    : {metrics.recall:.4f}")
    print(f"    F1-score  : {metrics.f1:.4f}")
    print(f"    ROC-AUC   : {metrics.roc_auc:.4f}")
    print()
    cm = metrics.confusion_matrix
    print("  Confusion Matrix:")
    print(f"    TN={cm[0][0]:>5d}  FP={cm[0][1]:>5d}")
    print(f"    FN={cm[1][0]:>5d}  TP={cm[1][1]:>5d}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 6. Generate evaluation figures (R16)
    # ------------------------------------------------------------------
    if not args.no_figures:
        y_true, y_pred, y_prob = predict(result, df, partition="test")
        figure_paths = generate_evaluation_figures(
            y_true, y_pred, y_prob, model_name=result.model_name
        )
        print(f"\n  Evaluation figures saved:")
        for p in figure_paths:
            print(f"    {p}")
    else:
        logger.info("Skipping figure generation (--no-figures).")

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    out_path = save_evaluation_results([metrics], output_dir=args.output_dir)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
