"""CLI entry point for multi-model training and advanced evaluation.

Trains Logistic Regression, Random Forest, and XGBoost with temporal
cross-validation, then performs full evaluation including:
  - Per-model metrics (Precision, Recall, F1, ROC-AUC)
  - McNemar's pairwise significance tests
  - Baseline comparisons (random, single-feature)
  - Bootstrap confidence intervals (1000 resamples)
  - Success tier validation
  - Final summary JSON

Usage:
    python run_training.py
    python run_training.py --data-path ../output/processed/labelled_dataset.csv
    python run_training.py --data-path ../output/processed/labelled_dataset.csv --output-dir ../output/evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure the src directory is on the path so surge_pipeline is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from surge_pipeline.config import PipelineConfig  # noqa: E402
from surge_pipeline.experiment_log import append_experiment  # noqa: E402
from surge_pipeline.features import compute_features, FEATURE_COLUMNS  # noqa: E402
from surge_pipeline.training import (  # noqa: E402
    train_models,
    get_training_summary,
    TrainingPipelineResult,
)
from surge_pipeline.evaluation import (  # noqa: E402
    ModelMetrics,
    compute_bootstrap_ci,
    evaluate_baselines,
    generate_evaluation_figures,
    mcnemar_pairwise_test,
    plot_roc_curve_combined,
    produce_final_summary,
    save_evaluation_results,
    validate_success_tiers,
    EvaluationMetrics,
)


def _resolve_default_data_path() -> str:
    """Resolve the latest labelled dataset path from latest_outputs.json."""
    latest_file = Path(__file__).resolve().parent.parent / "output" / "processed" / "latest_outputs.json"
    if latest_file.exists():
        try:
            data = json.loads(latest_file.read_text(encoding="utf-8"))
            rel_path = data.get("outputs", {}).get("labelled_dataset", "")
            if rel_path:
                # The stored path may be relative to a different CWD.
                # Try resolving from multiple bases:
                # 1. Relative to latest_outputs.json directory
                candidate = (latest_file.parent / rel_path).resolve()
                if candidate.exists():
                    return str(candidate)
                # 2. Relative to the src/ directory (where pipeline runs)
                candidate = (Path(__file__).resolve().parent / rel_path).resolve()
                if candidate.exists():
                    return str(candidate)
                # 3. Just use the filename in the processed directory
                filename = Path(rel_path).name
                candidate = latest_file.parent / filename
                if candidate.exists():
                    return str(candidate.resolve())
        except (json.JSONDecodeError, KeyError):
            pass
    return str(Path(__file__).resolve().parent.parent / "output" / "processed" / "labelled_dataset.csv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-model training and advanced evaluation pipeline."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=_resolve_default_data_path(),
        help="Path to the labelled dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../output/evaluation",
        help="Output directory for evaluation results.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="../output/models",
        help="Output directory for serialised models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--weight-sentiment",
        type=float,
        default=0.5,
        help="Sentiment weight (0.0 = phase1, >0 = phase2).",
    )
    parser.add_argument(
        "--threshold-tau",
        type=float,
        default=1.5,
        help="Surge threshold tau.",
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
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-text annotation for the experiment log.",
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
    print("MULTI-MODEL TRAINING & ADVANCED EVALUATION")
    print("=" * 60)

    pipeline_start = time.perf_counter()

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
    # 3. Multi-model training (LR, RF, XGBoost)
    # ------------------------------------------------------------------
    config = PipelineConfig(
        random_seed=args.seed,
        weight_sentiment=args.weight_sentiment,
        threshold_tau=args.threshold_tau,
    )

    training_result: TrainingPipelineResult = train_models(
        df, config, output_dir=args.models_dir
    )
    training_summary = get_training_summary(training_result)

    print(f"\n  Phase: {training_result.phase}")
    print(f"  Models trained: {list(training_result.models.keys())}")
    for name, tm in training_result.models.items():
        print(f"    {name}: CV AUC = {tm.best_cv_auc:.4f} | "
              f"{tm.n_configs_evaluated} configs | {tm.training_duration_seconds:.1f}s")

    # ------------------------------------------------------------------
    # 4. Evaluate all models on test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    # Prepare test partition
    test_mask = (df["partition"] == "test") & (~df["excluded"].astype(bool))
    test_df = df.loc[test_mask]
    X_test = test_df[FEATURE_COLUMNS].values.astype(np.float64)
    y_test = test_df["surge_label"].values.astype(np.int64)

    model_metrics: dict[str, ModelMetrics] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}

    for name, tm in training_result.models.items():
        X_scaled = tm.scaler.transform(X_test)
        y_pred = tm.model.predict(X_scaled)
        y_prob = tm.model.predict_proba(X_scaled)[:, 1]

        predictions[name] = y_pred
        probabilities[name] = y_prob

        # Compute metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix,
        )

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0.0))
        rec = float(recall_score(y_test, y_pred, zero_division=0.0))
        f1_val = float(f1_score(y_test, y_pred, zero_division=0.0))

        if len(np.unique(y_test)) < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y_test, y_prob))

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

        model_metrics[name] = ModelMetrics(
            model_name=name,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1_val,
            auc_roc=auc,
            confusion_matrix=cm,
            n_test_samples=len(y_test),
        )

        print(f"\n  {name}:")
        print(f"    Accuracy  : {acc:.4f}")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")
        print(f"    F1-score  : {f1_val:.4f}")
        print(f"    ROC-AUC   : {auc:.4f}")
        print(f"    Confusion : TN={cm[0][0]} FP={cm[0][1]} | FN={cm[1][0]} TP={cm[1][1]}")

    # ------------------------------------------------------------------
    # 5. McNemar's pairwise test (Phase 2.1)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("McNEMAR'S PAIRWISE TEST (Bonferroni-corrected)")
    print("=" * 60)

    mcnemar_results = mcnemar_pairwise_test(y_test, predictions)

    for r in mcnemar_results:
        sig_marker = "***" if r.is_significant else "   "
        print(f"  {r.model_a} vs {r.model_b}: "
              f"χ²={r.test_statistic:.3f}, p={r.p_value:.4f} "
              f"(adj.α={r.adjusted_alpha:.4f}) {sig_marker}")

    # ------------------------------------------------------------------
    # 6. Baseline comparisons (Phase 2.2)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE COMPARISONS")
    print("=" * 60)

    baseline_comparisons = evaluate_baselines(df, model_metrics)

    # Print single-feature AUCs (same across all models)
    first_comp = next(iter(baseline_comparisons.values()))
    print("\n  Single-feature baseline AUCs:")
    for feat, auc in sorted(first_comp.single_feature_aucs.items(), key=lambda x: -x[1]):
        print(f"    {feat:30s}: {auc:.4f}")
    print(f"\n  Best single feature: {first_comp.best_single_feature} "
          f"(AUC={first_comp.best_single_feature_auc:.4f})")

    print("\n  Model vs baseline comparison:")
    for name, comp in baseline_comparisons.items():
        print(f"    {name}: AUC={comp.model_auc:.4f} | "
              f"beats_random={'✓' if comp.beats_random else '✗'} | "
              f"improvement_over_best_feature={comp.improvement_over_best_feature:+.4f}")

    # ------------------------------------------------------------------
    # 7. Bootstrap confidence intervals (Phase 2.3)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({args.n_bootstrap} resamples)")
    print("=" * 60)

    bootstrap_results: dict[str, any] = {}
    for name in training_result.models:
        ci = compute_bootstrap_ci(
            y_test, predictions[name], probabilities[name],
            model_name=name,
            n_resamples=args.n_bootstrap,
            random_seed=args.seed,
        )
        bootstrap_results[name] = ci
        print(f"\n  {name}:")
        for m in ci.metric_cis:
            print(f"    {m.metric_name:12s}: {m.point_estimate:.4f} "
                  f"[{m.ci_lower:.4f}, {m.ci_upper:.4f}]")

    # ------------------------------------------------------------------
    # 8. Success tier validation (Phase 2.4)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUCCESS TIER VALIDATION")
    print("=" * 60)

    tier_results = validate_success_tiers(model_metrics)

    for name, tr in tier_results.items():
        tier_display = tr.tier_achieved.replace("_", " ").title()
        print(f"  {name}: AUC={tr.auc_roc:.4f} → {tier_display}")

    overall_pass = any(tr.achieves_minimum for tr in tier_results.values())
    best_model = max(model_metrics, key=lambda k: model_metrics[k].auc_roc)
    best_tier = tier_results[best_model].tier_achieved
    print(f"\n  Overall pass: {'✓' if overall_pass else '✗'}")
    print(f"  Best model: {best_model} ({best_tier})")

    # ------------------------------------------------------------------
    # 9. Final summary report (Phase 2.5)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    # Timestamp prefix for experiment comparison (YYYY-MM-DD_HH-MM)
    prefix = datetime.now().strftime("%Y-%m-%d_%H-%M")

    summary = produce_final_summary(
        model_metrics=model_metrics,
        mcnemar_results=mcnemar_results,
        baseline_comparisons=baseline_comparisons,
        tier_results=tier_results,
        config=config,
        output_dir=args.output_dir,
    )

    print(f"  Best model       : {summary.best_model}")
    print(f"  Best AUC-ROC     : {summary.best_auc_roc:.4f}")
    print(f"  Tier achieved    : {summary.success_tier_achieved}")
    print(f"  Overall pass     : {summary.overall_pass}")

    # ------------------------------------------------------------------
    # 10. Generate evaluation figures
    # ------------------------------------------------------------------
    if not args.no_figures:
        print("\n  Generating evaluation figures...")
        figures_dir = Path(args.output_dir).parent / "figures" / "evaluation"

        # Per-model figures
        for name in training_result.models:
            figure_paths = generate_evaluation_figures(
                y_test, predictions[name], probabilities[name],
                model_name=name, figures_dir=figures_dir,
            )
            for p in figure_paths:
                print(f"    {p}")

        # Combined ROC curve
        roc_data = [
            (name, y_test, probabilities[name])
            for name in training_result.models
        ]
        combined_path = plot_roc_curve_combined(roc_data, figures_dir=figures_dir)
        print(f"    {combined_path}")
    else:
        logger.info("Skipping figure generation (--no-figures).")

    # ------------------------------------------------------------------
    # 11. Save evaluation metrics JSON
    # ------------------------------------------------------------------
    # Build EvaluationMetrics for backward compatibility
    eval_metrics_list = []
    for name, mm in model_metrics.items():
        eval_metrics_list.append(EvaluationMetrics(
            model_name=name,
            partition="test",
            precision=mm.precision,
            recall=mm.recall,
            f1=mm.f1,
            roc_auc=mm.auc_roc,
            support_positive=int(y_test.sum()),
            support_negative=int(len(y_test) - y_test.sum()),
            confusion_matrix=mm.confusion_matrix,
        ))

    out_path = save_evaluation_results(eval_metrics_list, output_dir=args.output_dir, timestamp_prefix=prefix)
    print(f"\n  Evaluation metrics: {out_path}")
    print(f"  Final summary    : {Path(args.output_dir) / f'{prefix}_final_summary.json'}")

    # ------------------------------------------------------------------
    # 12. Latest outputs manifest (for downstream tool discovery)
    # ------------------------------------------------------------------
    eval_dir = Path(args.output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "evaluation_metrics": str(out_path),
        "final_summary": str(Path(args.output_dir) / f"{prefix}_final_summary.json"),
    }
    manifest_path = eval_dir / "latest_outputs.json"
    manifest_path.write_text(
        json.dumps({"prefix": prefix, "outputs": output_paths}, indent=2),
        encoding="utf-8",
    )
    print(f"  Latest manifest  : {manifest_path}")

    # ------------------------------------------------------------------
    # 13. Append to consolidated experiment log
    # ------------------------------------------------------------------
    total_duration = time.perf_counter() - pipeline_start

    print(f"\n  Total pipeline duration: {total_duration:.2f}s")

    append_experiment(
        run_id=prefix,
        pipeline="training",
        config={
            "seed": args.seed,
            "weight_sentiment": args.weight_sentiment,
            "threshold_tau": args.threshold_tau,
            "n_bootstrap": args.n_bootstrap,
            "data_path": str(Path(args.data_path).name),
        },
        outputs=list(output_paths.values()),
        summary={
            "best_model": summary.best_model,
            "best_auc_roc": summary.best_auc_roc,
            "tier_achieved": summary.success_tier_achieved,
            "overall_pass": summary.overall_pass,
            "total_duration_seconds": round(total_duration, 2),
        },
        notes=args.notes,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
