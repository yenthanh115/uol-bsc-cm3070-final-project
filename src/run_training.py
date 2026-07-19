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
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Ensure the src directory is on the path so surge_pipeline is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from surge_pipeline.cli_logging import resolve_log_path, tee_output  # noqa: E402
from surge_pipeline.config import PipelineConfig  # noqa: E402
from surge_pipeline.experiment_log import append_experiment  # noqa: E402
from surge_pipeline.features import compute_features, FEATURE_COLUMNS  # noqa: E402
from surge_pipeline.training import (  # noqa: E402
    train_models,
    get_training_summary,
    predict_with_threshold,
    TrainingPipelineResult,
)
from surge_pipeline.evaluation import (  # noqa: E402
    ModelMetrics,
    ThresholdResult,
    compute_bootstrap_ci,
    compute_builtin_importance,
    compute_feature_importance,
    evaluate_baselines,
    find_optimal_threshold,
    generate_evaluation_figures,
    mcnemar_pairwise_test,
    plot_feature_importance,
    plot_roc_curve_combined,
    produce_final_summary,
    save_evaluation_results,
    validate_success_tiers,
    EvaluationMetrics,
)


def _resolve_default_data_path() -> str:
    """Resolve the latest labelled dataset path from latest_outputs.json."""
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    latest_file = _PROJECT_ROOT / "output" / "processed" / "latest_outputs.json"
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
        default=str(Path(__file__).resolve().parent.parent / "output" / "evaluation"),
        help="Output directory for evaluation results.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "output" / "models"),
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
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path. Use 'auto' for timestamped filename in output/logs/.",
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

    # Resolve log file path and run the pipeline under tee_output
    log_path = resolve_log_path(args.log_file, pipeline="training")
    if log_path:
        logger.info("Logging output to: %s", log_path)

    with tee_output(log_path):
        _run_pipeline(args, logger)


def _run_pipeline(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Core pipeline logic, extracted so tee_output can wrap it."""
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
              f"improvement_over_best_feature={comp.improvement_over_best_single_feature:+.4f}")

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

    # Timestamp prefix for all output files (YYYY-MM-DD_HH-MM)
    prefix = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # ------------------------------------------------------------------
    # 7b. Classification threshold tuning (P1)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLASSIFICATION THRESHOLD TUNING (selected on validation fold)")
    print("=" * 60)

    threshold_results: dict[str, ThresholdResult] = {}
    tuned_metrics: dict[str, ModelMetrics] = {}

    for name, tm in training_result.models.items():
        if tm.val_y_true is not None and tm.val_y_prob is not None:
            # Select optimal threshold on the last CV validation fold
            tr = find_optimal_threshold(
                tm.val_y_true, tm.val_y_prob, model_name=name
            )
            threshold_results[name] = tr

            # Apply tuned threshold to test set
            y_pred_tuned = predict_with_threshold(probabilities[name], tr.optimal_threshold)

            prec_tuned = float(precision_score(y_test, y_pred_tuned, zero_division=0.0))
            rec_tuned = float(recall_score(y_test, y_pred_tuned, zero_division=0.0))
            f1_tuned = float(f1_score(y_test, y_pred_tuned, zero_division=0.0))
            cm_tuned = confusion_matrix(y_test, y_pred_tuned, labels=[0, 1]).tolist()

            tuned_metrics[name] = ModelMetrics(
                model_name=name,
                accuracy=float(accuracy_score(y_test, y_pred_tuned)),
                precision=prec_tuned,
                recall=rec_tuned,
                f1=f1_tuned,
                auc_roc=model_metrics[name].auc_roc,  # AUC doesn't change
                confusion_matrix=cm_tuned,
                n_test_samples=len(y_test),
            )

            f1_gain = f1_tuned - model_metrics[name].f1
            print(f"\n  {name}:")
            print(f"    Default (0.50): P={model_metrics[name].precision:.3f}  "
                  f"R={model_metrics[name].recall:.3f}  "
                  f"F1={model_metrics[name].f1:.3f}")
            print(f"    Tuned   ({tr.optimal_threshold:.2f}): P={prec_tuned:.3f}  "
                  f"R={rec_tuned:.3f}  "
                  f"F1={f1_tuned:.3f}  ← {f1_gain:+.3f} F1")
            print(f"    (Threshold selected on validation fold: "
                  f"val_F1={tr.f1_at_threshold:.3f})")
        else:
            logger.warning(
                "%s: No validation predictions available for threshold tuning.", name
            )

    # ------------------------------------------------------------------
    # 7c. Feature importance (P3)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (permutation, n_repeats=10, scoring=roc_auc)")
    print("=" * 60)

    importance_results: list = []

    for name, tm in training_result.models.items():
        fi = compute_feature_importance(
            model=tm.model,
            scaler=tm.scaler,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
            feature_names=FEATURE_COLUMNS,
            n_repeats=10,
            random_seed=args.seed,
        )
        importance_results.append(fi)

        ranked = fi.ranked()
        print(f"\n  {name}:")
        for rank, (feat, imp, std) in enumerate(ranked, 1):
            print(f"    {rank:2d}. {feat:<30s} {imp:+.4f} ± {std:.4f}")

    # Also get built-in importances for tree models
    builtin_results: dict = {}
    for name, tm in training_result.models.items():
        bi = compute_builtin_importance(tm.model, name, FEATURE_COLUMNS)
        if bi is not None:
            builtin_results[name] = bi

    if builtin_results:
        print("\n  Built-in (gain-based) importances:")
        for name, bi in builtin_results.items():
            ranked = bi.ranked()
            print(f"    {name}: top-3 = "
                  f"{ranked[0][0]} ({ranked[0][1]:.3f}), "
                  f"{ranked[1][0]} ({ranked[1][1]:.3f}), "
                  f"{ranked[2][0]} ({ranked[2][1]:.3f})")

    # Save feature importance JSON
    fi_output = {
        "experiment": "feature_importance",
        "method": "permutation",
        "scoring": "roc_auc",
        "n_repeats": 10,
        "timestamp": datetime.now().isoformat(),
        "models": {fi.model_name: fi.to_dict() for fi in importance_results},
    }
    if builtin_results:
        fi_output["builtin_importances"] = {
            name: bi.to_dict() for name, bi in builtin_results.items()
        }

    fi_path = Path(args.output_dir) / f"{prefix}_feature_importance.json"
    fi_path.parent.mkdir(parents=True, exist_ok=True)
    fi_path.write_text(json.dumps(fi_output, indent=2), encoding="utf-8")
    print(f"\n  Feature importance: {fi_path}")

    # Generate figure
    if not args.no_figures and importance_results:
        figures_dir = Path(args.output_dir).parent / "figures" / "evaluation"
        fi_fig_path = plot_feature_importance(importance_results, figures_dir=figures_dir)
        print(f"  Figure: {fi_fig_path}")

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

    summary = produce_final_summary(
        model_metrics=model_metrics,
        mcnemar_results=mcnemar_results,
        baseline_comparisons=baseline_comparisons,
        tier_results=tier_results,
        config=config,
        output_dir=args.output_dir,
        timestamp_prefix=prefix,
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

    # Save threshold tuning results as separate JSON
    if threshold_results:
        threshold_output = {
            "experiment": "threshold_tuning",
            "strategy": "max_f1",
            "selection_method": "last_cv_validation_fold",
            "timestamp": datetime.now().isoformat(),
            "models": {},
        }
        for name, tr in threshold_results.items():
            threshold_output["models"][name] = {
                "optimal_threshold": tr.optimal_threshold,
                "validation_fold_metrics": {
                    "precision": tr.precision_at_threshold,
                    "recall": tr.recall_at_threshold,
                    "f1": tr.f1_at_threshold,
                },
                "test_set_metrics_at_tuned_threshold": (
                    asdict(tuned_metrics[name]) if name in tuned_metrics else None
                ),
                "test_set_metrics_at_default_0_5": (
                    asdict(model_metrics[name]) if name in model_metrics else None
                ),
            }
            # Also add to model_metrics dict representation
            threshold_output["models"][name]["f1_improvement"] = (
                tuned_metrics[name].f1 - model_metrics[name].f1
                if name in tuned_metrics else 0.0
            )

        threshold_path = Path(args.output_dir) / f"{prefix}_threshold_tuning.json"
        threshold_path.write_text(
            json.dumps(threshold_output, indent=2, default=str), encoding="utf-8"
        )
        print(f"  Threshold tuning : {threshold_path}")

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
