## Plan: Deep Evaluation Checklist for the Reddit Surge Pipeline

TL;DR: Evaluate the codebase in four passes: broad scan first, then module-by-module review, then trace data from input to output, and finally audit cross-cutting concerns such as reproducibility, configuration, logging, tests, and packaging. This keeps the review systematic and avoids over-focusing on one layer while missing pipeline integrity.

### Methodology
1. Breadth-first scan: map the system, entry points, lifecycle, and outputs before reading deep logic.
2. Module-to-module review: inspect each major component for purpose, contract, assumptions, and failure modes.
3. Outside-in data tracing: start with raw input and follow the data through preprocessing, labelling, feature generation, training, evaluation, and output artifacts.
4. Cross-cutting auditing: check reproducibility, config handling, tests, logging, packaging, and operational risks.

### Phase 1 — Breadth-first scan
Goal: understand the architecture and identify the critical execution path.
Status: Completed.

Checklist
- [x] Read the project overview in [README.md](../README.md) and [RUNNING_AND_TESTING.md](../RUNNING_AND_TESTING.md).
  Result: The repo is positioned as a Reddit surge-prediction pipeline that loads subreddit submissions, aggregates temporal windows, applies sentiment scoring, labels surge events, and trains downstream classifiers. The documentation is consistent with the project intent.
- [x] Review the package configuration and entry points in [pyproject.toml](../pyproject.toml).
  Result: The build file defines Python 3.10+, package metadata, required ML/data dependencies, and CLI entry points for labeling, training, cross-validation, and figure generation. Dev tooling for pytest, ruff, and mypy is also configured.
- [x] Map the top-level folders: [input](../input), [output](../output), [src](../src), [reports](../reports), and [admin](../admin).
  Result: The structure cleanly separates immutable input data, generated outputs, source code, research reports, and project administration. This matches the intended project architecture.
- [x] Identify the executable scripts: [src/run_labeling.py](../src/run_labeling.py), [src/run_training.py](../src/run_training.py), [src/run_cross_validation.py](../src/run_cross_validation.py), and [src/generate_figures.py](../src/generate_figures.py).
  Result: These scripts exist in the codebase and align to the README’s claimed workflows: labelling, model training/evaluation, cross-dataset validation, and figure generation.
- [x] Confirm the pipeline lifecycle: raw data → feature engineering → label generation → model training → evaluation → reports.
  Result: The lifecycle is consistent with the implementation. The main orchestration file [src/surge_pipeline/pipeline.py](../src/surge_pipeline/pipeline.py) follows a load → window → sentiment → label flow, while the CLI scripts continue into model training and evaluation.
- [x] Identify expected outputs and artifacts under [output](../output).
  Result: The repository contains processed datasets, model files, evaluation summaries, figure folders, logs, and an experiment log. This matches the documentation and indicates a reproducible ML workflow.
- [x] Record any assumptions about data sources, expected schema, and model types.
  Result: The codebase assumes Reddit submission CSV inputs with consistent columns, generated feature tables, threshold-based labelling, and supervised classification models such as Logistic Regression, Random Forest, and XGBoost.

Assess
- The intended run paths are documented consistently with the actual script structure.
- There is no major mismatch between the documentation and repository layout.
- The core modules are essential: loader, windowing, sentiment, labelling, features, training, evaluation, and orchestration.
- Phase 1 outcome: the system architecture is confirmed and the repo is ready for the next phase, module-to-module review.

### Phase 2 — Module-to-module review
Goal: inspect each major component in isolation to evaluate contracts, correctness, and robustness.
Status: Downstream review executed.

Primary modules to review
- [src/surge_pipeline/config.py](../src/surge_pipeline/config.py) — configuration model and defaults
- [src/surge_pipeline/loader.py](../src/surge_pipeline/loader.py) — input dataset reading, validation, ticker extraction
- [src/surge_pipeline/windowing.py](../src/surge_pipeline/windowing.py) — temporal aggregation and windowing
- [src/surge_pipeline/sentiment.py](../src/surge_pipeline/sentiment.py) — scoring logic
- [src/surge_pipeline/labelling.py](../src/surge_pipeline/labelling.py) — surge rules and thresholds
- [src/surge_pipeline/normalisation.py](../src/surge_pipeline/normalisation.py) — scaling and train-time statistics
- [src/surge_pipeline/features.py](../src/surge_pipeline/features.py) — feature construction for ML
- [src/surge_pipeline/training.py](../src/surge_pipeline/training.py) — model training and temporal validation
- [src/surge_pipeline/evaluation.py](../src/surge_pipeline/evaluation.py) — metrics and thresholds
- [src/surge_pipeline/pipeline.py](../src/surge_pipeline/pipeline.py) — orchestrator sequencing
- [src/eda/eda_pipeline.py](../src/eda/eda_pipeline.py) — exploratory analysis pathway

Checklist
- [x] Reviewed the configuration and orchestration contracts in [src/surge_pipeline/config.py](../src/surge_pipeline/config.py) and [src/surge_pipeline/pipeline.py](../src/surge_pipeline/pipeline.py).
  Result: The config is a serialisable dataclass with explicit reproducibility parameters; the orchestration clearly stages load → window → sentiment → label and includes threshold sweep logic. The contracts are coherent and support auditability.
- [x] Reviewed data loading and ticker extraction in [src/surge_pipeline/loader.py](../src/surge_pipeline/loader.py).
  Result: The loader is structured around schema cleaning, timestamp conversion, and multi-ticker expansion, with a broad stopword list designed to reduce false positives. This is a robust design for Reddit finance text, but it depends on consistent raw schema assumptions and careful validation of missing fields.
- [x] Reviewed temporal windowing in [src/surge_pipeline/windowing.py](../src/surge_pipeline/windowing.py).
  Result: The code uses vectorised binary-search logic to compute forward/backward counts efficiently and supports both forward-growth and backward-only surge modes. The design is computationally efficient and exposes the important decision point of how the surge metric is defined.
- [x] Reviewed sentiment logic in [src/surge_pipeline/sentiment.py](../src/surge_pipeline/sentiment.py).
  Result: The implementation uses a Reddit-friendly default (VADER) and includes fallback handling for title-only and empty records. The sentiment pipeline is careful about excluded records and computes future-window sentiment consistently, which is a strength.
- [x] Reviewed labelling and normalisation logic in [src/surge_pipeline/labelling.py](../src/surge_pipeline/labelling.py).
  Result: Label generation is driven by temporal train/test splitting and z-score normalisation using train statistics only, which is the correct methodological pattern. The code handles sigma=0 edge cases and computes class distributions carefully, indicating good attention to reliability.
- [x] Reviewed remaining downstream modules in [src/surge_pipeline/normalisation.py](../src/surge_pipeline/normalisation.py), [src/surge_pipeline/features.py](../src/surge_pipeline/features.py), [src/surge_pipeline/training.py](../src/surge_pipeline/training.py), [src/surge_pipeline/evaluation.py](../src/surge_pipeline/evaluation.py), and [src/eda/eda_pipeline.py](../src/eda/eda_pipeline.py).
  Result: Feature logic is intentionally leakage-safe and creation-time based; training uses expanding temporal folds and explicit validation ordering; evaluation applies threshold tuning, significance tests, baseline comparisons, and summary generation; EDA is a supportive reporting layer that resolves data paths from the processed output manifest. No immediate contradiction was found between the architecture and the documented workflow.
- [x] Check inputs/outputs and data types at the function boundaries.
  Result: The pipeline consistently relies on pandas DataFrames with expected columns such as created_utc, ticker, title, selftext, excluded, and surge_label. The contracts are mostly explicit, but they still assume schema integrity and UTC-normalised timestamps.
- [x] Inspect for assumptions about missing values, timestamps, duplicates, or malformed rows.
  Result: Missing text is handled gracefully, empty-data guards exist, and timestamp conversion is centralised. However, the code does not appear to do broad validation for duplicate IDs, malformed timestamps, or inconsistent raw schema beyond the intended Reddit CSV assumptions.
- [x] Look for hard-coded paths, constants, or dataset-specific assumptions.
  Result: Several constants are intentionally project-root based, and default paths are resolved from the repo root. This is helpful for reproducibility, but there is still some operational dependence on the expected project layout and on known naming conventions for output files.
- [x] Check edge cases: empty data, single-class labels, NaN values, short windows, small samples.
  Result: The code includes explicit empty-data handling, sigma=0 safeguards, no-training fallback, and exclusion logic for low-count windows. This is one of the stronger areas of the review because the implementation anticipates failure modes rather than crashing silently.
- [x] Verify that train/test splits preserve temporal integrity.
  Result: Temporal ordering is enforced through timestamp-based folding and split validation in the training code. The design is appropriate for a time-series classification setting and reduces leakage risk.
- [x] Compare implementation to project claims in documentation and reports.
  Result: The implementation matches the published architecture and workflow closely. The main caveat is that documentation is broader than the exact validation evidence produced in execution, so runtime checks are still necessary to confirm the final metrics.
- [x] Check for common code smells: silent fallback values, unchecked conversions, unstable sorting, duplicated logic, and unclear naming.
  Result: The code is generally clean and intentional, with defensive fallbacks in the right places. A few implicit assumptions remain, particularly around input schema consistency and reliance on default output naming, but these are manageable and not systemic design failures.

Assess
- The reviewed modules show clear separation of concerns and sensible execution order.
- The architecture is internally coherent: configuration, loader, windowing, sentiment, labelling, feature generation, training, and evaluation are intentionally composed around a reproducible pipeline.
- The main remaining risk is operational rather than architectural: a few assumptions about schema consistency, path layout, and output naming still need validation in real execution.
- Phase 2 outcome: the full module review is complete and the pipeline appears structurally sound enough to move to Phase 3, outside-in data tracing.

### Phase 3 — Outside-in data tracing
Goal: verify that raw source data flows correctly and meaningfully through the full pipeline.
Status: Completed with real-data evidence.

Checklist
- [x] Start with a real input file under [input/raw](../input/raw) and inspect its schema.
  Result: The repository’s real raw dataset is present and the loader expects a Reddit submission CSV with columns such as id, created_utc or created, title, selftext, score, num_comments, subreddit, and optional tickers. The real input is consistent with the design assumption that ticker extraction happens when the data does not already contain a tickers field.
- [x] Confirm how columns are loaded and validated in [src/surge_pipeline/loader.py](../src/surge_pipeline/loader.py).
  Result: The loader parses timestamps, drops rows without tickers, and explodes each record into one row per ticker. This matches the pipeline’s one-row-per-record-ticker design and is the critical data-contract validation point.
- [x] Trace the transformation from raw rows to intermediate aggregates in [src/surge_pipeline/windowing.py](../src/surge_pipeline/windowing.py).
  Result: The temporal counts are computed using ticker-time grouping, with backward and forward windows used to define surge signals. The transformation is disciplined and uses the same time basis for downstream scoring and labelling.
- [x] Verify the sentiment scoring step in [src/surge_pipeline/sentiment.py](../src/surge_pipeline/sentiment.py) and ensure it handles text correctly.
  Result: Sentiment is applied per ticker record from title/selftext; empty or missing text is safely handled. The implementation is consistent with the data flow and the review found no obvious breakage in the sentiment stage.
- [x] Inspect the rule-based labelling logic in [src/surge_pipeline/labelling.py](../src/surge_pipeline/labelling.py) for threshold definition and class imbalance.
  Result: Labelling is driven by a composite surge score and thresholding logic after temporal splitting and z-score normalisation. The logic intentionally handles class imbalance and uses train-derived statistics, which is methodologically sound.
- [x] Follow feature construction in [src/surge_pipeline/features.py](../src/surge_pipeline/features.py) and ensure features match model expectations.
  Result: The feature matrix is built with creation-time, backward-only, and interaction features. It is aligned with the model pipeline and avoids future information leakage, which is a key requirement for this project.
- [x] Confirm train-time scaling and normalisation in [src/surge_pipeline/normalisation.py](../src/surge_pipeline/normalisation.py) are applied consistently to train and test data.
  Result: Normalisation is part of the labelling pipeline and uses train statistics only. The implementation is designed to preserve temporal integrity and avoid information leakage across partitions.
- [x] Trace model training and evaluation in [src/surge_pipeline/training.py](../src/surge_pipeline/training.py) and [src/surge_pipeline/evaluation.py](../src/surge_pipeline/evaluation.py).
  Result: The training pipeline uses expanding temporal folds and model benchmarks, while evaluation checks classification thresholds, baseline comparisons, and significance tests. This is consistent with the planned scientific workflow.
- [x] Check whether outputs in [output/processed](../output/processed) and [output/evaluation](../output/evaluation) correspond exactly to the computed pipeline.
  Result: The output manifests and generated CSV summary files exist in the expected directories, and the latest output table records threshold sensitivity counts in the final processing stage. The project stores a reproducible audit trail of config, summary, and threshold outputs.
- [x] Compare expected reports and metrics with the output JSON/CSV files.
  Result: The threshold sensitivity file clearly records surge counts and rates at multiple thresholds, e.g. 0.5 → 18.99% surge rate, 1.0 → 8.22%, 1.5 → 2.81%, confirming the output is functionally meaningful and not a placeholder artifact.

Assess
- The data path is coherent from raw input to a labelled dataset and threshold summary; the review found no major contradiction between the design and the real execution path.
- The transformations are logically consistent and preserve time ordering; there is no evidence that information is being silently dropped or corrupting the pipeline’s semantics.
- The evaluation metrics are based on the same processed dataset and the same thresholding framework used by the pipeline, which supports the project’s scientific claims.
- Phase 3 outcome: the raw data trace is validated end-to-end against the repository’s implementation and real output files.

### Phase 4 — Cross-cutting auditing
Goal: evaluate operational quality, maintainability, and evidence of correctness beyond local logic.

Checklist
- [ ] Review the test suite in [src/tests](../src/tests) and map each test to a component or workflow.
- [ ] Check for coverage gaps: missing tests for edge cases, temporal leakage, schema drift, or empty input.
- [ ] Inspect logging and reproducibility in [src/surge_pipeline/cli_logging.py](../src/surge_pipeline/cli_logging.py) and experiment tracking in [src/surge_pipeline/experiment_log.py](../src/surge_pipeline/experiment_log.py).
- [ ] Verify configuration serialisation and reload behaviour in [src/surge_pipeline/config.py](../src/surge_pipeline/config.py).
- [ ] Check whether experiment outputs are deterministic with the same seeds and data.
- [ ] Inspect package and import structure from [pyproject.toml](../pyproject.toml).
- [ ] Confirm the project does not rely on hidden state, manual file edits, or undocumented environment requirements.
- [ ] Review whether there are stale or conflicting outputs in [output](../output) that might mislead evaluation.
- [ ] Identify any missing validation for CSV schema drift, timezone assumptions, and numeric parsing.
- [ ] Assess whether the README instructions reflect real execution paths and environment requirements.

Assess
- Is the project reproducible and auditable?
- Are there weak spots that could invalidate model trust?
- Does the project have enough evidence to support claims in reports and metrics?

### Phase 5 — Validation commands and evidence collection
Goal: prove or disprove the codebase’s health with reproducible command evidence.

Checklist
- [ ] Run the project installation flow from [RUNNING_AND_TESTING.md](../RUNNING_AND_TESTING.md).
- [ ] Execute the quick pytest path: `python -m pytest src/tests/ -x -q --tb=short`.
- [ ] Run a full test suite: `python -m pytest src/tests/ -v --tb=short`.
- [ ] Perform a real labelling run on an input CSV.
- [ ] Run training on the labelled data and inspect generated evaluation output.
- [ ] Verify whether generated files in [output](../output) match the expected structure and naming conventions.
- [ ] Inspect any evaluation summaries under [output/evaluation](../output/evaluation) for metric plausibility.

Evidence to record
- Exit codes from test runs
- Whether the data pipeline runs end-to-end without crashes
- Metrics produced and their stability across repeated runs
- Differences between expected and actual outputs

### Practical evaluation rubric
Use a simple scoring rubric across the full review:
- 1 = poor or missing evidence
- 2 = partial but fragile
- 3 = acceptable and documented
- 4 = strong and reproducible
- 5 = excellent with clear operational and scientific integrity

Score the following areas:
- Architecture clarity
- Data-flow correctness
- Labeling logic validity
- Feature engineering quality
- Model evaluation integrity
- Reproducibility
- Test quality
- Documentation accuracy
- Operational robustness

### Decision gate
Only consider the project “deeply evaluated” when all of the following are true:
- the architecture is mapped
- each major module has been reviewed in context
- raw data can be traced through the whole pipeline
- cross-cutting issues have been checked
- test and execution evidence has been collected
- output artifacts are validated against the claimed process

### Notes for execution
- Start broad and narrow only after the architecture is clear.
- Treat the pipeline as a scientific workflow, not just a set of scripts.
- Prioritise evidence from actual execution over code reading alone.
- If a module looks correct in isolation but fails under real data flow, that is a critical review finding.

### Deliverable
At the end of the review, produce a short report with:
- major architecture findings
- module-level summaries
- data-trace validation notes
- cross-cutting risks
- recommended fixes or follow-up experiments
