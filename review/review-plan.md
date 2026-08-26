## Plan: Deep Evaluation Checklist for the Reddit Surge Pipeline

TL;DR: Evaluate the codebase in four passes: broad scan first, then module-by-module review, then trace data from input to output, and finally audit cross-cutting concerns such as reproducibility, configuration, logging, tests, and packaging. This keeps the review systematic and avoids over-focusing on one layer while missing pipeline integrity.

### Methodology
1. Breadth-first scan: map the system, entry points, lifecycle, and outputs before reading deep logic.
2. Module-to-module review: inspect each major component for purpose, contract, assumptions, and failure modes.
3. Outside-in data tracing: start with raw input and follow the data through preprocessing, labelling, feature generation, training, evaluation, and output artifacts.
4. Cross-cutting auditing: check reproducibility, config handling, tests, logging, packaging, and operational risks.

### Phase 1 — Breadth-first scan
Goal: understand the architecture and identify the critical execution path.

Checklist
- [ ] Read the project overview in [README.md](../README.md) and [RUNNING_AND_TESTING.md](../RUNNING_AND_TESTING.md).
- [ ] Review the package configuration and entry points in [pyproject.toml](../pyproject.toml).
- [ ] Map the top-level folders: [input](../input), [output](../output), [src](../src), [reports](../reports), and [admin](../admin).
- [ ] Identify the executable scripts: [src/run_labeling.py](../src/run_labeling.py), [src/run_training.py](../src/run_training.py), [src/run_cross_validation.py](../src/run_cross_validation.py), and [src/generate_figures.py](../src/generate_figures.py).
- [ ] Confirm the pipeline lifecycle: raw data → feature engineering → label generation → model training → evaluation → reports.
- [ ] Identify expected outputs and artifacts under [output](../output).
- [ ] Record any assumptions about data sources, expected schema, and model types.

Assess
- Are the intended run paths documented consistently?
- Are there mismatches between docs and actual scripts or folder structure?
- Which modules are essential versus optional?

### Phase 2 — Module-to-module review
Goal: inspect each major component in isolation to evaluate contracts, correctness, and robustness.

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
- [ ] For each module, confirm its role and public API.
- [ ] Check inputs/outputs and data types at the function boundaries.
- [ ] Inspect for assumptions about missing values, timestamps, duplicates, or malformed rows.
- [ ] Look for hard-coded paths, constants, or dataset-specific assumptions.
- [ ] Check edge cases: empty data, single-class labels, NaN values, short windows, small samples.
- [ ] Verify that train/test splits preserve temporal integrity.
- [ ] Compare implementation to project claims in documentation and reports.
- [ ] Check for common code smells: silent fallback values, unchecked conversions, unstable sorting, duplicated logic, and unclear naming.

Assess
- Are module contracts coherent and consistent with the rest of the pipeline?
- Are there hidden dependencies that break reproducibility?
- Are failure paths explicit and diagnosable?

### Phase 3 — Outside-in data tracing
Goal: verify that raw source data flows correctly and meaningfully through the full pipeline.

Checklist
- [ ] Start with a real input file under [input/raw](../input/raw) and inspect its schema.
- [ ] Confirm how columns are loaded and validated in [src/surge_pipeline/loader.py](../src/surge_pipeline/loader.py).
- [ ] Trace the transformation from raw rows to intermediate aggregates in [src/surge_pipeline/windowing.py](../src/surge_pipeline/windowing.py).
- [ ] Verify the sentiment scoring step in [src/surge_pipeline/sentiment.py](../src/surge_pipeline/sentiment.py) and ensure it handles text correctly.
- [ ] Inspect the rule-based labelling logic in [src/surge_pipeline/labelling.py](../src/surge_pipeline/labelling.py) for threshold definition and class imbalance.
- [ ] Follow feature construction in [src/surge_pipeline/features.py](../src/surge_pipeline/features.py) and ensure features match model expectations.
- [ ] Confirm train-time scaling and normalisation in [src/surge_pipeline/normalisation.py](../src/surge_pipeline/normalisation.py) are applied consistently to train and test data.
- [ ] Trace model training and evaluation in [src/surge_pipeline/training.py](../src/surge_pipeline/training.py) and [src/surge_pipeline/evaluation.py](../src/surge_pipeline/evaluation.py).
- [ ] Check whether outputs in [output/processed](../output/processed) and [output/evaluation](../output/evaluation) correspond exactly to the computed pipeline.
- [ ] Compare expected reports and metrics with the output JSON/CSV files.

Assess
- Is the data path coherent from raw input to final metrics?
- Are transformations logically consistent and not accidentally dropping or corrupting information?
- Are evaluation metrics based on the same processed features used for training?

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
