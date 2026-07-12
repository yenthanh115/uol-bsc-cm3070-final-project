## Improvement Plan

### Phase 1: Foundation & Folder Reorganisation (2–3 days)

**Goal:** Establish clean project structure, implement the full multi-model training and evaluation interface, and get the test suite passing.

#### ~~1.1 — Reorganise input/output folders~~ - DONE

Move from the current scattered layout to:

```
project-root/
├── input/                              # Everything the pipeline READS (immutable)
│   ├── raw/                            # Original Reddit CSV
│   │   └── r_pennystocks_submissions_reddit.csv
│   └── reference/                      # Static reference files (e.g., ticker list)
│       └── valid_tickers.csv           # (Phase 4 addition)
│
├── output/                             # Everything the pipeline WRITES (reproducible)
│   ├── processed/                      # Pipeline stage outputs
│   │   ├── labelled_dataset.csv
│   │   ├── pipeline_summary.json
│   │   ├── pipeline_config.json
│   │   └── threshold_sensitivity.csv
│   ├── models/                         # Serialised trained models
│   │   ├── logistic_regression_phase2_42.joblib
│   │   ├── random_forest_phase2_42.joblib
│   │   └── xgboost_phase2_42.joblib
│   ├── evaluation/                     # Metrics and final summary
│   │   ├── evaluation_metrics.json
│   │   └── final_summary.json
│   └── figures/                        # ALL generated figures
│       ├── eda/                        # EDA figures (01–09)
│       │   ├── 01_posting_frequency_over_time.png
│       │   ├── ...
│       │   └── 09_threshold_sensitivity_curve.png
│       └── evaluation/                 # Model evaluation figures (10+)
│           ├── 10_confusion_matrix_LogisticRegression.png
│           ├── 11_roc_curve_LogisticRegression.png
│           └── 12_classification_threshold_sensitivity_LogisticRegression.png
│
├── src/                                # Source code (unchanged internally)
├── reports/                            # Academic reports (unchanged)
├── admin/                              # Project admin (unchanged)
└── README.md
```

Steps:
- Create `input/raw/`, `input/reference/`.
- Create `output/processed/`, `output/models/`, `output/evaluation/`, `output/figures/eda/`, `output/figures/evaluation/`.
- Move `data/raw/*.csv` → `input/raw/`.
- Move `data/processed/labelled_dataset.csv` → `output/processed/`.
- Move `figures/01–09*.png` → `output/figures/eda/`.
- Move `figures/10–12*.png` → `output/figures/evaluation/`.
- Delete empty `data/interim/`, `output/demo/`.
- Remove old `data/` and `figures/` once contents are moved.
- Add `.gitkeep` in empty tracked folders.
- Update `.gitignore` (ignore large CSVs and model files, keep JSON configs tracked).

#### ~~1.2 — Update all code path references~~ - DONE

- `PipelineConfig` defaults: `file_path` → `input/raw/...`, `output_dir` → `output/processed`.
- `evaluation.py` `FIGURES_DIR` → `output/figures/evaluation`.
- `eda_pipeline.py` figure output → `output/figures/eda`.
- `run_pipeline.py` and `run_training.py` default CLI paths.
- `README.md` project structure and usage examples.

#### ~~1.3 — Refactor `training.py` to match test interface~~ - DONE

- Expose: `create_temporal_folds`, `get_expanding_window_splits`, `N_FOLDS`, `_verify_temporal_ordering`, `_get_lr_param_grid`, `_get_rf_param_grid`, `_get_xgb_param_grid`, `train_models`, `get_training_summary`.
- Introduce a `TrainedModel` dataclass (or similar) returned per model, with `.model`, `.scaler`, `.cv_scores`, `.best_params`.
- The existing LR logic stays intact — wrap it into the new multi-model structure.

#### 1.4 — ~~Implement Random Forest and XGBoost training~~ - DONE

- Add `RandomForestClassifier` with grid: `n_estimators(3) × max_depth(4) × min_samples_leaf(3) = 36` configs.
- Add `XGBClassifier` with a grid ≤50 configs.
- Reuse the same temporal CV infrastructure (expanding-window splits, AUC-ROC selection, retrain on full training set).
- Add `xgboost` to `requirements.txt`.

#### 1.5 — ~~Implement advanced evaluation functions in `evaluation.py`~~ - DONE

- Add dataclasses: `ModelMetrics`, `McNemarResult`, `BaselineComparison`, `BootstrapCI`, `MetricCI`, `FinalSummary`, `SuccessTierResult`.
- Add constants: `SUCCESS_TIER_MINIMUM = 0.60`, `SUCCESS_TIER_TARGET = 0.70`, `SUCCESS_TIER_STRETCH = 0.80`.
- Implement: `mcnemar_pairwise_test`, `evaluate_baselines`, `validate_success_tiers`, `produce_final_summary`.

#### ~~1.6 — Run pytest and fix until green~~ - DONE

- Resolve any import mismatches, type issues, or edge cases.
- Target: all 5 test files pass.

---

### Phase 2: Advanced Evaluation (1–2 days)

**Goal:** Rigorous model comparison and statistical validation.

**2.1 — McNemar's pairwise test**
- Build the 2×2 contingency table (record-level correct/incorrect per model pair).
- Apply Bonferroni correction (adjusted alpha = 0.05 / n_comparisons).
- Report test statistic, p-value, and significance decision per pair.

**2.2 — Baseline comparisons**
- Random baseline (AUC = 0.5 by definition).
- Single-feature baselines: for each of the 9 features, train a 1-feature LR and report AUC.
- Compare each trained model against random and best single-feature baseline.
- Flag `beats_random` and `improvement_over_best_single_feature`.

**2.3 — Bootstrap confidence intervals**
- 1000 bootstrap resamples of the test set.
- Report 95% CI for precision, recall, F1, and AUC-ROC per model.

**2.4 — Success tier validation**
- Minimum (AUC > 0.60), Target (AUC > 0.70), Stretch (AUC > 0.80).
- Map each model to its tier; report `overall_pass` if any model exceeds minimum.

**2.5 — Final summary report**
- Produce `output/evaluation/final_summary.json` with: best model, best AUC, tier achieved, McNemar results, baseline comparisons, and config used.
- This becomes the artifact referenced in the academic report.

---

### Phase 3: Improve Sentiment Signal (1 day)

**Goal:** Replace TextBlob with a model that understands financial/Reddit language.

**3.1 — Swap to VADER (quick win)**
- VADER handles social media text better (exclamation, capitalisation, slang).
- Add `vaderSentiment` to requirements.
- Replace `TextBlob(text).sentiment.polarity` with `SentimentIntensityAnalyzer().polarity_scores(text)['compound']`.
- Keep the same interface (`sentiment_polarity` column, same range [-1, 1]).

**3.2 — Optional: FinBERT (stretch goal)**
- If runtime budget allows, add a FinBERT path for better financial text understanding.
- Use `transformers` + `ProsusAI/finbert` for batch inference.
- Gate behind a config flag (`sentiment_model: "vader" | "finbert"`) so the default stays fast.

**3.3 — Optimise sentiment computation (engineering hygiene)**
- Deduplicate texts before scoring: exploded rows share the same post text (~80k rows from ~36k unique posts → 45% fewer VADER calls).
- Vectorise text preparation with `np.where` instead of per-row `.iloc` + conditionals.
- Expected improvement: sentiment stage from ~8 min → ~2 min. Not report-worthy on its own, but enables faster iteration during experimentation.

**3.4 — Re-run and compare**
- Run the full pipeline with VADER, compare surge rate and model AUC against TextBlob baseline.
- Log which sentiment model was used in the pipeline summary for reproducibility.

---

### Phase 4: Ticker Extraction Hardening (0.5 day)

**Goal:** Reduce false positives without adding external dependencies.

**4.1 — Add a known-ticker validation list**
- Place `valid_tickers.csv` in `input/reference/`.
- Bundle a static list of NASDAQ/OTC penny stock tickers (~5000 tickers).
- After regex extraction, validate candidates against this list.
- Keep unvalidated tickers only if they match the `$TICKER` dollar-sign pattern (high confidence).

**4.2 — Add extraction metrics**
- Log precision estimate: sample 50 records manually, count correct vs. false-positive extractions.
- Include this in the academic report as a known limitation or validation step.

---

### Phase 5: End-to-End Run & Artifact Generation (0.5 day) - DONE

**Goal:** Produce persisted results that prove the system works.

**5.1 — Full pipeline run on real data**
```
python src/run_pipeline.py --file-path input/raw/r_pennystocks_submissions_reddit.csv --output-dir output/processed
```
- Verify outputs: `labelled_dataset.csv`, `pipeline_summary.json`, `threshold_sensitivity.csv`, `pipeline_config.json`.

**5.2 — Full training and evaluation**
```
python src/run_training.py --data-path output/processed/labelled_dataset.csv --output-dir output/evaluation
```
- Verify outputs: `evaluation_metrics.json`, confusion matrix figure, ROC curve, threshold sensitivity plot.

**5.3 — Commit results**
- Commit the generated artifacts to `output/`.
- These become the evidence for the final report.

---

### Phase 6: CI & Housekeeping (0.5 day)

**6.1 — Add a `Makefile` for reproducibility**
- `make pipeline` — runs full labelling pipeline.
- `make train` — trains and evaluates models.
- `make test` — runs pytest.
- `make eda` — runs EDA figures.
- `make all` — pipeline → train → test.

**6.2 — Pin dependency versions**
- Change `requirements.txt` from `>=` to `==` for exact reproducibility.
- Add `xgboost`, `vaderSentiment` (and optionally `joblib` if not bundled with sklearn).

**6.3 — Clean up dead code**
- Remove `src/demo_scripts/` (currently empty).
- Remove any orphaned placeholder directories.

---

### Phase 7: Report Writing (3–4 days)

**Goal:** Produce a near-submission-quality academic report (90–95% complete) following the prescribed 6-section structure.

---

#### Report Structure

| Section | Content Focus |
|---|---|
| 1. Introduction | Motivation, aims, concept |
| 2. Literature Review | Revised review, feedback improvements, additional references |
| 3. Design | Architecture, technologies, methods |
| 4. Implementation | Features, algorithms, technical methods, progress |
| 5. Evaluation | Strengths, weaknesses, performance, improvements needed |
| 6. Conclusion | Achievements, findings, remaining work, future developments |

---

#### Input Data for Writing

Each report section draws from specific artifacts produced by the earlier phases. This mapping ensures nothing is written from memory — every claim is backed by a generated file.

| Report Section | Input Artifacts |
|---|---|
| Introduction | `reports/01-literature-review/`, `admin/decision-log.md`, proposal, preliminary report |
| Literature Review | `reports/01-literature-review/literature-review-v1.0.md`, previous feedback, new papers |
| Design | `src/surge_pipeline/config.py`, `src/surge_pipeline/features.py`, `output/processed/pipeline_config.json`, EDA figures |
| Implementation | `src/surge_pipeline/` (all modules), `output/models/*.joblib`, `output/processed/pipeline_summary.json` |
| Evaluation | `output/evaluation/evaluation_metrics.json`, `output/evaluation/final_summary.json`, `output/figures/evaluation/10–12*.png` |
| Conclusion | `admin/decision-log.md`, `reports/04-draft/risk-and-challenges.md`, `final_summary.json` |

---

#### 7.1 — Introduction (0.5 day)

Describe the project motivation, aims, and concept. Build directly on the proposal and preliminary report.

- Project motivation: why predicting posting-volume surges on r/pennystocks matters for market participants and researchers.
- Project aims: binary classification of ticker-level 24h posting-volume surges using backward-looking features.
- Project concept: composite surge metric, temporal CV methodology, multi-model comparison approach.
- Scope and research question.

**Input:** Proposal, preliminary report, literature review conclusions, `admin/decision-log.md`.

#### 7.2 — Literature Review (0.5 day)

Include the revised literature review with improvements from previous feedback and additional references.

- Polish existing literature review (already written in `reports/01-literature-review/`).
- Address improvements based on previous feedback (reviewer comments, gaps identified).
- Add additional references if needed (2–3 recent citations on social media prediction, penny stocks, NLP in finance).
- Tighten the research gap statement to directly motivate the methodology.

**Input:** `reports/01-literature-review/literature-review-v1.0.md`, previous feedback, any new papers found during implementation.

#### 7.3 — Design (0.75 day)

Present the system architecture, project design, technologies, and methods. Refine the design document developed earlier.

- System architecture: pipeline stages (loading → preprocessing → feature engineering → labelling → training → evaluation).
- Data flow diagram: input sources → intermediate outputs → final artifacts.
- Technology choices and justification: Python, scikit-learn, XGBoost, VADER, pandas.
- Method design:
  - 9 backward-looking features with leakage-prevention argument.
  - Composite surge metric formula: `S = w1 * z_volume + w2 * z_sentiment`.
  - Threshold τ selection via sensitivity sweep.
  - Expanding-window temporal cross-validation (k=4 folds, 3 splits).
  - Three models: Logistic Regression, Random Forest, XGBoost.
- Reproducibility design: fixed seeds, serialised models, config JSON, temporal train/test split (80/20).

**Input:** `src/surge_pipeline/config.py`, `features.py`, `labelling.py`, `normalisation.py`, `training.py`, `pipeline_config.json`, EDA figures 05–09.

#### 7.4 — Implementation (0.75 day)

Describe features completed, algorithms, technical methods, and implementation progress. Expand the implementation write-up created during development.

- Features completed:
  - Data loading and preprocessing (text cleaning, ticker extraction, sentiment scoring).
  - Feature engineering (9 features with temporal windowing).
  - Surge labelling with composite metric and configurable threshold.
  - Multi-model training with hyperparameter search (LR: 10, RF: 36, XGB: ≤50 configs).
  - Evaluation pipeline with statistical tests.
- Algorithms and technical methods:
  - Ticker extraction: regex + stopword filtering + known-ticker validation.
  - Sentiment analysis: VADER compound scoring.
  - Temporal CV: expanding-window splits preserving chronological ordering.
  - Model selection: mean validation AUC-ROC, final retraining on full training set.
- Implementation progress: all pipeline stages functional, end-to-end run producing artefacts.
- Key implementation decisions and trade-offs (referencing decision log).

**Input:** `src/surge_pipeline/` (all modules), `output/models/*.joblib`, `output/processed/pipeline_summary.json`, `admin/decision-log.md`.

#### 7.5 — Evaluation (0.75 day)

Evaluate the current implementation by discussing strengths, weaknesses, current performance, and improvements still required. Focus on the current project state rather than the finished system.

- Current performance:
  - Per-model metrics table: Precision, Recall, F1, AUC-ROC with 95% bootstrap CIs.
  - ROC curve figure (combined overlay).
  - Confusion matrices per model.
  - Best model identification and tier achieved (minimum/target/stretch).
- Statistical rigour:
  - McNemar's test pairwise significance table.
  - Baseline comparison table (random baseline, single-feature baselines).
- Strengths:
  - Temporal CV prevents data leakage.
  - Multi-model comparison provides robust conclusions.
  - Reproducibility via fixed seeds and serialised configs.
- Weaknesses:
  - Ticker extraction precision (false positives from common words).
  - VADER ceiling for financial language understanding.
  - Single subreddit scope limits generalisability.
  - Class imbalance effects on metrics.
- Improvements still required:
  - FinBERT for better sentiment signal.
  - Multi-subreddit expansion.
  - Real-time inference capability.
  - Graph-based diffusion features.

**Input:** `evaluation_metrics.json`, `final_summary.json`, all evaluation figures, confusion matrices, `risk-and-challenges.md`.

#### 7.6 — Conclusion (0.25 day)

Summarise current achievements, key findings, remaining work, and possible future developments.

- Current achievements: functional end-to-end pipeline, multi-model comparison, statistical validation.
- Key findings: best model performance, tier achieved, which features matter most.
- Remaining work: sentiment model upgrade, ticker validation refinement, deployment considerations.
- Possible future developments: FinBERT integration, graph-based diffusion, multi-subreddit, real-time inference system.

**Input:** `final_summary.json`, `admin/decision-log.md`, `risk-and-challenges.md`.

#### 7.7 — Formatting & Polish (0.5 day)

- ACM citation format.
- Figure captions with numbering.
- Table formatting (consistent decimal places).
- Abstract (written last).
- Proofread for grammar, flow, and consistency.
- Check all figures are referenced in text.
- Verify word count / page limit compliance.

---

### Summary Timeline

| Phase | Effort | Priority |
|-------|--------|----------|
| 1. Foundation & folder reorg | 2–3 days | Critical |
| 2. Advanced evaluation | 1–2 days | High |
| 3. Sentiment improvement | 1 day | Medium |
| 4. Ticker hardening | 0.5 day | Low |
| 5. End-to-end run | 0.5 day | Critical |
| 6. CI & housekeeping | 0.5 day | Medium |
| 7. Report writing | 3–4 days | Critical |

**Total: ~9–12 days of focused work.**

Phase 1 is critical — it establishes the path conventions all subsequent code changes will use and makes the test suite functional. Phase 5 validates the entire system. Phase 2 adds the academic rigour needed for the final report. Phases 3–4 improve signal quality. Phase 6 is polish. Phase 7 is the final deliverable — it cannot begin until Phase 5 produces the evaluation artifacts, but sections 7.1–7.5 (methodology) can be drafted in parallel with Phases 2–4.
