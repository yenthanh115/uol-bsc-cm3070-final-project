## Assignment Requirements (Draft Report)

The draft report consists of 6 chapters with a **strict total maximum of 9,500 words** (individual section limits are also strict, but their sum exceeds 9,500 to allow flexibility in distribution).

| Chapter | Max Words | Content |
|---------|-----------|---------|
| 1. Introduction | 1,000 | Project concept, motivation, which project template is used |
| 2. Literature Review | 2,500 | Revised from preliminary report |
| 3. Design | 2,000 | Revised from preliminary report |
| 4. Implementation | 2,000 | Major algorithms/techniques, key code explanations, visual results (screenshots/graphs) |
| 5. Evaluation | 2,500 | Testing results (unit testing, user studies, data testing), critical evaluation of achievements and areas for improvement |
| 6. Conclusion | 1,000 | Summary, broader themes, further work |

**Excluded from word count:** diagrams, figures, tables, references, title page.

**Key notes:**
- Figures/tables must be appropriate and clearly linked to written sections.
- The project does not need to be completed — the draft is for feedback on what will ultimately be submitted as the final report.
- Incorporate feedback from earlier submissions and peer reviews.
- The word limits are the same for the final report.

---

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
- `run_labeling.py` and `run_training.py` default CLI paths.
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
python src/run_labeling.py --file-path input/raw/r_pennystocks_submissions_reddit.csv --output-dir output/processed
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

### Phase 7: Report Writing (5 days)

**Goal:** Produce a near-submission-quality academic report (90–95% complete) following the prescribed 6-chapter structure. Strict total limit: **9,500 words** (references, title page, diagrams/figures/tables excluded from count).

---

#### Report Structure & Word Budget

| Chapter | Max Words | Instruction Requirement |
|---|---|---|
| 1. Introduction | 1,000 | Project concept, motivation, **state which project template is used** |
| 2. Literature Review | 2,500 | Revised from preliminary report |
| 3. Design | 2,000 | Revised from preliminary report |
| 4. Implementation | 2,000 | Major algorithms/techniques, key code explanation, visual results (screenshots/graphs) |
| 5. Evaluation | 2,500 | Initial evaluations carried out, results, critical evaluation of achievements and improvements |
| 6. Conclusion | 1,000 | Summary, broader themes, further work |

**Budget strategy:** Individual caps sum to 12,000 but total is capped at 9,500. Allocate aggressively to Evaluation (2,500) and Implementation (2,000) as these carry the most marks. Keep Introduction and Conclusion lean (~800 each) to preserve headroom.

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

#### 7.1 — Introduction (max 1,000 words · 0.5 day)

Explain the project concept and motivation. Based on the proposal but refined. **Must explicitly state the project template used.**

- Project motivation: why predicting posting-volume surges on r/pennystocks matters for market participants and researchers.
- Project aims: binary classification of ticker-level 24h posting-volume surges using backward-looking features.
- Project concept: composite surge metric, temporal CV methodology, multi-model comparison approach.
- **Project template declaration** (required by instruction): state which CM3070 project template this follows.
- Scope and research question.

**Input:** Proposal, preliminary report, literature review conclusions, `admin/decision-log.md`.

#### 7.2 — Literature Review (max 2,500 words · 0.5 day)

Revised version of the literature review submitted in the preliminary report. Incorporate any feedback received.

- Polish existing literature review (already written in `reports/01-literature-review/`).
- Address improvements based on previous feedback (reviewer comments, gaps identified).
- Add additional references if needed (2–3 recent citations on social media prediction, penny stocks, NLP in finance).
- Tighten the research gap statement to directly motivate the methodology.

**Input:** `reports/01-literature-review/literature-review-v1.0.md`, previous feedback, any new papers found during implementation.

#### 7.3 — Design (max 2,000 words · 0.75 day)

Revised version of the design chapter from the preliminary report. Focus on the *what* and *why* — design decisions, rationale, and justification.

- System architecture: pipeline stages (loading → preprocessing → feature engineering → labelling → training → evaluation).
- Data flow diagram: input sources → intermediate outputs → final artifacts.
- Technology choices and justification: Python, scikit-learn, XGBoost, VADER, pandas — why each was chosen over alternatives.
- Method design rationale:
  - Why backward-looking features (leakage prevention argument).
  - Why a composite surge metric rather than raw volume threshold.
  - Why expanding-window temporal CV rather than k-fold or random splits.
  - Why three model families (linear, ensemble, boosting) for comparison.
  - Why AUC-ROC as primary selection criterion given class imbalance.
- Reproducibility design: why fixed seeds, serialised models, and config JSON matter for this project.
- Threshold τ design: why sensitivity sweep, how the operating point was chosen.

**Input:** `src/surge_pipeline/config.py`, `features.py`, `labelling.py`, `normalisation.py`, `training.py`, `pipeline_config.json`, EDA figures 05–09.

#### 7.4 — Implementation (max 2,000 words · 0.75 day)

Describe the implementation of the project. Follow the style of the topic 6 peer review but expanded. Must include: major algorithms/techniques used, explanation of the most important parts of the code, and a visual representation of results (screenshots or graphs).

- Code organisation and module structure (`src/surge_pipeline/` layout).
- **Major algorithms/techniques:**
  - Ticker extraction: regex patterns, stopword filtering, known-ticker validation — specifics of implementation.
  - Sentiment analysis: VADER compound scoring, deduplication optimisation.
  - Temporal CV: expanding-window split logic, fold construction.
  - Model selection: GridSearchCV with custom scorer, final retraining procedure.
- **Key code explanation** (include annotated code snippets or pseudocode for the most important logic):
  - Feature engineering pipeline (9 features with temporal windowing).
  - Surge labelling with composite metric and configurable threshold.
  - Multi-model training with hyperparameter search (LR: 10 configs, RF: 36 configs, XGB: ≤50 configs).
- **Visual representation of results:**
  - Include relevant figures/screenshots showing pipeline output, EDA results, or model outputs.
  - Reference figures from `output/figures/eda/` and `output/figures/evaluation/`.
- Challenges encountered and how they were resolved:
  - Any deviations from original design (reference decision log).
  - Performance bottlenecks and optimisations applied.
  - Edge cases discovered during development.
- Implementation progress: all pipeline stages functional, end-to-end run producing artefacts.

**Input:** `src/surge_pipeline/` (all modules), `output/models/*.joblib`, `output/processed/pipeline_summary.json`, `admin/decision-log.md`.

#### 7.5 — Evaluation (max 2,500 words · 1.5 days) ⭐ KEY SECTION

Describe the evaluations carried out (unit testing, testing on data) and give results. Provide a **critical evaluation** of the project so far, making clear what has been achieved and what can be improved. Must extend beyond the feature prototype to cover the whole project.

---

##### 7.5.1 — Evaluate Against Project Objectives

Directly measure whether the project achieved its intended goals:

- **Objective 1: Predict posting-volume surges** — Did the models beat random and single-feature baselines? What tier was achieved (minimum 0.60 / target 0.70 / stretch 0.80)?
- **Objective 2: Compare multiple ML approaches** — Did the multi-model comparison reveal meaningful differences? Were differences statistically significant (McNemar's test)?
- **Objective 3: Demonstrate temporal validity** — Did the expanding-window CV and temporal train/test split prevent data leakage? Is there evidence the model generalises to unseen time periods?
- **Objective 4: Build a reproducible pipeline** — Can results be recreated from config JSON and fixed seeds? Are all artifacts traceable?

For each objective: state the goal, present the measured outcome, and give a clear verdict (met / partially met / not met).

---

##### 7.5.2 — Comprehensive Coverage

Evaluate ALL major features and components, not just the successful parts:

- **Data pipeline:** loading reliability, preprocessing quality, ticker extraction precision.
- **Feature engineering:** which of the 9 features contributed most? Feature importance from RF. Any features that added noise?
- **Surge labelling:** is the composite metric definition valid? Threshold sensitivity analysis — how sensitive are results to τ?
- **Model training:** convergence, hyperparameter sensitivity, training time.
- **Model performance:** all three models, not just the best one. Where each model succeeds and fails.
- **Statistical validation:** bootstrap CIs, McNemar comparisons, baseline comparisons.

---

##### 7.5.3 — Present Results Clearly

Present raw results BEFORE analysis. Readers need to see the evidence before the argument.

Use appropriate evidence for each claim:

- **Tables:**
  - Per-model metrics table (Precision, Recall, F1, AUC-ROC) with 95% bootstrap CIs.
  - McNemar's pairwise significance table (test statistic, p-value, significance after Bonferroni correction).
  - Baseline comparison table (random baseline AUC, best single-feature baseline AUC, improvement margin).
  - Success tier mapping table (model → tier achieved).
- **Figures:**
  - ROC curves (combined overlay showing all models + random baseline).
  - Confusion matrices per model (with actual counts, not just percentages).
  - Threshold sensitivity curve (metrics vs. classification threshold).
  - Feature importance bar chart (from Random Forest).
- **Clear narrative:**
  - For every table/figure: state what was measured, what the results show, and how conclusions were reached.
  - Cross-reference figures in the text — no orphaned visuals.

---

##### 7.5.4 — Critical Analysis

Go beyond reporting numbers. For each result, discuss WHY. The instruction requires a **critical evaluation** — demonstrate reflective judgement, not just metric reporting.

- **Why did the best model outperform others?**
  - Feature importance differences between models.
  - Decision boundary complexity (linear vs. tree-based).
  - Sensitivity to class imbalance.
- **What worked well?**
  - Temporal CV preventing overly optimistic estimates.
  - Composite surge metric capturing multi-dimensional signal.
  - Backward-looking features avoiding look-ahead bias.
- **What did NOT work?**
  - False positive patterns in confusion matrices — what types of records are misclassified?
  - Features with low importance — were they worth including?
  - VADER limitations on financial/Reddit slang.
- **Unexpected outcomes:**
  - Any model performing surprisingly well or poorly?
  - Threshold sensitivity — did small τ changes cause large performance shifts?
  - Class imbalance impact — precision vs. recall trade-off.
- **Relationship to objectives:**
  - Map each finding back to the stated research question.
  - If a model fails to reach minimum tier: explain why and what this means.

---

##### 7.5.5 — What Can Be Improved

The instruction explicitly asks to make clear "what you can improve." Identify limitations honestly, then propose concrete improvements:

| Limitation | Impact | Proposed Improvement |
|---|---|---|
| Ticker extraction false positives | Noisy labels reduce model signal | Known-ticker validation list, $-prefix confidence weighting |
| VADER ceiling for financial text | Sentiment feature underperforms | FinBERT fine-tuned on financial Reddit |
| Single subreddit scope | Limited generalisability | Multi-subreddit expansion (r/wallstreetbets, r/stocks) |
| Class imbalance (~15% positive) | Precision-recall trade-off | SMOTE, cost-sensitive learning, threshold tuning |
| Static feature window (24h) | May miss longer-term patterns | Multi-scale windows (6h, 24h, 72h) |

---

##### 7.5.6 — Originality & Contribution

Highlight what is novel (without overclaiming):

- Composite surge metric combining volume z-score and sentiment z-score — not found in prior literature for penny stock forums.
- Expanding-window temporal CV applied to social media prediction — addresses a common leakage mistake in similar studies.
- Multi-model comparison with statistical significance testing — goes beyond single-model reporting common in undergraduate projects.

Frame as contributions rather than "groundbreaking" — demonstrate awareness that these are incremental advances with clear academic value.

---

**Input:** `evaluation_metrics.json`, `final_summary.json`, all evaluation figures, confusion matrices, `risk-and-challenges.md`, `pipeline_config.json`, feature importance data.

#### 7.6 — Conclusion (max 1,000 words · 0.5 day)

Short summary of the project as a whole. Can also bring out broader themes or suggest further work.

- **Current achievements:**
  - Functional end-to-end pipeline from raw Reddit data to trained classifiers.
  - Multi-model comparison with statistical significance testing.
  - Reproducible results via fixed seeds, serialised configs, and automated pipeline.
  - Evaluation framework with bootstrap CIs, McNemar's test, and baseline comparisons.
- **Key findings:**
  - Best model performance and which tier was achieved.
  - Which features contributed most to prediction (link to RF feature importance).
  - Whether posting-volume surges are predictable from backward-looking features (answer to research question).
  - Relationship between findings and existing literature.
- **Broader themes:**
  - Implications for social media-based financial prediction research.
  - Temporal validity as a general concern in time-series ML projects.
- **Further work:**
  - Sentiment model upgrade (VADER → FinBERT) to address financial language ceiling.
  - Ticker validation refinement to reduce extraction false positives.
  - Class imbalance handling (cost-sensitive learning, threshold optimisation).
  - Multi-subreddit expansion for generalisability.
  - Real-time inference system with streaming Reddit data.
  - Integration with market data for downstream trading signal validation.

**Input:** `final_summary.json`, `admin/decision-log.md`, `risk-and-challenges.md`.

#### 7.7 — Formatting & Polish (0.5 day)

- Diagrams, figures, and tables: ensure all are appropriate and clearly linked to written sections (instruction requirement).
- Figure captions with numbering.
- Table formatting (consistent decimal places).
- ACM citation format.
- Abstract (written last).
- Proofread for grammar, flow, and consistency.
- Check all figures are referenced in text.
- **Word count verification:** total ≤ 9,500; each section within its individual cap.
- Incorporate feedback from earlier submissions and peer reviews (instruction requirement).

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
| 7. Report writing | 5 days | Critical |

**Total: ~10–13 days of focused work.**

Phase 1 is critical — it establishes the path conventions all subsequent code changes will use and makes the test suite functional. Phase 5 validates the entire system. Phase 2 adds the academic rigour needed for the final report. Phases 3–4 improve signal quality. Phase 6 is polish. Phase 7 is the final deliverable — it cannot begin until Phase 5 produces the evaluation artifacts, but sections 7.1–7.4 can be drafted in parallel with earlier phases.

---

#### Writing Order & Dependencies

- **7.1–7.4** (Introduction, Literature Review, Design, Implementation) can be drafted while Phase 5 artifacts are being generated. These sections describe what was planned and built, not the final results.
- **7.5** (Evaluation) requires final evaluation outputs from Phase 5 (`evaluation_metrics.json`, `final_summary.json`, all figures). This is the last major section to be written.
- **7.6** (Conclusion) is written after 7.5 — it summarises findings that only exist once evaluation is complete.
- **7.7** (Formatting & Polish) is done last, once all content is in place.

Within 7.5 (Evaluation), follow this writing order: present raw results first (7.5.3 — "here's what happened"), then critically analyse them (7.5.4 — "here's what it means"). Do not interleave presentation with interpretation — readers need to see the evidence before the argument.
