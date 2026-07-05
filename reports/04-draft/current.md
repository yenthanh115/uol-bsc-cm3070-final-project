Here's my evaluation of the project:

---

## Project Evaluation: Surge-Labelling Pipeline for Reddit Trend Prediction

**Overall verdict: A solid, well-architected prototype that demonstrates strong software engineering discipline for an academic data science project. The data processing pipeline is complete and functional; the ML modelling layer is partially implemented.**

---

### Strengths

**1. Architecture & Modularity (Excellent)**
- Clean separation of concerns: 10 focused modules (loader, windowing, sentiment, labelling, normalisation, features, training, evaluation, pipeline, config).
- Well-defined pipeline orchestrator (`pipeline.py`) that chains stages deterministically.
- JSON-serialisable `PipelineConfig` dataclass with full audit trail support.
- Two CLI entry points (`run_pipeline.py`, `run_training.py`) with sensible argument parsing.

**2. Data Science Methodology (Strong)**
- Temporal train/test split (not random) — prevents data leakage.
- Z-score normalisation fit only on training partition.
- 9 backward-only features with explicit leakage prevention (no future information, no engagement metrics).
- Threshold sensitivity sweep with viability analysis (surge rate 5–10%).
- Composite surge metric using weighted z-scores of volume growth and sentiment change.

**3. Code Quality (Very Good)**
- Consistent docstrings with parameter/return documentation.
- Explicit traceability to requirements (R1, R2, R3... referenced throughout).
- Logging at appropriate levels with informative messages.
- Reproducibility handled via seeded randomness everywhere.
- Vectorised binary search approach (`np.searchsorted`) for windowing — computationally sound.

**4. Documentation (Good)**
- README covers setup, usage, structure, and outputs.
- Decision log and journal entries show the evolution of thinking.
- Multiple academic reports (literature review, design report, preliminary report).

---

### Weaknesses & Gaps

**1. Tests vs. Implementation Mismatch (Critical)**
- `test_training.py` imports `train_models`, `create_temporal_folds`, `get_expanding_window_splits`, `N_FOLDS`, `_get_rf_param_grid`, `_get_xgb_param_grid` — none of these exist in the current `training.py`.
- `test_evaluation_significance.py` imports `ModelMetrics`, `BaselineComparison`, `BootstrapCI`, `McNemarResult`, `FinalSummary`, `SuccessTierResult`, `evaluate_baselines`, `mcnemar_pairwise_test`, `produce_final_summary`, `validate_success_tiers` — none of these exist in the current `evaluation.py`.
- **The tests were written ahead of the implementation.** This means the test suite cannot currently pass, and there's no CI feedback loop.

**2. Model Coverage (Partial)**
- Only Logistic Regression is implemented in `training.py`. The tests expect Random Forest and XGBoost as well.
- The existing LR implementation is solid (expanding-window temporal CV, hyperparameter grid search, AUC-ROC selection, retrain on full training set).
- Missing: multi-model comparison, ensemble approaches, or any non-linear baseline.

**3. Evaluation Module (Incomplete)**
- Basic metrics (precision, recall, F1, ROC-AUC) and figure generation (confusion matrix, ROC curve, threshold sensitivity) are implemented.
- Missing the planned advanced evaluation: McNemar's test for statistical significance, baseline comparisons (random, single-feature), success tier validation, bootstrap CIs, final summary report.
- The `output/demo/evaluation/` directory is empty — no persisted evaluation results.

**4. Sentiment Analysis (Simplistic)**
- TextBlob polarity is a lightweight choice. For financial text from Reddit (slang, sarcasm, ticker mentions), it's unlikely to capture real sentiment well.
- No domain-specific sentiment model (FinBERT, etc.) or even VADER.
- This is fine for a prototype, but the feature likely contributes little signal.

**5. Ticker Extraction (Heuristic-based)**
- Regex + stopword filtering is reasonable for a prototype but noisy.
- No validation against an actual ticker list (e.g., from SEC or a financial API).
- Could produce false positives despite the extensive stopword set.

**6. No Results Yet**
- No evaluation metrics persisted anywhere.
- No evidence the pipeline has been run end-to-end on the real dataset (`r_pennystocks_submissions_reddit.csv`) with training and evaluation.
- Figures exist (12 PNGs) suggesting EDA and possibly one evaluation run, but no model performance artifacts.

---

### Maturity Assessment

| Aspect | Status |
|--------|--------|
| Data ingestion & preprocessing | ✅ Complete |
| Temporal windowing | ✅ Complete |
| Sentiment scoring | ✅ Complete (basic) |
| Surge labelling & normalisation | ✅ Complete |
| Feature engineering | ✅ Complete |
| Threshold sensitivity analysis | ✅ Complete |
| EDA & visualisation | ✅ Complete |
| Logistic Regression training | ✅ Complete |
| Model evaluation (basic) | ✅ Complete |
| Random Forest / XGBoost | ❌ Tests written, not implemented |
| Statistical significance testing | ❌ Tests written, not implemented |
| Baseline comparisons | ❌ Tests written, not implemented |
| Success tier validation | ❌ Tests written, not implemented |
| End-to-end run with results | ❌ No persisted outputs |
| CI / passing test suite | ❌ Tests will fail on import |

---

### Recommendations (Priority Order)

1. **Implement the multi-model training** (`train_models` with RF + XGBoost) to match `test_training.py` expectations. The tests already define the interface — treat them as a spec.
2. **Implement the advanced evaluation functions** (`mcnemar_pairwise_test`, `evaluate_baselines`, `validate_success_tiers`, `produce_final_summary`) to match `test_evaluation_significance.py`.
3. **Run the full pipeline end-to-end** on the real dataset and persist results. This validates the whole chain and gives you actual numbers to report.
4. **Get the test suite green.** Right now the imports will fail. Either implement the missing code or temporarily remove the forward-looking tests.
5. **Consider a better sentiment model** (VADER at minimum, FinBERT if runtime allows) — TextBlob on Reddit financial text is likely near-random.

---

**In summary:** The engineering scaffold is impressive for a BSc final project. The design decisions (temporal CV, leakage prevention, threshold sweeps, requirement traceability) show mature thinking. The gap is between planned and implemented functionality — roughly 60% of the ML modelling/evaluation layer exists in tests but not in code. Closing that gap and running a complete end-to-end experiment would elevate this from a well-structured prototype to a complete deliverable.