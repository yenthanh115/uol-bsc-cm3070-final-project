# Draft Report Phase Plan

**Status as of 2026-07-19:** Phases 1–3 are COMPLETE. All experiments executed, all success criteria met. Focus is now entirely on Phase 4 (writing) and Phase 5 (review).

## Project Status Summary

| Phase                    | Goal                                     | Status                   |
| ------------------------ | ---------------------------------------- | ------------------------ |
| 1. Finish implementation | Complete the entire ML pipeline          | **DONE** ✓               |
| 2. Perform experiments   | Produce all evaluation results           | **DONE** ✓               |
| 3. Analyse results       | Explain why the model behaves as it does | **DONE** ✓ (data ready)  |
| 4. Write draft report    | Produce almost the entire report         | **IN PROGRESS** →        |
| 5. Internal review       | Find weaknesses before final report      | Pending                  |

---

## Phase 1 — Finish Implementation (DONE ✓)

### Data pipeline

- [x] Final dataset (r/pennystocks 80,212 records; r/wallstreetbets 1,293,981 records)
- [x] Data cleaning (text normalisation, deduplication)
- [x] Feature engineering (11 features: 9 base + 2 interaction terms)
- [x] Surge label generation (composite metric with configurable τ and weights)
- [x] Dataset splitting (80/20 temporal split)
- [x] Reproducible pipeline (config JSON, fixed seeds, CLI interface)

### Models

- [x] Logistic Regression (baseline, 10 hyperparameter configs)
- [x] Random Forest (ensemble, 36 configs)
- [x] XGBoost (boosting, 75 configs)

### Engineering

- [x] Configuration files (pipeline_config.json per run)
- [x] Reproducible experiments (fixed seeds, serialised models as .joblib)
- [x] Logging (timestamped log files in output/logs/)
- [x] Saved models (per-experiment directories: output/models/{exp_id}/)
- [x] Evaluation scripts (run_training.py, run_cross_validation.py, generate_figures.py)
- [x] Figures generated automatically (ROC curves, confusion matrices, threshold sensitivity)
- [x] Experiment log (append-only JSONL for consolidated history)

---

## Phase 2 — Perform Experiments (DONE ✓)

### Produced

- [x] Confusion matrices (per model, per dataset)
- [x] ROC curves (individual + combined overlay)
- [x] Feature importance (permutation-based + built-in gain-based)
- [x] Class distribution analysis (surge rates across configurations)
- [x] Threshold sensitivity curves (surge rate vs τ for 5 thresholds)
- [x] Classification threshold tuning (validation-fold selected thresholds)
- [x] Prediction examples (TP/FP/FN/TN records with features and context — generated via `src/generate_prediction_examples.py`)

### Evaluated using

- [x] Precision
- [x] Recall
- [x] F1
- [x] ROC-AUC (primary metric)
- [x] Bootstrap confidence intervals (1000 resamples)

### Comparisons performed

- [x] Random baseline (AUC=0.50)
- [x] Single-feature baselines (11 features evaluated individually)
- [x] Phase 1 vs Phase 2 (volume-only vs composite surge metric)
- [x] Threshold sensitivity (τ=1.0 vs τ=1.5)
- [x] Weight sensitivity sweep (w_sent: 0.0, 0.25, 0.5, 0.75, 1.0)

### Validation performed

- [x] Expanding-window temporal cross-validation (k=4 folds, 3 splits)
- [x] Cross-dataset transfer (D1: WSB→pennystocks; D2: pennystocks→WSB)
- [x] Multi-seed robustness (5 seeds: 42, 123, 456, 789, 2024)
- [x] McNemar's pairwise significance test (Bonferroni-corrected)
- [x] Statistical significance via bootstrap CIs

---

## Phase 3 — Analyse Results (DONE ✓ — data ready for write-up)

Key analytical findings available:

- [x] XGBoost dominates on WSB (dense data); Random Forest best on pennystocks (sparse data)
- [x] ticker_post_rate_24h is the most important feature across all models
- [x] Default classification threshold produces near-zero precision (threshold tuning essential)
- [x] Class imbalance is extreme (1.4% surge rate on WSB, 2.8% on pennystocks)
- [x] Cross-dataset transfer is asymmetric (pennystocks→WSB much better than reverse)
- [x] Sentiment in label definition adds +0.182 AUC on WSB (Phase 2 >> Phase 1)
- [x] Weight sweep shows non-monotonic curve: balanced 50/50 is optimal
- [x] Results are reproducible: 5-seed std = 0.008
- [x] VADER sentiment limitations visible in feature importance (sentiment_score ranks low)

---

### Completed experiments

| Machine | Experiments | Status |
|---------|------------|--------|
| Machine 1 (WSB core) | A2, B1, C1 | Done |
| Machine 2 (Pennystocks) | A1, B3, C2, F2×4 seeds | Done |
| Machine 3 (Weight sweep) | G2, G4, G5 | Done |
| Consolidation | D1, D2, figures | Done |

### Success criteria (all met)

| # | Criterion | Result |
|---|-----------|--------|
| 1 | ≥0.70 AUC on each dataset | WSB: 0.892 (XGBoost); Pennystocks: 0.753 (RF) |
| 2 | Phase 2 > Phase 1 | WSB: +0.182; Pennystocks: +0.042 |
| 3 | Cross-dataset AUC > 0.60 | D1: 0.684; D2: 0.871 |
| 4 | Multi-seed std < 0.03 | std = 0.008 (5 seeds) |

---

## Phase 4 — Write the Draft Report

Aim for a report that is **90–95% complete**.

### Report structure (ACM style, matching `draft-report-v0.1.md`)

1. Abstract
2. Introduction (1.1 Motivation, 1.2 Aims, 1.3 Concept, 1.4 Scope/RQ)
3. Literature Review (2.1–2.5)
4. Design (3.1 Architecture, 3.2 Technology, 3.3 Method Design, 3.4 Reproducibility)
5. Implementation (4.1–4.8)
6. Evaluation (5.1 Objectives, 5.2 Results, 5.3 Critical Analysis, 5.4 Limitations, 5.5 Originality)
7. Conclusion (6.1 Achievements, 6.2 Key Findings, 6.3 Remaining Work, 6.4 Future)
8. References

### Writing order (by dependency, not section order)

Write results first because they're the foundation everything else references.

| Step | Section(s) | Content | Time est. | Dependencies |
|------|-----------|---------|-----------|--------------|
| 1 | 5.2 Results | All metrics tables, figures, raw numbers | 4h | Experiment data (have it) |
| 2 | 5.1 Objectives | Verdict per objective citing 5.2 | 2h | Step 1 |
| 3 | 5.3 Critical Analysis | Why results behave as they do | 3h | Steps 1–2, feature importance JSONs |
| 4 | 5.4–5.5 Limitations + Originality | Honest weaknesses, contribution claim | 2h | Steps 1–3 |
| 5 | 3.3 Method Design | Formal feature/surge/CV definitions | 3h | Source code, pipeline configs |
| 6 | 3.1–3.2 Architecture + Tech | System diagram, tech justification | 2h | src/ structure |
| 7 | 4.1–4.6 Implementation | Code walkthrough, hyperparameter grids | 4h | Source modules |
| 8 | 4.7–4.8 Challenges + Progress | What went wrong, how fixed | 2h | Journal entries, decision log |
| 9 | 2. Literature Review | 15–20 sources, position against prior work | 6h | Academic search |
| 10 | 1. Introduction | Motivation, aims, scope, RQ | 2h | Lit review gaps + results |
| 11 | 6. Conclusion | Answer RQ, summarize, future work | 2h | Everything above |
| 12 | Abstract | Rewrite with final numbers | 1h | Everything above |
| 13 | References + polish | ACM format, consistency pass | 2h | Full draft exists |

**Total estimated: ~35 hours of writing**

### Key content per results subsection

**5.2.1 Model Performance Metrics**
- Per-model table: Precision, Recall, F1, AUC-ROC with 95% bootstrap CIs
- Two datasets side by side (WSB vs pennystocks)
- Success tier mapping

**5.2.2 Model Comparison**
- ROC curves (combined overlay, already generated as figures)
- McNemar's pairwise significance table

**5.2.3 Baseline Comparisons**
- Random baseline (AUC=0.50)
- Best single-feature baseline (ticker_post_rate_24h on WSB, hour_of_day on pennystocks)
- Multi-feature model improvement margin

**5.2.4 Threshold Sensitivity & Feature Importance**
- Threshold tuning results (default vs tuned F1)
- Feature importance rankings per model
- Weight sensitivity sweep curve (5 data points: w_sent 0.0→1.0)

**5.2.5 Cross-Dataset Transfer**
- D1 and D2 results table (per-model AUC)
- Asymmetry analysis (pennystocks→WSB transfers better)

**5.2.6 Robustness**
- 5-seed results table (mean ± std)
- Threshold sensitivity (τ=1.0 vs τ=1.5 comparison)

### Critical analysis (what makes average reports strong)

Instead of saying "XGBoost achieved 0.892 AUC", discuss:

* Why it outperformed others (feature importance differences, decision boundary complexity)
* Which features mattered most (ticker_post_rate_24h dominates)
* Where it failed (precision near zero at default threshold)
* Effect of class imbalance (1.4% surge rate on WSB)
* Why cross-dataset transfer is asymmetric
* Why sentiment weight matters (weight sweep curve shape)
* Implications for real-world deployment (threshold calibration is essential)

Then relate findings back to the research question.

### Figures needed in report

| Fig # | Content | Source |
|-------|---------|--------|
| 1 | Pipeline architecture diagram | Draw new |
| 2 | Temporal CV split illustration | Draw new |
| 3 | ROC curves combined (A2/WSB) | `output/figures/evaluation/A2/` |
| 4 | ROC curves combined (A1/pennystocks) | `output/figures/evaluation/A1/` |
| 5 | Confusion matrices (best model per dataset) | `output/figures/evaluation/` |
| 6 | Feature importance bar chart | Feature importance JSONs |
| 7 | Weight sensitivity curve (AUC vs w_sentiment) | Generate from G-series data |
| 8 | Threshold sensitivity curve (surge rate vs τ) | `threshold_sensitivity.csv` |
| 9 | Class distribution / EDA summary | `output/figures/eda/` |

---

## Phase 5 — Review Against the Rubric

### Literature (target: 15–20 sources)

- [ ] 8–12 peer-reviewed papers minimum
- [ ] Critical comparison (not just description)
- [ ] Research gap clearly identified
- [ ] Each source contributes to justifying a design decision

### Design

- [ ] Pipeline architecture diagram (clear, professional)
- [ ] Feature engineering flow diagram
- [ ] Model training workflow diagram
- [ ] Temporal validation design diagram
- [ ] All design choices justified (not just described)

### Implementation

- [ ] Working end-to-end pipeline (demonstrated via results)
- [ ] Clean code structure described
- [ ] Reproducible experiments (config JSON, fixed seeds, serialised models)
- [ ] Challenges and deviations documented

### Evaluation

- [ ] Multiple metrics (Precision, Recall, F1, AUC-ROC, with CIs)
- [ ] Baseline comparison (random + single-feature)
- [ ] Statistical significance (McNemar's, bootstrap CIs)
- [ ] Error analysis (confusion matrices, false positive patterns)
- [ ] Cross-dataset generalisation (D1, D2)
- [ ] Multi-seed robustness (F2, 5 seeds)
- [ ] Weight sensitivity analysis (G-series sweep)
- [ ] Threshold sensitivity (τ=1.0 vs τ=1.5)
- [ ] Discussion of limitations
- [ ] Threshold tuning analysis (default vs optimised)

### Originality

Demonstrated through:
- Composite surge metric (volume + sentiment, not in prior literature for Reddit penny stock forums)
- Expanding-window temporal CV (addresses common leakage mistake in social media ML)
- Multi-dataset comparison (sparse vs dense communities)
- Cross-dataset transfer evaluation (directional asymmetry finding)
- Systematic weight sensitivity analysis (5-point sweep)
- Leakage-free feature design (backward-looking only, no engagement metrics)

---

## Deliverables Checklist

By draft submission:

- [x] Complete end-to-end implementation
- [x] Final experimental results (14 experiments across 3 machines)
- [x] Baseline and improved model comparison
- [x] Cross-dataset evaluation (D1, D2)
- [x] Multi-seed robustness check (5 seeds, std=0.008)
- [x] Feature importance analysis
- [x] Weight sensitivity sweep (5 data points)
- [x] Evaluation figures generated (A1, A2, B1)
- [ ] Nearly complete literature review (15–20 sources)
- [ ] Complete methodology sections (Design + Implementation)
- [ ] Complete evaluation chapter with critical analysis
- [ ] Complete conclusions and future work
- [ ] High-quality diagrams (architecture, CV design, feature flow)
- [ ] Proper ACM citations throughout
- [ ] Consistent numbering and cross-references
- [ ] Abstract rewritten with final numbers
- [ ] Report at 90–95% completion, ready for tutor feedback
