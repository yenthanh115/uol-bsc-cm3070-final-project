# Decision Log

This document records key design decisions made during the development of the Engagement and Sentiment Surge Prediction Pipeline. Each entry captures the context, decision, rationale, and alternatives considered for traceability and academic reporting.

## Table of Contents

| ID | Decision | Date | Status |
|----|----------|------|--------|
| DEC-001 | Composite surge threshold default value | 2026-06-08 | Superseded → DEC-013 |
| DEC-002 | Unit of analysis — per-ticker scoping | 2026-06-08 | Accepted |
| DEC-003 | Why combine engagement and sentiment (composite metric) | 2026-06-08 | Accepted |
| DEC-004 | Phased experimental approach (baseline → composite) | 2026-06-08 | Accepted |
| DEC-005 | Dataset selection — Reddit Finance Data (Kaggle) | 2026-06-08 | Accepted |
| DEC-006 | Statistical robustness in evaluation | 2026-06-08 | Accepted |
| DEC-007 | Tiered success criteria | 2026-06-08 | Accepted |
| DEC-008 | Literature review requires critical evaluation | 2026-06-08 | Accepted |
| DEC-009 | Report structure — add surge definition and unit of analysis | 2026-06-08 | Accepted |
| DEC-010 | Composite metric scale mismatch — z-score normalisation with configurable weighting | 2026-06-08 | Superseded → DEC-013 |
| DEC-011 | Per-ticker sparsity — minimum window record count | 2026-06-08 | Accepted |
| DEC-013 | Z-score normalisation of composite surge metric | 2026-06-11 | Accepted |
| DEC-012 | Eliminate snapshot engagement values — use posting volume instead | 2026-06-12 | Accepted |
| DEC-014 | Logistic Regression as baseline model | 2026-06-12 | Accepted |
| DEC-015 | Model selection rationale (LR + RF + XGBoost) | 2026-06-12 | Accepted |
| DEC-016 | TextBlob as initial sentiment tool | 2026-06-13 | Accepted |
| DEC-017 | Temporal train/test split (80/20 by timestamp) | 2026-06-13 | Accepted |
| DEC-018 | Expanding-window cross-validation for hyperparameter tuning | 2026-06-15 | Accepted |
| DEC-019 | Feature set design (9 leakage-free features) | 2026-06-15 | Accepted |
| DEC-020 | Hyperparameter grid sizing and search strategy | 2026-06-17 | Accepted (revised in implementation) |
| DEC-021 | Class imbalance handling — balanced class weights | 2026-07-03 | Accepted |
| DEC-022 | Feature scaling — uniform StandardScaler across all models | 2026-07-03 | Accepted |

---

## DEC-001: Composite surge threshold default value

- **Date:** 2026-06-08
- **Context:** The composite surge metric (`engagement_growth + |sentiment_change|`) requires a threshold to produce binary labels. The choice of threshold directly affects class distribution and model trainability.
- **Decision:** Set default threshold to 2.0, with configurable parameter and planned sensitivity analysis.
- **Rationale:** Engagement growth of 1.0 = doubling; sentiment change bounded by [0, 2.0]. A threshold of 2.0 requires substantial combined shift (e.g., tripling engagement with no sentiment change, or doubling engagement with full polarity reversal). Lower values (1.5) risk capturing routine noise; higher values (3.0) produce too few positives for model learning.
- **Alternatives considered:**
  - 1.5 — rejected: too permissive, inflates positive class with non-exceptional events
  - 3.0 — rejected: too restrictive, produces extremely sparse positives (<2%)
- **Status:** Accepted (subject to sensitivity analysis at {1.0, 1.5, 2.0, 2.5, 3.0})

---

## DEC-002: Unit of analysis — per-ticker scoping

- **Date:** 2026-06-08
- **Context:** Ambiguity about whether the prediction target represents a surge in a single discussion thread, a surge in discussion about a specific stock ticker, or a surge across the entire subreddit.
- **Decision:** Scope surge computation to same-ticker records within the 24-hour prediction window. For a record mentioning ticker $X at time t, only future records also mentioning $X within (t, t+24h] are considered.
- **Rationale:**
  1. Conceptual coherence — "a surge in stock-related discussion" most naturally means intensifying activity around a specific stock
  2. Practical utility — stakeholders care about per-security surges, not aggregate forum traffic
  3. Data availability — ticker symbols are extracted per record in the dataset
  4. Literature alignment — popularity/cascade prediction operates at content/topic level, not platform level
- **Alternatives considered:**
  - Per-thread scoping — rejected: comment-level thread data not available in dataset
  - Global (subreddit-wide) scoping — rejected: conflates unrelated events (e.g., simultaneous $GME and $AAPL activity)
- **Status:** Accepted

---

## DEC-003: Why combine engagement and sentiment (composite metric)

- **Date:** 2026-06-08
- **Context:** Question of whether engagement and sentiment should be treated as separate prediction targets or combined into a single composite.
- **Decision:** Combine into a single composite binary target, with a phased experimental approach to validate.
- **Rationale:**
  1. Neither signal alone captures a meaningful surge — high engagement without sentiment shift (memes, algorithms) or sentiment shift without visibility are both non-actionable
  2. Literature treats these signals independently; combining them tests an under-explored hypothesis
  3. Single binary target simplifies evaluation and enables direct model comparison
- **Alternatives considered:**
  - Two separate classifiers (one for engagement surge, one for sentiment surge) — rejected: doubles complexity without clear benefit at this stage
  - Multi-output model — rejected: unnecessary complexity for initial investigation
- **Validation plan:** Phase 1 trains on engagement-only target as baseline; Phase 2 introduces composite target. Comparison measures marginal contribution of sentiment.
- **Status:** Accepted

---

## DEC-004: Phased experimental approach (baseline → composite)

- **Date:** 2026-06-08
- **Context:** Need to empirically validate whether combining engagement and sentiment is actually beneficial, rather than assuming it.
- **Decision:** Adopt two-phase modelling strategy:
  - Phase 1 (Baseline): Engagement-only surge target
  - Phase 2 (Advanced): Composite engagement + sentiment target
- **Rationale:** Provides controlled evidence for/against sentiment's marginal contribution. If composite outperforms, validates the design. If not, produces equally valuable finding about signal redundancy in this domain.
- **Alternatives considered:**
  - Single composite model only — rejected: cannot attribute performance to specific signal components
  - Ablation study post-hoc — considered but phased approach is cleaner and more interpretable
- **Status:** Accepted

---

## DEC-005: Dataset selection — Reddit Finance Data (Kaggle)

- **Date:** 2026-06-08
- **Context:** Need a pre-collected static dataset of stock-related social media discussions with engagement metrics and text content for sentiment computation.
- **Decision:** Use the Reddit Finance Data dataset from Kaggle (https://www.kaggle.com/datasets/leukipp/reddit-finance-data), with `pennystocks/submissions_reddit.csv` as primary development dataset.
- **Rationale:**
  - ~1.38M total records across 9 subreddits (2021 calendar year)
  - Includes engagement metrics (score, num_comments), timestamps, text, and ticker symbols
  - pennystocks (54,785 records) selected for: high completeness (20.7% selftext missing — lowest among larger subs), sufficient volume, diverse ticker coverage (2,912 tickers)
  - Covers GameStop era — high retail activity period
- **Secondary datasets:** wallstreetbets (775K records) and gme (273K records) for generalisability testing
- **Alternatives considered:**
  - StockTwits data — not readily available as static dataset
  - Twitter financial data — API access restrictions (post-2023 policy changes)
- **Status:** Accepted

---

## DEC-006: Statistical robustness in evaluation

- **Date:** 2026-06-08
- **Context:** Single-run point estimates are insufficient for academic claims about model performance, especially on imbalanced datasets.
- **Decision:** Implement three statistical procedures:
  1. Bootstrap confidence intervals (95%, 1,000 iterations) for all metrics
  2. Multiple-seed evaluation (5 seeds: 42, 123, 256, 512, 1024) to assess initialisation sensitivity
  3. McNemar's test for paired model comparison (α = 0.05, Bonferroni correction)
- **Rationale:** Two models differ meaningfully only if CIs don't overlap. Seed variation quantifies stability. Significance testing prevents over-interpreting noise.
- **Alternatives considered:**
  - k-fold cross-validation — rejected: temporal ordering makes random folds inappropriate (leakage risk)
  - Permutation tests — considered: may add as supplementary if time allows
- **Status:** Accepted

---

## DEC-007: Tiered success criteria

- **Date:** 2026-06-08
- **Context:** Original success criterion (AUC > 0.60) was a single pass/fail threshold. Needed more nuanced framing.
- **Decision:** Three-tier success framework:
  - Minimum: AUC-ROC > 0.60 (validates feasibility)
  - Target: AUC-ROC > 0.70 (moderate discriminative power, comparable to literature)
  - Stretch: AUC-ROC > 0.80 (strong contribution)
- **Rationale:** Minimum ensures the project demonstrates *something* beyond random. Target aligns with early-stage results in related work (Szabo ~0.7 correlations, Cheng ~0.877 AUC on cascades). Stretch acknowledges that financial text prediction is harder than image cascades.
- **Alternatives considered:**
  - Single threshold only — rejected: too binary, doesn't communicate ambition
  - F1-based criteria — considered: added as supplementary (threshold sensitivity produces best-F1 operating point)
- **Status:** Accepted

---

## DEC-008: Literature review requires critical evaluation

- **Date:** 2026-06-08
- **Context:** Initial literature summary was descriptive ("Paper A did X, Paper B did Y") rather than critically evaluative.
- **Decision:** Rewrite to assess each work on methodological strengths, validity threats, generalisability concerns, and specific relevance/limitations for this project.
- **Rationale:** Academic requirement for critical evaluation. Descriptive summaries don't demonstrate understanding or justify design choices. Critical assessment connects literature gaps directly to project design decisions.
- **Key critiques identified:**
  - Szabo & Huberman: assumes content already has engagement; not applicable to pre-engagement prediction
  - Bollen et al.: short evaluation period, no out-of-sample validation, lexicon tools miss financial jargon
  - Bandari et al.: 84% accuracy on 4-class task is less impressive than headline suggests; features designed for news not forums
  - Cheng et al.: Facebook photo cascades don't transfer to text-based financial discussion
- **Status:** Accepted

---

## DEC-009: Report structure — add surge definition and unit of analysis

- **Date:** 2026-06-08
- **Context:** Report lacked formal definition of what constitutes a "surge" and what the unit of prediction is. Reviewer feedback identified these as conceptual gaps.
- **Decision:** Add dedicated sections: 1.4 (Unit of Analysis and Prediction Scope) and 1.5 (Surge Definition) with formal mathematical definition.
- **Rationale:** Without these, the report is ambiguous about what is being predicted. Explicit definitions prevent misinterpretation and strengthen the methodology.
- **Status:** Accepted

---

## DEC-010: Composite metric scale mismatch — empirical investigation over a priori normalisation

- **Date:** 2026-06-08
- **Context:** The composite surge formula (`engagement_growth + |sentiment_change|`) combines two components on different scales. Engagement growth is an unbounded ratio (values of 10+ are common for low-engagement posts), while |sentiment_change| is bounded by [0, 2.0] due to TextBlob's polarity range. This means the composite is likely dominated by the engagement component in practice, making the "composite" nature potentially illusory.
- **Decision:** ~~Retain the raw additive formulation as the starting point.~~ **SUPERSEDED by DEC-013** (2026-06-11). The original approach of deferring normalisation to post-hoc analysis was found to be methodologically unsound — it would allow the volume component to structurally dominate the target label, making the Phase 1 vs Phase 2 comparison unable to detect sentiment's contribution even if one existed.
- **Status:** ~~Accepted~~ Superseded → DEC-013

---

## DEC-011: Per-ticker sparsity — minimum window record count

- **Date:** 2026-06-08
- **Context:** The per-ticker surge computation requires future records mentioning the same ticker within a 24-hour window. Many tickers (likely the majority of the 2,912 in pennystocks) have very sparse posting activity — their 24h window may contain 0, 1, or 2 future records. With so few data points, engagement growth becomes a ratio of near-zero denominators, and mean sentiment is computed from 1–2 samples, making the composite metric numerically unstable and the surge label essentially random.
- **Decision:** Enforce a minimum record count (default N ≥ 3) within each ticker's 24-hour forward window. Records that do not meet this threshold are excluded from surge labelling entirely (they receive no label and are dropped from training/evaluation).
- **Rationale:**
  1. A ratio computed from 0–2 future observations is not statistically meaningful — a single outlier can flip the label
  2. Mean sentiment from 1–2 records has no interpretive value (no regression toward any true mean)
  3. Excluding unstable labels is preferable to training on noise
  4. The exclusion rate quantifies how much of the dataset is affected, which is itself informative about the dataset's suitability for per-ticker analysis
- **Alternatives considered:**
  - No minimum (include all records) — rejected: would produce noisy labels that degrade model learning
  - Higher minimum (N ≥ 5 or N ≥ 10) — deferred: may exclude too much data, but will be tested in sensitivity analysis
  - Imputation or smoothing (e.g., Bayesian shrinkage toward global mean) — considered for future work: adds complexity without clear benefit at this stage
  - Fallback to subreddit-scoped windowing for sparse tickers — rejected: breaks conceptual coherence of per-ticker design
- **Validation plan:** During EDA, compute the distribution of per-ticker window sizes across the dataset. Report: (a) what fraction of records are excluded at N ≥ 3, (b) how exclusion rate changes at N ∈ {1, 2, 3, 5, 10}, (c) whether excluded records are systematically different from retained records.
- **Risk classification:** Elevated to formal risk (Risk #9, Likelihood: High, Impact: High) because ticker frequency follows a power-law distribution — most tickers are mentioned rarely.
- **Status:** Accepted (N ≥ 3 as default, subject to EDA sensitivity analysis)

---

## DEC-013: Z-score normalisation of composite surge metric

- **Date:** 2026-06-11
- **Context:** The composite surge formula (`posting_volume_growth + |sentiment_change|`) combines two components on fundamentally different scales. Posting volume growth is an unbounded ratio (values of 10+ are common for tickers going from 1–2 posts/day to 10+), while |sentiment_change| is bounded by [0, 2.0] due to TextBlob's polarity range of [-1, +1]. Without normalisation, the volume component structurally dominates the composite — rendering sentiment unable to influence the surge label. This undermines the core research question (does sentiment add predictive value?) because Phase 2 would effectively use the same label as Phase 1 regardless of sentiment dynamics.
- **Decision:** Replace the raw additive formula with z-score normalised components and configurable weighting:
  ```
  z_volume    = (posting_volume_growth − μ_vol) / σ_vol
  z_sentiment = (|sentiment_change| − μ_sent) / σ_sent
  composite   = (w₁ × z_volume) + (w₂ × z_sentiment)
  ```
  where μ and σ are computed from the training partition only, and w₁ = w₂ = 0.5 by default.
- **Rationale:**
  1. **Equal contribution by construction** — both components become zero-mean, unit-variance; neither dominates regardless of their raw scale differences
  2. **Interpretable threshold** — τ expressed in standard deviation units has consistent statistical meaning across all operating points ("the combined signal exceeds τ SDs above typical")
  3. **Principled Phase 1 vs Phase 2 comparison** — Phase 1 sets w₂ = 0; Phase 2 sets w₂ = 0.5. Because z-score normalisation guarantees sentiment carries equal weight in Phase 2, any performance difference reflects a genuine signal contribution, not a scale artefact
  4. **Configurable weight** — w₁/w₂ split becomes a hyperparameter for sensitivity analysis (sweep w₂ ∈ {0, 0.25, 0.5, 0.75, 1.0}), providing empirical evidence for optimal weighting
  5. **No circular dependency** — normalisation statistics are fitted on training data only; test records use training-set μ/σ, preventing leakage
- **Supersedes:** DEC-010 (which deferred normalisation to post-hoc analysis). The original reasoning — that normalisation "requires assumptions about relative importance" — was incorrect. Equal weighting (w₁ = w₂ = 0.5) is the *neutral* assumption (agnostic prior), and the weight sweep provides the empirical grounding that DEC-010 sought to defer to. Deferring normalisation would have meant that the Phase 1 vs Phase 2 comparison was structurally incapable of detecting sentiment's contribution even if one existed.
- **Also supersedes:** DEC-001 threshold value (2.0). The threshold is now expressed in standard deviation units rather than raw composite units, so the numeric value will differ. Default τ to be determined empirically during EDA.
- **Alternatives considered:**
  - Min-max normalisation — rejected: sensitive to outliers; volume growth has no natural maximum so min-max requires clipping
  - Rank-based normalisation — rejected: loses magnitude information that may be predictively useful
  - Multiplicative combination (volume_growth × |sentiment_change|) — rejected: produces zero whenever either component is zero, which is too restrictive (a pure volume surge with neutral sentiment would score 0)
  - Separate thresholds per component (volume > X AND sentiment > Y) — rejected: creates a rectangular decision boundary that doesn't capture the intuition of "combined" surge strength
- **Implementation notes:**
  - μ_vol, σ_vol, μ_sent, σ_sent computed from training partition only (first 80% by timestamp)
  - Test records normalised with training-set statistics (no test leakage)
  - If σ = 0 for either component (degenerate case, e.g., all training records have identical volume growth), that component receives z = 0 and the composite reduces to the other component alone
- **Impact on report sections:**
  - Section 1.5 (Surge Definition): formula updated, "Scale considerations" paragraph removed (problem solved by design), normalisation rationale and configurable weighting paragraphs added
  - Section 3.5 (Composite Target Design): steps updated to include standardisation
  - Section 4.7: renamed to "Threshold and Weight Sensitivity Analysis"; threshold values updated to SD units; weight sweep added
  - Section 8 (Success Criteria): weight sweep added to analytical criteria
  - Risk Register #6: updated description
- **Status:** Accepted

---

## DEC-012: Eliminate snapshot engagement values from features and target — use posting volume instead

- **Date:** 2026-06-12
- **Context:** The Reddit Finance dataset provides engagement metrics (score, num_comments) as **final snapshot values** collected at crawl time, not as point-in-time values at post creation. This means these values incorporate all future engagement — including engagement generated *by* the surge being predicted. Using them as features (`engagement_rate = score / hours_since_posting`) or in the target formula (`engagement_growth = (future_engagement − current_engagement) / max(current_engagement, 1)`) constitutes temporal data leakage. The model would effectively "see" the outcome it is trying to predict.
- **Decision:** 
  1. **Features:** Remove `engagement_rate` and `has_ticker` from the feature set. Do not use `score` or `num_comments` as prediction-time features. Replace with leakage-free alternatives: `ticker_post_rate_24h` (backward-looking post count), `ticker_post_acceleration` (ratio of recent to older posting frequency), `title_length`, and `num_tickers_mentioned`.
  2. **Target:** Replace score-based engagement growth with **posting volume growth** — the ratio of posts mentioning the same ticker in the future 24h window vs. the prior 24h window. This is derived entirely from creation timestamps, which are fixed at post creation and uncontaminated by future activity.
  3. **Constraint:** No dataset field that represents a post-creation-time aggregate (score, num_comments) may be used as a feature or in the target formula unless it can be proven to represent a value available at observation time *t*.
- **Rationale:**
  1. Score and num_comments are accumulated over the post's lifetime — they are outcomes, not inputs available at prediction time
  2. `engagement_rate` = score / hours_since_posting gives an average rate using the final score, which includes all future votes including those from the surge period
  3. In a hypothetical deployment scenario, these values would not be available at observation time — only the text, timestamp, and historical posting patterns would be observable
  4. Posting volume (count of posts) is derived from timestamps alone and is conceptually aligned with what a surge means: more people creating posts about a stock
  5. `has_ticker` removed because all records in the modelling dataset mention at least one ticker (prerequisite for ticker-scoped analysis) — the feature would be constant = 1
- **Impact on composite target formula:**
  - Old: `composite = engagement_growth + |sentiment_change|` where engagement_growth used score values
  - New: `composite = posting_volume_growth + |sentiment_change|` where posting_volume_growth = (count of $X posts in (t, t+24h]) / max(count of $X posts in (t−24h, t], 1)) − 1
  - Default threshold (previously 2.0) will be redetermined empirically during EDA since the scale of posting volume growth differs from score-based growth
- **Impact on feature set:**
  - Removed: `engagement_rate`, `has_ticker`
  - Added: `ticker_post_rate_24h`, `ticker_post_acceleration`, `title_length`, `num_tickers_mentioned`
  - Retained unchanged: `sentiment_score`, `hour_of_day`, `day_of_week`, `time_since_previous` (scope clarified to per-ticker), `word_count`
- **Alternatives considered:**
  - Keep score in target but exclude from features (Option B) — rejected: still introduces circular dependency in labelling (surge drives high scores on future posts, which inflates the "future engagement" term)
  - Use score as a feature with a fixed early-window proxy (e.g., score at 1 hour) — rejected: dataset does not provide temporal score snapshots, only final values
  - Remove engagement entirely and predict sentiment-only surges — rejected: loses the volume dimension which is central to the surge concept
- **Status:** Accepted


---

## DEC-014: Logistic Regression as baseline model

- **Date:** 2026-06-12
- **Context:** The project requires a baseline classifier against which more complex models (RF, XGBoost) are compared. The choice of baseline affects the interpretability of comparative results — it must be simple enough to represent a "lower bound" while being credible enough that beating it is meaningful.
- **Decision:** Use Logistic Regression (L2-regularised) as the primary baseline model.
- **Rationale:**
  1. **Interpretability** — LR coefficients directly indicate feature importance direction and magnitude, valuable for understanding which features drive surge prediction
  2. **Linear assumption as lower bound** — if the relationship between features and surge is non-linear, tree-based models should demonstrably outperform LR; if LR performs comparably, it suggests the problem is approximately linear and simpler models suffice
  3. **Standard in literature** — LR is the conventional baseline in binary classification studies across ML and NLP (Bandari et al., Cheng et al. both compare against logistic baselines)
  4. **Fast training** — enables rapid iteration during development and supports the full sensitivity sweep (150 configurations) within reasonable compute budgets
  5. **Well-calibrated probabilities** — LR outputs are naturally calibrated, making threshold sensitivity analysis (varying the classification cutoff) directly interpretable
- **Alternatives considered:**
  - Naive Bayes — rejected: assumes feature independence, which is known to be violated (e.g., `ticker_post_rate_24h` and `ticker_post_acceleration` are correlated by construction)
  - SVM (linear kernel) — rejected: produces only decision boundary, not calibrated probabilities; makes threshold analysis less natural; no significant advantage over LR for linear problems
  - Majority-class classifier — too trivial: included as a sanity check (random baseline with AUC = 0.5) but not a credible "model" baseline
  - Single-feature LR — included as part of baseline comparisons (DEC-006) but distinct from the full-feature LR baseline
- **Status:** Accepted

---

## DEC-015: Model selection rationale (LR + RF + XGBoost)

- **Date:** 2026-06-12
- **Context:** Need to select a set of classifiers that covers different modelling paradigms (linear, ensemble-bagging, ensemble-boosting) while remaining tractable for a single-researcher project with temporal CV and full sensitivity analysis.
- **Decision:** Train and compare three models: Logistic Regression, Random Forest, and XGBoost.
- **Rationale:**
  1. **Paradigm coverage** — LR (linear), RF (bagging/variance reduction), XGBoost (boosting/bias reduction) represent the three dominant approaches for tabular classification; if one paradigm is clearly superior, this design will detect it
  2. **Tabular data suitability** — with 9 numeric features and ~54K records, the problem is squarely in the regime where tree-based ensembles excel. Neural networks offer no structural advantage on low-dimensional tabular data and require more tuning (Grinsztajn et al., 2022)
  3. **Practical compute budget** — three models × temporal CV × grid search × sensitivity sweep is feasible on a single machine. Adding more models would expand the comparison without proportional insight
  4. **XGBoost over LightGBM** — XGBoost was chosen over LightGBM because: (a) more established in academic literature with broader citation base, (b) functionally equivalent on datasets of this size (LightGBM's speed advantage manifests on larger datasets), (c) slightly simpler API for reproducibility
  5. **Complementary failure modes** — LR fails on non-linear interactions, RF can overfit noisy features via deep splits, XGBoost can overfit via excessive boosting rounds. McNemar's test (DEC-006) identifies which models make systematically different errors
- **Alternatives considered:**
  - SVM (RBF kernel) — rejected: expensive grid search (C × γ), no native probability output (requires Platt scaling), no feature importance
  - Neural network (MLP) — rejected: no structural advantage on 9-feature tabular data; introduces architecture/optimisation decisions that complicate reproducibility; overfits easily on 54K records without careful regularisation
  - LightGBM — rejected: functionally near-identical to XGBoost for this data size; using both would produce near-duplicate results without scientific value
  - Naive Bayes, k-NN — rejected: too simple/limited to add meaningful comparison beyond LR
  - Stacking/meta-learner — rejected: adds complexity without directly addressing the research question (does sentiment help?); considered as future work
- **Status:** Accepted

---

## DEC-016: TextBlob as initial sentiment tool

- **Date:** 2026-06-13
- **Context:** The pipeline requires sentiment polarity scores for each post's text content. The choice of sentiment tool affects signal quality (and therefore model performance), runtime, and reproducibility.
- **Decision:** Use TextBlob's `sentiment.polarity` as the initial sentiment computation method, with planned migration to VADER in Phase 3.
- **Rationale:**
  1. **Simplicity and reproducibility** — TextBlob is a deterministic, rule-based tool with no model weights to download or GPU requirements. Results are perfectly reproducible across environments
  2. **Sufficient for pipeline validation** — Phase 1 focuses on establishing the infrastructure (windowing, labelling, temporal CV). A "good enough" sentiment signal validates the pipeline; signal quality improvement is orthogonal and addressed in Phase 3
  3. **Known baseline** — TextBlob's limitations are well-documented (misses sarcasm, financial jargon, Reddit slang). This makes it a conservative baseline — if the composite target shows value with TextBlob, it will likely improve with a better sentiment tool
  4. **Zero-configuration** — no API keys, no model downloads, no tokeniser setup. Reduces friction during initial development
- **Known limitations:**
  - Lexicon-based: misses context-dependent sentiment (e.g., "to the moon" is positive in WSB context but neutral in TextBlob)
  - Trained on product reviews, not financial text
  - Doesn't handle Reddit-specific conventions (capitalisation for emphasis, emoji, ticker symbols as emotional proxies)
  - These limitations are documented in the risk register (Risk #2)
- **Migration plan:** Phase 3 swaps to VADER (handles social media conventions: exclamation, capitalisation, slang) with optional FinBERT path for financial domain understanding. Interface is preserved (`sentiment_polarity` column, range [-1, 1]).
- **Alternatives considered:**
  - VADER from the start — deferred: slightly more setup, and the priority was pipeline correctness over signal quality
  - FinBERT from the start — rejected: requires `transformers` dependency, GPU for reasonable speed, and model download (~400MB). Over-investment before validating the pipeline works end-to-end
  - No sentiment (volume-only throughout) — rejected: defeats the research question. Even a noisy sentiment signal allows validating the composite vs. volume-only comparison
  - Custom fine-tuned model — rejected: no labelled financial sentiment training data available; out of scope for this project
- **Status:** Accepted

---

## DEC-017: Temporal train/test split (80/20 by timestamp)

- **Date:** 2026-06-13
- **Context:** The dataset spans January–December 2021. A splitting strategy is needed that respects temporal ordering (no future information in training) while providing a sufficiently large test set for reliable evaluation.
- **Decision:** Split data chronologically: first 80% of records (by timestamp) form the training set; final 20% form the held-out test set. No shuffling, no stratification.
- **Rationale:**
  1. **No temporal leakage** — a random split would allow the model to train on December posts and predict January posts, which violates the real-world deployment scenario (you can only predict future surges using past data)
  2. **Realistic evaluation** — mirrors actual use: model trained on historical data, evaluated on unseen future data. This is the gold standard for time-series-adjacent prediction tasks
  3. **80/20 ratio** — with ~54K records, 20% ≈ 10,900 test records. This provides sufficient statistical power for bootstrap CIs (DEC-006) and McNemar's test while keeping 80% ≈ 43,600 records for training (enough for temporal CV within training)
  4. **No stratification** — class imbalance in the test set reflects the natural surge rate in the evaluation period. Stratification would artificially equalise positive/negative rates across train and test, hiding potential concept drift
  5. **Concept drift acknowledged** — training ≈ Jan–Oct (includes meme-stock mania Q1), test ≈ Nov–Dec (post-squeeze normalisation). If models trained on high-activity periods generalise to quieter periods, this strengthens the validity claim. If not, it's a documented limitation rather than a flaw
- **Alternatives considered:**
  - Random 80/20 split — rejected: temporal leakage makes results uninterpretable for a forecasting task
  - Stratified temporal split — rejected: would require matching positive class rates across periods, which masks the concept drift that evaluators should know about
  - 70/30 split — rejected: reduces training data by ~5,400 records without proportional benefit to test reliability
  - 90/10 split — rejected: ~5,400 test records is borderline for bootstrap CI stability with 1,000 iterations
  - Sliding-window evaluation (multiple test periods) — considered as future work: provides more robust evaluation but multiplies compute cost and complicates reporting. Acknowledged in report limitations
- **Status:** Accepted

---

## DEC-018: Expanding-window cross-validation for hyperparameter tuning

- **Date:** 2026-06-15
- **Context:** Hyperparameter tuning requires internal validation within the training partition. Standard k-fold CV randomly shuffles data, violating temporal ordering. A time-respecting CV strategy is needed.
- **Decision:** Use expanding-window (anchored) temporal cross-validation with k=4 folds within the 80% training partition. The training window starts at the first record and grows with each fold; the validation window is always the next temporal block.
- **Rationale:**
  1. **Respects temporal ordering** — each fold trains on earlier data and validates on later data, preventing future information from influencing model selection
  2. **Expanding (anchored) over sliding** — expanding windows use all available historical data for training, which is more data-efficient. With only ~43K training records, discarding early data (as sliding windows do) reduces training set size unnecessarily
  3. **k=4 folds** — balances between: (a) enough folds for stable hyperparameter selection, (b) each validation fold having enough records for meaningful AUC estimation (~2,700 records per fold), and (c) computational cost (each fold requires full grid search evaluation)
  4. **Consistent with final evaluation** — the CV scheme mirrors the global train/test split philosophy: always train on past, evaluate on future
- **Fold structure (approximate, for ~43,600 training records):**
  - Fold 1: Train on records 1–8,700; Validate on 8,701–17,400
  - Fold 2: Train on records 1–17,400; Validate on 17,401–26,100
  - Fold 3: Train on records 1–26,100; Validate on 26,101–34,800
  - Fold 4: Train on records 1–34,800; Validate on 34,801–43,600
- **Alternatives considered:**
  - Standard k-fold CV — rejected: shuffles temporal order, introduces leakage
  - Sliding-window CV — rejected: discards early training data (fold 3 wouldn't use data from fold 1 period). Acceptable for concept-drift-heavy domains, but here we want maximum training data per fold
  - Leave-one-out temporal (daily/weekly blocks) — rejected: too many folds, computationally expensive, and individual folds may have too few positive examples for stable AUC
  - Nested CV (outer temporal split + inner temporal CV) — rejected: adds a layer of complexity without clear benefit given the single held-out test set design
  - k=5 folds — rejected: validation folds become ~2,200 records each; with potentially low positive class rates, some folds might have <50 positive examples, making per-fold AUC unstable
- **Status:** Accepted

---

## DEC-019: Feature set design (9 leakage-free features)

- **Date:** 2026-06-15
- **Context:** After eliminating snapshot engagement values (DEC-012), the feature set needed redesigning from scratch. Features must be: (a) available at observation time *t* without future information, (b) derivable from the dataset fields (text, timestamp, ticker), (c) diverse enough to give tree-based models interaction opportunities, and (d) interpretable for academic reporting.
- **Decision:** Use the following 9 features for all models:

  | # | Feature | Type | Source | Rationale |
  |---|---------|------|--------|-----------|
  | 1 | `sentiment_polarity` | Continuous [-1, 1] | TextBlob on post text | Core research variable — sentiment signal |
  | 2 | `word_count` | Integer | len(text.split()) | Proxy for post effort/detail; longer posts may indicate conviction |
  | 3 | `title_length` | Integer | len(title) | Headline engagement signal; short punchy titles vs. detailed analysis |
  | 4 | `hour_of_day` | Integer [0–23] | Timestamp extraction | Temporal pattern — market hours vs. after-hours posting |
  | 5 | `day_of_week` | Integer [0–6] | Timestamp extraction | Weekly seasonality — weekday trading vs. weekend speculation |
  | 6 | `time_since_previous` | Float (hours) | Same-ticker time delta | Per-ticker posting acceleration — short gaps indicate building momentum |
  | 7 | `ticker_post_rate_24h` | Float | Count of same-ticker posts in prior 24h | Backward-looking activity level for this ticker |
  | 8 | `ticker_post_acceleration` | Float | Ratio of recent (12h) to older (12–24h) posting rate | Momentum — is activity accelerating or decelerating? |
  | 9 | `num_tickers_mentioned` | Integer | Count of tickers extracted from post | Multi-ticker posts may indicate cross-stock momentum events |

- **Rationale:**
  1. **No leakage** — all features derived from information available at post creation time (text content, timestamp, historical same-ticker activity). No feature uses score, num_comments, or any post-creation aggregate
  2. **Three signal categories** — content (1, 2, 3, 9), temporal (4, 5, 6), and ticker-activity (7, 8). Provides diversity for tree-based models to find interactions
  3. **Sentiment as explicit feature** — rather than only in the target, including `sentiment_polarity` as a feature lets models learn non-linear relationships between post sentiment and surge likelihood (e.g., extreme negative sentiment might predict surges as well as extreme positive)
  4. **Per-ticker activity features** — `ticker_post_rate_24h` and `ticker_post_acceleration` are the strongest leakage-free proxies for "building attention." They capture the backward-looking momentum that may predict forward-looking surges
  5. **Minimal feature count** — 9 features is modest, which: (a) reduces overfitting risk on 54K records, (b) ensures each feature can be meaningfully analysed for importance, (c) is comparable to feature counts in related literature (Bandari used 7, Cheng used ~12)
- **Excluded features (with reasoning):**
  - `engagement_rate` (score/hours) — snapshot leakage (DEC-012)
  - `has_ticker` — constant = 1 for all records in modelling set (DEC-012)
  - `score`, `num_comments` — snapshot values, not available at prediction time
  - `is_selftext` — considered but provides minimal signal (binary, most pennystocks posts are self-text)
  - `sentiment_subjectivity` — listed as candidate addition (journal 2026-06-12a, concern #4) but deferred to avoid scope creep; can be added if tree models underperform
- **Status:** Accepted

---

## DEC-020: Hyperparameter grid sizing and search strategy

- **Date:** 2026-06-17
- **Context:** Each model requires hyperparameter tuning via temporal CV (DEC-018). The grid size directly affects compute time (grid_size × 4 folds × n_models) and the risk of overfitting to the validation folds. Need to balance thorough search against practical runtime constraints.
- **Decision:** Use exhaustive grid search for LR and RF (small grids), and randomised search (≤50 configurations) for XGBoost (large hyperparameter space).
- **Grid specifications:**

  **Logistic Regression (12 configurations):**
  - `C`: [0.01, 0.1, 1.0, 10.0] — regularisation strength
  - `penalty`: ['l1', 'l2'] — sparsity vs. ridge
  - `solver`: matched to penalty (saga for l1, lbfgs for l2)

  **Random Forest (36 configurations):**
  - `n_estimators`: [100, 200, 500] — ensemble size
  - `max_depth`: [5, 10, 20, None] — tree complexity
  - `min_samples_leaf`: [1, 5, 10] — regularisation via minimum leaf size

  **XGBoost (≤50 random configurations from):**
  - `n_estimators`: [100, 200, 300, 500]
  - `max_depth`: [3, 5, 7, 9]
  - `learning_rate`: [0.01, 0.05, 0.1, 0.2]
  - `subsample`: [0.7, 0.8, 0.9, 1.0]
  - `colsample_bytree`: [0.7, 0.8, 0.9, 1.0]
  - Full grid = 4×4×4×4×4 = 1,024 configurations → sampled to 50

- **Rationale:**
  1. **LR grid is small enough for exhaustive search** — 12 configs × 4 folds = 48 fits; LR trains in seconds per fit, total <1 minute
  2. **RF grid is moderate** — 36 configs × 4 folds = 144 fits; RF is embarrassingly parallel and each fit takes ~5–15 seconds on 43K records, total ~15–30 minutes
  3. **XGBoost full grid is impractical** — 1,024 × 4 = 4,096 fits at ~10–30 seconds each = 11–34 hours. Randomised search at 50 configs × 4 = 200 fits ≈ 30–60 minutes while still exploring the space adequately
  4. **50 random configs provides good coverage** — Bergstra & Bengio (2012) showed that random search with 60 trials finds configurations within the top 5% of the search space with >95% probability for typical hyperparameter response surfaces. 50 is conservative but sufficient for 5 parameters
  5. **Grid ranges are informed by defaults and literature** — XGBoost learning rates below 0.01 require impractically many estimators; depths above 9 overfit on 9-feature data; subsample below 0.7 discards too much data per tree
- **Selection criterion:** Mean AUC-ROC across 4 temporal CV folds. Best configuration is retrained on the full training partition before final evaluation.
- **Alternatives considered:**
  - Bayesian optimisation (Optuna/Hyperopt) — rejected: adds dependency complexity; marginal benefit over random search for ≤50 trials; harder to reproduce exactly across runs
  - Exhaustive grid for XGBoost — rejected: 4,096 fits is impractical for iterative development and the full sensitivity sweep
  - Smaller XGBoost sample (20 configs) — rejected: 20 samples over 5 dimensions gives sparse coverage; risk of missing good regions
  - Larger RF grid (include `max_features`) — deferred: with only 9 features, `max_features` variations (e.g., sqrt(9)=3 vs. all 9) have modest impact. Can be added if RF underperforms expectations
- **Implementation revision (2026-07-03):** The grids were narrowed during implementation for practical runtime reasons:
  - LR: changed from separate L1/L2 penalties with matched solvers to ElasticNet with `l1_ratio ∈ {0.0, 1.0}` using saga solver uniformly. Added `C=100.0` to the sweep → 10 configs (was 12).
  - RF: `n_estimators` narrowed to [50, 100, 200] (was [100, 200, 500]); `max_depth` narrowed to [3, 5, 10, None] (was [5, 10, 20, None]); `min_samples_leaf` changed to [1, 2, 5] (was [1, 5, 10]). Total remains 36 configs.
  - XGBoost: `colsample_bytree` dropped (minimal impact with 9 features); `learning_rate` changed to [0.01, 0.1, 0.3]; grid structure changed to 3×3×3×2 = 54, capped at 50.
  - Rationale for changes: smaller estimator counts reduce per-config runtime; shallower RF depths better suit 9-feature data; ElasticNet with l1_ratio endpoints is functionally equivalent to pure L1/L2 while simplifying solver selection.
- **Status:** Accepted (revised in implementation)

---

## DEC-021: Class imbalance handling — balanced class weights

- **Date:** 2026-07-03
- **Context:** The surge prediction task produces an imbalanced dataset — the majority of records are non-surge (negative class). Without mitigation, classifiers optimise for overall accuracy by predicting the majority class, producing high accuracy but near-zero recall on the minority (surge) class. A strategy for handling class imbalance during training is required.
- **Decision:** Use `class_weight="balanced"` for Logistic Regression and Random Forest. XGBoost uses its default internal handling (no explicit `scale_pos_weight` set, relying on the boosting mechanism's sensitivity to misclassification).
- **Rationale:**
  1. **Simplicity** — `class_weight="balanced"` automatically adjusts sample weights inversely proportional to class frequency (`n_samples / (n_classes × n_samples_per_class)`). No manual calculation or separate resampling step required
  2. **No synthetic data** — avoids introducing artificial samples that could distort feature distributions or create misleading patterns in temporal features (e.g., SMOTE interpolating between records from different time periods)
  3. **Preserves temporal integrity** — resampling techniques (SMOTE, random oversampling) create new records that violate the temporal ordering assumptions of the expanding-window CV. Weighting adjusts the loss function without altering the data
  4. **Consistent with evaluation metrics** — AUC-ROC (primary metric) is threshold-independent and rank-based, so it naturally accommodates the re-weighted decision boundary. The balanced weights ensure the model's internal threshold reflects the true task rather than the class ratio
  5. **XGBoost robustness** — gradient boosting inherently focuses on hard-to-classify examples (high-loss samples), which in an imbalanced setting are predominantly minority-class records. Adding explicit `scale_pos_weight` on top of this focus risks over-correcting. If XGBoost underperforms on recall, `scale_pos_weight` can be added as a tuned hyperparameter
- **Alternatives considered:**
  - SMOTE (Synthetic Minority Over-sampling) — rejected: creates synthetic temporal records that break temporal ordering; interpolated features (e.g., `time_since_previous`, `ticker_post_rate_24h`) have no valid temporal interpretation
  - Random oversampling — rejected: duplicates minority samples, inflating their influence without adding information; can cause overfitting in tree-based models (identical records appear in multiple leaves)
  - Random undersampling — rejected: discards majority-class data, reducing training set size from ~43K to potentially <5K. Unacceptable loss of information for a modestly-sized dataset
  - Threshold adjustment (post-hoc) — complementary, not alternative: the threshold sensitivity analysis (DEC-020) already sweeps classification thresholds. Balanced weights set the training objective; threshold sweep optimises the operating point
  - `scale_pos_weight` for XGBoost — deferred: can be added to the hyperparameter grid if XGBoost shows poor minority-class performance during initial evaluation
- **Status:** Accepted

---

## DEC-022: Feature scaling — uniform StandardScaler across all models

- **Date:** 2026-07-03
- **Context:** The 9 features have different scales: `sentiment_polarity` ∈ [-1, 1], `hour_of_day` ∈ [0, 23], `ticker_post_rate_24h` can be arbitrarily large. Logistic Regression requires feature scaling for proper regularisation behaviour (L1/L2 penalties are scale-sensitive). Tree-based models (RF, XGBoost) are scale-invariant — they split on rank order, not magnitude.
- **Decision:** Apply StandardScaler (zero-mean, unit-variance) uniformly to all features for all three model types. The scaler is fit on training data only and applied to validation/test data using training statistics.
- **Rationale:**
  1. **LR requirement** — without scaling, LR's regularisation penalty disproportionately penalises features with larger raw magnitudes. `ticker_post_rate_24h` (potentially 0–100+) would be penalised far more heavily than `sentiment_polarity` (−1 to 1), distorting feature selection
  2. **Uniform pipeline** — applying the same preprocessing to all models simplifies the training infrastructure. A single `StandardScaler` per fold/final-train means feature matrices are identical across models, ensuring fair comparison
  3. **No harm to tree models** — StandardScaler is a monotonic transformation that preserves rank order. RF and XGBoost produce identical split decisions before and after scaling. The only cost is a negligible O(n×d) transform — acceptable for pipeline simplicity
  4. **Consistent with temporal CV** — the scaler is fit within each CV fold's training portion and applied to the validation portion, then fit on the full training set for final model evaluation. This prevents leakage of test-set statistics into training
  5. **Reproducibility** — storing a single scaler per trained model (alongside the model weights) fully specifies the inference pipeline. No ambiguity about whether a given model expects raw or scaled inputs
- **Alternatives considered:**
  - Scale only for LR, raw features for RF/XGBoost — rejected: complicates the training loop (separate preprocessing paths), increases risk of applying wrong features to wrong model, and provides no performance benefit since scaling doesn't harm tree models
  - MinMaxScaler — rejected: sensitive to outliers in unbounded features (`ticker_post_rate_24h`, `ticker_post_acceleration`); would compress most values into a narrow range near 0
  - RobustScaler (median/IQR) — considered: more robust to outliers, but StandardScaler is the conventional choice and the feature distributions are not heavily skewed after the minimum-window-count exclusion (DEC-011)
  - No scaling (even for LR) — rejected: LR with regularisation produces unreliable coefficients when features differ by orders of magnitude
- **Status:** Accepted
