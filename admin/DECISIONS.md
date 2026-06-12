# Decision Log

This document records key design decisions made during the development of the Engagement and Sentiment Surge Prediction Pipeline. Each entry captures the context, decision, rationale, and alternatives considered for traceability and academic reporting.

## Table of Contents

| ID | Decision | Date | Status |
|----|----------|------|--------|
| DEC-001 | Composite surge threshold default value | 2026-06-08 | Accepted |
| DEC-002 | Unit of analysis — per-ticker scoping | 2026-06-08 | Accepted |
| DEC-003 | Why combine engagement and sentiment (composite metric) | 2026-06-08 | Accepted |
| DEC-004 | Phased experimental approach (baseline → composite) | 2026-06-08 | Accepted |
| DEC-005 | Dataset selection — Reddit Finance Data (Kaggle) | 2026-06-08 | Accepted |
| DEC-006 | Statistical robustness in evaluation | 2026-06-08 | Accepted |
| DEC-007 | Tiered success criteria | 2026-06-08 | Accepted |
| DEC-008 | Literature review requires critical evaluation | 2026-06-08 | Accepted |
| DEC-009 | Report structure — add surge definition and unit of analysis | 2026-06-08 | Accepted |
| DEC-010 | Composite metric scale mismatch — empirical investigation over a priori normalisation | 2026-06-08 | Accepted |
| DEC-011 | Per-ticker sparsity — minimum window record count | 2026-06-08 | Accepted |
| DEC-012 | Eliminate snapshot engagement values — use posting volume instead | 2026-06-12 | Accepted |

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
- **Decision:** Retain the raw additive formulation as the starting point. Do not apply a priori normalisation or weighting. Instead, analyse the relative influence of each component experimentally through the existing phased design and threshold sensitivity analysis.
- **Rationale:**
  1. Any normalisation scheme (z-scores, min-max, weighting) requires assumptions about relative importance that are not empirically grounded at this stage
  2. The Phase 1 vs Phase 2 comparison already quantifies sentiment's marginal contribution — if it adds nothing, the scale mismatch is the likely explanation
  3. The threshold sensitivity analysis will reveal how the effective contribution of each component varies across operating points
  4. Reporting the dominance pattern (if confirmed) is itself a valid finding rather than a flaw to be hidden
- **Alternatives considered:**
  - Z-score normalisation of both components — rejected: requires computing population statistics before labelling, introduces circular dependency with threshold
  - Weighted sum (e.g., `α * engagement_growth + β * |sentiment_change|`) — rejected: choice of weights would be arbitrary without prior evidence
  - Separate thresholds per component (engagement > X AND sentiment > Y) — deferred to future work if composite proves inadequate
- **If analysis confirms dominance:** Will report as finding and discuss alternative formulations (standardised z-scores, weighted sums, multiplicative combination) as future work directions.
- **Status:** Accepted

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
