# Decision Log

This document records key design decisions made during the development of the Engagement and Sentiment Surge Prediction Pipeline. Each entry captures the context, decision, rationale, and alternatives considered for traceability and academic reporting.

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
