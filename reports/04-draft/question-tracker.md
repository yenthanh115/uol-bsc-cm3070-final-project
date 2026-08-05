# Report Question Tracker

Tracks which questions a reader might ask, where they are (or should be) addressed, and current status.

**Status key:** DONE = fully addressed | PARTIAL = touched but incomplete | MISSING = not addressed | DEFERRED = belongs in another section

---

## Section 1: Introduction

| # | Question | Status | Where Addressed | Priority | Action |
|---|----------|--------|-----------------|----------|--------|
| 1.1 | What is the problem? | DONE | §1.2 | MUST-HAVE | No action needed |
| 1.2 | Why does it matter? Who benefits? | PARTIAL | §1.2 | MUST-HAVE | Add 1–2 sentences clarifying use case (narrative monitoring, manipulation surveillance, or attention research — pick one concrete scenario) |
| 1.3 | What exactly are you predicting? | DONE | §1.3 | MUST-HAVE | No action needed |
| 1.4 | Why binary classification (not regression, multi-class, anomaly detection)? | DEFERRED | §3.5 | — | One clause in §1.3 is enough; full justification moved to §3.5 |
| 1.5 | Why ML over simpler alternatives (rule-based threshold, z-score alert)? | MISSING | — | MUST-HAVE | Add one sentence: a fixed threshold cannot combine heterogeneous signals or adapt to non-linear interactions |
| 1.6 | Why Reddit specifically? | DEFERRED | §3.2 | — | Already sufficient with forward reference; full justification moved to §3.2 |
| 1.7 | Why these two subreddits? | DEFERRED | §3.2 | — | Already present at intro level; density explanation moved to §3.2 |
| 1.8 | What is the hypothesis / expected outcome? | MISSING | — | SHOULD-HAVE | Add one sentence in §1.1: "The hypothesis is that backward-looking temporal and textual features carry sufficient signal to discriminate surges, and that performance scales with data density." |
| 1.9 | What are the objectives? | DONE | §1.1 | MUST-HAVE | No action needed |
| 1.10 | What's in/out of scope? | DONE | §1.4 | MUST-HAVE | No action needed |
| 1.11 | How does this relate to the CM3005 template? | PARTIAL | §1.1 | NICE-TO-HAVE | One clarifying sentence for the marker; low priority for external readers |
| 1.12 | What's the structure of the rest of the report? | MISSING | — | SHOULD-HAVE | Add 3–4 line roadmap paragraph at end of Section 1 |
| 1.13 | Why exclude engagement scores (upvotes)? | DEFERRED | §3.4 | — | One sentence stating the principle in intro; detailed reasoning moved to §3.4 |
| 1.14 | Why use a combined metric (volume + sentiment) not just volume? | MISSING | — | MUST-HAVE (brief) | Add one sentence: "Volume alone misses cases where sentiment intensifies without a proportional posting increase; combining both captures a richer phenomenon." Full justification in §3.3 and §5.2.5 |

---

## Section 2: Literature Review

| # | Question | Status | Where Addressed | Notes |
|---|----------|--------|-----------------|-------|
| 2.1 | Can online attention be predicted at all? | DONE | §2.1 | Szabo & Huberman, Lerman & Hogg |
| 2.2 | Can prediction happen before engagement accumulates? | DONE | §2.2 | Bandari et al., Cheng et al. |
| 2.3 | Does sentiment carry predictive value in finance? | DONE | §2.3 | Bollen et al., VADER, FinBERT |
| 2.4 | Do Reddit financial communities generate useful signal? | DONE | §2.4 | Long et al., Costola et al., Mancini et al. |
| 2.5 | What methodological weaknesses exist in prior work? | DONE | §2.5 | Temporal leakage across reviewed studies |
| 2.6 | What specific gap does this project fill? | DONE | §2.6 | Four gaps identified and mapped to design |
| 2.7 | How do the reviewed studies relate to each other (synthesis)? | PARTIAL | §2.1–2.4 | Mostly sequential; cross-study synthesis could be stronger |
| 2.8 | Are there conflicting findings across studies? | MISSING | — | No contradictions discussed |

---

## Section 3: Design

| # | Question | Status | Where Addressed | Notes |
|---|----------|--------|-----------------|-------|
| 3.1 | What is the overall pipeline architecture? | DONE | §3.1 | Six-stage linear pipeline with diagram |
| 3.2 | Why Reddit (platform justification)? | DONE | §3.2 | Subreddit structure, public archives, text suitability |
| 3.2a | Why Reddit specifically? (delegated from §1, Q1.6) | DONE | §3.2 | Full justification lives here; intro has brief forward reference |
| 3.2b | Why these two subreddits? (delegated from §1, Q1.7) | DONE | §3.2 | Density spectrum rationale explained here |
| 3.3 | Why these two subreddits? | DONE | §3.2 | Density spectrum + cross-dataset transfer motivation |
| 3.4 | How is "surge" defined formally? | DONE | §3.3 | Composite metric with z-scores, threshold, weights |
| 3.5 | Why use a composite metric? | PARTIAL | §3.3 | Stated that sentiment captures agitation without volume increase, but could be stronger |
| 3.6 | How are features chosen and justified? | DONE | §3.4 | 11 features with backward-looking constraint |
| 3.7 | Why exclude engagement scores? | DONE | §3.4 | Accumulate after posting, would introduce leakage |
| 3.7a | Why exclude engagement scores? (delegated from §1, Q1.13) | DONE | §3.4 | Full reasoning here; intro states principle only |
| 3.8 | Why these three models? | DONE | §3.5 | Complexity spectrum: linear → bagged → boosted |
| 3.8a | Why binary classification not regression/multi-class/anomaly detection? (delegated from §1, Q1.4) | PARTIAL | §3.5 | Currently not explicitly justified in §3.5 either — needs adding |
| 3.9 | Why AUC-ROC as primary metric? | DONE | §3.5 | Accuracy uninformative at 1–5% surge rate |
| 3.10 | How is class imbalance handled? | DONE | §3.5 | Cost-sensitive learning (no SMOTE for temporal data) |
| 3.11 | How does temporal validation work? | DONE | §3.6 | Expanding-window CV, 80/20 temporal split |
| 3.12 | Why k=4 (not 5 or 10)? | PARTIAL | §3.6 | Implied (2.5-month windows for enough surges) but not explicitly justified |
| 3.13 | What are the success criteria? | DONE | §3.7 | Three tiers: minimum/target/stretch |
| 3.14 | How are models compared statistically? | DONE | §3.7 | Bootstrap CI + McNemar's with Bonferroni |
| 3.15 | Is the study reproducible? | DONE | §3.1, §4.1 | Fixed seed, config dataclass, deterministic pipeline |
| 3.16 | What about data drift / temporal non-stationarity? | MISSING | — | Not discussed in design (appears only in §5.3.4 as a finding) |
| 3.17 | Why not use an API for live data? | MISSING | — | Implied by "static CSV" but not justified |
| 3.18 | What are the ethical considerations? | DONE | §3.2 | Public data, aggregated analysis, no user identification |

---

## Section 4: Implementation

| # | Question | Status | Where Addressed | Notes |
|---|----------|--------|-----------------|-------|
| 4.1 | How is the code organised? | DONE | §4.1 | Module-by-module breakdown |
| 4.2 | How does data loading work? | DONE | §4.2 | Text cleaning, ticker extraction, explosion |
| 4.3 | How are features computed? | DONE | §4.3 | Searchsorted approach, interaction terms |
| 4.4 | How does labelling prevent leakage? | DONE | §4.4 | Frozen z-score params from training only |
| 4.5 | How are models trained and evaluated? | DONE | §4.5 | Expanding-window CV, grid search, threshold tuning |
| 4.6 | What problems were encountered and solved? | DONE | §4.6 | Timestamp bug, sparsity, threshold collapse |
| 4.7 | Is the implementation complete? | DONE | §4.7 | All six stages functional, tests pass |

---

## Section 5: Evaluation

| # | Question | Status | Where Addressed | Notes |
|---|----------|--------|-----------------|-------|
| 5.1 | Were objectives met? | DONE | §5.1 | Each objective revisited with evidence |
| 5.2 | What are the raw performance numbers? | DONE | §5.2.1 | Tables 11–13 |
| 5.3 | Are model differences statistically significant? | DONE | §5.2.2 | McNemar's test, all significant |
| 5.4 | Do models transfer across communities? | DONE | §5.2.3 | Cross-dataset AUC table |
| 5.5 | Which features matter most? | DONE | §5.2.4 | Permutation importance |
| 5.6 | Does sentiment actually help? | DONE | §5.2.5 | Phase 1 vs Phase 2 comparison |
| 5.7 | Why does XGBoost lose on pennystocks? | DONE | §5.3.1 | Insufficient positives for boosting |
| 5.8 | Why is sentiment important if it's weak alone? | DONE | §5.3.2 | Interactive signal with activity features |
| 5.9 | Why does transfer work asymmetrically? | DONE | §5.3.3 | Feature over-reliance on word_count |
| 5.10 | What are the limitations? | DONE | §5.4.1 | Sample size, single year, VADER, calibration |
| 5.11 | What would improve results? | DONE | §5.4.2 | Prioritised improvement table |
| 5.12 | What's original about this work? | DONE | §5.5 | Three contributions stated |

---

## Section 6: Conclusion

| # | Question | Status | Where Addressed | Notes |
|---|----------|--------|-----------------|-------|
| 6.1 | What was achieved? | DONE | §6.1 | Pipeline works end-to-end, question answered |
| 6.2 | What are the key findings? | DONE | §6.2 | Data density > model complexity; sentiment interactive |
| 6.3 | What are the limitations? | DONE | §6.3 | Four limitations bounded |
| 6.4 | What's next? | DONE | §6.3 | Multi-scale windows, live testing |

---

## Cross-Cutting Questions (should be addressed somewhere)

| # | Question | Status | Best Section | Notes |
|---|----------|--------|--------------|-------|
| C.1 | Why not anomaly detection instead of classification? | MISSING | §1 or §3.5 | Never justified |
| C.2 | Could a simple threshold rule work just as well? | MISSING | §1 or §5 | The baseline comparison (Table 15) partially addresses this but doesn't frame it as "would a rule suffice?" |
| C.3 | What about data drift over time? | PARTIAL | §3 or §5 | Found as a result (§5.3.4) but not anticipated in design |
| C.4 | Why 24-hour windows specifically? | PARTIAL | §3.3 | Stated as scope, acknowledged as limitation in §5.4.2, but not justified |
| C.5 | How would this work in practice (deployment scenario)? | MISSING | §1.2 or §6 | Out of scope declared but no sketch of how it *could* work |
