# Report Question Tracker

Tracks which questions a reader might ask, where they are (or should be) addressed, and current status.

**Status key:** DONE = fully addressed | PARTIAL = touched but incomplete | MISSING = not addressed | DEFERRED = belongs in another section

---

## Section 1: Introduction

| # | Question | Status | Where Addressed | Priority | Action |
|---|----------|--------|-----------------|----------|--------|
| 1.1 | What is the problem? | DONE | §1.2 | MUST-HAVE | No action needed |
| 1.2 | Why does it matter? Who benefits? | DONE | §1.2 | MUST-HAVE | Added concrete scenario: a compliance team monitoring covered stocks uses automated surge flags to focus on the few tickers most likely to dominate tomorrow's discussion, rather than scanning thousands of threads manually. |
| 1.3 | What exactly are you predicting? | DONE | §1.3 | MUST-HAVE | No action needed |
| 1.4 | Why binary classification (not regression, multi-class, anomaly detection)? | DEFERRED | §3.5 | — | One clause in §1.3 is enough; full justification moved to §3.5 |
| 1.5 | Why ML over simpler alternatives (rule-based threshold, z-score alert)? | DONE | §1.2 | MUST-HAVE | Added sentence at end of §1.2: a simple threshold cannot combine heterogeneous signals or adapt to non-linear interactions; a learning-based approach is needed. |
| 1.6 | Why Reddit specifically? | DEFERRED | §3.2 | — | Already sufficient with forward reference; full justification moved to §3.2 |
| 1.7 | Why these two subreddits? | DEFERRED | §3.2 | — | Already present at intro level; density explanation moved to §3.2 |
| 1.8 | What is the hypothesis / expected outcome? | DONE | §1.1 | SHOULD-HAVE | Added after objectives: "The underlying hypothesis is that backward-looking temporal and textual features carry sufficient signal to discriminate surges from baseline activity, and that predictive performance scales with data density rather than model complexity." |
| 1.9 | What are the objectives? | DONE | §1.1 | MUST-HAVE | No action needed |
| 1.10 | What's in/out of scope? | DONE | §1.4 | MUST-HAVE | No action needed |
| 1.11 | How does this relate to the CM3005 template? | DONE | §1.1 | NICE-TO-HAVE | Added clarifying sentence: "The template calls for predicting when online content will gain traction; this project instantiates that brief by targeting posting-volume surges on Reddit financial communities." |
| 1.12 | What's the structure of the rest of the report? | DONE | §1.6 | SHOULD-HAVE | Added §1.6 "Report Structure" with 4-line roadmap paragraph after the Gantt chart. |
| 1.13 | Why exclude engagement scores (upvotes)? | DEFERRED | §3.4 | — | One sentence stating the principle in intro; detailed reasoning moved to §3.4 |
| 1.14 | Why use a combined metric (volume + sentiment) not just volume? | DONE | §1.3 | MUST-HAVE (brief) | Added sentence in §1.3: "Volume alone misses cases where a community becomes markedly more agitated without a proportional posting increase; combining both signals captures a richer, more structured phenomenon (confirmed empirically in Section 5.2.5)." |

---

## Section 2: Literature Review

| # | Question | Status | Where Addressed | Priority | Action |
|---|----------|--------|-----------------|----------|--------|
| 2.1 | Can online attention be predicted at all? | DONE | §2.1 | MUST-HAVE | No action needed |
| 2.2 | Can prediction happen before engagement accumulates? | DONE | §2.2 | MUST-HAVE | No action needed |
| 2.3 | Does sentiment carry predictive value in finance? | DONE | §2.3 | MUST-HAVE | No action needed |
| 2.4 | Do Reddit financial communities generate useful signal? | DONE | §2.4 | MUST-HAVE | No action needed |
| 2.5 | What methodological weaknesses exist in prior work? | DONE | §2.5 | MUST-HAVE | No action needed |
| 2.6 | What specific gap does this project fill? | DONE | §2.6 | MUST-HAVE | No action needed |
| 2.7 | How do the reviewed studies relate to each other (synthesis)? | DONE | §2.4 | SHOULD-HAVE | Added cross-reference sentences linking Costola's consensus formation to Lerman & Hogg's network dynamics, and Long et al.'s volume-precedes-trading observation to Cheng et al.'s early-propagation-predicts-growth finding. |
| 2.8 | Are there conflicting findings across studies? | DONE | §2.1 | NICE-TO-HAVE | Added one sentence noting the Szabo/Cheng tension: early popularity strongly predicts final outcome [1] vs. cascade prediction accuracy plateaus after initial phase [5]. |
| 2.9 | What is the current state of surge/burst detection specifically? | DONE | §2.1 | SHOULD-HAVE | Added sentence after Kong et al. [7] distinguishing their burst detection (hashtag-level, contemporaneous, real-time) from this project's surge prediction (ticker-level, pre-onset). |
| 2.10 | Why are these 17 sources sufficient? (scope of review) | DONE | §2.1 | NICE-TO-HAVE | Added scope statement at start of §2.1: "This review covers three intersecting research strands... The seventeen sources cited here represent the seminal and most-cited works within each strand rather than an exhaustive survey." |
| 2.11 | Are there studies that tried surge prediction and failed? | DONE | §2.6 | NICE-TO-HAVE | Changed gap 1 wording to "No study was found that defines or predicts..." — making the absence of prior attempts explicit rather than implied. |

---

## Section 3: Design

| # | Question | Status | Where Addressed | Priority | Action |
|---|----------|--------|-----------------|----------|--------|
| 3.1 | What is the overall pipeline architecture? | DONE | §3.1 | MUST-HAVE | No action needed |
| 3.2 | Why Reddit (platform justification)? | DONE | §3.2 | MUST-HAVE | No action needed |
| 3.2a | Why Reddit specifically? (delegated from §1, Q1.6) | DONE | §3.2 | MUST-HAVE | No action needed; full justification lives here |
| 3.2b | Why these two subreddits? (delegated from §1, Q1.7) | DONE | §3.2 | MUST-HAVE | No action needed; density spectrum rationale explained here |
| 3.3 | Why these two subreddits? | DONE | §3.2 | MUST-HAVE | No action needed |
| 3.4 | How is "surge" defined formally? | DONE | §3.3 | MUST-HAVE | No action needed |
| 3.5 | Why use a composite metric? | DONE | §3.3 | MUST-HAVE | Added sentence stating volume-only surges are noisier and harder to predict, with Phase 1 empirical confirmation (AUC 0.710 vs 0.892). |
| 3.6 | How are features chosen and justified? | DONE | §3.4 | MUST-HAVE | No action needed |
| 3.7 | Why exclude engagement scores? | DONE | §3.4 | MUST-HAVE | No action needed |
| 3.7a | Why exclude engagement scores? (delegated from §1, Q1.13) | DONE | §3.4 | MUST-HAVE | No action needed; full reasoning here |
| 3.8 | Why these three models? | DONE | §3.5 | MUST-HAVE | No action needed |
| 3.8a | Why binary classification not regression/multi-class/anomaly detection? (delegated from §1, Q1.4) | DONE | §3.5 | MUST-HAVE | Added "Why binary classification?" paragraph at top of §3.5 covering: surges are present-or-absent (binary is natural), regression conflates occurrence with magnitude, multi-class introduces arbitrary magnitude boundaries, anomaly detection can't leverage labelled history. |
| 3.9 | Why AUC-ROC as primary metric? | DONE | §3.5 | MUST-HAVE | No action needed |
| 3.10 | How is class imbalance handled? | DONE | §3.5 | MUST-HAVE | No action needed |
| 3.11 | How does temporal validation work? | DONE | §3.6 | MUST-HAVE | No action needed |
| 3.12 | Why k=4 (not 5 or 10)? | DONE | §3.6 | SHOULD-HAVE | Added sentences: four blocks produce ~2.5-month validation windows with enough surge events for stable AUC; higher k would thin folds below reliable evaluation. |
| 3.13 | What are the success criteria? | DONE | §3.7 | MUST-HAVE | No action needed |
| 3.14 | How are models compared statistically? | DONE | §3.7 | MUST-HAVE | No action needed |
| 3.15 | Is the study reproducible? | DONE | §3.1, §4.1 | MUST-HAVE | No action needed |
| 3.16 | What about data drift / temporal non-stationarity? | DONE | §3.6 | SHOULD-HAVE | Added sentence in §3.6 acknowledging non-stationarity as a known risk, with forward reference to §5.3.4 for empirical evidence. |
| 3.17 | Why not use an API for live data? | DONE | §3.2 | NICE-TO-HAVE | Added sentence in §3.2 explaining that static CSVs ensure reproducibility while live API scraping would introduce temporal variability. |
| 3.18 | What are the ethical considerations? | DONE | §3.2 | MUST-HAVE | No action needed |

---

## Section 4: Implementation

| # | Question | Status | Where Addressed | Priority | Action |
|---|----------|--------|-----------------|----------|--------|
| 4.1 | How is the code organised? | DONE | §4.1 | MUST-HAVE | No action needed |
| 4.2 | How does data loading work? | DONE | §4.2 | MUST-HAVE | No action needed |
| 4.3 | How are features computed? | DONE | §4.3 | MUST-HAVE | No action needed |
| 4.4 | How does labelling prevent leakage? | DONE | §4.4 | MUST-HAVE | No action needed |
| 4.5 | How are models trained and evaluated? | DONE | §4.5 | MUST-HAVE | No action needed |
| 4.6 | What problems were encountered and solved? | DONE | §4.6 | MUST-HAVE | No action needed |
| 4.7 | Is the implementation complete? | DONE | §4.7 | MUST-HAVE | No action needed |
| 4.8 | What are the key dependencies and their versions? | DONE | §4.1 | SHOULD-HAVE | Added version constraints (pandas ≥2.0, scikit-learn ≥1.3, XGBoost ≥2.0, vaderSentiment ≥3.3.2, NumPy ≥1.24) and noted exact pins are in requirements.txt and experiment logs. |
| 4.9 | How long does the pipeline take to run? | DONE | §4.7 | NICE-TO-HAVE | Added sentence in §4.7: approximately 8 minutes for pennystocks and 19 minutes for WSB on a standard laptop CPU, with sentiment computation as the dominant cost. |
| 4.10 | How is the ticker stopword list maintained/validated? | DONE | §4.2 | NICE-TO-HAVE | Added clause: "built through iterative false-positive analysis on early pipeline runs and manually reviewed for completeness." |
| 4.11 | What happens when a ticker has too few posts for windowing? | DONE | §4.4 | SHOULD-HAVE | Added: records with fewer than two same-ticker posts in the forward window are excluded as unlabellable. Noted this accounts for the 577K to 457K attrition on WSB (tickers near data boundary). |
| 4.12 | How are the hyperparameter grids justified? | DONE | §4.5 | SHOULD-HAVE | Added sentence: "Ranges were chosen from common defaults in the scikit-learn and XGBoost documentation, then narrowed by preliminary runs on the first validation fold to exclude values that consistently underperformed." |

---

## Section 5: Evaluation

| # | Question | Status | Where Addressed | Priority | Action |
|---|----------|--------|-----------------|----------|--------|
| 5.1 | Were objectives met? | DONE | §5.1 | MUST-HAVE | No action needed |
| 5.2 | What are the raw performance numbers? | DONE | §5.2.1 | MUST-HAVE | No action needed |
| 5.3 | Are model differences statistically significant? | DONE | §5.2.2 | MUST-HAVE | No action needed |
| 5.4 | Do models transfer across communities? | DONE | §5.2.3 | MUST-HAVE | No action needed |
| 5.5 | Which features matter most? | DONE | §5.2.4 | MUST-HAVE | No action needed |
| 5.6 | Does sentiment actually help? | DONE | §5.2.5 | MUST-HAVE | No action needed |
| 5.7 | Why does XGBoost lose on pennystocks? | DONE | §5.3.1 | MUST-HAVE | No action needed |
| 5.8 | Why is sentiment important if it's weak alone? | DONE | §5.3.2 | MUST-HAVE | No action needed |
| 5.9 | Why does transfer work asymmetrically? | DONE | §5.3.3 | MUST-HAVE | No action needed |
| 5.10 | What are the limitations? | DONE | §5.4.1 | MUST-HAVE | No action needed |
| 5.11 | What would improve results? | DONE | §5.4.2 | MUST-HAVE | No action needed |
| 5.12 | What's original about this work? | DONE | §5.5 | MUST-HAVE | No action needed |
| 5.13 | How do results compare to published baselines? | DONE | §5.5 | SHOULD-HAVE | Added sentence contextualising AUC range (0.753–0.892) alongside Cheng's 0.877 and Bandari's ~84%, while explicitly noting protocol differences make direct ranking invalid. |
| 5.14 | Is the precision (0.217) actually useful in practice? | DONE | §5.3.4 | SHOULD-HAVE | Added sentence framing 0.217 precision as a screening trade-off: monitoring 500 tickers yields ~20 flags with ~4 real surges — manageable for human review but unsuitable for automated action. |
| 5.15 | Could the validation-test gap (F1 0.911 → 0.145) indicate overfitting rather than non-stationarity? | DONE | §5.3.4 | SHOULD-HAVE | Added sentences distinguishing: AUC remains high (0.880) on test, so ranking transfers; F1 collapse reflects threshold miscalibration under distribution shift, not wholesale model failure. Overfitting would degrade both metrics. |
| 5.16 | Are the bootstrap CIs on pennystocks wide enough to overlap between models? | DONE | §5.2.1 | NICE-TO-HAVE | Added sentence after Table 11 noting RF [0.673, 0.824] and XGB [0.641, 0.821] overlap substantially, so model rankings are not distinguishable by CI alone, though McNemar's confirms they differ in prediction pattern. |
| 5.17 | Is the structural correlation between sentiment feature and target a problem? | DONE | §5.4.1 | SHOULD-HAVE | Expanded to explicitly draw the Phase 1 connection: with sentiment removed from target (w₂=0), XGBoost still achieves 0.710 (Table 18), proving the pipeline works without sentiment. Also states what a conclusive test would require (dropping sentiment from features while keeping the composite target). |
| 5.18 | How sensitive are results to the specific threshold τ=1.0? | DONE | §5.2.5 | MUST-HAVE | No action needed; Table 19 weight sweep covers this. |
| 5.19 | Would the models work on a different time period? | DONE | §5.4.1 | SHOULD-HAVE | Added sentence estimating likely degradation: 2021 includes unprecedented retail speculation; models may underperform on calmer periods where surges are rarer, less structured, and not reinforced by coordinated retail enthusiasm. |

---

## Section 6: Conclusion

| # | Question | Status | Where Addressed | Priority | Action |
|---|----------|--------|-----------------|----------|--------|
| 6.1 | What was achieved? | DONE | §6.1 | MUST-HAVE | No action needed |
| 6.2 | What are the key findings? | DONE | §6.2 | MUST-HAVE | No action needed |
| 6.3 | What are the limitations? | DONE | §6.3 | MUST-HAVE | No action needed |
| 6.4 | What's next? | DONE | §6.3 | MUST-HAVE | No action needed |
| 6.5 | Does the conclusion introduce new information? | DONE | — | MUST-HAVE (negative) | No action needed. The conclusion correctly synthesises without introducing new data or arguments. This is the right approach. |
| 6.6 | Does it answer the research question explicitly? | DONE | §6.2 | MUST-HAVE | No action needed. "The short answer to the central research question is yes" — clear and direct. |
| 6.7 | Are the three contributions restated concisely? | DONE | §6.1 | SHOULD-HAVE | Added sentence at end of §6.1: "The project contributes a leakage-free methodology, a composite surge metric, and empirical evidence that data density is the binding constraint on prediction quality." |
| 6.8 | Is there a clear "so what" for the reader? | DONE | §6.2 | SHOULD-HAVE | Added sentence at end of §6.2: "For teams monitoring financial communities, the practical takeaway is to invest in data coverage before model sophistication. A sparse community needs more history, not a better algorithm." |
| 6.9 | Does the future work connect back to limitations? | DONE | §6.3 | MUST-HAVE | No action needed. Each future direction maps to a stated limitation (VADER → FinBERT, fixed window → multi-scale, retrospective → live). |
| 6.10 | Is there a final closing statement? | DONE | §6.3 | MUST-HAVE | No action needed. "The question this project set out to answer... has been answered. What remains is finding out how far that answer extends." — effective closing. |
| 6.11 | Does the conclusion acknowledge what was NOT achieved? | DONE | §6.3 | NICE-TO-HAVE | Added opening sentences to §6.3: all objectives met, but pennystocks results are tentative (31 surges) and the methodology is validated for retrospective prediction only, not live deployment. |

---

## Cross-Cutting Questions (should be addressed somewhere)

| # | Question | Status | Best Section | Priority | Action |
|---|----------|--------|--------------|----------|--------|
| C.1 | Why not anomaly detection instead of classification? | DONE | §3.5 | MUST-HAVE | Resolved by Q3.8a fix. "Why binary classification?" paragraph in §3.5 explicitly addresses anomaly detection: it is unsupervised, cannot leverage labelled surge history, and would treat every rare surge as an anomaly regardless of whether it is genuinely distinct from the baseline distribution. |
| C.2 | Could a simple threshold rule work just as well? | DONE | §5.2.2 | SHOULD-HAVE | Added two sentences after Table 15 explicitly framing single-feature predictors as threshold-rule equivalents: WSB best single feature (0.805) is strong but multi-feature adds 8.7 AUC points; pennystocks gap is even wider (+0.162), where the best single feature (0.591) barely beats chance and multi-feature combination is what makes the task solvable. |
| C.3 | What about data drift over time? | DONE | §3.6 | SHOULD-HAVE | Covered by Q3.16. When that fix is applied (one sentence acknowledging drift as a design-stage risk), the design-to-evaluation arc is complete. No separate action beyond completing 3.16. |
| C.4 | Why 24-hour windows specifically? | DONE | §3.3 | SHOULD-HAVE | Add one sentence in §3.3: "A 24-hour window aligns with the daily trading cycle and captures overnight-to-open discussion patterns; shorter windows (6h) risk insufficient post counts for stable statistics, while longer windows (72h) blur the distinction between surge onset and sustained activity." Acknowledged as a limitation in §5.4.2 — this provides the forward justification. |
| C.5 | How would this work in practice (deployment scenario)? | DONE | §6.3 | NICE-TO-HAVE | Added two sentences at end of §6.3 sketching deployment as a rolling hourly screening layer, while noting it is beyond scope. Bridges the gap between retrospective research and practical use without overstepping. |
