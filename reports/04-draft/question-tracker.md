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

| # | Question | Status | Where Addressed | Priority | Action |
|---|----------|--------|-----------------|----------|--------|
| 2.1 | Can online attention be predicted at all? | DONE | §2.1 | MUST-HAVE | No action needed |
| 2.2 | Can prediction happen before engagement accumulates? | DONE | §2.2 | MUST-HAVE | No action needed |
| 2.3 | Does sentiment carry predictive value in finance? | DONE | §2.3 | MUST-HAVE | No action needed |
| 2.4 | Do Reddit financial communities generate useful signal? | DONE | §2.4 | MUST-HAVE | No action needed |
| 2.5 | What methodological weaknesses exist in prior work? | DONE | §2.5 | MUST-HAVE | No action needed |
| 2.6 | What specific gap does this project fill? | DONE | §2.6 | MUST-HAVE | No action needed |
| 2.7 | How do the reviewed studies relate to each other (synthesis)? | PARTIAL | §2.1–2.4 | SHOULD-HAVE | Add 2–3 cross-reference sentences linking studies across subsections (e.g., Cheng's temporal propagation speed relates to Bollen's mood timing; Costola's consensus formation connects to Lerman & Hogg's network dynamics) |
| 2.8 | Are there conflicting findings across studies? | MISSING | — | NICE-TO-HAVE | One sentence noting the Szabo/Cheng tension: early popularity strongly predicts final outcome [1] vs. cascade prediction accuracy plateaus after initial phase [5]. Optional: Bollen's 87.6% claim vs. Fernández-Delgado's caution about evaluation methodology |
| 2.9 | What is the current state of surge/burst detection specifically? | MISSING | — | SHOULD-HAVE | Kong et al. [7] does hashtag bursts but the review doesn't distinguish their burst detection (hashtag-level, real-time) from this project's surge prediction (ticker-level, pre-onset). One sentence would sharpen the gap claim. |
| 2.10 | Why are these 17 sources sufficient? (scope of review) | MISSING | — | NICE-TO-HAVE | A brief scope statement at the start of §2 ("This review covers three intersecting strands...") would signal deliberate selectivity rather than accidental sparseness |
| 2.11 | Are there studies that tried surge prediction and failed? | MISSING | — | NICE-TO-HAVE | Negative results strengthen a gap argument. If none exist, stating "no study was found that attempts..." explicitly in §2.6 is useful. Currently implied but not stated. |

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
| 3.5 | Why use a composite metric? | PARTIAL | §3.3 | MUST-HAVE | Strengthen justification: currently says sentiment captures agitation without volume increase, but should also state why a single-signal definition is insufficient (volume-only surges are noisier and harder to predict — Phase 1 results confirm this empirically). One sentence. |
| 3.6 | How are features chosen and justified? | DONE | §3.4 | MUST-HAVE | No action needed |
| 3.7 | Why exclude engagement scores? | DONE | §3.4 | MUST-HAVE | No action needed |
| 3.7a | Why exclude engagement scores? (delegated from §1, Q1.13) | DONE | §3.4 | MUST-HAVE | No action needed; full reasoning here |
| 3.8 | Why these three models? | DONE | §3.5 | MUST-HAVE | No action needed |
| 3.8a | Why binary classification not regression/multi-class/anomaly detection? (delegated from §1, Q1.4) | DONE | §3.5 | MUST-HAVE | Added "Why binary classification?" paragraph at top of §3.5 covering: surges are present-or-absent (binary is natural), regression conflates occurrence with magnitude, multi-class introduces arbitrary magnitude boundaries, anomaly detection can't leverage labelled history. |
| 3.9 | Why AUC-ROC as primary metric? | DONE | §3.5 | MUST-HAVE | No action needed |
| 3.10 | How is class imbalance handled? | DONE | §3.5 | MUST-HAVE | No action needed |
| 3.11 | How does temporal validation work? | DONE | §3.6 | MUST-HAVE | No action needed |
| 3.12 | Why k=4 (not 5 or 10)? | PARTIAL | §3.6 | SHOULD-HAVE | Add one sentence: "Four blocks produce ~2.5-month validation windows, each containing enough surge events for stable AUC estimates while keeping the minimum training set large enough for meaningful model fitting. Higher k would thin the validation folds below reliable evaluation." |
| 3.13 | What are the success criteria? | DONE | §3.7 | MUST-HAVE | No action needed |
| 3.14 | How are models compared statistically? | DONE | §3.7 | MUST-HAVE | No action needed |
| 3.15 | Is the study reproducible? | DONE | §3.1, §4.1 | MUST-HAVE | No action needed |
| 3.16 | What about data drift / temporal non-stationarity? | MISSING | — | SHOULD-HAVE | Add one sentence in §3.6 acknowledging it as a known risk: "Temporal non-stationarity (shifting surge dynamics across the year) is a known risk; the expanding-window design partially mitigates it by always training on the longest available history, though it cannot adapt to regime changes within the test period." Findings then appear naturally in §5.3.4. |
| 3.17 | Why not use an API for live data? | MISSING | — | NICE-TO-HAVE | One sentence in §3.2: "Static archival CSVs ensure exact reproducibility; live API scraping would introduce temporal variability between runs and complicate replication." Low priority — most readers won't ask this. |
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
| 4.8 | What are the key dependencies and their versions? | PARTIAL | §4.1 | SHOULD-HAVE | §4.1 lists libraries (pandas, scikit-learn, XGBoost, vaderSentiment, NumPy) but no version pins. One sentence or a small table of pinned versions would aid reproducibility claims. |
| 4.9 | How long does the pipeline take to run? | MISSING | — | NICE-TO-HAVE | Runtime context helps readers gauge feasibility. One sentence: "The full pipeline completes in ~X minutes on [hardware spec] for the WSB dataset." Not critical but strengthens the practical contribution. |
| 4.10 | How is the ticker stopword list maintained/validated? | PARTIAL | §4.2 | NICE-TO-HAVE | §4.2 mentions 297 terms across 8 categories but doesn't explain how the list was built or validated. One sentence on methodology (manual curation from false-positive analysis) would suffice. |
| 4.11 | What happens when a ticker has too few posts for windowing? | PARTIAL | §4.4 | SHOULD-HAVE | §4.4 mentions "records with too few posts in their forward window are excluded as unlabellable" but doesn't state the minimum threshold or how many records are lost. Table 9 shows attrition at loading stage but not at labelling exclusion. |
| 4.12 | How are the hyperparameter grids justified? | PARTIAL | §4.5 | SHOULD-HAVE | Table 10 lists the grids but doesn't explain why those specific ranges (e.g., why max_depth goes to 10 but not 15, why learning_rate stops at 0.3). One sentence: "Ranges were chosen from common defaults in the scikit-learn/XGBoost documentation, narrowed by preliminary runs on the first validation fold." |

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
| 5.13 | How do results compare to published baselines? | PARTIAL | §5.5 | SHOULD-HAVE | §5.5 states direct comparison is not possible (different datasets/protocols). This is honest but could be strengthened: one sentence comparing the AUC range (0.75–0.89) to the broader literature's reported figures (Cheng 0.877, Bandari ~84%) while noting protocol differences make direct comparison invalid. Contextualises rather than claims superiority. |
| 5.14 | Is the precision (0.217) actually useful in practice? | PARTIAL | §5.3.4 | SHOULD-HAVE | §5.3.4 raises the concern ("four of five flags are false alarms") but doesn't frame it as a practical trade-off: at what alert volume does this become useful? If a system monitors 500 tickers, 0.217 precision means ~4 real surges per 20 flags — is that acceptable for a screening tool? One sentence contextualising the operating point would help. |
| 5.15 | Could the validation-test gap (F1 0.911 → 0.145) indicate overfitting rather than non-stationarity? | PARTIAL | §5.3.4 | SHOULD-HAVE | §5.3.4 attributes the gap to temporal non-stationarity but doesn't rule out overfitting to validation-fold patterns. One sentence distinguishing: "AUC remains high (0.880) on test, indicating ranking ability transfers; the F1 collapse reflects threshold miscalibration under distribution shift rather than wholesale model failure." |
| 5.16 | Are the bootstrap CIs on pennystocks wide enough to overlap between models? | PARTIAL | §5.2.1 | NICE-TO-HAVE | Table 11 shows RF [0.673–0.824] and XGB [0.641–0.821] — these overlap substantially. One sentence acknowledging that "model rankings on pennystocks are not statistically distinguishable by CI overlap alone, though McNemar's confirms they differ in prediction pattern" would pre-empt the critique. |
| 5.17 | Is the structural correlation between sentiment feature and target a problem? | PARTIAL | §5.4.1 | SHOULD-HAVE | Listed as a limitation ("not leakage but a circularity") — good. But could be stronger: explicitly state what would be needed to rule it out (a feature set excluding sentiment entirely achieving comparable AUC). Phase 1 partially does this but the connection isn't drawn explicitly. |
| 5.18 | How sensitive are results to the specific threshold τ=1.0? | DONE | §5.2.5 | MUST-HAVE | No action needed; Table 19 weight sweep covers this. |
| 5.19 | Would the models work on a different time period? | PARTIAL | §5.4.1 | SHOULD-HAVE | Listed as a limitation (single calendar year) but no estimate of likely degradation. One sentence: "The 2021 dataset includes an unprecedented retail speculation event (GameStop); models may underperform on calmer periods where surges are rarer and less structured." |

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
| 6.7 | Are the three contributions restated concisely? | PARTIAL | §6.1–6.2 | SHOULD-HAVE | §5.5 states three contributions explicitly; §6 weaves them into narrative prose but doesn't re-enumerate them. A reader skimming only the conclusion might miss one. Consider a single sentence: "The project contributes a leakage-free methodology, a composite surge metric, and empirical evidence that data density is the binding constraint." |
| 6.8 | Is there a clear "so what" for the reader? | PARTIAL | §6.2 | SHOULD-HAVE | The findings are stated but the implication for practitioners is implicit. One sentence bridging to action: "For teams monitoring financial communities, the practical takeaway is to invest in data coverage before model sophistication — a sparse community needs more history, not a better algorithm." |
| 6.9 | Does the future work connect back to limitations? | DONE | §6.3 | MUST-HAVE | No action needed. Each future direction maps to a stated limitation (VADER → FinBERT, fixed window → multi-scale, retrospective → live). |
| 6.10 | Is there a final closing statement? | DONE | §6.3 | MUST-HAVE | No action needed. "The question this project set out to answer... has been answered. What remains is finding out how far that answer extends." — effective closing. |
| 6.11 | Does the conclusion acknowledge what was NOT achieved? | PARTIAL | §6.3 | NICE-TO-HAVE | Limitations are stated but there's no explicit statement about objectives that fell short. §5.1 shows all objectives were met, so this may not apply — but one could note that pennystocks results are tentative (31 test surges) and that the methodology is validated but not the deployment case. Already implicit; making it explicit would strengthen intellectual honesty. |

---

## Cross-Cutting Questions (should be addressed somewhere)

| # | Question | Status | Best Section | Priority | Action |
|---|----------|--------|--------------|----------|--------|
| C.1 | Why not anomaly detection instead of classification? | DONE | §3.5 | MUST-HAVE | Resolved by Q3.8a fix. "Why binary classification?" paragraph in §3.5 explicitly addresses anomaly detection: it is unsupervised, cannot leverage labelled surge history, and would treat every rare surge as an anomaly regardless of whether it is genuinely distinct from the baseline distribution. |
| C.2 | Could a simple threshold rule work just as well? | PARTIAL | §5.2.2 | SHOULD-HAVE | Table 15 shows multi-feature models beat the best single-feature baseline (+0.087 to +0.162 AUC). Reframe one sentence: "The best single-feature predictor (equivalent to a threshold rule on one signal) achieves 0.805 on WSB — strong, but the multi-feature models add 8.7 AUC points by combining signals a single rule cannot integrate." This closes the "would a rule suffice?" question explicitly. |
| C.3 | What about data drift over time? | PARTIAL | §3.6 | SHOULD-HAVE | Covered by Q3.16. When that fix is applied (one sentence acknowledging drift as a design-stage risk), the design-to-evaluation arc is complete. No separate action beyond completing 3.16. |
| C.4 | Why 24-hour windows specifically? | PARTIAL | §3.3 | SHOULD-HAVE | Add one sentence in §3.3: "A 24-hour window aligns with the daily trading cycle and captures overnight-to-open discussion patterns; shorter windows (6h) risk insufficient post counts for stable statistics, while longer windows (72h) blur the distinction between surge onset and sustained activity." Acknowledged as a limitation in §5.4.2 — this provides the forward justification. |
| C.5 | How would this work in practice (deployment scenario)? | MISSING | §6.3 | NICE-TO-HAVE | The project explicitly declares deployment out of scope. However, one sentence sketching a plausible use case would strengthen motivation without overstepping: "In deployment, the pipeline would ingest a rolling stream of posts, recompute features hourly, and flag tickers whose predicted surge probability crosses a tuned threshold — functioning as a screening layer that reduces thousands of tickers to a manageable watchlist for human review." Place at end of §6.3 or in §1.2 alongside the stakeholder discussion. |
