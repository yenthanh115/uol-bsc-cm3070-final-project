Based on where the project stands today, the **Draft Report** should be treated as the point where the project becomes **nearly complete**. Unlike the preliminary report, which focused on design and a feasibility prototype, the draft report is expected to contain a complete implementation, comprehensive evaluation, critical analysis, and a report that is already close to submission quality. 

For your project (predicting engagement and sentiment surge of stock-related Reddit discussions), I would organise the milestone into five phases.

| Phase                    | Goal                                     | Expected completion      |
| ------------------------ | ---------------------------------------- | ------------------------ |
| 1. Finish implementation | Complete the entire ML pipeline          | 100% functional          |
| 2. Perform experiments   | Produce all evaluation results           | Complete                 |
| 3. Analyse results       | Explain why the model behaves as it does | Complete                 |
| 4. Write draft report    | Produce almost the entire report         | 90–95%                   |
| 5. Internal review       | Find weaknesses before final report      | Ready for tutor feedback |

---

## Phase 1 — Complete the implementation

This should be the highest priority.

### Data pipeline

* Final dataset
* Data cleaning
* Feature engineering
* Surge label generation
* Dataset splitting
* Reproducible pipeline

### Models

At minimum:

* Logistic Regression (baseline)
* Random Forest

Recommended additions:

* XGBoost or LightGBM
* Optional ablation experiments

### Engineering

* configuration files
* reproducible experiments
* logging
* saved models
* evaluation scripts
* figures generated automatically

---

## Phase 2 — Complete the evaluation

This section carries a large proportion of the marks.

Produce:

* confusion matrices
* ROC curve
* Precision-Recall curve
* feature importance
* learning curves (if useful)
* class distribution
* prediction examples

Evaluate using

* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC

Also compare

* baseline
* improved model

Perform

* cross validation
* error analysis
* robustness checks

The draft rubric expects evaluation to cover the major issues rather than only reporting a few metrics. 

---

## Phase 3 — Critical analysis

This is usually where average reports become strong reports.

Instead of saying

> Random Forest achieved 0.87 F1.

Discuss

* why it performed better
* which features mattered
* where it failed
* effect of class imbalance
* examples of false positives
* examples of false negatives
* implications for real-world use

Then relate the findings back to the original research question.

---

## Phase 4 — Draft report

Aim for a report that is already **90–95% complete**.

Suggested structure:

1. Introduction
2. Background
3. Literature Review
4. Problem Definition
5. Requirements
6. Project Design
7. Dataset
8. Data Preprocessing
9. Feature Engineering
10. Surge Definition
11. Model Development
12. Experimental Setup
13. Results
14. Discussion
15. Evaluation
16. Limitations
17. Future Work
18. Conclusion

Include high-quality figures throughout, as clear presentation and diagrams are explicitly assessed.

---

## Phase 5 — Review against the rubric

Before submission, review every section against the draft rubric.

**Literature**

* 8–12 quality papers
* Critical comparison
* Research gap identified

**Design**

* Clear pipeline diagrams
* Architecture diagrams
* Feature engineering flow
* Model workflow

**Implementation**

* Working end-to-end pipeline
* Clean code
* Reproducible experiments

**Evaluation**

* Multiple metrics
* Baseline comparison
* Error analysis
* Discussion of limitations

**Originality**
Your project can demonstrate originality through:

* a well-justified definition of "surge"
* custom feature engineering
* combining engagement and sentiment prediction
* practical analysis of stock-related Reddit discussions

These align well with the higher bands of the rubric, which reward originality, technically challenging implementation, and evidence-driven evaluation.

## Deliverables for the draft milestone

By the time you submit the draft report, you should ideally have:

* Complete end-to-end implementation
* Final experimental results
* Baseline and improved model comparison
* Error analysis and discussion
* Nearly complete literature review
* Complete methodology chapter
* Complete evaluation chapter
* Complete conclusions and future work
* High-quality diagrams and tables
* Proper ACM citations
* A polished report requiring only refinement after tutor feedback

Given your current progress, your primary effort should now shift from **building new functionality** to **strengthening experiments, analysis, and the quality of the written report**. Those areas contribute heavily to the draft and final report marks and are where the largest gains can be made.
