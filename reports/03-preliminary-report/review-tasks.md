# Preliminary Report — Review Tasks

## Submission Structure (4 Chapters, 6000 words max total)

| Chapter | Content | Word Limit |
|---------|---------|------------|
| 1. Introduction | Project concept, motivation, project template used | max 1000 words |
| 2. Literature Review | Revised version from peer review 2 | max 2500 words |
| 3. Design | Revised version from peer review 3 | max 2000 words |
| 4. Feature Prototype | Implementation + evaluation of a key technical feature | max 1500 words |

The per-chapter limits are strict. The total of chapter limits (7000) exceeds the overall limit (6000), giving flexibility to balance shorter/longer chapters.

---

## Deliverables

1. **Report** (PDF or compatible format) — 4 chapters as described above
2. **Video demonstration** — 3–5 minute MP4 showing the prototype with motivation and discussion

---

## Feature Prototype Requirements (Chapter 4)

- Implement at least one of the most important technical features to demonstrate feasibility
- Must work as designed (acceptable if the feature proves less effective than expected)
- Describe the prototype in the final chapter
- Include an evaluation of how well the prototype works
- Describe how it would be improved
- Evaluation approach must be appropriate for the project area (Machine Learning)
- Link the evaluation approach to related background work where possible

---

## Marking Criteria

1. Report clarity, formatting, and coherence
2. Knowledge of area of study, previous work, and academic literature
3. Critical evaluation of previous work / academic literature
4. Proper citation and referencing
5. Design clarity and quality
6. Project concept justified based on domain and users
7. Workplan explained in sufficient detail
8. Workplan feasibility
9. Evaluation strategy appropriate to project aims
10. Feature prototype quality
11. Feature prototype technical challenge
12. Demonstration effectiveness and impact (video)
13. Evaluation of the feature prototype with suitable improvements
14. Innovation and excellence beyond the above

---

## Current Status

All 9 original tasks are marked complete. The existing draft (`preliminary-report-v0.35.md`) covers:

- Sections 1–2: Project definition, scope, dataset (Task 1)
- Section 3: Methodology, pipeline, features, composite target (Task 3)
- Section 4: Evaluation strategy with metrics, temporal CV, robustness (Task 7)
- Section 5: Risk register (Task 2)
- Section 6: Timeline/Gantt (Task 4)
- Section 7: Literature review (Task 5)
- Section 8: Success criteria (Task 8)
- Section 9: References (Task 9)

---

## Gap Analysis — Outstanding Work

### [x] 1. Chapter 4: Feature Prototype — DONE

Written as Section 9 in `preliminary-report-v0.36.md`. Covers:

- **9.1** Prototype scope — 7-module pipeline architecture with technical challenges
- **9.2** Execution results — data flow, class distributions, normalisation parameters
- **9.3** Model evaluation — AUC-ROC 0.733 (exceeds target 0.70), confusion matrix, precision-recall analysis
- **9.4** Threshold sensitivity — 5 thresholds swept, τ=1.0 identified as viable operating point
- **9.5** Effectiveness evaluation — connected to literature [1][3][5], validates feasibility
- **9.6** Limitations and improvements — 7 concrete improvements planned
- **9.7** Reproducibility — CLI commands to reproduce results

### [x] 2. Report Restructuring — DONE

Restructured into 4 chapters in `preliminary-report-v1.00.md`:

| Chapter | Content |
|---------|---------|
| Chapter 1: Introduction | Project concept, motivation, problem statement, prediction scope, surge definition, scope |
| Chapter 2: Literature Review | Critical evaluation of 8 papers across 4 research areas, synthesis, research gap |
| Chapter 3: Design | Pipeline architecture, feature design, composite target, evaluation strategy, risk register, timeline |
| Chapter 4: Feature Prototype | Implementation, execution results, model performance, threshold sensitivity, evaluation + improvements |

### [x] 3. Word Count Compliance — DONE

Word counts per chapter (all within limits):

| Chapter | Word Count | Limit | Status |
|---------|-----------|-------|--------|
| Ch1: Introduction | ~456 | 1000 | OK |
| Ch2: Literature Review | ~1067 | 2500 | OK |
| Ch3: Design | ~880 | 2000 | OK |
| Ch4: Feature Prototype | ~692 | 1500 | OK |
| **Total (excl. refs)** | **~3381** | **6000** | **OK** |

Note: PowerShell word counts underestimate by ~10-15% due to markdown tables. Actual rendered word count is closer to ~3800.

### [ ] 4. Video Demonstration (MP4, 3–5 minutes)

- Script and record a 3–5 minute video demonstrating the prototype
- Must be MP4 format (no other formats accepted)
- Should include: motivation, prototype walkthrough, key results, brief discussion
- Demonstrate the pipeline running and producing evaluation metrics

### [ ] 5. Final Review Against Marking Criteria

Before submission, verify:

- [ ] Report is clear, well-formatted, and coherent
- [ ] Displays knowledge of area and academic literature
- [ ] Critically evaluates previous work
- [ ] Uses proper citation and referencing (IEEE format)
- [ ] Design is clear and high quality
- [ ] Project concept justified based on domain and users
- [ ] Workplan explained in detail and is feasible
- [ ] Evaluation strategy appropriate to aims
- [ ] Feature prototype is high quality and technically challenging
- [ ] Prototype evaluation shows improvements if appropriate
- [ ] Demonstration is effective and impactful

---

## Suggested Priority Order

1. ~~**Write the feature prototype** (Chapter 4) — this is the only new content required~~ DONE
2. ~~**Restructure** into 4 chapters~~ DONE
3. ~~**Trim word count** to compliance~~ DONE
4. **Record video** demonstrating the prototype
5. **Final review** against marking criteria
