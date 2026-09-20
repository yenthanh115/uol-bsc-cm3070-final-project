# CM3070 Final Project — Submission Checklist

Checklist for getting the report, code repository, and demo video submission-ready
against `reports/05-final/instruction.md`.

**Overall state:** The final report (`reports/05-final/final-report-v0.0.2.md`) is
mature and complete across all six chapters. 43 of 45 requirements are PASS
(`requirements.md`). The two open requirements (R44 demo video, R45 packaged
submission) plus a per-section word overflow are the remaining work.

## Word count status

Measured with `python scripts/wordcount.py reports/05-final/final-report-v0.0.2.md --split-level 1`.

| Section | Words | Per-section max | Status |
| --- | --- | --- | --- |
| Introduction | 890 | 1000 | OK |
| Literature Review | 1911 | 2500 | OK |
| Design | 1917 | 2000 | OK |
| Implementation | 2614 | 2500 | OVER by 114 |
| Evaluation | 2140 | 2500 | OK |
| Conclusion | 824 | 1000 | OK |
| **Total** | **10,296** | **10,500** | OK |

Total is under the strict 10,500 limit, but Implementation breaches its own 2,500-word cap.

## Checklist

- [ ] **1. Trim Implementation section to ≤2500 words** (114+ words over the strict per-section limit).
  Tighten prose without cutting the technical substance the rubric rewards. Re-run
  `scripts/wordcount.py` to confirm Implementation ≤2500 and total ≤10,500.
- [ ] **2. Verify report format and formal constraints (X20).**
  All six chapters present; per-section and 10,500 total word limits met; references in
  ACM style; figures/tables captioned; code repo link present in the report; PDF renders correctly.
- [ ] **3. Reconcile all report numbers against one canonical results artifact (X17).**
  Cross-check every metric/table/figure against `output/evaluation` and the `facts/` files so
  no stale numbers from earlier experiment versions remain. Note: uncommitted changes to
  `output/evaluation/latest_outputs.json`, `output/experiment_log.jsonl`,
  `output/processed/latest_outputs.json`, `pyproject.toml`.
- [ ] **4. Add the public code repository link into the final report.**
  `instruction.md` requires a link to a publicly viewable repo in the submission. Confirm it
  appears in the report body.
- [ ] **5. Verify the GitHub repository is public and reproducible (X18, R37).**
  Confirm `https://github.com/yenthanh115/uol-bsc-cm3070-final-project` is publicly viewable,
  README run instructions work from a clean clone, and the final branch/commit is pushed.
- [ ] **6. Record the final results and commit/push the working branch (R38).**
  Currently on branch `review/p2` with uncommitted output changes. Decide final branch/merge,
  commit, and push so the public repo reflects the submitted state.
- [ ] **7. Produce the 3–5 minute demo video (R44).**
  Spoken audio (no AI voice, no speed-up). Show raw input data, pipeline running end-to-end
  (`surge-label` → `surge-train` / `run_cross_validation.py`), predictions/outputs, and
  evaluation (metrics/figures) for the final two-dataset, three-model system. Include a link
  in the report if hosted. **Currently absent** — only a superseded preliminary-stage demo
  *plan* exists (see `facts/r44.md`).
- [ ] **8. Assemble and independently check the submission package (R45, X19).**
  Bundle report PDF + code archive/link + demo video + references. After upload, verify each
  opens correctly: PDF renders, code extracts/clones, video plays with audio, references
  resolve. Confirm file naming/type/deadline against `instruction.md`.

## Two real blockers

`facts/r44.md` and `facts/r45.md` diagnose these:

1. **Demo video does not exist yet** (R44) — only a preliminary-stage demo plan describing an
   older single-dataset prototype.
2. **Assembled/verified submission package** (R45) — cannot exist until the video does.

Critical-checklist items X17–X20 have no `facts/` files yet. Everything else is verification
and packaging around an already-strong report.
