# Recommendations Implementation Plan

## Objective
Turn the review findings into a focused, low-risk improvement plan that strengthens schema validation, test coverage, reproducibility, and operational portability without changing the project’s core scientific approach.

## Scope
This plan covers the recommendations from the final review:
- add schema validation before pipeline processing
- add integration tests for CSV drift and timezone edge cases
- improve portability of outputs and project-root assumptions
- tighten auditability and stale output hygiene
- document and automate the validation loop

## Priority order
### P1 — Immediate reliability improvements
These are the highest-value items because they reduce the chance of silent data errors.

#### 1. Add schema validation at the data-entry boundary
- Goal: fail clearly when the raw CSV does not match expectations.
- Files to update:
  - [src/surge_pipeline/loader.py](../src/surge_pipeline/loader.py)
  - [src/tests/test_loader.py](../src/tests/test_loader.py)
- Tasks:
  - [ ] Check required columns before processing (`id`, `title`, `selftext`, timestamp field, etc.)
  - [ ] Validate timestamp parsing and convert malformed values to explicit errors
  - [ ] Validate `tickers` field when present and ensure mixed-null/empty values are handled consistently
  - [ ] Log a clear diagnostic message when schema drift is detected
  - [ ] Add a regression test for missing required columns
  - [ ] Add a regression test for malformed timestamp input
- Done when:
  - invalid CSVs stop early with a readable error
  - the pipeline remains robust when expected inputs are valid

#### 2. Add drift and edge-case integration tests
- Goal: make the review recommendations real through automated regression protection.
- Files to update:
  - [src/tests/test_loader.py](../src/tests/test_loader.py)
  - [src/tests/test_pipeline_integration.py](../src/tests/test_pipeline_integration.py)
- Tasks:
  - [ ] Add a test for missing `created_utc` / `created` field
  - [ ] Add a test for empty title/selftext handling with valid tickers
  - [ ] Add a test for timezone-normalisation edge cases
  - [ ] Add a test for malformed or partially missing ticker strings
  - [ ] Add a test for duplicate or non-unique IDs if applicable
- Done when:
  - each major data-contract risk has a corresponding automated test

### P2 — Operational hardening
These improve portability and reduce ambiguity during real-world use.

#### 3. Reduce dependency on project-root assumptions
- Goal: make the code easier to run in different environments and avoid hidden path coupling.
- Files to update:
  - [src/surge_pipeline/config.py](../src/surge_pipeline/config.py)
  - [src/run_labeling.py](../src/run_labeling.py)
  - [src/run_training.py](../src/run_training.py)
  - [src/eda/eda_pipeline.py](../src/eda/eda_pipeline.py)
- Tasks:
  - [ ] Review all absolute/project-root based path resolution logic
  - [ ] Standardise output directory resolution from explicit CLI args where possible
  - [ ] Add a clearer error message if input/output paths are invalid
  - [ ] Document any remaining repo-root assumptions in the developer docs
- Done when:
  - most path logic is explicit and predictable
  - users can run the workflow from a consistent environment without hidden assumptions

#### 4. Improve stale-output hygiene and manifest discipline
- Goal: reduce confusion between historical artifacts and current outputs.
- Files to update:
  - [output/processed/latest_outputs.json](../output/processed/latest_outputs.json)
  - [src/surge_pipeline/experiment_log.py](../src/surge_pipeline/experiment_log.py)
  - [README.md](../README.md)
- Tasks:
  - [ ] Define a clear naming convention for runs and latest outputs
  - [ ] Document how to identify the current experiment snapshot
  - [ ] Review whether stale files should be archived or cleaned automatically
  - [ ] Add a guard to warn when multiple candidate outputs exist for the same stage
- Done when:
  - users can unambiguously tell which output is current
  - stale artifact confusion is reduced

### P3 — Quality and release validation
These are the finishing tasks to make the recommendations sustainable.

#### 5. Add a release-level validation checklist
- Goal: codify the review evidence into repeatable project hygiene.
- Files to update:
  - [RUNNING_AND_TESTING.md](../RUNNING_AND_TESTING.md)
  - [review/review-plan.md](review-plan.md)
  - [review/final-review-report.md](final-review-report.md)
- Tasks:
  - [ ] Document the exact validation commands to run before a release
  - [ ] Add a short pre-flight checklist for raw-data and output validation
  - [ ] Record the expected artifacts after a successful labelling run
  - [ ] Add note on when to re-run full validation after config or schema changes
- Done when:
  - the project has a repeatable validation recipe for future reviewers

## Suggested execution sequence
1. Implement schema validation in the loader and add the first failing tests.
2. Add drift/edge-case regression tests for malformed timestamps and missing columns.
3. Review and harden path resolution for CLI scripts and output manifests.
4. Tighten output hygiene and update docs for current artifacts.
5. Finalise the validation checklist and keep it in the review documentation.

## Definition of done
The recommendations are considered implemented when all of the following are true:
- raw-data schema drift is detected early with a clear message
- critical edge cases are covered by automated tests
- path handling is explicit and documented
- output manifest and historical file usage are unambiguous
- a repeatable validation checklist is included in project documentation

## Suggested effort estimate
- P1 items: 1–2 working sessions
- P2 items: 1 working session
- P3 items: 0.5 working session

Total: approximately 2–4 focused development sessions, depending on how much refactoring is needed for path portability.
