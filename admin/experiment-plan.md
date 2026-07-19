# Experiment Plan — Distributed Across 3 Machines

## Overview

This plan assumes **zero prior results**. All experiments will be run fresh using the two selected datasets and the current pipeline codebase, distributed across **3 machines** running in parallel with no cross-dependencies.

**Datasets:**
- **r/pennystocks** — 80,212 raw records (sparse, niche community)
- **r/wallstreetbets** — 1,293,981 raw records (dense, mainstream community)

**Pipeline settings (defaults):**
- `surge_method`: forward_growth
- `threshold_tau`: 1.5
- `weight_volume`: 0.5, `weight_sentiment`: 0.5
- `temporal_split_ratio`: 0.8
- `sentiment_model`: vader
- `min_window_count`: 1
- `random_seed`: 42
- Features: 11 (9 base + 2 interaction terms)
- Models: Logistic Regression, Random Forest, XGBoost
- CV: Expanding-window temporal, k=4 folds (3 splits)
- Evaluation: AUC-ROC (primary), Precision, Recall, F1, confusion matrix, bootstrap CIs, McNemar's test, threshold tuning

---

## Output File Conflict Analysis

Before running experiments, the following conflicts must be understood and mitigated:

### Conflict 1: Model Files (CRITICAL)

Model files are named `{model_name}_{phase}_{seed}.joblib` where `phase` is `phase1` if `weight_sentiment=0.0`, else `phase2`. Multiple experiments with the same phase+seed produce **identical filenames** that would overwrite each other.

**Affected experiments:**
- A2 (phase2, seed=42) vs G2, G4, G5 — all produce `*_phase2_42.joblib`
- G2, G4, G5 would overwrite each other if sharing a directory

**Mitigation:** Each experiment uses `--models-dir output/models/{experiment_id}/` to isolate model files.

### Conflict 2: Evaluation Figures (MODERATE)

Figure filenames are NOT timestamped (e.g., `10_confusion_matrix_xgboost.png`). If two training runs execute on the same machine, the second overwrites the first.

**Mitigation:** Use `--no-figures` during batch runs. Regenerate figures selectively during consolidation.

### Conflict 3: `latest_outputs.json` (LOW)

Each labelling run overwrites `output/processed/latest_outputs.json`. Since we use explicit `--data-path` for all training commands, this is harmless but noted.

**Mitigation:** Always pass explicit `--data-path` to `run_training.py`. Never rely on `latest_outputs.json`.

### Conflict 4: `experiment_log.jsonl` (LOW)

Append-only file. Each machine writes its own log. Merge during consolidation.

**Mitigation:** After all machines finish, concatenate the three `experiment_log.jsonl` files into one.

---

## Machine Assignments

| Machine | Role | Experiments | Est. Runtime |
|---------|------|-------------|--------------|
| **Machine 1** | WSB core (baseline + Phase 1 + threshold) | A2, B1, C1 | ~6 hours |
| **Machine 2** | Pennystocks (all experiments + multi-seed) | A1, B3, C2, F2 (4 seeds) | ~3 hours |
| **Machine 3** | WSB weight sensitivity sweep | G2, G4, G5 | ~6 hours |
| **Consolidation** | Cross-dataset + merge | D1, D2 + merge outputs | ~30 min |

### Dependency Graph

```
Machine 1 (WSB core)          Machine 2 (Pennystocks)       Machine 3 (WSB weight sweep)
─────────────────────         ───────────────────────       ────────────────────────────
A2 (baseline)                 A1 (baseline)                 G2 (w=0.75,0.25)
B1 (Phase 1)                  B3 (Phase 1)                  G4 (w=0.25,0.75)
C1 (τ=1.0)                   C2 (τ=1.0)                   G5 (w=0.0,1.0)
                              F2-seed123
                              F2-seed456
                              F2-seed789
                              F2-seed2024
         │                            │
         └────────────────────────────┘
                        │
              Consolidation Phase
              D1: WSB models → pennystocks data
              D2: pennystocks models → WSB data
              Merge experiment logs + generate figures
```

**Key design decisions:**
- G1 (w=1.0/0.0) is identical to B1 — reuse B1's AUC as the w_sent=0.0 data point in the weight sweep
- G3 (w=0.5/0.5) is identical to A2 — reuse A2's AUC as the w_sent=0.5 data point in the weight sweep
- A1 is run as a standalone step (not as F2-seed42) to keep its `--models-dir` clean for cross-dataset eval (D2)
- F2 runs seeds 123, 456, 789, 2024 only — A1 provides the seed=42 data point for robustness
- Phase D requires models from Machine 1 (A2) AND Machine 2 (A1) → runs post-merge
- No machine depends on another machine's output

---

## Pre-flight Checklist (All Machines)

Before running any experiments, on each machine:

```bash
# 1. Clone repository at the SAME commit
git clone <repository-url>
cd uol-bsc-cm3070-final-project
git checkout <frozen-commit-sha>

# 2. Create and activate virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate.bat
# Linux/Mac: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify input data exists
ls input/raw/r_pennystocks_submissions_reddit.csv
ls input/raw/r_wallstreetbets_submissions_reddit.csv

# 5. Create output directories
mkdir -p output/processed output/evaluation output/models output/logs

# 6. Verify pipeline works (dry run with small data)
python src/run_labeling.py --file-path input/raw/r_pennystocks_submissions_reddit.csv --sweep-only
```

**Important:** All commands are run from the **project root** directory (not `src/`). This ensures that relative paths in `latest_outputs.json` resolve correctly (they are stored relative to project root).

---

## Machine 1: WSB Core Experiments

**Assigned experiments:** A2, B1, C1
**Estimated total runtime:** ~6 hours
**Dataset:** r/wallstreetbets only

### Step 1.1 — Experiment A2: WSB Baseline (Phase 2)

**Description:** Label and train on r/wallstreetbets with default settings (τ=1.5, w_vol=0.5, w_sent=0.5, seed=42). This is the primary WSB baseline and produces models needed for cross-dataset evaluation later.

**Commands:**
```bash
# Labelling
python src/run_labeling.py \
    --file-path input/raw/r_wallstreetbets_submissions_reddit.csv \
    --threshold-tau 1.5 \
    --weight-volume 0.5 \
    --weight-sentiment 0.5 \
    --random-seed 42 \
    --log-file auto \
    --notes "A2: WSB baseline (phase2, seed=42)"

# Training (replace <TIMESTAMP> with the prefix from labelling output, e.g., 2026-07-18_10-30)
python src/run_training.py \
    --data-path output/processed/<TIMESTAMP>_labelled_dataset.csv \
    --models-dir output/models/A2 \
    --seed 42 \
    --weight-sentiment 0.5 \
    --threshold-tau 1.5 \
    --no-figures \
    --log-file auto \
    --notes "A2: WSB baseline (phase2, seed=42)"
```

**Expected outputs:**
- `output/processed/<ts>_labelled_dataset.csv` (~457K records, surge_rate ~1.44%)
- `output/processed/<ts>_pipeline_summary.json`
- `output/processed/<ts>_pipeline_config.json`
- `output/processed/<ts>_threshold_sensitivity.csv`
- `output/evaluation/<ts>_evaluation_metrics.json`
- `output/evaluation/<ts>_threshold_tuning.json`
- `output/evaluation/<ts>_feature_importance.json`
- `output/evaluation/<ts>_final_summary.json`
- `output/models/A2/logistic_regression_phase2_42.joblib`
- `output/models/A2/random_forest_phase2_42.joblib`
- `output/models/A2/xgboost_phase2_42.joblib`
- `output/logs/<ts>_labelling.log`
- `output/logs/<ts>_training.log`

**Estimated runtime:** ~2 hours (labelling ~18 min, training ~90 min)

---

### Step 1.2 — Experiment B1: WSB Phase 1 (Volume-Only)

**Description:** Label WSB with `weight_sentiment=0.0` (volume-only surge definition) and train. This answers whether sentiment in the label definition improves predictability compared to pure volume surges.

**Commands:**
```bash
# Labelling (volume-only)
python src/run_labeling.py \
    --file-path input/raw/r_wallstreetbets_submissions_reddit.csv \
    --threshold-tau 1.5 \
    --weight-volume 1.0 \
    --weight-sentiment 0.0 \
    --random-seed 42 \
    --log-file auto \
    --notes "B1: WSB Phase 1 volume-only (w_sent=0.0, tau=1.5)"

# Training
python src/run_training.py \
    --data-path output/processed/<TIMESTAMP>_labelled_dataset.csv \
    --models-dir output/models/B1 \
    --seed 42 \
    --weight-sentiment 0.0 \
    --threshold-tau 1.5 \
    --no-figures \
    --log-file auto \
    --notes "B1: WSB Phase 1 volume-only (w_sent=0.0, tau=1.5)"
```

**Expected outputs:**
- `output/processed/<ts>_labelled_dataset.csv` (surge_rate ~0.95% — fewer surges without sentiment boost)
- `output/processed/<ts>_pipeline_summary.json`
- `output/processed/<ts>_pipeline_config.json`
- `output/processed/<ts>_threshold_sensitivity.csv`
- `output/evaluation/<ts>_evaluation_metrics.json`
- `output/evaluation/<ts>_threshold_tuning.json`
- `output/evaluation/<ts>_feature_importance.json`
- `output/evaluation/<ts>_final_summary.json`
- `output/models/B1/logistic_regression_phase1_42.joblib`
- `output/models/B1/random_forest_phase1_42.joblib`
- `output/models/B1/xgboost_phase1_42.joblib`

**Estimated runtime:** ~2 hours

---

### Step 1.3 — Experiment C1: WSB Threshold τ=1.0

**Description:** Label WSB with a lower surge threshold (τ=1.0 instead of 1.5). This increases the surge rate (~8% vs ~1.4%), creating an easier classification problem. Tests whether the pipeline performs better with more balanced classes.

**Commands:**
```bash
# Labelling (lower threshold)
python src/run_labeling.py \
    --file-path input/raw/r_wallstreetbets_submissions_reddit.csv \
    --threshold-tau 1.0 \
    --weight-volume 0.5 \
    --weight-sentiment 0.5 \
    --random-seed 42 \
    --log-file auto \
    --notes "C1: WSB threshold tau=1.0"

# Training
python src/run_training.py \
    --data-path output/processed/<TIMESTAMP>_labelled_dataset.csv \
    --models-dir output/models/C1 \
    --seed 42 \
    --weight-sentiment 0.5 \
    --threshold-tau 1.0 \
    --no-figures \
    --log-file auto \
    --notes "C1: WSB threshold tau=1.0"
```

**Expected outputs:**
- `output/processed/<ts>_labelled_dataset.csv` (surge_rate ~8.2%)
- `output/processed/<ts>_pipeline_summary.json`
- `output/processed/<ts>_pipeline_config.json`
- `output/processed/<ts>_threshold_sensitivity.csv`
- `output/evaluation/<ts>_evaluation_metrics.json`
- `output/evaluation/<ts>_threshold_tuning.json`
- `output/evaluation/<ts>_feature_importance.json`
- `output/evaluation/<ts>_final_summary.json`
- `output/models/C1/logistic_regression_phase2_42.joblib`
- `output/models/C1/random_forest_phase2_42.joblib`
- `output/models/C1/xgboost_phase2_42.joblib`

**Estimated runtime:** ~2 hours

---

## Machine 2: Pennystocks All Experiments

**Assigned experiments:** A1, B3, C2, F2 (4 seeds: 123, 456, 789, 2024)
**Estimated total runtime:** ~2.5 hours
**Dataset:** r/pennystocks only

### Step 2.1 — Experiment A1: Pennystocks Baseline (Phase 2)

**Description:** Label and train on r/pennystocks with default settings. This is the primary pennystocks baseline. Models are needed for cross-dataset evaluation (D2) during consolidation.

**Commands:**
```bash
# Labelling
python src/run_labeling.py \
    --file-path input/raw/r_pennystocks_submissions_reddit.csv \
    --threshold-tau 1.5 \
    --weight-volume 0.5 \
    --weight-sentiment 0.5 \
    --random-seed 42 \
    --log-file auto \
    --notes "A1: Pennystocks baseline (phase2, seed=42)"

# Training
python src/run_training.py \
    --data-path output/processed/<TIMESTAMP>_labelled_dataset.csv \
    --models-dir output/models/A1 \
    --seed 42 \
    --weight-sentiment 0.5 \
    --threshold-tau 1.5 \
    --no-figures \
    --log-file auto \
    --notes "A1: Pennystocks baseline (phase2, seed=42)"
```

**Expected outputs:**
- `output/processed/<ts>_labelled_dataset.csv` (~24.8K records, surge_rate ~2.8%)
- `output/processed/<ts>_pipeline_summary.json`
- `output/processed/<ts>_pipeline_config.json`
- `output/processed/<ts>_threshold_sensitivity.csv`
- `output/evaluation/<ts>_evaluation_metrics.json`
- `output/evaluation/<ts>_threshold_tuning.json`
- `output/evaluation/<ts>_feature_importance.json`
- `output/evaluation/<ts>_final_summary.json`
- `output/models/A1/logistic_regression_phase2_42.joblib`
- `output/models/A1/random_forest_phase2_42.joblib`
- `output/models/A1/xgboost_phase2_42.joblib`

**Estimated runtime:** ~15 min

---

### Step 2.2 — Experiment B3: Pennystocks Phase 1 (Volume-Only)

**Description:** Label pennystocks with `weight_sentiment=0.0` (volume-only). Companion to B1 on WSB — tests whether sentiment contribution holds on sparse data.

**Commands:**
```bash
# Labelling (volume-only)
python src/run_labeling.py \
    --file-path input/raw/r_pennystocks_submissions_reddit.csv \
    --threshold-tau 1.5 \
    --weight-volume 1.0 \
    --weight-sentiment 0.0 \
    --random-seed 42 \
    --log-file auto \
    --notes "B3: Pennystocks Phase 1 volume-only (w_sent=0.0, tau=1.5)"

# Training
python src/run_training.py \
    --data-path output/processed/<TIMESTAMP>_labelled_dataset.csv \
    --models-dir output/models/B3 \
    --seed 42 \
    --weight-sentiment 0.0 \
    --threshold-tau 1.5 \
    --no-figures \
    --log-file auto \
    --notes "B3: Pennystocks Phase 1 volume-only (w_sent=0.0, tau=1.5)"
```

**Expected outputs:**
- `output/processed/<ts>_labelled_dataset.csv` (lower surge rate than A1)
- `output/evaluation/<ts>_evaluation_metrics.json`
- `output/evaluation/<ts>_threshold_tuning.json`
- `output/evaluation/<ts>_feature_importance.json`
- `output/evaluation/<ts>_final_summary.json`
- `output/models/B3/logistic_regression_phase1_42.joblib`
- `output/models/B3/random_forest_phase1_42.joblib`
- `output/models/B3/xgboost_phase1_42.joblib`

**Estimated runtime:** ~15 min

---

### Step 2.3 — Experiment C2: Pennystocks Threshold τ=1.0

**Description:** Label pennystocks with lower threshold (τ=1.0). Tests threshold sensitivity on sparse data — companion to C1 on WSB.

**Commands:**
```bash
# Labelling
python src/run_labeling.py \
    --file-path input/raw/r_pennystocks_submissions_reddit.csv \
    --threshold-tau 1.0 \
    --weight-volume 0.5 \
    --weight-sentiment 0.5 \
    --random-seed 42 \
    --log-file auto \
    --notes "C2: Pennystocks threshold tau=1.0"

# Training
python src/run_training.py \
    --data-path output/processed/<TIMESTAMP>_labelled_dataset.csv \
    --models-dir output/models/C2 \
    --seed 42 \
    --weight-sentiment 0.5 \
    --threshold-tau 1.0 \
    --no-figures \
    --log-file auto \
    --notes "C2: Pennystocks threshold tau=1.0"
```

**Expected outputs:**
- `output/processed/<ts>_labelled_dataset.csv` (higher surge rate ~8%)
- `output/evaluation/<ts>_evaluation_metrics.json`
- `output/evaluation/<ts>_threshold_tuning.json`
- `output/evaluation/<ts>_feature_importance.json`
- `output/evaluation/<ts>_final_summary.json`
- `output/models/C2/logistic_regression_phase2_42.joblib`
- `output/models/C2/random_forest_phase2_42.joblib`
- `output/models/C2/xgboost_phase2_42.joblib`

**Estimated runtime:** ~15 min

---

### Step 2.4 — Experiment F2: Pennystocks Multi-Seed Robustness

**Description:** Run the pennystocks baseline (same config as A1) with 4 additional seeds to measure result variance. Combined with A1 (seed=42), this gives 5 seeds total for robustness analysis.

**Commands:**
```bash
# --- Seed 123 ---
python src/run_labeling.py \
    --file-path input/raw/r_pennystocks_submissions_reddit.csv \
    --random-seed 123 \
    --log-file auto \
    --notes "F2-seed123: Pennystocks robustness"
python src/run_training.py \
    --data-path output/processed/<TIMESTAMP>_labelled_dataset.csv \
    --models-dir output/models/F2_seed123 \
    --seed 123 \
    --no-figures \
    --log-file auto \
    --notes "F2-seed123: Pennystocks robustness"

# --- Seed 456 ---
python src/run_labeling.py \
    --file-path input/raw/r_pennystocks_submissions_reddit.csv \
    --random-seed 456 \
    --log-file auto \
    --notes "F2-seed456: Pennystocks robustness"
python src/run_training.py \
    --data-path output/processed/<TIMESTAMP>_labelled_dataset.csv \
    --models-dir output/models/F2_seed456 \
    --seed 456 \
    --no-figures \
    --log-file auto \
    --notes "F2-seed456: Pennystocks robustness"

# --- Seed 789 ---
python src/run_labeling.py \
    --file-path input/raw/r_pennystocks_submissions_reddit.csv \
    --random-seed 789 \
    --log-file auto \
    --notes "F2-seed789: Pennystocks robustness"
python src/run_training.py \
    --data-path output/processed/<TIMESTAMP>_labelled_dataset.csv \
    --models-dir output/models/F2_seed789 \
    --seed 789 \
    --no-figures \
    --log-file auto \
    --notes "F2-seed789: Pennystocks robustness"

# --- Seed 2024 ---
python src/run_labeling.py \
    --file-path input/raw/r_pennystocks_submissions_reddit.csv \
    --random-seed 2024 \
    --log-file auto \
    --notes "F2-seed2024: Pennystocks robustness"
python src/run_training.py \
    --data-path output/processed/<TIMESTAMP>_labelled_dataset.csv \
    --models-dir output/models/F2_seed2024 \
    --seed 2024 \
    --no-figures \
    --log-file auto \
    --notes "F2-seed2024: Pennystocks robustness"
```

**Expected outputs per seed:**
- `output/processed/<ts>_labelled_dataset.csv`
- `output/evaluation/<ts>_evaluation_metrics.json`
- `output/evaluation/<ts>_final_summary.json`
- `output/models/F2_seed{N}/{model}_phase2_{seed}.joblib` (3 models each)

**Note:** A1 (seed=42) provides the fifth data point for robustness. During consolidation, combine A1's AUC with the 4 seeds here to compute mean ± std across 5 seeds.

**Estimated runtime:** ~1.2 hours (4 seeds × ~18 min each)

---

## Machine 3: WSB Weight Sensitivity Sweep

**Assigned experiments:** G2, G4, G5
**Estimated total runtime:** ~6 hours
**Dataset:** r/wallstreetbets only

**Notes:**
- G1 (w_vol=1.0, w_sent=0.0) is identical to B1 (Machine 1). Reuse B1's AUC as the w_sent=0.0 data point.
- G3 (w_vol=0.5, w_sent=0.5) is identical to A2 (Machine 1). Reuse A2's AUC as the w_sent=0.5 data point.
- Only G2, G4, G5 need to actually run here.

### Step 3.1 — Experiment G2: WSB Weight Sweep (w_vol=0.75, w_sent=0.25)

**Description:** 75% volume, 25% sentiment in labelling formula. Tests whether a smaller sentiment contribution still adds value.

**Commands:**
```bash
# Labelling
python src/run_labeling.py \
    --file-path input/raw/r_wallstreetbets_submissions_reddit.csv \
    --threshold-tau 1.5 \
    --weight-volume 0.75 \
    --weight-sentiment 0.25 \
    --random-seed 42 \
    --log-file auto \
    --notes "G2: WSB weight sweep (w_vol=0.75, w_sent=0.25)"

# Training
python src/run_training.py \
    --data-path output/processed/<TIMESTAMP>_labelled_dataset.csv \
    --models-dir output/models/G2 \
    --seed 42 \
    --weight-sentiment 0.25 \
    --threshold-tau 1.5 \
    --no-figures \
    --log-file auto \
    --notes "G2: WSB weight sweep (w_vol=0.75, w_sent=0.25)"
```

**Expected outputs:**
- `output/processed/<ts>_labelled_dataset.csv`
- `output/evaluation/<ts>_evaluation_metrics.json`
- `output/evaluation/<ts>_final_summary.json`
- `output/models/G2/{model}_phase2_42.joblib`

**Estimated runtime:** ~2 hours

---

### Step 3.2 — Experiment G4: WSB Weight Sweep (w_vol=0.25, w_sent=0.75)

**Description:** 25% volume, 75% sentiment in labelling formula. Tests whether over-weighting sentiment improves or degrades performance.

**Commands:**
```bash
# Labelling
python src/run_labeling.py \
    --file-path input/raw/r_wallstreetbets_submissions_reddit.csv \
    --threshold-tau 1.5 \
    --weight-volume 0.25 \
    --weight-sentiment 0.75 \
    --random-seed 42 \
    --log-file auto \
    --notes "G4: WSB weight sweep (w_vol=0.25, w_sent=0.75)"

# Training
python src/run_training.py \
    --data-path output/processed/<TIMESTAMP>_labelled_dataset.csv \
    --models-dir output/models/G4 \
    --seed 42 \
    --weight-sentiment 0.75 \
    --threshold-tau 1.5 \
    --no-figures \
    --log-file auto \
    --notes "G4: WSB weight sweep (w_vol=0.25, w_sent=0.75)"
```

**Expected outputs:**
- `output/processed/<ts>_labelled_dataset.csv`
- `output/evaluation/<ts>_evaluation_metrics.json`
- `output/evaluation/<ts>_final_summary.json`
- `output/models/G4/{model}_phase2_42.joblib`

**Estimated runtime:** ~2 hours

---

### Step 3.3 — Experiment G5: WSB Weight Sweep (w_vol=0.0, w_sent=1.0)

**Description:** Full sentiment, zero volume in labelling formula. Tests the extreme of sentiment-only surge definition.

**Commands:**
```bash
# Labelling
python src/run_labeling.py \
    --file-path input/raw/r_wallstreetbets_submissions_reddit.csv \
    --threshold-tau 1.5 \
    --weight-volume 0.0 \
    --weight-sentiment 1.0 \
    --random-seed 42 \
    --log-file auto \
    --notes "G5: WSB weight sweep (w_vol=0.0, w_sent=1.0)"

# Training
python src/run_training.py \
    --data-path output/processed/<TIMESTAMP>_labelled_dataset.csv \
    --models-dir output/models/G5 \
    --seed 42 \
    --weight-sentiment 1.0 \
    --threshold-tau 1.5 \
    --no-figures \
    --log-file auto \
    --notes "G5: WSB weight sweep (w_vol=0.0, w_sent=1.0)"
```

**Expected outputs:**
- `output/processed/<ts>_labelled_dataset.csv`
- `output/evaluation/<ts>_evaluation_metrics.json`
- `output/evaluation/<ts>_final_summary.json`
- `output/models/G5/{model}_phase2_42.joblib`

**Estimated runtime:** ~2 hours

---

## Consolidation Phase (After All Machines Complete)

**Prerequisites:** All 3 machines have finished. Collect outputs from each machine into a single repository copy.
**Estimated runtime:** ~30 minutes

### Step 4.1 — Collect Outputs

**Description:** Copy all output files from each machine into a single consolidated repository. Since labelling/evaluation outputs use timestamp prefixes, there will be no filename collisions. Model directories are already namespaced per experiment.

**Commands:**
```bash
# On the consolidation machine (can be any of the 3):

# Copy from Machine 1
cp -r machine1/output/processed/*       output/processed/
cp -r machine1/output/evaluation/*      output/evaluation/
cp -r machine1/output/models/A2         output/models/A2
cp -r machine1/output/models/B1         output/models/B1
cp -r machine1/output/models/C1         output/models/C1
cp -r machine1/output/logs/*            output/logs/

# Copy from Machine 2
cp -r machine2/output/processed/*       output/processed/
cp -r machine2/output/evaluation/*      output/evaluation/
cp -r machine2/output/models/A1         output/models/A1
cp -r machine2/output/models/B3         output/models/B3
cp -r machine2/output/models/C2         output/models/C2
cp -r machine2/output/models/F2_*      output/models/
cp -r machine2/output/logs/*            output/logs/

# Copy from Machine 3
cp -r machine3/output/processed/*       output/processed/
cp -r machine3/output/evaluation/*      output/evaluation/
cp -r machine3/output/models/G*        output/models/
cp -r machine3/output/logs/*            output/logs/

# Merge experiment logs
cat machine1/output/experiment_log.jsonl \
    machine2/output/experiment_log.jsonl \
    machine3/output/experiment_log.jsonl > output/experiment_log.jsonl
```

**Expected result:** All outputs in a single `output/` directory tree with no filename conflicts.

---

### Step 4.2 — Experiment D1: Cross-Dataset (WSB → Pennystocks)

**Description:** Evaluate models trained on WSB (from A2, Machine 1) against the pennystocks labelled dataset (from A1, Machine 2). Tests whether surge dynamics generalise across communities.

**Prerequisites:**
- `output/models/A2/` contains WSB-trained models (from Machine 1)
- A1's labelled dataset CSV is available in `output/processed/` (from Machine 2)

**Commands:**
```bash
# Identify A1's labelled dataset (from Machine 2's output)
# Look for the pennystocks labelled CSV with tau=1.5, w_sent=0.5, seed=42
# Example: output/processed/2026-07-18_14-30_labelled_dataset.csv

python src/run_cross_validation.py \
    --model-dir output/models/A2 \
    --eval-data output/processed/2026-07-18_17-42_labelled_dataset.csv \
    --partition test \
    --output-dir output/evaluation \
    --notes "D1: WSB-trained models → pennystocks test set"
```

**Expected outputs:**
- `output/evaluation/<ts>_cross_validation.json`
- Entry in `output/experiment_log.jsonl`

**What this answers:** Whether WSB-trained models rank pennystocks surges correctly (AUC > 0.60 = partial generalisation).

**Estimated runtime:** ~5 min

---

### Step 4.3 — Experiment D2: Cross-Dataset (Pennystocks → WSB)

**Description:** Evaluate models trained on pennystocks (from A1, Machine 2) against the WSB labelled dataset (from A2, Machine 1). Tests the reverse transfer direction.

**Prerequisites:**
- `output/models/A1/` contains pennystocks-trained models (from Machine 2)
- A2's labelled dataset CSV is available in `output/processed/` (from Machine 1)

**Commands:**
```bash
# Identify A2's labelled dataset (from Machine 1's output)
python src/run_cross_validation.py \
    --model-dir output/models/A1 \
    --eval-data output/processed/2026-07-19_07-49_labelled_dataset.csv \
    --partition test \
    --output-dir output/evaluation \
    --notes "D2: Pennystocks-trained models → WSB test set"
```

**Expected outputs:**
- `output/evaluation/<ts>_cross_validation.json`
- Entry in `output/experiment_log.jsonl`

**What this answers:** Directional transfer asymmetry (does dense→sparse or sparse→dense generalise better?).

**Estimated runtime:** ~5 min

---

### Step 4.4 — Generate Evaluation Figures

**Description:** Generate evaluation figures for key experiments using `generate_figures.py`, which loads saved models and produces plots without retraining.

**Commands:**
```bash
# Generate figures for A2 (WSB baseline — the primary result)
python src/generate_figures.py \
    --data-path output/processed/2026-07-19_07-49_labelled_dataset.csv \
    --models-dir output/models/A2 \
    --phase phase2 \
    --seed 42 \
    --figures-dir output/figures/evaluation/A2

# Generate figures for A1 (Pennystocks baseline)
python src/generate_figures.py \
    --data-path output/processed/2026-07-18_17-42_labelled_dataset.csv \
    --models-dir output/models/A1 \
    --phase phase2 \
    --seed 42 \
    --figures-dir output/figures/evaluation/A1

# Generate figures for B1 (WSB Phase 1 — for comparison)
python src/generate_figures.py \
    --data-path output/processed/2026-07-19_08-36_labelled_dataset.csv \
    --models-dir output/models/B1 \
    --phase phase1 \
    --seed 42 \
    --figures-dir output/figures/evaluation/B1
```

**Note:** Each experiment gets its own `--figures-dir` subdirectory to prevent `11_roc_curves_combined.png` (which has no prefix) from being overwritten across runs.

**Expected outputs per experiment:**
- `output/figures/evaluation/{exp}/10_confusion_matrix_{model}.png` (×3 models)
- `output/figures/evaluation/{exp}/11_roc_curve_{model}.png` (×3 models)
- `output/figures/evaluation/{exp}/11_roc_curves_combined.png`
- `output/figures/evaluation/{exp}/12_classification_threshold_sensitivity_{model}.png` (×3 models)

**Estimated runtime:** ~5 min (no retraining — just loads models and generates plots)

---

### Step 4.5 — Verify Multi-Seed Results & Assemble Weight Sweep

**Description:** Collect evaluation metrics for robustness and weight sensitivity analyses.

**Multi-seed robustness (F2):**
```bash
# Collect AUC-ROC from 5 seeds on pennystocks:
# Seed 42 = A1 (Machine 2, step 2.1)
# Seeds 123, 456, 789, 2024 = F2 (Machine 2, step 2.4)
# Look in output/evaluation/ for each run's evaluation_metrics.json (match by --notes field in experiment_log.jsonl)
```

**Weight sensitivity sweep (G series):**
```
# 5 data points for the curve (w_sentiment → AUC-ROC):
# w_sent=0.0  → B1 result (Machine 1)
# w_sent=0.25 → G2 result (Machine 3)
# w_sent=0.5  → A2 result (Machine 1)
# w_sent=0.75 → G4 result (Machine 3)
# w_sent=1.0  → G5 result (Machine 3)
```

**Expected analysis output:**
- Robustness: "Best model achieves AUC X.XX ± Y.YY across 5 seeds on pennystocks"
- Weight sweep: table/plot of AUC vs w_sentiment showing optimal weighting
- Success criterion: multi-seed std < 0.03

---

## Success Criteria

| Tier | AUC-ROC | Interpretation |
|------|---------|----------------|
| Minimum | >= 0.60 | Better than random; pipeline works |
| Target | >= 0.70 | Meaningful predictive signal exists |
| Stretch | >= 0.80 | Strong discriminative performance |

**Report claims require:**
1. At least one model meets Target tier on each dataset → "surges are predictable"
2. Phase 2 outperforms Phase 1 on at least one dataset → "sentiment adds value"
3. Cross-dataset AUC > 0.60 → "partial generalisation demonstrated"
4. Multi-seed std < 0.03 → "results are reproducible"

---

## Summary: All Experiments by Machine

| Machine | Exp ID | Dataset | Key Config | Research Question |
|---------|--------|---------|------------|-------------------|
| 1 | A2 | WSB | baseline (τ=1.5, w=0.5/0.5, s=42) | Can surges be predicted on dense data? |
| 1 | B1 | WSB | w_sent=0.0 | Does sentiment improve over volume-only? |
| 1 | C1 | WSB | τ=1.0 | How does threshold choice affect performance? |
| 2 | A1 | pennystocks | baseline (τ=1.5, w=0.5/0.5, s=42) | Can surges be predicted on sparse data? |
| 2 | B3 | pennystocks | w_sent=0.0 | Does sentiment improve (sparse data)? |
| 2 | C2 | pennystocks | τ=1.0 | Threshold sensitivity (sparse data)? |
| 2 | F2×4 | pennystocks | seeds 123,456,789,2024 | Are results reproducible? (+ A1 for seed=42) |
| 3 | G2 | WSB | w=0.75/0.25 | Weight sensitivity sweep |
| 3 | G4 | WSB | w=0.25/0.75 | Weight sensitivity sweep |
| 3 | G5 | WSB | w=0.0/1.0 | Weight sensitivity sweep |
| Consol. | D1 | WSB→penny | cross-dataset | Do models generalise across communities? |
| Consol. | D2 | penny→WSB | cross-dataset | Directional transfer asymmetry? |

**Notes:**
- G1 (w=1.0/0.0) = B1 — reuse B1's AUC as the w_sent=0.0 data point
- G3 (w=0.5/0.5) = A2 — reuse A2's AUC as the w_sent=0.5 data point
- Weight sweep curve has 5 points: B1(0.0), G2(0.25), A2(0.5), G4(0.75), G5(1.0)
- F2 seed=42 = A1 — use A1's AUC as the fifth robustness data point
- E1/E2 (feature importance) is computed as part of A2/A1 training — no separate step needed

---

## Output Directory Structure (After Consolidation)

```
output/
├── processed/                          # All labelled datasets (timestamp-prefixed, no conflicts)
│   ├── <ts1>_labelled_dataset.csv     # A2 (Machine 1)
│   ├── <ts1>_pipeline_config.json
│   ├── <ts2>_labelled_dataset.csv     # B1 (Machine 1)
│   ├── <ts3>_labelled_dataset.csv     # C1 (Machine 1)
│   ├── <ts4>_labelled_dataset.csv     # A1 (Machine 2)
│   ├── <ts5>_labelled_dataset.csv     # B3 (Machine 2)
│   ├── ...                            # (all timestamp-prefixed)
│   └── latest_outputs.json            # (ignore — use explicit paths)
│
├── evaluation/                         # All evaluation results (timestamp-prefixed, no conflicts)
│   ├── <ts1>_evaluation_metrics.json  # A2
│   ├── <ts1>_threshold_tuning.json    # A2
│   ├── <ts1>_feature_importance.json  # A2 (includes E1)
│   ├── <ts1>_final_summary.json       # A2
│   ├── <ts2>_evaluation_metrics.json  # B1
│   ├── ...
│   └── <tsN>_cross_validation.json    # D1, D2
│
├── models/                             # Per-experiment model directories (no conflicts)
│   ├── A1/                            # Pennystocks baseline models (used for D2)
│   ├── A2/                            # WSB baseline models (used for D1)
│   ├── B1/                            # WSB phase1 models
│   ├── B3/                            # Pennystocks phase1 models
│   ├── C1/                            # WSB tau=1.0 models
│   ├── C2/                            # Pennystocks tau=1.0 models
│   ├── F2_seed123/                    # Pennystocks robustness
│   ├── F2_seed456/
│   ├── F2_seed789/
│   ├── F2_seed2024/
│   ├── G2/                            # Weight sweep
│   ├── G4/
│   └── G5/
│
├── figures/                            # Generated during consolidation only
│   ├── eda/
│   └── evaluation/
│
├── logs/                               # All CLI logs (timestamped, no conflicts)
│
└── experiment_log.jsonl               # Merged from all 3 machines
```

---

## Timing Estimates

| Machine | Sequential Runtime | Parallelism Benefit |
|---------|-------------------|---------------------|
| Machine 1 | ~6 hours | Runs simultaneously with M2 and M3 |
| Machine 2 | ~2.5 hours | Finishes first |
| Machine 3 | ~6 hours | Tied with M1 for longest |
| Consolidation | ~30 min | After all machines finish |

**Wall-clock time (parallel):** ~6.5 hours (limited by Machine 1 and Machine 3)
**Wall-clock time (sequential, single machine):** ~15 hours

**Parallelism savings:** ~8.5 hours (~57% reduction)

---

## Batch Scripts

Executable scripts are in `admin/`:
- `admin/m1.sh` — Machine 1 (A2, B1, C1)
- `admin/m2.sh` — Machine 2 (A1, B3, C2, F2)
- `admin/m3.sh` — Machine 3 (G2, G4, G5)

**Run from project root using Git Bash:**
```bash
bash admin/m1.sh
```

---

## Consolidation Checklist

After all machines finish:

- [ ] Copy all `output/` trees to consolidation machine
- [ ] Verify no filename collisions in `output/processed/` and `output/evaluation/`
- [ ] Merge `experiment_log.jsonl` files (cat + sort by timestamp)
- [ ] Run D1: WSB models (A2) → pennystocks eval
- [ ] Run D2: Pennystocks models (A1) → WSB eval
- [ ] Generate figures using `generate_figures.py` for A1, A2, B1
- [ ] Collect F2 multi-seed AUC values (A1 as seed=42 + seeds 123,456,789,2024 from F2)
- [ ] Compute mean ± std for robustness reporting
- [ ] Assemble weight sensitivity curve: B1(w_sent=0.0), G2(0.25), A2(0.5), G4(0.75), G5(1.0)
- [ ] Compare B1 vs A2 (Phase 1 vs Phase 2 on WSB)
- [ ] Compare B3 vs A1 (Phase 1 vs Phase 2 on pennystocks)
- [ ] Compare C1 vs A2 (τ=1.0 vs τ=1.5 on WSB)
- [ ] Compare C2 vs A1 (τ=1.0 vs τ=1.5 on pennystocks)
