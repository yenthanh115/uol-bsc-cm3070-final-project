# Experiment Plan — From Scratch

## Overview

This plan assumes **zero prior results**. All experiments will be run fresh using the two selected datasets and the current pipeline codebase.

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

## Experiment Schedule

### Phase A: Baseline Runs (Both Datasets, Default Settings)

| ID | Dataset | Config | Purpose | Est. Runtime |
|----|---------|--------|---------|--------------|
| A1 | r/pennystocks | τ=1.5, w_vol=0.5, w_sent=0.5, seed=42 | Label + train + evaluate on sparse dataset | ~15 min |
| A2 | r/wallstreetbets | τ=1.5, w_vol=0.5, w_sent=0.5, seed=42 | Label + train + evaluate on dense dataset | ~2 hours |

**Commands:**
```bash
# A1: Pennystocks labelling
python run_labeling.py --file-path ../input/raw/r_pennystocks_submissions_reddit.csv

# A1: Pennystocks training
python run_training.py --data-path <A1_labelled_output>

# A2: WSB labelling
python run_labeling.py --file-path ../input/raw/r_wallstreetbets_submissions_reddit.csv

# A2: WSB training
python run_training.py --data-path <A2_labelled_output>
```

**Expected outputs per run:**
- Labelled dataset CSV (with surge_label, partition, features)
- Pipeline config JSON (audit trail)
- Threshold sensitivity sweep CSV
- Evaluation metrics JSON (per-model P/R/F1/AUC)
- Threshold tuning JSON (optimal thresholds on validation fold)
- Final summary JSON (success tiers, McNemar's, bootstrap CIs)
- Evaluation figures (confusion matrices, ROC curves, threshold sensitivity)
- Saved models (.joblib)

**What this answers:**
- Baseline AUC-ROC for each model on each dataset
- Whether surges can be predicted from backward-looking features alone
- Usable record counts / exclusion rates / surge rates per dataset

---

### Phase B: Research Question — Phase 1 vs Phase 2 (Sentiment Contribution)

The core research question: *does adding sentiment (Phase 2) improve over volume-only (Phase 1)?*

| ID | Dataset | Config Change | Purpose | Est. Runtime |
|----|---------|--------------|---------|--------------|
| B1 | r/wallstreetbets | `w_sentiment=0.0` (volume-only labelling) | Phase 1 baseline — pure volume | ~2 hours |
| B2 | r/wallstreetbets | `w_sentiment=0.5` (same as A2) | Phase 2 — volume + sentiment | (reuse A2) |
| B3 | r/pennystocks | `w_sentiment=0.0` | Phase 1 on sparse dataset | ~15 min |
| B4 | r/pennystocks | `w_sentiment=0.5` | Phase 2 on sparse dataset | (reuse A1) |

**Commands:**
```bash
# B1: WSB volume-only labelling
python run_labeling.py --file-path ../input/raw/r_wallstreetbets_submissions_reddit.csv --weight-sentiment 0.0

# B1: WSB volume-only training
python run_training.py --data-path <B1_labelled_output>

# B3: Pennystocks volume-only labelling
python run_labeling.py --file-path ../input/raw/r_pennystocks_submissions_reddit.csv --weight-sentiment 0.0

# B3: Pennystocks volume-only training
python run_training.py --data-path <B3_labelled_output>
```

**Analysis:**
- Compare AUC-ROC of each model: Phase 1 (B1) vs Phase 2 (A2) on WSB
- Compare AUC-ROC: Phase 1 (B3) vs Phase 2 (A1) on pennystocks
- McNemar's test between Phase 1 and Phase 2 best models
- Determines whether sentiment contributes signal beyond volume alone

---

### Phase C: Threshold Sensitivity (Operating Point Selection)

| ID | Dataset | Config Change | Purpose | Est. Runtime |
|----|---------|--------------|---------|--------------|
| C1 | r/wallstreetbets | τ=1.0 | Lower threshold → more surges, easier task | ~2 hours |
| C2 | r/pennystocks | τ=1.0 | Compare threshold effect on sparse data | ~15 min |

**Commands:**
```bash
# C1
python run_labeling.py --file-path ../input/raw/r_wallstreetbets_submissions_reddit.csv --threshold-tau 1.0
python run_training.py --data-path <C1_labelled_output>

# C2
python run_labeling.py --file-path ../input/raw/r_pennystocks_submissions_reddit.csv --threshold-tau 1.0
python run_training.py --data-path <C2_labelled_output>
```

**Analysis:**
- How surge rate changes with τ (τ=1.0 yields ~8% vs τ=1.5 yields ~1.4-2.8%)
- Whether higher surge rate improves model performance
- Whether the pipeline remains effective under different class balance regimes

---

### Phase D: Cross-Dataset Generalisation

| ID | Source | Target | Purpose | Est. Runtime |
|----|--------|--------|---------|--------------|
| D1 | WSB-trained models (from A2) | r/pennystocks labelled (from A1) | Transfer: dense → sparse | ~5 min |
| D2 | Pennystocks-trained models (from A1) | r/wallstreetbets labelled (from A2) | Transfer: sparse → dense | ~5 min |

**Commands:**
```bash
# D1: WSB models → pennystocks evaluation
python run_cross_validation.py \
    --model-dir ../output/models \
    --eval-data <A1_labelled_output> \
    --notes "D1: WSB-trained → pennystocks"

# D2: Pennystocks models → WSB evaluation
# (Need to swap saved models — save A1 models separately first)
python run_cross_validation.py \
    --model-dir ../output/models_pennystocks \
    --eval-data <A2_labelled_output> \
    --notes "D2: Pennystocks-trained → WSB"
```

**Analysis:**
- Whether surge dynamics are community-universal or community-specific
- Directional asymmetry (does dense→sparse transfer better than sparse→dense?)
- Supports generalisation claims in the report

---

### Phase E: Feature Importance (P3)

| ID | Dataset | Purpose | Est. Runtime |
|----|---------|---------|--------------|
| E1 | r/wallstreetbets (from A2) | Built-in importance (RF/XGB) + permutation importance | ~30 min |
| E2 | r/pennystocks (from A1) | Same analysis on sparse data | ~10 min |

This is computed as part of `run_training.py` — the evaluation module already produces feature importance rankings. No separate command needed; it's part of A1/A2 outputs.

**Analysis:**
- Which features drive predictions (temporal vs textual vs interaction)
- Whether the same features matter across datasets
- Validates design decisions (are interaction features worthwhile?)

---

### Phase F: Multi-Seed Robustness

| ID | Dataset | Seeds | Purpose | Est. Runtime |
|----|---------|-------|---------|--------------|
| F1 | r/wallstreetbets | 42, 123, 456, 789, 2024 | Variance of results across seeds | ~10 hours |
| F2 | r/pennystocks | 42, 123, 456, 789, 2024 | Same for sparse dataset | ~1.5 hours |

**Commands:**
```bash
# For each seed in [42, 123, 456, 789, 2024]:
python run_labeling.py --file-path ../input/raw/r_wallstreetbets_submissions_reddit.csv --random-seed <seed>
python run_training.py --data-path <labelled_output> --random-seed <seed>
```

**Analysis:**
- Mean ± std of AUC-ROC across 5 seeds per model
- Confirms that results are not an artifact of a lucky split
- Reports robustness: "XGBoost achieves AUC 0.88 ± 0.01 across 5 seeds"

---

### Phase G: Weight Sensitivity Sweep (Optional)

| ID | Dataset | Weights (w_vol, w_sent) | Purpose |
|----|---------|------------------------|---------|
| G1–G5 | r/wallstreetbets | (1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0) | Full sensitivity curve |

**Analysis:**
- How the volume/sentiment weighting affects classification performance
- Identifies optimal weighting (or confirms equal weighting is reasonable)
- Lower priority — run only if time permits

---

## Execution Order (Critical Path)

```
A1 (pennystocks baseline)  ──┐
                              ├──→ D1, D2 (cross-dataset)
A2 (WSB baseline)          ──┘
        │
        ├──→ B1 (WSB Phase 1)  ──→ Compare B1 vs A2
        │
        ├──→ B3 (Pennystocks Phase 1) ──→ Compare B3 vs A1
        │
        ├──→ C1 (WSB τ=1.0)
        │
        └──→ E1, E2 (feature importance — from A1/A2 outputs)

F1, F2 (multi-seed) ← run in parallel / overnight
G1–G5 (weight sweep) ← optional, lowest priority
```

**Total estimated runtime (critical path):** ~4.5 hours for A+B+C+D+E
**Multi-seed addition:** +11.5 hours (can run overnight)

---

## Success Criteria

| Tier | AUC-ROC | Interpretation |
|------|---------|----------------|
| Minimum | ≥ 0.60 | Better than random; pipeline works |
| Target | ≥ 0.70 | Meaningful predictive signal exists |
| Stretch | ≥ 0.80 | Strong discriminative performance |

**Report claims require:**
1. At least one model meets Target tier on each dataset → "surges are predictable"
2. Phase 2 outperforms Phase 1 on at least one dataset → "sentiment adds value"
3. Cross-dataset AUC > 0.60 → "partial generalisation demonstrated"
4. Multi-seed std < 0.03 → "results are reproducible"

---

## File/Model Management

Each experiment generates timestamped output files. To support cross-dataset evaluation (Phase D), models from different datasets need separate storage:

```
output/models/              ← models from most recent training
output/models_wsb/          ← copy WSB models here after A2
output/models_pennystocks/  ← copy pennystocks models here after A1
```

---

## Summary Table of All Experiments

| ID | Dataset | Key Variable | Research Question Addressed |
|----|---------|-------------|---------------------------|
| A1 | pennystocks | baseline | Can surges be predicted on sparse data? |
| A2 | WSB | baseline | Can surges be predicted on dense data? |
| B1 | WSB | w_sent=0 | Does sentiment improve over volume-only? |
| B3 | pennystocks | w_sent=0 | Same question on sparse data |
| C1 | WSB | τ=1.0 | How does threshold choice affect performance? |
| C2 | pennystocks | τ=1.0 | Same on sparse data |
| D1 | WSB→pennystocks | transfer | Do models generalise across communities? |
| D2 | pennystocks→WSB | transfer | Directional transfer asymmetry? |
| E1 | WSB | importance | Which features drive predictions? |
| E2 | pennystocks | importance | Same question on sparse data |
| F1 | WSB | 5 seeds | Are results reproducible? |
| F2 | pennystocks | 5 seeds | Same on sparse data |
| G1–5 | WSB | weight sweep | What's the optimal vol/sent weighting? |
