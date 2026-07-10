Here's what the evaluation output tells us:

---

## Model Performance

| Model | AUC-ROC | Accuracy | Precision | Recall | F1 | Tier |
|-------|---------|----------|-----------|--------|-----|------|
| Random Forest | 0.770 | 42.6% | 1.98% | 85.7% | 3.87% | Target |
| XGBoost | 0.707 | 98.7% | 0.0% | 0.0% | 0.0% | Target |
| Logistic Regression | 0.646 | 43.0% | 1.67% | 71.4% | 3.27% | Minimum |

**What's going on here:**

The test set has 519 samples with only **7 surges** (1.35% positive rate). This extreme imbalance explains the metric split:

- **AUC-ROC is meaningful** (0.65–0.77): The models can *rank* surge cases higher than non-surge cases. Random Forest does this best.
- **Precision is near-zero**: When LR/RF predict "surge", they're almost always wrong because they set aggressive thresholds to catch the rare positives. Out of ~300 predicted surges, only 5–6 are correct.
- **Recall is high for LR/RF** (71–86%): They find most of the 7 actual surges — but at the cost of massive false positives.
- **XGBoost predicts all-negative**: It learned that predicting "no surge" every time gives 98.7% accuracy. Its AUC (0.707) shows it *internally* ranks surges higher, but its default threshold never triggers.

---

## Baseline Comparisons

The single-feature baselines reveal something important:

| Feature | AUC (single-feature LR) |
|---------|------------------------|
| **word_count** | **0.809** |
| hour_of_day | 0.745 |
| num_tickers_mentioned | 0.731 |
| title_length | 0.677 |
| ticker_post_acceleration | 0.596 |
| day_of_week | 0.579 |
| time_since_previous | 0.562 |
| ticker_post_rate_24h | 0.561 |
| sentiment_score | 0.465 |

**Key finding:** A single-feature LR using just `word_count` achieves AUC=0.809 — *better than all three multi-feature models*. This means:
1. Word count is a very strong predictor (longer posts get more attention → more surge)
2. The multi-feature models aren't capturing synergies well, possibly hurt by noise from weaker features
3. Sentiment contributes essentially nothing (AUC=0.465, below random)

---

## McNemar's Test

- LR vs RF: **not significant** (p=0.83) — they make similar errors
- LR vs XGBoost: **significant** (p≈0) — fundamentally different prediction strategies
- RF vs XGBoost: **significant** (p≈0) — same

---

## Implications for the Report

This is honest, reportable output. The discussion should address:
1. **Why word_count dominates** — likely a proxy for post effort/quality, not directly causal
2. **Why sentiment fails** — VADER compound score may not capture Reddit-specific financial sentiment meaningfully
3. **Class imbalance problem** — 7 test surges is too few for reliable precision/recall; AUC is the robust metric here
4. **Multi-feature models underperform single-feature** — a finding worth discussing (overfitting? noise features diluting signal?)