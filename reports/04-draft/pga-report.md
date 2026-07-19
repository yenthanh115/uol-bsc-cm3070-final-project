# Prototype Surge Detection Pipeline for Reddit Financial Communities

## Template

This project uses the **Data Science** template (CM3070 Final Project).

## Project Overview

This prototype implements a machine learning pipeline that detects emerging "surge" trends in Reddit penny stock discussions. The system processes historical submission data from r/wallstreetbets (1.29 million records) and r/pennystocks (80,212 records), computing temporal features and sentiment scores to predict whether a given ticker-mention represents the start of a coordinated attention surge.

The prototype constitutes the core deliverable of the project: a complete, end-to-end pipeline from raw data ingestion through to model evaluation. It addresses the research question: *Can social media posting patterns be used to predict emerging surge behaviour in penny stock communities, and does incorporating sentiment alongside volume improve detection accuracy?*

## Features Implemented

The prototype implements a five-stage pipeline:

1. **Data Loading and Ticker Extraction** — parses Reddit CSV exports, normalises timestamps, and extracts stock ticker symbols from post titles.
2. **Temporal Windowing** — computes forward and backward post counts within configurable time windows (24h) per ticker.
3. **Sentiment Analysis** — applies VADER sentiment scoring to post text, producing polarity scores at observation time.
4. **Composite Surge Labelling** — z-score normalises volume and sentiment metrics using training-partition statistics only, then combines them into a composite score with a configurable threshold τ for binary surge classification.
5. **Model Training and Evaluation** — trains three classifiers (Logistic Regression, Random Forest, XGBoost) using expanding-window temporal cross-validation, evaluates with AUC-ROC, bootstrap confidence intervals, McNemar's pairwise significance tests, and classification threshold tuning.

### Prediction Features (11 total)

The pipeline engineers 11 backward-only ML features, ensuring no temporal leakage from future information:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `sentiment_score` | VADER polarity at observation time |
| 2 | `hour_of_day` | Hour (0–23) of post creation |
| 3 | `day_of_week` | Day (0=Mon–6=Sun) of post creation |
| 4 | `time_since_previous` | Hours since last post mentioning same ticker |
| 5 | `ticker_post_rate_24h` | Backward 24h post count for the ticker |
| 6 | `ticker_post_acceleration` | Ratio of recent-12h to older-12h activity |
| 7 | `word_count` | Token count of title + body text |
| 8 | `title_length` | Token count of title only |
| 9 | `num_tickers_mentioned` | Distinct tickers in the original post |
| 10 | `word_count_x_hour` | Interaction: word_count × hour_of_day |
| 11 | `accel_x_time_since_prev` | Interaction: acceleration × time since previous |

Features 1–9 capture temporal, textual, and community signals. Features 10–11 are interaction terms that model combined effects (e.g., long posts at peak hours, rapid acceleration after a period of silence).

## Algorithms, Techniques and Methods

### Composite Surge Metric

The core labelling mechanism uses z-score normalisation followed by a weighted composite score. Given a record's volume growth $g_v$ and sentiment change $g_s$, training-partition statistics $\mu_v, \sigma_v, \mu_s, \sigma_s$ are used to compute:

$$z_v = \frac{g_v - \mu_v}{\sigma_v}, \quad z_s = \frac{g_s - \mu_s}{\sigma_s}$$

The composite metric combines these with configurable weights:

$$C = w_v \cdot z_v + w_s \cdot z_s$$

A record is labelled as a surge if $C > \tau$, where $\tau$ is the threshold parameter (default 1.5). Crucially, normalisation statistics are computed from the training partition only, preventing data leakage from the test set.

### Expanding-Window Temporal Cross-Validation

Standard k-fold cross-validation is inappropriate for time-series data because it allows future information to leak into training. Instead, the pipeline divides training data into k=4 chronological folds and creates expanding-window splits: split $i$ trains on folds $0..i$ and validates on fold $i+1$. This produces 3 train/validation splits that respect temporal ordering.

The following diagram illustrates the expanding-window scheme:

```
Fold:   [  1  ][  2  ][  3  ][  4  ]  ← chronological order
Split 1: TRAIN   VAL
Split 2: TRAIN   TRAIN  VAL
Split 3: TRAIN   TRAIN  TRAIN  VAL
```

### Class Imbalance Handling

The surge class is extremely rare (1.4% on WSB, 2.8% on pennystocks). The pipeline addresses this through: (1) `class_weight="balanced"` for Logistic Regression and Random Forest, which inversely weights classes by frequency; and (2) `scale_pos_weight` for XGBoost, which applies a configurable multiplier to the minority class loss contribution.

### Classification Threshold Tuning

Rather than using the default 0.5 probability threshold, the pipeline sweeps thresholds from 0.01 to 0.99 on a validation fold and selects the operating point that maximises F1-score. This is essential given the extreme class imbalance — the default threshold produces near-zero precision.

## Code Explanation

### Pipeline Orchestration

The pipeline is controlled by a `PipelineConfig` dataclass that captures all parameters (threshold, weights, seed, paths) and serialises to JSON for reproducibility:

```python
@dataclass
class PipelineConfig:
    threshold_tau: float = 1.5
    weight_volume: float = 0.5
    weight_sentiment: float = 0.5
    temporal_split_ratio: float = 0.8
    random_seed: int = 42
    # ... additional parameters
```

Every experiment run saves its configuration alongside results, enabling exact reproduction of any historical experiment.

### Feature Engineering (Backward-Only)

All 11 features are computed using only information available at or before observation time, preventing temporal leakage. The most important feature — `ticker_post_rate_24h` — simply reuses the backward window count from the windowing stage:

```python
# Feature 5: ticker_post_rate_24h (backward-only)
df = df.assign(ticker_post_rate_24h=df["backward_count"].values.copy())

# Feature 6: ticker_post_acceleration (ratio of recent vs older activity)
# Uses searchsorted on sorted timestamps for O(n log n) computation
ticker_post_acceleration = _compute_ticker_post_acceleration(df, created_utc)
```

The acceleration feature captures whether posting about a ticker is accelerating (ratio of 12h-recent to 12h-older backward counts), providing the model with momentum information beyond raw volume.

### Temporal Cross-Validation and Model Training

The training module performs grid search across all hyperparameter configurations using the expanding-window splits. For each configuration, it trains the model on each expanding window and averages validation AUC scores:

```python
for params in param_grid:
    fold_aucs = []
    for train_idx, val_idx in splits:
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X_train_full[train_idx])
        X_fold_val = scaler.transform(X_train_full[val_idx])

        model = make_model_fn(params, random_seed)
        model.fit(X_fold_train, y_fold_train)

        y_prob = model.predict_proba(X_fold_val)[:, 1]
        auc = roc_auc_score(y_fold_val, y_prob)
        fold_aucs.append(auc)

    mean_auc = np.mean(fold_aucs)
    if mean_auc > best_mean_auc:
        best_mean_auc = mean_auc
        best_params = params
```

The best configuration is retrained on the full training partition before final test-set evaluation. This ensures the model has seen all available training data while hyperparameters were selected without test-set contamination.

### Statistical Evaluation

McNemar's pairwise test compares models at the record level by building a 2×2 contingency table of correct/incorrect predictions for each model pair, with Bonferroni correction for multiple comparisons:

```python
# Record-level correct/incorrect comparison
correct_a = (predictions[name_a] == y_true)
correct_b = (predictions[name_b] == y_true)

# Cells: b = A wrong & B right; c = A right & B wrong
b = int(((~correct_a) & correct_b).sum())
c = int((correct_a & (~correct_b)).sum())
```

## Results

### Model Performance

The pipeline was evaluated on two Reddit communities with 14 experiments across 3 machines:

| Dataset | Model | AUC-ROC | Tier |
|---------|-------|---------|------|
| r/wallstreetbets (τ=1.0) | XGBoost | 0.892 | Stretch |
| r/wallstreetbets (τ=1.0) | Random Forest | 0.876 | Stretch |
| r/wallstreetbets (τ=1.5) | XGBoost | 0.862 | Stretch |
| r/wallstreetbets (τ=1.5) | Random Forest | 0.854 | Stretch |
| r/pennystocks (τ=1.5) | Random Forest | 0.746 | Target |
| r/pennystocks (τ=1.5) | XGBoost | 0.734 | Target |

All models significantly outperform the random baseline (AUC=0.50) with p < 0.001 (McNemar's test, Bonferroni-corrected).

### Cross-Dataset Transfer

Models trained on WSB and evaluated on pennystocks (D1) achieved AUC 0.676, demonstrating partial generalisability. The reverse transfer (D2: pennystocks→WSB) performed better at AUC 0.871, suggesting patterns learned from the sparse community transfer well to the dense community.

### Robustness

Multi-seed experiments (5 seeds: 42, 123, 456, 789, 2024) produced standard deviation of 0.008, confirming result stability.

### Key Finding: Sentiment Improves Detection

The Phase 1 (volume-only) vs Phase 2 (composite volume+sentiment) comparison shows sentiment adds +0.18 AUC on WSB, validating the composite approach. A weight sensitivity sweep confirmed that balanced 50/50 weighting is optimal.

> **Note:** ROC curves, confusion matrices, and threshold sensitivity plots are generated automatically by the pipeline and stored in `output/figures/evaluation/`. Representative figures are included on the following pages.

## Evaluation and Improvements

### Successes

The prototype successfully achieves the project's "stretch" success criterion (AUC > 0.80) on r/wallstreetbets with both XGBoost and Random Forest. The pipeline is fully reproducible — configuration serialisation, fixed seeds, and timestamped outputs ensure any experiment can be exactly replicated. The modular architecture allows individual stages to be swapped or extended independently.

### Limitations

1. **Extreme class imbalance** — the surge rate is only 1.4–2.8%, making precision at any threshold very low. Even the best model achieves practical precision of only ~4% at default settings.
2. **VADER sentiment limitations** — feature importance analysis shows `sentiment_score` ranks low across all models. A more domain-specific financial sentiment model could improve this.
3. **Single-subreddit training** — cross-dataset transfer is asymmetric and imperfect, suggesting community-specific patterns that don't fully generalise.

### Planned Improvements

1. **Domain-specific sentiment** — replace VADER with a financial sentiment model (e.g., FinBERT) that understands stock-specific language.
2. **Threshold tuning on validation set** — current experiments show threshold tuning dramatically improves F1 (e.g., XGBoost F1 from 0.00 at default to 0.12 at tuned threshold of 0.16). Systematic per-dataset tuning should be part of the final pipeline.
3. **Additional features** — incorporate cross-ticker correlation features and subreddit-level activity baselines to capture community-wide momentum shifts.
4. **Real-time inference** — extend the batch pipeline to support streaming Reddit data with online prediction, enabling practical early-warning alerts.
