# Preliminary Report

## 1. Project Definition

### 1.1 Title

Predicting Engagement and Sentiment Surges in Stock-Related Social Media Discussions

This project follows **Template 2: Predictive Modelling Prototype.**

### 1.2 Objectives

- Develop a predictive model to forecast whether a stock-related social media discussion will experience a significant engagement and sentiment surge within 24 hours
- Engineer features from raw discussion data including temporal, textual, activity-frequency, and sentiment signals
- Compare traditional ML approaches (Logistic Regression, Random Forest, XGBoost) for binary surge classification
- Evaluate model performance using standard classification metrics (accuracy, precision, recall, F1, AUC-ROC)

### 1.3 Problem Statement

Financial discussions on social media platforms often experience sudden increases in posting activity and emotional intensity. These surges can develop within hours, making them difficult to anticipate through manual monitoring. Stakeholders — financial analysts, market surveillance teams, quantitative researchers, and platform moderators — need early warning of discussions gaining momentum around specific stocks.

Existing research focuses on predicting eventual popularity [1][3] or sentiment-to-market correlations [4], rather than forecasting whether a specific stock's discussion is about to surge within a bounded time window.

### 1.4 Unit of Analysis

The prediction operates at the **record level** (one Reddit submission) but measures surges **scoped to the same ticker**. For a record mentioning ticker $X at time *t*, the system asks: "Will discussion about $X experience a surge within the next 24 hours?"

Subsequent records used in surge computation are those mentioning the same ticker within *(t, t + 24h]*. Records without identifiable tickers are excluded. A minimum record count (N ≥ 3) within the ticker window is enforced to ensure metric stability.

### 1.5 Surge Definition

A **surge** is a statistically significant increase in the composite posting-volume-and-sentiment metric for a specific ticker within a 24-hour window:

1. **Posting volume growth**: *(count of $X posts in (t, t + 24h]) / max(count in (t − 24h, t], 1)) − 1*
2. **Sentiment change**: *|mean(future sentiments) − current sentiment|*
3. **Standardise** both components using z-scores (training-set μ and σ)
4. **Combine**: *composite = w₁ × z_volume + w₂ × z_sentiment* (default w₁ = w₂ = 0.5)
5. **Label**: surge (1) if composite > threshold *τ*, else no-surge (0)

Posting volume (derived from timestamps) is used instead of engagement scores (score, num_comments) because those are final snapshot values that would introduce circular dependency.

The threshold *τ* will be determined empirically during EDA, targeting a surge rate of approximately 5–10%.

**Phased approach:**
- **Phase 1** — Volume-only target (w₂ = 0) establishes a baseline
- **Phase 2** — Full composite (w₂ = 0.5) tests whether sentiment adds predictive value

### 1.6 Motivation

- Rapid posting surges with sentiment shifts often influence information diffusion and investor behaviour [4]
- Thousands of daily financial posts make manual surge identification impractical
- Early detection enables proactive monitoring for financial risk assessment and market surveillance
- The 2021 GameStop event demonstrated real market impact from social media discussion surges [5]
- Addresses a gap in short-term composite surge prediction integrating both engagement and sentiment signals

---

## 2. Scope and Boundaries

### 2.1 In Scope

- Pre-collected static dataset of stock-related Reddit discussions (CSV)
- Feature engineering: temporal, textual, sentiment, and activity-frequency features
- Binary classification: surge (1) vs no-surge (0) within 24-hour window
- Traditional ML models: Logistic Regression, Random Forest, XGBoost
- Standard evaluation metrics and visualisations
- Reproducible pipeline with seeded randomness

### 2.2 Dataset

The **Reddit Finance Data** dataset (Kaggle) contains submissions from nine stock-related subreddits over calendar year 2021, totalling ~1.38 million records. Each record includes post ID, timestamp, title, body text, engagement metrics (score, num_comments), subreddit, and extracted ticker symbols.

| Subreddit | Records | Tickers | Mean Score | Selftext Missing |
|-----------|---------|---------|-----------|-----------------|
| wallstreetbets | 775,326 | 4,451 | 116.0 | 33.8% |
| gme | 273,327 | 340 | 101.3 | 48.7% |
| stocks | 75,857 | 1,963 | 30.2 | 0.1% |
| StockMarket | 72,620 | 1,597 | 6.3 | 29.2% |
| pennystocks | 54,785 | 2,912 | 29.4 | 20.7% |
| stockmarket | 43,809 | 1,475 | 35.0 | 39.1% |
| investing | 41,912 | 990 | 17.8 | 0.0% |
| robinhoodpennystocks | 23,304 | 855 | 30.8 | 36.0% |
| robinhood | 18,893 | 294 | 5.5 | 25.3% |

**Primary subset:** `pennystocks` (54,785 records, 2,912 tickers) — selected for high data completeness, sufficient volume, and diverse ticker coverage. Secondary validation on `wallstreetbets` and `gme` for generalisability testing.

Preliminary analysis confirms viable surge definitions exist: at the target operating point, surge rates of 5–9% produce imbalance ratios of 10:1 to 18:1.

### 2.3 Out of Scope

- Real-time data ingestion or API streaming
- Production deployment
- Trading signals or financial advice
- Multi-class or regression targets
- Cross-platform data fusion

---

## 3. Proposed Methodology

### 3.1 Data Pipeline Stages

1. **Data Loading** — Read static dataset from disk (CSV)
2. **Preprocessing** — Deduplicate, parse timestamps, normalise text, remove nulls
3. **Feature Engineering** — Compute sentiment, temporal, activity-frequency, and text features
4. **Target Labelling** — Compute composite surge target using 24-hour prediction window
5. **Model Training** — Train LR, RF, XGBoost with temporal train-test split
6. **Evaluation** — Compute metrics, generate confusion matrices and ROC curves

### 3.2 Tools and Technologies

- Python 3.10+
- pandas, numpy, scikit-learn, XGBoost
- TextBlob (sentiment analysis)
- matplotlib (visualisation)
- Jupyter notebooks (exploration)
- pytest + Hypothesis (testing)

### 3.3 System Architecture

The pipeline follows a linear staged architecture with a centralised configuration module. All stages produce deterministic outputs via seeded randomness.

<figure align="center">
  <img src="figures/02-data-pipeline-v0.1.png" alt="Data Pipeline" width="1000">
  <figcaption>Figure 2: Data Pipeline.</figcaption>
</figure>

The system is implemented as a Python package (`surge_pipeline`) with a CLI entry point, supporting both notebook-based exploration and script-based batch execution.

### 3.4 Feature Design

Only information available at observation time *t* may be used as features. The dataset's engagement metrics (score, num_comments) are final snapshot values and are excluded to prevent data leakage.

| Feature | Type | Description |
|---------|------|-------------|
| `sentiment_score` | Continuous [-1, 1] | TextBlob polarity of post text |
| `hour_of_day` | Discrete [0–23] | Hour when the post was created |
| `day_of_week` | Discrete [0–6] | Day when the post was created |
| `time_since_previous` | Continuous ≥ 0 | Hours since previous post mentioning the same ticker |
| `ticker_post_rate_24h` | Continuous ≥ 0 | Posts mentioning this ticker in the prior 24 hours |
| `ticker_post_acceleration` | Continuous | Ratio of post count in prior 12h to prior 12–24h |
| `word_count` | Discrete ≥ 0 | Token count of post body text |
| `title_length` | Discrete ≥ 0 | Token count of post title |
| `num_tickers_mentioned` | Discrete ≥ 1 | Count of distinct tickers in the post |

All features use backward-looking or creation-time information only. Activity-frequency features are correlated with the target by design (current momentum predicts future momentum) but introduce no temporal leakage.

---

## 4. Evaluation Strategy

### 4.1 Performance Metrics

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of predicted surges that are actual surges |
| Recall | Proportion of actual surges correctly predicted |
| F1-Score | Harmonic mean of precision and recall |
| AUC-ROC | Discriminative ability across all thresholds |

Given expected class imbalance, AUC-ROC and F1-Score are prioritised over raw accuracy.

### 4.2 Train-Test Split

The dataset is split using **temporal ordering** (not random sampling) to prevent data leakage:

- Records sorted chronologically; first 80% for training, remaining 20% for testing
- No future information leaks into training, reflecting realistic deployment conditions

### 4.3 Hyperparameter Tuning

Expanding-window temporal cross-validation (k=4 folds, 3 validation splits) within the training partition. Best configuration selected by mean validation AUC-ROC, then retrained on the full training set.

| Model | Tuned hyperparameters |
|-------|----------------------|
| Logistic Regression | C, penalty type (L1/L2) |
| Random Forest | n_estimators, max_depth, min_samples_leaf |
| XGBoost | n_estimators, max_depth, learning_rate, subsample, colsample_bytree |

### 4.4 Baselines

- **Random baseline** — AUC-ROC of 0.5
- **Majority-class baseline** — always predicting "no surge"
- **Single-feature baselines** — individual features as lone predictors

A model demonstrates meaningful signal if AUC-ROC > 0.60 on the test set.

### 4.5 Statistical Robustness

- 95% confidence intervals via bootstrap resampling (1,000 iterations)
- Multiple-seed evaluation (5 seeds) to assess initialisation sensitivity
- McNemar's test for pairwise model comparison (α = 0.05, Bonferroni-corrected)

### 4.6 Sensitivity Analysis

- **Threshold sweep**: τ ∈ {0.5, 1.0, 1.5, 2.0, 2.5} standard deviations — targeting 5–10% surge rate at the recommended operating point
- **Weight sweep**: w₂ ∈ {0, 0.25, 0.5, 0.75, 1.0} to assess sentiment's marginal contribution
- Phase 1 (w₂ = 0) vs Phase 2 (w₂ = 0.5) comparison quantifies the value of composite targets

---

## 5. Risk Register

*L = Likelihood, I = Impact. H = High, M = Medium, L = Low, C = Critical.*

| # | Risk | L | I | Mitigation | Status |
|---|------|---|---|------------|--------|
| 1 | Class imbalance (few surge events) | H | H | Stratified evaluation, SMOTE/class weighting, precision-recall curves | Open |
| 2 | TextBlob sentiment accuracy | M | M | Document limitations; FinBERT as alternative if time allows | Open |
| 3 | Data quality (missing fields, noise) | M | M | Robust preprocessing with logging; document exclusion criteria | Open |
| 4 | Temporal data leakage | M | H | Strict temporal split; no future data in features or labels | Open |
| 5 | Overfitting | M | H | Temporal CV, regularisation, report train vs test gaps | Open |
| 6 | Threshold/weight sensitivity | M | M | Sensitivity analysis across τ and w₂ configurations | Open |
| 7 | Per-ticker sparsity | H | H | Minimum record count (N ≥ 3) in ticker window; report exclusion rate | Open |
| 8 | Temporal concept drift | M | M | Report train/test surge rate differences; acknowledge as limitation | Open |
| 9 | Reproducibility failures | L | M | Pin dependencies, fixed random seeds, documented setup | Open |
| 10 | Snapshot engagement values (leakage) | H | C | Eliminated — features and target use only timestamps and text | Mitigated |

---

## 6. Project Plan and Timeline

<figure align="center">
  <img src="figures/01-gantt-chart-v0.1.png" alt="Project Timeline" width="1000">
  <figcaption>Figure 1: Project Timeline.</figcaption>
</figure>

---

## 7. Initial Literature Review Summary

### 7.1 Key Research Areas

The project draws on four research areas:

1. **Early popularity prediction** — Early engagement signals correlate strongly with future popularity [1][2], establishing feasibility of forecasting online attention from initial behavioural data.

2. **Content-based prediction** — Content metadata (source, category, subjectivity) can predict popularity before engagement accumulates [3], though feature sets are designed for news rather than financial discussion.

3. **NLP and sentiment for finance** — Aggregate social media mood carries predictive signal for financial outcomes [4], though evaluation periods are short and domain-specific language (e.g., "short", "bearish") challenges general-purpose sentiment tools.

4. **Information diffusion and cascades** — Early propagation patterns predict whether content will continue growing [5], with temporal features (propagation speed) showing particular value.

### 7.2 Identified Research Gap

The literature establishes that: (a) early signals predict future attention [1][5]; (b) multiple feature types each contribute explanatory power [2][3][4]; and (c) popularity follows identifiable temporal dynamics [6][7][8].

Three critical gaps remain:

1. **Prediction target mismatch** — Most studies predict eventual outcomes (final popularity, cascade size, market direction) rather than detecting the onset of rapid growth within a bounded time window. No reviewed study predicts a composite surge within a fixed short-term window.

2. **Single-signal approaches** — Each strand demonstrates one feature category's value, yet few combine temporal, content, and sentiment signals into an integrated framework.

3. **Domain transfer** — Studies draw on general social media (YouTube, Facebook, Twitter) rather than finance-specific platforms with event-driven reactions, speculative behaviour, and domain-specific language.

This project addresses all three gaps: a composite binary surge target within a 24-hour window, multi-signal features, applied specifically to stock-related discussions.

---

## 8. Success Criteria

Success is defined at three tiers to distinguish between a viable proof-of-concept, a strong result, and an exceptional outcome:

| Tier | AUC-ROC | Interpretation |
|------|---------|----------------|
| **Minimum success** | > 0.60 | Demonstrates predictive signal above random; validates that surge prediction from early features is feasible |
| **Target success** | > 0.70 | Indicates moderate discriminative power; comparable to early-stage results in related popularity prediction literature [1][5] |
| **Stretch goal** | > 0.80 | Strong predictive performance; would represent a notable contribution to the field |

### Functional criteria

- Pipeline executes end-to-end on the static dataset without errors
- All pipeline stages produce deterministic outputs with fixed random seed (verified by running twice and comparing)
- Temporal train-test split contains no data leakage (max training timestamp ≤ min test timestamp)

### Analytical criteria

- At least one model achieves minimum success (AUC-ROC > 0.60) on the temporally held-out test set
- Composite model (Phase 2, w₂ = 0.5) is compared against volume-only baseline (Phase 1, w₂ = 0) with statistical significance testing
- Threshold sensitivity analysis produces at least 3 viable operating points with documented precision-recall trade-offs
- Weight sensitivity sweep quantifies the marginal contribution of sentiment across the w₂ spectrum
- Confidence intervals are reported for all metrics

### Academic criteria

- Code is well-documented, modular, and structured for reproducibility
- Report clearly articulates methodology, results, limitations, and threats to validity
- All claims about model performance are supported by statistical evidence (confidence intervals, significance tests)

---

## 9. References

[1] G. Szabo and B. A. Huberman, "Predicting the popularity of online content," *Communications of the ACM*, vol. 53, no. 8, pp. 80–88, 2010. doi: 10.1145/1787234.1787254

[2] K. Lerman and T. Hogg, "Using a model of social dynamics to predict popularity of news," in *Proceedings of the 19th International Conference on World Wide Web (WWW '10)*, ACM, 2010, pp. 621–630. doi: 10.1145/1772690.1772758

[3] R. Bandari, S. Asur, and B. A. Huberman, "The pulse of news in social media: Forecasting popularity," in *Proceedings of the International AAAI Conference on Web and Social Media*, vol. 6, no. 1, AAAI Press, 2012, pp. 26–33.

[4] J. Bollen, H. Mao, and X.-J. Zeng, "Twitter mood predicts the stock market," *Journal of Computational Science*, vol. 2, no. 1, pp. 1–8, 2011. doi: 10.1016/j.jocs.2010.12.007

[5] J. Cheng, L. Adamic, P. A. Dow, J. Kleinberg, and J. Leskovec, "Can cascades be predicted?" in *Proceedings of the 23rd International Conference on World Wide Web (WWW '14)*, ACM, 2014, pp. 925–936. doi: 10.1145/2566486.2567997

[6] C. Wang and B. A. Huberman, "Long trend dynamics in social media," *EPJ Data Science*, vol. 1, no. 1, Article 2, 2012. doi: 10.1140/epjds2

[7] Q. Kong, W. Mao, G. Chen, and D. Zeng, "Exploring trends and patterns of popularity stage evolution in social media," *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, vol. 48, no. 12, pp. 2408–2420, 2018. doi: 10.1109/TSMC.2017.2719279

[8] D. Yuan and Y. Li, "Discovering and early predicting popularity evolution patterns of social media emergency information," *Aslib Journal of Information Management*, vol. 77, no. 1, pp. 115–137, 2025. doi: 10.1108/AJIM-06-2024-0288

