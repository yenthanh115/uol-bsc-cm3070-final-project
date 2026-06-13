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

| # | Risk | Likelihood | Impact | Mitigation | Status |
|---|------|-----------|--------|------------|--------|
| 1 | Class imbalance (few surge events) | High | High | Use stratified evaluation, consider SMOTE/class weighting, report precision-recall curves | Open |
| 2 | Sentiment analysis accuracy (TextBlob limitations) | Medium | Medium | Document limitations, consider FinBERT as alternative if time allows | Open |
| 3 | Data quality issues (missing fields, noise) | Medium | Medium | Robust preprocessing with logging, document exclusion criteria | Open |
| 4 | Temporal data leakage via train-test split | Medium | High | Strict temporal split, no future data in features or labels | Open |
| 5 | Overfitting on small dataset | Medium | High | Temporal cross-validation (Section 4.4), regularisation, report train vs test gaps | Open |
| 6 | Composite target threshold and weight sensitivity | Medium | Medium | Z-score normalisation ensures equal component contribution; sensitivity analysis across multiple thresholds and weight configurations | Open |
| 7 | Time constraints for deep learning baseline | Medium | Low | Mark as optional, prioritise traditional ML models | Open |
| 8 | Reproducibility failures across environments | Low | Medium | Pin all dependencies, use fixed random seeds, document setup | Open |
| 9 | Per-ticker sparsity destabilises surge metric | High | High | Enforce minimum record count (N ≥ 3) within 24h ticker window; exclude or flag records where window contains 0–2 future records; report exclusion rate; conduct sensitivity analysis on minimum-N threshold during EDA | Open |
| 10 | Snapshot engagement values as features (data leakage) | High | Critical | Dataset provides score/num_comments as final snapshot values, not point-in-time. Using them as features would leak future information. Eliminated entirely — features use only timestamps, text, and backward-looking post counts | Mitigated |
| 11 | Snapshot engagement values in target formula (circular labelling) | High | Critical | Replaced score-based engagement growth with posting volume growth derived from timestamps only. Label interpretation paragraph acknowledges labels represent observed outcomes | Mitigated |
| 12 | Temporal concept drift (Q1 meme-stock era vs Q3–Q4 normalisation) | Medium | Medium | Report train/test surge rate differences; acknowledge as limitation of single temporal split; sliding-window evaluation deferred to future work | Open |

---

## 6. Project Plan and Timeline

<figure align="center">
  <img src="figures/01-gantt-chart-v0.1.png" alt="Project Timeline" width="1000">
  <figcaption>Figure 1: Project Timeline.</figcaption>
</figure>

---

## 7. Initial Literature Review Summary

### 7.1 Key Research Areas

The project draws on four established research areas within social media prediction and computational finance:

1. **Early popularity prediction** — Foundational work demonstrating that early engagement signals (views, votes, reposts) correlate strongly with future popularity, establishing the feasibility of forecasting online attention from initial behavioural data.

2. **Machine learning and content-based prediction** — Studies extending prediction beyond temporal signals by incorporating content metadata, source features, and structured classification pipelines to predict online attention before substantial engagement occurs.

3. **NLP and sentiment analysis for financial prediction** — Research applying natural language processing to extract emotional and semantic signals from social media text, particularly in financial contexts where public mood may carry predictive value for market-related outcomes.

4. **Information diffusion and cascade prediction** — Work examining how information spreads through social networks, using early propagation patterns and structural properties to forecast whether content will continue growing.

### 7.2 Critical Evaluation of Foundational Works

#### Early Popularity Prediction

Szabo and Huberman [1] demonstrated strong log-linear correlations between early and later popularity on YouTube and Digg, showing that simple regression on early view counts can predict future attention with high accuracy. However, their model assumes a stationary growth process and relies on content that has already accumulated measurable engagement. This limits applicability to *pre-engagement* prediction — the model cannot make forecasts at or near the time of posting, which is precisely the regime of interest for early surge detection. Furthermore, their evaluation was limited to platforms with specific ranking algorithms (Digg's front-page mechanism), raising questions about generalisability to finance-focused forums where content discovery differs fundamentally.

Lerman and Hogg [2] modelled the interplay between social network structure and content discovery, highlighting that popularity depends on behavioural dynamics beyond simple cumulative counts. Their agent-based approach provided mechanistic insight but required detailed knowledge of platform-specific network topology — data rarely available for financial discussion platforms. The model also assumed homogeneous user behaviour, which is unrealistic in stock forums where institutional participants, retail traders, and bots exhibit very different engagement patterns.

#### Machine Learning and Content-Based Prediction

Bandari et al. [3] advanced the field by demonstrating that content metadata (source, category, subjectivity, named entities) could predict popularity *before* engagement accumulates, achieving ~84% classification accuracy. This was a methodologically important shift toward pre-publication prediction. However, the study used coarse popularity bins rather than continuous or binary surge targets, and the feature set was designed for news articles rather than user-generated financial discussion. Their reliance on manually engineered features also limits transferability — features like "news source reputation" have no direct analogue in anonymous forum posts. The 84% accuracy figure, while frequently cited, should also be interpreted cautiously: it was measured on a four-class classification task with uneven class sizes, meaning that majority-class baselines already achieve substantial accuracy.

#### NLP and Sentiment Analysis

Bollen et al. [4] demonstrated that aggregate Twitter mood (particularly the "Calm" dimension) predicted Dow Jones movements with ~87.6% directional accuracy. This was influential in establishing sentiment as a predictive signal for finance. However, the study has significant methodological limitations that subsequent literature has noted: the evaluation period was short (approximately one month of trading days), no out-of-sample validation was reported, and the causal mechanism is unclear — external events may simultaneously drive both social media mood and market outcomes without one causing the other. The lexicon-based mood measurement tool (OpinionFinder and GPOMS) also lacks domain specificity for financial language, where terms like "short," "bearish," or "moon" carry specialised meaning that general-purpose sentiment tools misclassify. For this project, TextBlob shares similar lexicon-based limitations, which is acknowledged in the risk register and motivates the choice of a configurable sentiment component.

#### Information Diffusion and Cascade Prediction

Cheng et al. [5] achieved ~79.5% accuracy (AUC 0.877) predicting whether Facebook photo cascades would double in size, using only early resharing observations. The methodological rigour was strong: large sample size (millions of cascades), temporal features derived from propagation speed, and structural virality metrics. However, the study focused exclusively on image resharing on Facebook — a platform with explicit social graph structure and algorithmic content distribution that differs markedly from text-based financial forums. The cascade framework also assumes discrete, traceable sharing events, whereas engagement on discussion platforms (upvotes, comments) often lacks explicit propagation chains. The concept of "early propagation speed" nevertheless informs this project's `time_since_previous` feature as a proxy for activity acceleration.

#### Lifecycle and Temporal Evolution

Wang and Huberman [6] and Kong et al. [7] characterised popularity as following identifiable temporal lifecycles (emergence → growth → peak → decline). While these frameworks provide useful conceptual grounding, both studies are primarily descriptive rather than predictive — they identify patterns retrospectively but do not offer methods for real-time forecasting. Yuan and Li [8] extended this by suggesting that early-stage signals may predict later evolution, but their work focused on emergency information diffusion rather than financial contexts, and the temporal granularity (days to weeks) is coarser than the 24-hour window relevant to stock discussion surges.

### 7.3 Synthesis and Identified Research Gap

Collectively, the literature establishes three important findings: (a) early behavioural signals contain predictive information about future online attention [1][5]; (b) multiple feature types (temporal, content, sentiment, structural) each contribute explanatory power [2][3][4]; and (c) popularity follows identifiable temporal dynamics that can theoretically be detected early [7][8].

However, three critical gaps remain:

1. **Prediction target mismatch** — Most studies predict *eventual outcomes* (final popularity, total cascade size, market direction) rather than detecting the *onset* of rapid growth within a bounded time window. This distinction matters practically: an analyst does not need to know that a post will eventually receive 10,000 upvotes — they need to know that a surge is developing *now*, within an actionable timeframe. None of the reviewed studies define or predict a composite engagement-and-sentiment surge within a fixed short-term window.

2. **Single-signal approaches** — Each research strand demonstrates the value of one feature category (Szabo: temporal; Bandari: content; Bollen: sentiment; Cheng: structural), yet few studies combine these signals into an integrated predictive framework. The literature suggests that multiple signal types interact during trend formation [2][7], but empirical integration remains limited. This project's multi-signal feature set (Section 3.4) is motivated directly by this gap.

3. **Domain transfer problem** — The reviewed studies draw on general social media (YouTube, Digg, Facebook, Twitter broadly) rather than finance-specific discussion platforms. Financial discussions have distinctive characteristics — event-driven reactions, domain-specific language, speculative behaviour, and regulatory sensitivity — that may invalidate assumptions from general popularity research. Bollen et al. [4] address financial context but predict market outcomes rather than social media dynamics themselves. No reviewed study predicts engagement-and-sentiment surges specifically within stock-related discussion data.

This project addresses all three gaps by defining a composite binary surge target within a fixed 24-hour window, combining temporal, engagement, sentiment, and textual features, and applying the framework specifically to stock-related social media discussions.

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

