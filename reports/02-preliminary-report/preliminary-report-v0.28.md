# Preliminary Report

## 1. Project Definition

### 1.1 Title

Predicting Engagement and Sentiment Surges in Stock-Related Social Media Discussions

### 1.2 Objectives

- Develop a predictive model using early-stage discussion features to forecast whether a stock-related social media discussion will experience a significant engagement and sentiment surge within 24 hours
- Engineer meaningful features from raw social media discussion data including temporal, textual, activity-frequency, and sentiment signals
- Compare traditional ML approaches (Logistic Regression, Random Forest, XGBoost) for binary surge classification
- Evaluate model performance using standard classification metrics (accuracy, precision, recall, F1, AUC-ROC)

### 1.3 Problem Statement

Financial discussions on social media platforms often experience sudden increases in public attention and emotional intensity. Discussions surrounding specific stocks can rapidly attract large numbers of comments, interactions, and strong sentiment, particularly following news events, earnings announcements, rumours, or speculative activity. These surges can develop within hours, making them difficult to anticipate through manual monitoring alone.

This problem affects multiple stakeholder groups:

- **Financial analysts and portfolio managers** who monitor social sentiment as a supplementary signal for investment decisions and need early warning of discussions gaining momentum.
- **Market surveillance teams and regulators** who track potential market manipulation, coordinated pump-and-dump activity, or rumour-driven volatility on social platforms.
- **Quantitative researchers** studying the relationship between social media dynamics and market microstructure, who require reproducible methods for identifying surge events in historical data.
- **Platform operators and content moderators** who allocate resources to discussions experiencing rapid growth in activity and emotional intensity.

In large social media environments, thousands of stock-related discussions occur every day. The volume makes it impractical to manually assess which discussions are likely to experience substantial growth before that growth becomes obvious. Automated prediction allows these stakeholders to focus attention on the small subset of discussions showing early surge signals.

Existing research frequently focuses on predicting overall popularity or analysing already-popular content [1][3], providing less emphasis on forecasting whether a stock-related discussion is about to experience a significant surge within a clearly defined future time window. Studies that do address financial social media often focus on sentiment-to-market correlations [4] rather than on predicting the social media dynamics themselves.

### 1.4 Unit of Analysis and Prediction Scope

A central design question is: *what exactly is being predicted?* The project title refers to "surges in stock-related social media discussions," but this phrase is ambiguous — it could mean a surge within a single thread, a surge across all discussion about a specific ticker, or a surge across an entire subreddit. This section defines the prediction scope precisely.

#### Terminology

- **Record**: A single row in the dataset representing one Reddit submission (post). Each record has its own timestamp, title, body text, engagement counts (score, num_comments), and extracted ticker symbols. This is the atomic unit of the dataset.

- **Ticker-window**: The set of all records mentioning the same stock ticker within a defined time interval. This is the primary analytical grouping.

- **Discussion thread**: A Reddit post and its associated comments. Thread-level data is not available in the dataset (comments are not linked to parent submissions), so thread-level prediction is not feasible.

#### Prediction target: Per-ticker surge detection

**The prediction operates at the record level but measures surges scoped to the same ticker.** For a given record mentioning ticker $X observed at time *t*, the system asks:

> *"Will discussion about ticker $X experience a surge in engagement and sentiment within the next 24 hours?"*

Specifically, the "subsequent records within *(t, t + 24h]*" used in the composite surge computation are **records in the dataset that mention the same ticker** within that time window. This means:

- A record about $TSLA is evaluated against future $TSLA activity, not against unrelated $AAPL posts
- The prediction is per-ticker rather than per-subreddit — it detects whether a specific stock's discussion is about to surge
- Records mentioning multiple tickers contribute to the window of each mentioned ticker independently

This scoping is the most defensible interpretation because:

1. **Conceptual coherence** — "A surge in stock-related discussion" most naturally refers to intensifying activity around a specific stock, not to an entire forum becoming more active. A subreddit-wide surge would conflate unrelated events (e.g., $GME and $AAPL surging simultaneously for different reasons).

2. **Practical utility** — Stakeholders (analysts, surveillance teams) care about surges in discussion around specific securities, not about aggregate forum traffic. A per-ticker prediction directly answers: "Should I pay attention to what's happening with this stock right now?"

3. **Data availability** — The dataset includes extracted ticker symbols per record, making ticker-scoped windowing feasible without requiring thread-linking metadata.

4. **Alignment with the literature** — Early popularity prediction [1][5] and cascade prediction [5] both operate at the level of individual content items or topics, not at the level of entire platforms.

#### Implications and limitations

- Records that do not mention any identifiable ticker are excluded from surge labelling (they lack a grouping key).
- **Ticker sparsity (Risk #9).** For tickers with very few records in the dataset, the 24-hour window may contain 0, 1, or 2 future records — too few to compute a stable posting volume growth ratio or meaningful mean sentiment. A single outlier post in a sparse window could flip the surge label arbitrarily. This is likely to affect the majority of the 2,912 tickers in the pennystocks dataset, as ticker frequency distributions in social media follow a heavy-tailed power law (a small number of tickers dominate discussion volume). Mitigation: enforce a minimum record count (default N ≥ 3) within each ticker's 24-hour window; records failing this threshold will be excluded from surge labelling. The exclusion rate and its effect on class distribution will be quantified during EDA.
- Records mentioning multiple tickers receive a label based on the combined activity across all mentioned tickers — a simplification that could be refined by computing per-ticker labels independently.
- The model still makes predictions at the record level (one prediction per record), but the target label reflects ticker-scoped dynamics rather than subreddit-wide dynamics.
- If future work uses a single-ticker filtered dataset (e.g., only $GME posts), the ticker scoping becomes equivalent to global scoping within that subset.

### 1.5 Surge Definition

In this project, a **surge** is defined as a statistically significant increase in the composite posting-volume-and-sentiment metric for a specific stock ticker within a fixed 24-hour prediction window. Specifically, for a given record mentioning ticker $X observed at time *t*, the system computes:

- **Posting volume growth**: the relative change in the number of posts mentioning ticker $X between the prior 24 hours and the subsequent 24 hours — specifically, *(count of $X posts in (t, t + 24h]) / max(count of $X posts in (t − 24h, t], 1)) − 1*.
- **Sentiment change**: the absolute difference between the sentiment polarity at observation time and the mean sentiment polarity of subsequent $X-mentioning records within the window.

The **composite surge metric** is defined using z-score normalisation to ensure both components contribute equally:

> *z_volume = (posting_volume_growth − μ_vol) / σ_vol*
>
> *z_sentiment = (|sentiment_change| − μ_sent) / σ_sent*
>
> *composite = (w₁ × z_volume) + (w₂ × z_sentiment)*

where μ and σ are the mean and standard deviation of each raw component computed from the training partition only, and w₁ = w₂ = 0.5 by default (equal weighting).

A record is labelled as a surge (1) if the composite metric exceeds a configurable threshold *τ* (in standard deviation units), and no-surge (0) otherwise.

**Why z-score normalisation.** The two raw components operate on fundamentally different scales: posting volume growth is an unbounded ratio (where 1.0 represents doubling, but values of 10+ are common for tickers that go from 1–2 posts/day to 10+), while |sentiment_change| is bounded by approximately [0, 2.0] given TextBlob's polarity range of [−1, +1]. Without normalisation, the composite metric would be dominated by the volume component — rendering sentiment structurally unable to influence the surge label. This would undermine the core research question (does sentiment add predictive value?) by preventing sentiment from affecting the target definition in Phase 2. Z-score normalisation places both components on a common zero-mean, unit-variance scale, ensuring that each contributes proportionally to the composite regardless of its raw magnitude. The threshold *τ* then has a clear statistical interpretation: "the combined signal exceeds *τ* standard deviations above the typical joint activity level."

**Why posting volume rather than engagement scores.** The dataset provides engagement metrics (score, num_comments) as final snapshot values at crawl time, not as point-in-time values at post creation. Using these values in the surge formula would introduce a circular dependency: posts that eventually experience a surge accumulate high scores *because* of the surge, so measuring score growth would be measuring the surge's effect rather than detecting its onset. Posting volume growth — the increase in the number of posts about a ticker — is derived entirely from creation timestamps, which are fixed at post creation and uncontaminated by future activity. A value of 1.0 means the number of posts about $X doubled; 2.0 means it tripled.

**Normalisation statistics and data leakage prevention.** The mean (μ) and standard deviation (σ) for each component are computed exclusively from the training partition (the first 80% of records by timestamp). Test-set records are normalised using these training-set statistics, not their own. This prevents information about the test distribution from leaking into the labelling process. Because the normalisation is fitted on training data, the z-scores on test records may not be perfectly centred at zero — this is expected and mirrors realistic deployment conditions where future distributional shifts are unknown.

**Threshold determination.** The composite threshold *τ* will be determined empirically during EDA rather than set a priori. Because both components are standardised, *τ* is expressed in units of standard deviations of the combined metric. The threshold sensitivity analysis (Section 4.7) will sweep a range of candidate values and select the operating point that balances class distribution against model trainability. The goal is to produce a surge rate of approximately 5–10% (imbalance ratio 10:1 to 18:1) as indicated by the preliminary viability analysis (Section 2.2).

**Configurable weighting.** The default equal weighting (w₁ = w₂ = 0.5) reflects an agnostic prior about the relative importance of volume versus sentiment in defining a surge. The weight sensitivity analysis (Section 4.7) will sweep w₂ ∈ {0, 0.25, 0.5, 0.75, 1.0} (with w₁ = 1 − w₂) to empirically assess how the relative contribution of each component affects class distribution and model performance. This directly supports the Phase 1 vs Phase 2 comparison: Phase 1 uses w₂ = 0 (volume only), while Phase 2 uses w₂ = 0.5 (equal composite).

**Rationale for a composite metric.** Engagement and sentiment are combined into a single target rather than treated as separate prediction tasks for three reasons:

1. **Neither signal alone captures a meaningful surge.** A discussion topic can attract high posting volume through controversy, memes, or platform algorithms without any genuine shift in investor sentiment. Conversely, sentiment can shift sharply in a low-visibility topic that never gains posting traction. Neither event in isolation constitutes the kind of surge relevant to financial monitoring — it is the *co-occurrence* of rising posting activity and intensifying emotion that distinguishes actionable surges from routine noise. The composite metric requires both dimensions to contribute before the threshold is reached.

2. **The literature supports signal interaction.** Existing work demonstrates that engagement signals [1] and sentiment signals [4] each carry independent predictive value, but studies examining their interaction are scarce (see Section 7.3, Gap 2). By defining the target as a joint function, this project directly tests the hypothesis that combined engagement-and-sentiment events are more meaningful and more predictable than either component alone. The sensitivity analysis will decompose performance by varying the relative contribution of each component.

3. **A single binary target simplifies the prediction task.** Treating engagement and sentiment as separate targets would require either two independent classifiers (doubling the modelling complexity without clear benefit at this stage) or a multi-output model. A composite binary label provides a single, well-defined classification task that can be evaluated with standard metrics (precision, recall, F1, AUC-ROC) and compared directly across models. If the composite approach proves effective, decomposition into separate or weighted components is a natural direction for future work.

**Phased experimental approach.** To empirically validate the composite design, the project adopts a two-phase modelling strategy:

- **Phase 1 (Baseline): Volume-only prediction.** The initial models will be trained using a posting-volume-only surge target — setting w₂ = 0 so that the composite reduces to z_volume alone, labelling records based solely on standardised posting volume growth exceeding the threshold. This establishes a performance baseline grounded in the most directly observable signal and aligns with the early popularity prediction literature [1][5], which demonstrates that activity-based features carry strong predictive power on their own.

- **Phase 2 (Advanced): Composite volume + sentiment prediction.** The full composite target (w₁ × z_volume + w₂ × z_sentiment, with w₁ = w₂ = 0.5) will then be introduced, with models trained on the combined feature set including sentiment scores. Because the z-score normalisation guarantees that sentiment carries equal weight in the label definition, any performance difference between Phase 1 and Phase 2 reflects a genuine contribution (or lack thereof) from the sentiment signal — not an artefact of scale asymmetry. If the composite model outperforms the volume-only baseline, this provides empirical evidence that sentiment integration adds value beyond posting activity alone — supporting the theoretical motivation. If performance is equivalent or worse, this informs a critical discussion about whether sentiment signals are redundant or too noisy in this domain.

This phased design strengthens the project's contribution by providing controlled evidence for (or against) the value of composite targets, rather than assuming that combining signals is inherently beneficial.

This definition captures cases where discussions experience rapid growth in both public posting activity and emotional intensity, distinguishing them from topics that attract posting volume without sentiment shifts or vice versa.

**Label interpretation.** The surge label is derived from observed future outcomes (posting volume growth and sentiment shift in the 24 hours following observation). This is standard for supervised classification on historical data — the label represents "a surge was observed to have occurred" rather than an independently verified ground truth. The label's validity rests on the assumption that a substantial increase in per-ticker posting volume accompanied by a sentiment shift constitutes a meaningful surge event. This assumption cannot be externally validated within the scope of this project (there is no independent "surge registry" to compare against), but the definition is operationally grounded: stakeholders monitoring social media activity would recognise a doubling or tripling of posting frequency about a specific stock as noteworthy. The threshold sensitivity analysis (Section 4.7) further tests the robustness of the labelling scheme across different operating points.

### 1.6 Motivation

- Discussions experiencing rapid posting volume growth and sentiment shifts often attract broader public attention and may influence information diffusion, investor behaviour, and market perception [4]
- Platforms hosting financial discussions (e.g., Reddit, StockTwits) process thousands of new posts daily, making manual identification of emerging surges impractical for analysts and researchers
- Early detection of surge-prone discussions enables proactive monitoring rather than reactive analysis, with applications in financial risk assessment, market surveillance, and social media analytics
- The 2021 GameStop short squeeze demonstrated how rapidly escalating social media discussion can translate into real market impact, underscoring the need for early warning systems [5]
- Addresses a gap in the academic literature around short-term composite surge prediction that integrates both engagement and sentiment signals within a clearly bounded time window

---

## 2. Scope and Boundaries

### 2.1 In Scope

- Pre-collected static dataset of stock-related social media discussions (CSV/Parquet)
- Feature engineering: temporal, textual, sentiment, and activity-frequency features
- Binary classification: surge (1) vs no-surge (0) within 24-hour window
- Traditional ML models: Logistic Regression, Random Forest, XGBoost
- Optional deep learning baseline (LSTM/Transformer) for comparison
- Standard evaluation metrics and visualisations
- Reproducible pipeline with seeded randomness

### 2.2 Dataset

The primary data source is the **Reddit Finance Data** dataset published on Kaggle (https://www.kaggle.com/datasets/leukipp/reddit-finance-data). This dataset contains submissions from nine stock-related subreddits collected over the calendar year 2021, totalling approximately **1.38 million records** across the following communities:

| Subreddit | Records | Tickers | Engagement (mean score) | Selftext Missing |
|-----------|---------|---------|------------------------|-----------------|
| wallstreetbets | 775,326 | 4,451 | 116.0 | 33.8% |
| gme | 273,327 | 340 | 101.3 | 48.7% |
| stocks | 75,857 | 1,963 | 30.2 | 0.1% |
| StockMarket | 72,620 | 1,597 | 6.3 | 29.2% |
| pennystocks | 54,785 | 2,912 | 29.4 | 20.7% |
| stockmarket | 43,809 | 1,475 | 35.0 | 39.1% |
| investing | 41,912 | 990 | 17.8 | 0.0% |
| robinhoodpennystocks | 23,304 | 855 | 30.8 | 36.0% |
| robinhood | 18,893 | 294 | 5.5 | 25.3% |

Each record includes a unique post ID, creation timestamp, title, body text (selftext), engagement metrics (score, num_comments), subreddit, and extracted ticker symbols. The date range covers January–December 2021, a period of high retail trading activity that includes the GameStop short squeeze and subsequent meme-stock phenomena.

**Primary dataset selection.** Based on exploratory data analysis (see Appendix: EDA Report), `pennystocks/submissions_reddit.csv` (54,785 records, 2,912 tickers) is recommended as the primary development dataset due to high data completeness (20.7% selftext missing — lowest among larger subreddits), sufficient volume for model training, and diverse ticker coverage. The `wallstreetbets` and `gme` datasets are retained as secondary validation sets to test generalisability across communities with different engagement distributions.

**Surge viability.** Preliminary surge analysis across 81 threshold configurations confirms that viable composite surge definitions exist (44 configurations produce ≥2% positive class rate). At the recommended operating point (engagement ≥ 90th percentile, sentiment shift ≥ 0.5 standard deviations, 24-hour window), surge rates range from 5–9% across most subreddits, producing imbalance ratios of 10:1 to 18:1 — challenging but tractable with appropriate evaluation strategies (see Risk Register, Risk #1).

### 2.3 Out of Scope

- Real-time or live data ingestion from APIs
- Deployment as a production service
- Trading signals or financial advice
- Multi-class or regression targets
- Cross-platform data fusion (single source dataset)

---

## 3. Proposed Methodology

### 3.1 Data Pipeline Stages

1. **Data Loading** — Read static dataset from disk (CSV/Parquet)
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

The pipeline follows a linear staged architecture where each stage receives the output of its predecessor. All stages share a centralised configuration module and produce deterministic outputs via seeded randomness.

<figure align="center">
  <img src="figures/02-data-pipeline-v0.1.png" alt="Data Pipeline" width="1000">
  <figcaption>Figure 2: Data Pipeline.</figcaption>
</figure>


The system is implemented as a Python package (`surge_pipeline`) with a corresponding CLI entry point (`run_pipeline.py`). Each module exposes a well-defined function interface, allowing both notebook-based exploration and script-based batch execution.

### 3.4 Feature Design

The feature engineering module computes features for each discussion record. A critical design constraint is that **only information available at observation time *t*** may be used as a prediction feature. The Reddit Finance dataset provides engagement metrics (score, num_comments) as final snapshot values collected at crawl time, not as point-in-time values at post creation. Because these snapshot values incorporate all future engagement — including engagement generated *by* the surge being predicted — they cannot be used as features without introducing data leakage. Instead, the feature set relies on temporal, textual, and activity-frequency signals that are fully determined at observation time.

| Feature | Type | Description | Rationale |
|---------|------|-------------|-----------|
| `sentiment_score` | Continuous [-1, 1] | TextBlob polarity of post text | Captures emotional tone; strong sentiment may precede surges [4] |
| `hour_of_day` | Discrete [0–23] | Hour when the post was created | Trading hours and after-hours activity show different surge patterns |
| `day_of_week` | Discrete [0–6] | Day when the post was created | Weekend vs weekday discussion dynamics differ |
| `time_since_previous` | Continuous ≥ 0 | Hours since previous post mentioning the same ticker | Rapid successive posting about the same ticker signals emerging activity [1] |
| `ticker_post_rate_24h` | Continuous ≥ 0 | Number of posts mentioning this ticker in the 24 hours before time *t* | Measures current per-ticker discussion intensity using only historical data |
| `ticker_post_acceleration` | Continuous | Ratio of post count in prior 12h to post count in prior 12–24h | Captures whether per-ticker discussion frequency is already increasing |
| `word_count` | Discrete ≥ 0 | Number of whitespace-separated tokens in post text | Longer posts may carry more informational content [3] |
| `title_length` | Discrete ≥ 0 | Number of whitespace-separated tokens in post title | Short urgent titles vs. detailed titles may signal different discussion types |
| `num_tickers_mentioned` | Discrete ≥ 1 | Count of distinct ticker symbols in the post | Multi-ticker posts may indicate broader market discussion vs. focused analysis |

**Excluded features.** The dataset fields `score` and `num_comments` are explicitly excluded from the feature set because they represent final snapshot values that are not available at observation time. Using them would constitute temporal data leakage — the model would effectively "see" the outcome it is trying to predict. The previously considered `engagement_rate` (score / hours since posting) is excluded for the same reason. The previously considered `has_ticker` feature is excluded because all records in the modelling dataset are required to mention at least one identifiable ticker (see Section 1.4); the feature would be a constant of 1 with zero predictive value.

These features combine temporal, activity-frequency, sentiment, and textual signals using only backward-looking or creation-time information, as supported by the literature [1][3][4][5].

**Note on feature–target correlation.** The activity-frequency features (`time_since_previous`, `ticker_post_rate_24h`, `ticker_post_acceleration`) are correlated with the target by design — current posting momentum is expected to predict future posting momentum. This is analogous to using current temperature to predict tomorrow's temperature: the correlation is informative rather than circular, because the features are strictly backward-looking (computed from timestamps prior to *t*) while the target is strictly forward-looking (computed from timestamps after *t*). No future information leaks into the features. If these features dominate model importance rankings, this expected relationship will be discussed in the evaluation.

**Candidate additional features.** The initial feature set of 9 is deliberately compact to establish a clear baseline. During implementation, the following low-cost additions will be evaluated if initial model performance suggests the hypothesis space is too constrained for tree-based models (RF, XGBoost) to exploit feature interactions effectively:

- `sentiment_subjectivity` — TextBlob subjectivity score (computed alongside polarity at no additional cost)
- `is_selftext` — binary indicator of whether the post contains body text or is a link-only submission
- `ticker_7d_mean_rate` — mean daily post count for this ticker over the preceding 7 days (longer-term activity baseline)

These will be added incrementally and their marginal contribution assessed via feature importance and ablation.

### 3.5 Composite Target Design

The binary surge target is computed at the record level using a forward-looking 24-hour window, scoped to the same ticker (see Section 1.4 for the unit of analysis). Critically, the target uses **posting volume** (record counts derived from timestamps) rather than engagement scores, because score and num_comments in the dataset are snapshot values that are not available at observation time.

1. For each record mentioning ticker $X at observation time *t*, identify all subsequent records **that also mention $X** within *(t, t + 24h]*
2. Compute posting volume growth: *(count of $X posts in (t, t + 24h]) / max(count of $X posts in (t − 24h, t], 1)) − 1*
3. Compute sentiment change: *mean(future_sentiments) − current_sentiment*
4. Standardise: *z_volume = (posting_volume_growth − μ_vol) / σ_vol*; *z_sentiment = (|sentiment_change| − μ_sent) / σ_sent* (using training-set statistics)
5. Combine: *composite = (w₁ × z_volume) + (w₂ × z_sentiment)* where w₁ = w₂ = 0.5 by default
6. Label: *1* if composite > threshold *τ* (configurable, in standard deviation units), else *0*

**Rationale for volume-based engagement.** The dataset provides engagement metrics (score, num_comments) only as final snapshot values, not as point-in-time observations. Using these values in the target formula would create a circular dependency: posts that eventually surge accumulate high scores *because* of the surge, so measuring score growth would be measuring the surge's effect rather than predicting its onset. Posting volume growth — the increase in the *number* of posts about a ticker — uses only timestamps, which are reliable creation-time values unaffected by future activity. A doubling in the number of posts about $X represents a genuine surge in community attention toward that stock.

This approach measures whether the discussion around a specific stock exhibits a substantial combined shift in posting activity and sentiment following the observation point. It captures records that precede per-ticker surges — periods where a specific stock attracts simultaneous growth in both discussion frequency and emotional intensity — rather than detecting subreddit-wide activity spikes that may conflate unrelated events.

---

## 4. Evaluation Strategy

### 4.1 Performance Metrics

Each trained model will be evaluated on the temporally held-out test set using the following classification metrics:

| Metric | Purpose |
|--------|---------|
| **Accuracy** | Overall proportion of correct predictions |
| **Precision** | Proportion of predicted surges that are actual surges (minimises false alarms) |
| **Recall** | Proportion of actual surges that are correctly predicted (minimises missed surges) |
| **F1-Score** | Harmonic mean of precision and recall, balancing both concerns |
| **AUC-ROC** | Area under the Receiver Operating Characteristic curve; measures discriminative ability across all classification thresholds |

Given the expected class imbalance (surges are rare events), precision-recall trade-offs and AUC-ROC will be prioritised over raw accuracy as primary evaluation criteria.

### 4.2 Evaluation Artefacts

The evaluation module will produce the following artefacts for inclusion in the final report:

- **Confusion matrices** — One per model, visualising true positives, false positives, true negatives, and false negatives
- **Combined ROC curve plot** — All models on a single figure with AUC values for direct comparison
- **Metrics summary table** — Structured CSV/JSON file with all metrics per model, suitable for tabular inclusion in the report
- **Feature importance rankings** — For tree-based models (Random Forest, XGBoost), documenting which features contribute most to predictions

### 4.3 Train-Test Split Strategy

The dataset will be split using **temporal ordering** rather than random sampling to prevent data leakage:

- Records are sorted chronologically by timestamp
- The first 80% (configurable) form the training set
- The remaining 20% form the test set
- This ensures no future information leaks into training, reflecting realistic deployment conditions

This approach is critical because random splitting would allow the model to observe future activity patterns during training, artificially inflating performance [5].

**Temporal concept drift.** The dataset spans January–December 2021, a period of significant regime change in retail trading activity. The first quarter (GameStop short squeeze, meme-stock mania) exhibits fundamentally different engagement dynamics than Q3–Q4 (post-squeeze normalisation, declining retail participation). With an 80/20 temporal split, the training set covers approximately January–October and the test set covers November–December. These periods may differ in base surge rates, active ticker composition, and posting patterns — a form of temporal concept drift that could depress test performance regardless of model quality. To characterise this risk, the evaluation will:

- Report the surge rate (positive class proportion) separately for the training and test partitions
- If rates differ substantially (>50% relative difference), discuss the implications for model generalisation
- Note this as a limitation inherent to the single temporal split design; a sliding-window evaluation across multiple time periods would provide a more complete picture but is deferred to future work due to computational scope

### 4.4 Hyperparameter Tuning via Temporal Cross-Validation

Hyperparameter selection for each model is conducted within the training partition using **expanding-window temporal cross-validation**. This ensures that tuning decisions respect chronological ordering and do not leak future information into model configuration.

**Procedure:**

1. The training set (first 80% of records by timestamp) is divided into *k* = 4 sequential folds of approximately equal size.
2. For each fold *i* (i = 2, 3, 4):
   - Training: all records from folds 1 through *i − 1* (expanding window)
   - Validation: records from fold *i*
3. This produces 3 train/validation splits, each progressively larger on the training side.
4. Candidate hyperparameter configurations are evaluated by mean validation AUC-ROC across the 3 splits.
5. The configuration with the highest mean validation AUC-ROC is selected and retrained on the full training partition before final evaluation on the held-out test set.

**Scope of tuning per model:**

| Model | Tuned hyperparameters |
|-------|----------------------|
| Logistic Regression | Regularisation strength (C), penalty type (L1/L2) |
| Random Forest | n_estimators, max_depth, min_samples_leaf |
| XGBoost | n_estimators, max_depth, learning_rate, subsample, colsample_bytree |

A small grid or randomised search (≤50 configurations per model) keeps computational cost manageable while preventing default-hyperparameter overfitting. Logistic Regression requires minimal tuning; the primary beneficiaries are the tree-based models where default settings rarely coincide with the optimal operating point for imbalanced binary classification.

### 4.5 Baseline Comparison

Model performance will be compared against:

- **Random baseline** — AUC-ROC of 0.5 (no discriminative power)
- **Majority-class baseline** — Always predicting "no surge" (establishes the floor for accuracy)
- **Single-feature baselines** — Individual features used alone as predictors to assess marginal contribution

A model is considered to demonstrate meaningful predictive signal if it achieves AUC-ROC > 0.60 on the test set.

### 4.6 Statistical Robustness

Single-run point estimates are insufficient for drawing conclusions about model performance, particularly on imbalanced datasets where small changes in the test set composition can produce large metric fluctuations. The evaluation strategy therefore incorporates the following statistical procedures:

**Confidence intervals.** All reported metrics (accuracy, precision, recall, F1, AUC-ROC) will be accompanied by 95% confidence intervals computed via bootstrap resampling (1,000 iterations) on the test set predictions. This quantifies the uncertainty around each estimate and allows meaningful comparison between models — two models are considered to differ meaningfully only if their confidence intervals do not overlap.

**Multiple-seed evaluation.** To assess sensitivity to random initialisation, each model will be trained and evaluated across 5 different random seeds (42, 123, 256, 512, 1024). The temporal split is deterministic (order-based), so seed variation affects model initialisation (Random Forest bootstrap samples, XGBoost column subsampling) rather than the data split itself. Results will report the mean and standard deviation of each metric across seeds. If standard deviations exceed 0.05 for AUC-ROC, this signals instability warranting investigation.

**Paired statistical tests.** To determine whether performance differences between models are statistically significant rather than due to chance, McNemar's test will be applied to paired predictions on the same test set. This is appropriate for comparing two classifiers on the same data without independence assumptions. A significance level of α = 0.05 will be used, with Bonferroni correction applied when comparing multiple model pairs.

### 4.7 Threshold and Weight Sensitivity Analysis

The composite surge threshold *τ* and weight parameters (w₁, w₂) directly control the class distribution and therefore influence model behaviour and evaluation. To characterise this sensitivity — identified as a key risk (Risk Register, Risk #6) — the following experiments will be conducted:

**Threshold sweep.** The full pipeline will be executed at threshold values of *τ* ∈ {0.5, 1.0, 1.5, 2.0, 2.5} (in standard deviation units of the composite metric), producing five distinct labelling configurations. The pipeline is designed for parameterised batch execution, allowing all threshold × model × seed × phase combinations to run without manual intervention. For each threshold:

- Record the resulting class distribution (surge rate, imbalance ratio)
- Train all three models (LR, RF, XGBoost) on the relabelled data
- Evaluate on the corresponding test set and report metrics with confidence intervals

Because both components are z-score normalised, the threshold has a consistent statistical interpretation across all operating points: *τ* = 1.5 means "the combined signal exceeds 1.5 standard deviations above the training-set mean." This eliminates the interpretability problem of the previous raw additive formulation.

**Expected outcomes and decision criteria:**

| Threshold (τ) | Expected surge rate | Expected behaviour |
|---------------|--------------------|--------------------|
| 0.5 | ~20–30% | Many positives; high recall, low precision |
| 1.0 | ~10–18% | Moderate imbalance; potentially best precision-recall balance |
| 1.5 | ~5–10% | Target operating range; aligned with viability analysis |
| 2.0 | ~2–5% | Sparse positives; models may struggle with minority class |
| 2.5 | <2% | Extreme imbalance; likely below viable training threshold |

The threshold producing the highest F1-score on the test set will be reported as the recommended operating point. If multiple thresholds produce similar F1 but different precision-recall trade-offs, both will be presented with guidance on which is preferable depending on the use case (surveillance favours recall; alert systems favour precision).

**Weight sensitivity sweep.** To empirically assess the relative contribution of each component to predictive performance, the pipeline will be executed with w₂ ∈ {0, 0.25, 0.5, 0.75, 1.0} (where w₁ = 1 − w₂) at the recommended threshold *τ*. This produces a spectrum from pure volume (w₂ = 0, equivalent to Phase 1) through equal weighting (w₂ = 0.5, Phase 2 default) to pure sentiment (w₂ = 1.0). For each weight configuration:

- Record the resulting class distribution
- Train and evaluate all three models
- Report metrics with confidence intervals

This sweep directly answers whether the optimal weighting differs from the default equal split, and whether sentiment's contribution is monotonically positive or exhibits diminishing/negative returns at high weightings. The weight producing the highest F1-score will be reported alongside the default, with discussion of practical implications.

**Interaction with phased approach.** Phase 1 corresponds to w₂ = 0 (volume only); Phase 2 corresponds to w₂ = 0.5 (equal composite). The weight sweep generalises this comparison across the full spectrum, providing a richer picture of how sentiment weight affects predictive performance. The threshold sweep is conducted independently at each phase's default weighting.

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

