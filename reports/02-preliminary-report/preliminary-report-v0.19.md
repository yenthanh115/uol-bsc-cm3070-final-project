# Preliminary Report

## 1. Project Definition

### 1.1 Title

Predicting Engagement and Sentiment Surges in Stock-Related Social Media Discussions

### 1.2 Objectives

- Develop a predictive model using early-stage discussion features to forecast whether a stock-related social media discussion will experience a significant engagement and sentiment surge within 24 hours
- Engineer meaningful features from raw social media discussion data including temporal, textual, engagement-rate, and sentiment signals
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
- For tickers with very few records in the dataset, the 24-hour window may contain insufficient data to compute meaningful engagement growth. A minimum record count within the window may be required (to be determined during EDA).
- Records mentioning multiple tickers receive a label based on the combined activity across all mentioned tickers — an simplification that could be refined by computing per-ticker labels independently.
- The model still makes predictions at the record level (one prediction per record), but the target label reflects ticker-scoped dynamics rather than subreddit-wide dynamics.
- If future work uses a single-ticker filtered dataset (e.g., only $GME posts), the ticker scoping becomes equivalent to global scoping within that subset.

### 1.5 Surge Definition

In this project, a **surge** is defined as a statistically significant increase in the composite engagement-and-sentiment metric for a specific stock ticker within a fixed 24-hour prediction window. Specifically, for a given record mentioning ticker $X observed at time *t*, the system computes:

- **Engagement growth**: the relative change in cumulative engagement (score, num_comments) across all $X-mentioning records between time *t* and *t + 24h*.
- **Sentiment change**: the absolute difference between the sentiment polarity at observation time and the mean sentiment polarity of subsequent $X-mentioning records within the window.

The **composite surge metric** is defined as:

> *composite = engagement_growth + |sentiment_change|*

A record is labelled as a surge (1) if the composite metric exceeds a configurable threshold (default: 2.0), and no-surge (0) otherwise.

**Threshold justification.** The default threshold of 2.0 is motivated by the scale of each component. Engagement growth is a ratio where 1.0 represents a doubling of interactions, while sentiment change is bounded by approximately [0, 2.0] given that TextBlob polarity ranges from −1 to +1. A composite threshold of 2.0 therefore requires a substantial combined shift — for example, engagement tripling with no sentiment movement, or engagement doubling alongside a full polarity reversal. Lower thresholds (e.g., 1.5) risk labelling routine fluctuations as surges, inflating the positive class with non-exceptional events. Higher thresholds (e.g., 3.0) would produce very few positive labels, limiting the model's ability to learn meaningful patterns from sparse examples. The value 2.0 balances selectivity against sufficient sample size for model training. Critically, this parameter is configurable and will be subject to a sensitivity analysis (see Risk Register, Risk #6) to assess how threshold variation affects class distribution and model performance across the dataset.

**Rationale for a composite metric.** Engagement and sentiment are combined into a single target rather than treated as separate prediction tasks for three reasons:

1. **Neither signal alone captures a meaningful surge.** A discussion can attract high engagement through controversy, memes, or platform algorithms without any genuine shift in investor sentiment. Conversely, sentiment can shift sharply in a low-visibility post that never gains traction. Neither event in isolation constitutes the kind of surge relevant to financial monitoring — it is the *co-occurrence* of rising attention and intensifying emotion that distinguishes actionable surges from routine noise. The composite metric requires both dimensions to contribute before the threshold is reached.

2. **The literature supports signal interaction.** Existing work demonstrates that engagement signals [1] and sentiment signals [4] each carry independent predictive value, but studies examining their interaction are scarce (see Section 7.3, Gap 2). By defining the target as a joint function, this project directly tests the hypothesis that combined engagement-and-sentiment events are more meaningful and more predictable than either component alone. The sensitivity analysis will decompose performance by varying the relative contribution of each component.

3. **A single binary target simplifies the prediction task.** Treating engagement and sentiment as separate targets would require either two independent classifiers (doubling the modelling complexity without clear benefit at this stage) or a multi-output model. A composite binary label provides a single, well-defined classification task that can be evaluated with standard metrics (precision, recall, F1, AUC-ROC) and compared directly across models. If the composite approach proves effective, decomposition into separate or weighted components is a natural direction for future work.

**Phased experimental approach.** To empirically validate the composite design, the project adopts a two-phase modelling strategy:

- **Phase 1 (Baseline): Engagement-only prediction.** The initial models will be trained using an engagement-only surge target — labelling records based solely on engagement growth exceeding a threshold, without incorporating sentiment change. This establishes a performance baseline grounded in the most directly observable signal and aligns with the early popularity prediction literature [1][5], which demonstrates that engagement-based features carry strong predictive power on their own.

- **Phase 2 (Advanced): Composite engagement + sentiment prediction.** The full composite target (engagement growth + |sentiment change|) will then be introduced, with models trained on the combined feature set including sentiment scores. Comparing Phase 2 against the Phase 1 baseline directly measures the marginal predictive contribution of sentiment signals. If the composite model outperforms the engagement-only baseline, this provides empirical evidence that sentiment integration adds value beyond engagement alone — supporting the theoretical motivation. If performance is equivalent or worse, this informs a critical discussion about whether sentiment signals are redundant or too noisy in this domain.

This phased design strengthens the project's contribution by providing controlled evidence for (or against) the value of composite targets, rather than assuming that combining signals is inherently beneficial.

This definition captures cases where discussions experience rapid growth in both public attention and emotional intensity, distinguishing them from discussions that attract engagement without sentiment shifts or vice versa.

### 1.6 Motivation

- Discussions experiencing rapid engagement and sentiment growth often attract broader public attention and may influence information diffusion, investor behaviour, and market perception [4]
- Platforms hosting financial discussions (e.g., Reddit, StockTwits) process thousands of new posts daily, making manual identification of emerging surges impractical for analysts and researchers
- Early detection of surge-prone discussions enables proactive monitoring rather than reactive analysis, with applications in financial risk assessment, market surveillance, and social media analytics
- The 2021 GameStop short squeeze demonstrated how rapidly escalating social media discussion can translate into real market impact, underscoring the need for early warning systems [5]
- Addresses a gap in the academic literature around short-term composite surge prediction that integrates both engagement and sentiment signals within a clearly bounded time window

---

## 2. Scope and Boundaries

### 2.1 In Scope

- Pre-collected static dataset of stock-related social media discussions (CSV/Parquet)
- Feature engineering: temporal, textual, sentiment, and engagement-rate features
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
3. **Feature Engineering** — Compute sentiment, temporal, engagement-rate, and text features
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

The feature engineering module computes seven features for each discussion record:

| Feature | Type | Description | Rationale |
|---------|------|-------------|-----------|
| `sentiment_score` | Continuous [-1, 1] | TextBlob polarity of post text | Captures emotional tone; strong sentiment may precede surges [4] |
| `hour_of_day` | Discrete [0–23] | Hour when the post was created | Trading hours and after-hours activity show different surge patterns |
| `day_of_week` | Discrete [0–6] | Day when the post was created | Weekend vs weekday discussion dynamics differ |
| `time_since_previous` | Continuous ≥ 0 | Hours since previous post in dataset | Rapid successive posting may signal emerging activity [1] |
| `engagement_rate` | Continuous ≥ 0 | Total engagement / hours since posting | Normalises engagement by exposure time |
| `word_count` | Discrete ≥ 0 | Number of whitespace-separated tokens | Longer posts may carry more informational content [3] |
| `has_ticker` | Binary | Presence of $TICKER pattern | Ticker mentions signal explicit stock focus |

These features combine temporal, behavioural, sentiment, and textual signals as supported by the literature [1][3][4][5].

### 3.5 Composite Target Design

The binary surge target is computed at the record level using a forward-looking 24-hour window, scoped to the same ticker (see Section 1.4 for the unit of analysis):

1. For each record mentioning ticker $X at observation time *t*, identify all subsequent records **that also mention $X** within *(t, t + 24h]*
2. Compute engagement growth: *(future_engagement − current_engagement) / max(current_engagement, 1)*
3. Compute sentiment change: *mean(future_sentiments) − current_sentiment*
4. Combine: *composite = engagement_growth + |sentiment_change|*
5. Label: *1* if composite > threshold (default 2.0), else *0*

This approach measures whether the discussion around a specific stock exhibits a substantial combined shift in engagement and sentiment following the observation point. It captures records that precede per-ticker surges — periods where a specific stock attracts simultaneous growth in both attention and emotional intensity — rather than detecting subreddit-wide activity spikes that may conflate unrelated events.

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

This approach is critical because random splitting would allow the model to observe future engagement patterns during training, artificially inflating performance [5].

### 4.4 Baseline Comparison

Model performance will be compared against:

- **Random baseline** — AUC-ROC of 0.5 (no discriminative power)
- **Majority-class baseline** — Always predicting "no surge" (establishes the floor for accuracy)
- **Single-feature baselines** — Individual features used alone as predictors to assess marginal contribution

A model is considered to demonstrate meaningful predictive signal if it achieves AUC-ROC > 0.60 on the test set.

### 4.5 Statistical Robustness

Single-run point estimates are insufficient for drawing conclusions about model performance, particularly on imbalanced datasets where small changes in the test set composition can produce large metric fluctuations. The evaluation strategy therefore incorporates the following statistical procedures:

**Confidence intervals.** All reported metrics (accuracy, precision, recall, F1, AUC-ROC) will be accompanied by 95% confidence intervals computed via bootstrap resampling (1,000 iterations) on the test set predictions. This quantifies the uncertainty around each estimate and allows meaningful comparison between models — two models are considered to differ meaningfully only if their confidence intervals do not overlap.

**Multiple-seed evaluation.** To assess sensitivity to random initialisation, each model will be trained and evaluated across 5 different random seeds (42, 123, 256, 512, 1024). The temporal split is deterministic (order-based), so seed variation affects model initialisation (Random Forest bootstrap samples, XGBoost column subsampling) rather than the data split itself. Results will report the mean and standard deviation of each metric across seeds. If standard deviations exceed 0.05 for AUC-ROC, this signals instability warranting investigation.

**Paired statistical tests.** To determine whether performance differences between models are statistically significant rather than due to chance, McNemar's test will be applied to paired predictions on the same test set. This is appropriate for comparing two classifiers on the same data without independence assumptions. A significance level of α = 0.05 will be used, with Bonferroni correction applied when comparing multiple model pairs.

### 4.6 Threshold Sensitivity Analysis

The composite surge threshold (default: 2.0) directly controls the class distribution and therefore influences model behaviour and evaluation. To characterise this sensitivity — identified as a key risk (Risk Register, Risk #6) — the following experiment will be conducted:

**Threshold sweep.** The full pipeline will be executed at threshold values of {1.0, 1.5, 2.0, 2.5, 3.0}, producing five distinct labelling configurations. For each threshold:

- Record the resulting class distribution (surge rate, imbalance ratio)
- Train all three models (LR, RF, XGBoost) on the relabelled data
- Evaluate on the corresponding test set and report metrics with confidence intervals

**Expected outcomes and decision criteria:**

| Threshold | Expected surge rate | Expected behaviour |
|-----------|--------------------|--------------------|
| 1.0 | ~15–25% | More positive labels; models may achieve high recall but low precision (many false positives) |
| 1.5 | ~8–15% | Moderate imbalance; potentially best balance between precision and recall |
| 2.0 | ~3–8% | Default operating point; higher precision but recall may suffer |
| 2.5 | ~1–4% | Sparse positives; models may struggle to learn the minority class |
| 3.0 | <2% | Extreme imbalance; likely below viable training threshold |

The threshold producing the highest F1-score on the test set will be reported as the recommended operating point. If multiple thresholds produce similar F1 but different precision-recall trade-offs, both will be presented with guidance on which is preferable depending on the use case (surveillance favours recall; alert systems favour precision).

**Interaction with phased approach.** The threshold sensitivity analysis will be conducted independently for both the engagement-only baseline (Phase 1) and the composite target (Phase 2), enabling comparison of how each target definition responds to threshold variation.

---

## 5. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | Class imbalance (few surge events) | High | High | Use stratified evaluation, consider SMOTE/class weighting, report precision-recall curves |
| 2 | Sentiment analysis accuracy (TextBlob limitations) | Medium | Medium | Document limitations, consider FinBERT as alternative if time allows |
| 3 | Data quality issues (missing fields, noise) | Medium | Medium | Robust preprocessing with logging, document exclusion criteria |
| 4 | Temporal data leakage | Medium | High | Strict temporal split, no future data in features or labels |
| 5 | Overfitting on small dataset | Medium | High | Cross-validation, regularisation, report train vs test gaps |
| 6 | Composite target threshold sensitivity | Medium | Medium | Sensitivity analysis across multiple thresholds |
| 7 | Time constraints for deep learning baseline | Medium | Low | Mark as optional, prioritise traditional ML models |
| 8 | Reproducibility failures across environments | Low | Medium | Pin all dependencies, use fixed random seeds, document setup |

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
- Composite model (Phase 2) is compared against engagement-only baseline (Phase 1) with statistical significance testing
- Threshold sensitivity analysis produces at least 3 viable operating points with documented precision-recall trade-offs
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

