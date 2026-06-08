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

### 1.4 Unit of Analysis

The terms *record* and *discussion* are used with distinct meanings throughout this report:

- **Record (Discussion Record)**: A single row in the dataset representing one individual post or comment on a social media platform. Each record has its own timestamp, text body, engagement counts, and unique identifier. This is the atomic unit of analysis — every feature is computed per record, and every prediction is made per record.

- **Discussion (Thread)**: A broader conversational context in which multiple records participate — for example, a Reddit thread or a StockTwits conversation about a particular ticker. The dataset may contain multiple records belonging to the same discussion.

**The prediction target is defined at the record level, not the discussion level.** For a given record observed at time *t*, the system asks: *"Will the surrounding activity in this dataset show a surge pattern within the next 24 hours relative to this record's baseline?"* The "subsequent records within *(t, t + 24h]*" used in the composite computation are all records in the dataset (regardless of thread membership) that fall within that time window. This is a deliberate simplification: rather than modelling thread-level dynamics (which would require reliable thread-linking metadata and substantially more complex labelling logic), the pipeline treats the dataset as a time-ordered stream of individual contributions and measures whether future activity — in aggregate — exhibits growth relative to each observation point.

**Implications and limitations of this choice:**

- The model predicts whether a *record* precedes a surge in overall dataset activity, not whether a specific discussion thread will go viral.
- Records from the same thread will share overlapping prediction windows, potentially receiving similar labels. This correlation is expected and does not constitute data leakage because the temporal split separates training and test sets chronologically.
- If the dataset is filtered to a single stock ticker, the prediction effectively becomes: "Will discussion around this ticker surge in the next 24 hours?" — closer to a discussion-level interpretation.
- Future work could extend this to thread-level aggregation, where features and labels are computed per discussion rather than per record, given sufficiently rich metadata.

### 1.5 Surge Definition

In this project, a **surge** is defined as a statistically significant increase in the composite engagement-and-sentiment metric of a stock-related social media discussion within a fixed 24-hour prediction window. Specifically, for a given discussion record observed at time *t*, the system computes:

- **Engagement growth**: the relative change in cumulative engagement (likes, comments, shares, upvotes) between time *t* and *t + 24h*.
- **Sentiment change**: the absolute difference between the sentiment polarity at observation time and the mean sentiment polarity of subsequent records within the window.

The **composite surge metric** is defined as:

> *composite = engagement_growth + |sentiment_change|*

A discussion is labelled as a surge (1) if the composite metric exceeds a configurable threshold (default: 2.0), and no-surge (0) otherwise.

**Threshold justification.** The default threshold of 2.0 is motivated by the scale of each component. Engagement growth is a ratio where 1.0 represents a doubling of interactions, while sentiment change is bounded by approximately [0, 2.0] given that TextBlob polarity ranges from −1 to +1. A composite threshold of 2.0 therefore requires a substantial combined shift — for example, engagement tripling with no sentiment movement, or engagement doubling alongside a full polarity reversal. Lower thresholds (e.g., 1.5) risk labelling routine fluctuations as surges, inflating the positive class with non-exceptional events. Higher thresholds (e.g., 3.0) would produce very few positive labels, limiting the model's ability to learn meaningful patterns from sparse examples. The value 2.0 balances selectivity against sufficient sample size for model training. Critically, this parameter is configurable and will be subject to a sensitivity analysis (see Risk Register, Risk #6) to assess how threshold variation affects class distribution and model performance across the dataset.

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

### 2.2 Out of Scope

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

```
TODO: data pipeline diagram
```

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

The binary surge target is computed at the record level using a forward-looking 24-hour window (see Section 1.4 for the unit of analysis):

1. For each record at observation time *t*, identify all subsequent records in the dataset within *(t, t + 24h]* — regardless of thread membership
2. Compute engagement growth: *(future_engagement − current_engagement) / max(current_engagement, 1)*
3. Compute sentiment change: *mean(future_sentiments) − current_sentiment*
4. Combine: *composite = engagement_growth + |sentiment_change|*
5. Label: *1* if composite > threshold (default 2.0), else *0*

This approach treats the dataset as a chronologically ordered stream and measures whether the aggregate activity following a given record exhibits a substantial combined shift in engagement and sentiment. It captures records that precede periods of simultaneous growth in both attention and emotional intensity.

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

- Pipeline executes end-to-end on the static dataset without errors
- At least one model achieves AUC-ROC > 0.60 on the test set (indicating predictive signal above random)
- All pipeline stages are reproducible with fixed random seed
- Code is well-documented and structured for academic submission
- Report clearly articulates methodology, results, and limitations

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

