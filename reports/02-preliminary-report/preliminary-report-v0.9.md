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

### 1.4 Surge Definition

In this project, a **surge** is defined as a statistically significant increase in the composite engagement-and-sentiment metric of a stock-related social media discussion within a fixed 24-hour prediction window. Specifically, for a given discussion record observed at time *t*, the system computes:

- **Engagement growth**: the relative change in cumulative engagement (likes, comments, shares, upvotes) between time *t* and *t + 24h*.
- **Sentiment change**: the absolute difference between the sentiment polarity at observation time and the mean sentiment polarity of subsequent records within the window.

The **composite surge metric** is defined as:

> *composite = engagement_growth + |sentiment_change|*

A discussion is labelled as a surge (1) if the composite metric exceeds a configurable threshold (default: 2.0), and no-surge (0) otherwise. This definition captures cases where discussions experience rapid growth in both public attention and emotional intensity, distinguishing them from discussions that attract engagement without sentiment shifts or vice versa.

### 1.5 Motivation

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

The binary surge target is computed using a forward-looking 24-hour window:

1. For each record at observation time *t*, identify all subsequent records within *(t, t + 24h]*
2. Compute engagement growth: *(future_engagement − current_engagement) / max(current_engagement, 1)*
3. Compute sentiment change: *mean(future_sentiments) − current_sentiment*
4. Combine: *composite = engagement_growth + |sentiment_change|*
5. Label: *1* if composite > threshold (default 2.0), else *0*

This approach captures discussions that experience simultaneous growth in both attention and emotional intensity, rather than those with only engagement spikes or only sentiment shifts.

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

### 6.1 Key Research Areas

The project draws on four established research areas within social media prediction and computational finance:

1. **Early popularity prediction** — Foundational work demonstrating that early engagement signals (views, votes, reposts) correlate strongly with future popularity, establishing the feasibility of forecasting online attention from initial behavioural data.

2. **Machine learning and content-based prediction** — Studies extending prediction beyond temporal signals by incorporating content metadata, source features, and structured classification pipelines to predict online attention before substantial engagement occurs.

3. **NLP and sentiment analysis for financial prediction** — Research applying natural language processing to extract emotional and semantic signals from social media text, particularly in financial contexts where public mood may carry predictive value for market-related outcomes.

4. **Information diffusion and cascade prediction** — Work examining how information spreads through social networks, using early propagation patterns and structural properties to forecast whether content will continue growing.

### 6.2 Foundational Papers

The following papers form the primary foundations for this project:

| # | Authors | Year | Contribution |
|---|---------|------|-------------|
| 1 | Szabo & Huberman | 2010 | Demonstrated strong correlations between early and later popularity on YouTube/Digg using simple statistical models |
| 2 | Lerman & Hogg | 2010 | Highlighted the role of social dynamics and user interaction in shaping content popularity |
| 3 | Bandari, Asur & Huberman | 2012 | Showed that content and metadata features can predict news popularity with ~84% accuracy before strong engagement occurs |
| 4 | Bollen, Mao & Zeng | 2011 | Demonstrated that collective mood from Twitter (especially "Calm") predicted Dow Jones movements with ~87.6% directional accuracy |
| 5 | Cheng, Adamic, Dow, Kleinberg & Leskovec | 2014 | Showed that large cascades can be predicted from early resharing behaviour with AUC of 0.877 |
| 6 | Wang & Huberman | 2012 | Identified that collective attention follows identifiable temporal dynamics in long-term trends |
| 7 | Kong, Mao, Chen & Zeng | 2018 | Described popularity as evolving through stages (emergence, growth, peak, decline) |
| 8 | Yuan & Li | 2025 | Indicated that early stages of popularity evolution contain predictive signals before large-scale diffusion |

### 6.3 Identified Research Gap

Three gaps emerge from the literature that this project addresses:

1. **Focus on eventual outcomes** — Most studies predict final popularity, cascade size, or market movement rather than detecting the earliest transition from ordinary discussion to emerging trend.

2. **Single-feature reliance** — Many approaches rely on one category of features (temporal, sentiment, or structural alone), even though trend formation is likely driven by interactions among multiple signal types.

3. **Limited finance-specific attention** — Despite the importance of sentiment, speculation, and rapid event-driven reactions in financial discussions, relatively little work targets finance-specific surge emergence.

This project addresses these gaps by combining temporal, engagement, and sentiment features within a composite surge target framework, using a clearly defined 24-hour prediction window applied to stock-related social media discussions.

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

