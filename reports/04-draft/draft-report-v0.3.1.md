# Predicting Posting-Volume Surges on Reddit Financial Communities Using Machine Learning

## Abstract

Rapid surges in social media posting about specific stocks can signal coordinated retail trading activity, yet detecting these surges in advance remains an open challenge due to temporal data leakage risks and the sparsity of ticker-level discussion. This project investigates whether posting-volume surges on Reddit financial communities can be predicted using only backward-looking features available at observation time. A composite surge metric combines z-score normalised posting-volume growth with sentiment change magnitude, producing binary labels from raw submission data. The pipeline is evaluated on two subreddits at opposite ends of the data density spectrum: r/pennystocks (80,212 records) and r/wallstreetbets (1,293,981 records).

Three classifiers are compared: Logistic Regression, Random Forest, and XGBoost, trained with expanding-window temporal cross-validation (k=4 folds) and evaluated on a held-out future partition (80/20 temporal split). Eleven features are engineered from timestamps and text content; no post-creation engagement metrics are used, eliminating look-ahead bias.

On r/wallstreetbets, XGBoost achieves AUC-ROC of 0.889 (stretch tier), while Random Forest reaches 0.754 on r/pennystocks (target tier). Validation-fold threshold tuning proves essential for practical utility, lifting XGBoost from zero positive predictions to F1=0.178. Cross-dataset transfer (WSB-trained models evaluated on r/pennystocks) yields AUC=0.694, demonstrating partial generalisation while confirming that thresholds require community-specific recalibration.

These results demonstrate that posting-volume surges are predictable from observation-time features alone, with data density being the primary determinant of model performance. The methodology provides a reproducible, leakage-free framework applicable to any timestamped discussion forum, offering a foundation for real-time surge detection systems relevant to market surveillance and retail investor research.

<!-- 
SIDE NOTE (DELETE LATER)
- why use combined/composite metric
- why use 2 datasets, why pick pennystocks and wsb
- why pick 3 model: LR, RF, XGB
- why use validation-fold
- why not k=5 or k=10 but k=4?
- what are stretch tier vs target tier
- have we set goal for this project (musthave tier, target tier, and stretch tier)
- discuss about data drift ? aware of it and provide solution 
-->
---

## 1. Introduction

### 1.1 Project Concept and Objectives

This project follows the **CM3005 Data Science** project template, *Predictive Modelling of Social Media Trend Emergence*. It builds a machine learning system that predicts whether a stock ticker's Reddit discussion is about to surge, using only backward-looking features available at observation time. Three classifiers (Logistic Regression, Random Forest, and XGBoost) are trained and compared on this binary task.

The project has three objectives:

- Build a predictive model from early-stage discussion features (temporal patterns, activity frequency, sentiment) that can forecast per-ticker surges before they happen
- Compare multiple ML approaches to find out whether more complex models actually improve prediction over simpler baselines
- Confirm that predictions hold up on unseen future time periods by using temporal evaluation protocols that prevent data leakage, a common methodological weakness in social media prediction studies

### 1.2 Problem Statement and Motivation

Stock-related discussions on Reddit can go from quiet to frenzied within hours. A ticker attracting two posts yesterday might appear in fifty today, triggered by earnings surprises, speculative momentum, or coordinated retail interest. These surges develop too quickly for manual monitoring, particularly across forums where thousands of tickers are discussed daily.

This is primarily a research question: can the onset of a social media surge be detected from the discussion patterns that precede it? Answering this question also has practical relevance for financial analysts seeking early warning of emerging narratives, surveillance teams watching for manipulation, and quantitative researchers studying how attention propagates through online communities.

Prior work in this area tends to focus on related but distinct problems: forecasting eventual content reach rather than detecting rapid onset, or predicting price movements rather than social media dynamics themselves. In the reviewed literature, predicting the onset of a volume-and-sentiment surge for individual tickers within a short-term window remains largely unaddressed (see Section 2.5).

This project explores whether such surges are predictable from the discussion patterns that precede them.

### 1.3 Prediction Scope and Surge Definition

A **surge** is a statistically significant increase in both posting volume and sentiment intensity for a specific ticker within a 24-hour window, measured by a composite metric combining normalised volume growth with sentiment change magnitude. The target derives from posting volume (timestamp-based record counts) rather than engagement scores like upvotes, which are future-contaminated snapshot values that would introduce look-ahead bias. Z-scores use training-partition statistics only, preventing leakage. The formal definition, weighting, and threshold selection are detailed in Section 3.3.2.

### 1.4 Scope

**In scope:** Two pre-collected Reddit datasets representing opposite ends of the data density spectrum — r/pennystocks (80,212 records), a sparse niche community, and r/wallstreetbets (1,293,981 records), a high-volume mainstream forum. This dual-dataset design tests whether the methodology generalises across community sizes or whether data density is a binding constraint. Also in scope: feature engineering from text and timestamps, binary classification, a reproducible pipeline with seeded randomness, and cross-dataset transfer evaluation.

**Out of scope:** Real-time ingestion, production deployment, trading signal generation, multi-class targets, cross-platform fusion.

The system achieves AUC-ROC of 0.889 on the high-density dataset and 0.754 on the sparse dataset, demonstrating that surges are predictable from observation-time features but that data density significantly affects performance.

---

## 2. Literature Review

Research on predicting online attention has established that early behavioural signals carry predictive power, that multiple feature types (temporal, content, sentiment, structural) each contribute, and that social media discussion patterns in financial communities correlate with subsequent market activity. However, no reviewed study combines these findings into a system that predicts the *onset* of a composite engagement surge within a bounded time window, applied to financial discussion, and evaluated with temporal protocols that prevent data leakage. The following sections trace how each finding was established, identify the methodological limitations of each strand, and converge on the four gaps this project aims to address.

### 2.1. Early Popularity Prediction

Szabo and Huberman [1] demonstrated strong log-linear correlations between early and later popularity on YouTube and Digg, showing that simple regression on early view counts can predict future attention with high accuracy. However, their model assumes a stationary growth process and relies on content that has already accumulated measurable engagement. This limits applicability to *pre-engagement* prediction — the model cannot make forecasts at or near the time of posting, which is precisely the regime of interest for early surge detection. Furthermore, their evaluation was limited to platforms with specific ranking algorithms (Digg's front-page mechanism), raising questions about generalisability to finance-focused forums where content discovery differs fundamentally.

Lerman and Hogg [2] modelled the interplay between social network structure and content discovery, highlighting that popularity depends on behavioural dynamics beyond simple cumulative counts. Their agent-based approach provided mechanistic insight but required detailed knowledge of platform-specific network topology — data rarely available for financial discussion platforms. The model also assumed homogeneous user behaviour, which is unrealistic in stock forums where institutional participants, retail traders, and bots exhibit very different engagement patterns. Neither study addresses the temporal validity of their predictions — both evaluate on data drawn from the same time period as training, leaving open the question of whether models would generalise to future periods with different platform dynamics.

### 2.2. Machine Learning and Content-Based Prediction

Bandari et al. [3] advanced the field by demonstrating that content metadata (source, category, subjectivity, named entities) could predict popularity *before* engagement accumulates, achieving ~84% classification accuracy. This was a methodologically important shift toward pre-publication prediction. However, the study used coarse popularity bins rather than continuous or binary surge targets, and the feature set was designed for news articles rather than user-generated financial discussion. Their reliance on manually engineered features also limits transferability — features like "news source reputation" have no direct analogue in anonymous forum posts. The 84% accuracy figure, while frequently cited, should also be interpreted cautiously: it was measured on a four-class classification task with uneven class sizes, meaning that majority-class baselines already achieve substantial accuracy. Importantly, the evaluation used random train-test splits rather than temporal partitions, meaning the model may have been tested on articles published *before* some of its training data — a form of temporal leakage that inflates reported performance.

Fernández-Delgado et al. [16] evaluated 179 classifier implementations across 121 datasets and found that random forests achieved the highest overall accuracy, followed by support vector machines and boosting ensembles. While the study did not address social media prediction specifically, it provides empirical justification for the model family selection in this project: a linear baseline (Logistic Regression), a strong ensemble method (Random Forest), and a gradient boosting approach (XGBoost) cover the three top-performing classifier families identified in that large-scale comparison.

### 2.3. NLP and Sentiment Analysis

Bollen et al. [4] demonstrated that aggregate Twitter mood (particularly the "Calm" dimension) predicted Dow Jones movements with ~87.6% directional accuracy, establishing sentiment as a viable predictive signal in financial contexts. However, the study has significant methodological limitations: the evaluation period was short (approximately one month of trading days), no out-of-sample validation was reported, and the causal mechanism is unclear — external events may simultaneously drive both social media mood and market outcomes without one causing the other. The lexicon-based mood measurement tools (OpinionFinder and GPOMS) also lack domain specificity for financial language, where terms like "short," "bearish," or "moon" carry specialised meaning that general-purpose sentiment tools misclassify. Despite these limitations, the core finding — that collective sentiment extracted from social media text carries measurable predictive information — provides the rationale for incorporating sentiment change as a component of this project's composite surge metric. The specific sentiment tool selection and its trade-offs are discussed in Section 3.3.

### 2.4. Penny Stocks, Reddit, and Retail Investor Communities

Penny stocks — typically low-capitalisation equities trading below $5 per share — occupy a distinctive position in the social media prediction landscape. Their low liquidity and limited analyst coverage mean that social media discussion can constitute a disproportionate share of available information, amplifying the potential for discussion-driven price and volume effects [9][10].

Reddit has emerged as a primary venue for retail investor coordination. Unlike Twitter's broadcast model, Reddit's subreddit structure creates focused communities with shared norms, persistent threads, and upvote-driven visibility. Long et al. [9] demonstrated that r/WallStreetBets posting volume correlated with abnormal trading volume and returns for discussed stocks, with effects concentrated in small-cap equities. Their analysis showed that increased Reddit attention preceded trading activity rather than merely reflecting it, suggesting predictive value in discussion patterns. However, the study focused on *market* outcomes (returns, volume) rather than predicting *social media dynamics* themselves — the question of whether discussion will continue to escalate remains unaddressed.

Costola et al. [10] examined the GameStop episode as a case study of collective coordination on Reddit, finding that consensus formation within r/WallStreetBets followed measurable patterns in posting frequency and sentiment alignment before reaching critical mass. Their network analysis revealed that a small number of committed users drove broader community engagement, suggesting that early activity patterns may carry predictive signal. This is directly relevant to this project's approach: if pre-surge discussion exhibits detectable temporal and sentiment signatures, early-stage features should capture them.

Mancini et al. [11] studied pump-and-dump schemes in online forums, building predictive models from the language and timing of social media posts associated with manipulated stocks. Their work demonstrated that text-based features from discussion forums could predict anomalous stock activity, achieving classification performance significantly above random baselines. The study is particularly relevant because it targets small-cap stocks susceptible to social media influence — the same market segment this project examines. However, their target was market manipulation detection (a retrospective labelling task) rather than real-time surge prediction from discussion patterns alone.

A common limitation across this literature is the absence of temporal evaluation protocols. Most studies use random or chronological train-test splits without addressing whether models trained on past patterns generalise to future periods with potentially different market regimes and community dynamics.

### 2.5. Information Diffusion and Cascade Prediction

Cheng et al. [5] achieved ~79.5% accuracy (AUC 0.877) predicting whether Facebook photo cascades would double in size, using only early resharing observations. The methodological rigour was strong: large sample size (millions of cascades), temporal features derived from propagation speed, and structural virality metrics. However, the study focused exclusively on image resharing on Facebook — a platform with explicit social graph structure and algorithmic content distribution that differs markedly from text-based financial forums. The cascade framework also assumes discrete, traceable sharing events, whereas engagement on discussion platforms (upvotes, comments) often lacks explicit propagation chains. The concept of "early propagation speed" nevertheless informs this project's `time_since_previous` feature as a proxy for activity acceleration. However, like most cascade prediction studies, Cheng et al. used random sampling of cascades for evaluation rather than strict temporal ordering, meaning that cascades from earlier time periods could appear in the test set while later ones were used for training.

Wang and Huberman [6] and Kong et al. [7] characterised popularity as following identifiable temporal lifecycles (emergence → growth → peak → decline). While these frameworks provide useful conceptual grounding, both studies are primarily descriptive rather than predictive — they identify patterns retrospectively but do not offer methods for real-time forecasting. Yuan and Li [8] extended this by suggesting that early-stage signals may predict later evolution, but their work focused on emergency information diffusion rather than financial contexts, and the temporal granularity (days to weeks) is coarser than the 24-hour window relevant to stock discussion surges.

### 2.6. Synthesis and Identified Research Gap

The literature establishes four findings: (a) early behavioural signals contain predictive information about future online attention [1][5]; (b) multiple feature types (temporal, content, sentiment, structural) each contribute explanatory power [2][3][4]; (c) popularity follows identifiable temporal dynamics that can theoretically be detected early [7][8]; and (d) Reddit discussion patterns correlate with subsequent trading activity in small-cap stocks, suggesting that social media dynamics carry predictive signal in financial contexts [9][10].

Four critical gaps remain:

1. **Prediction target mismatch** — Most studies predict *eventual outcomes* (final popularity, total cascade size, market direction) rather than detecting the *onset* of rapid growth within a bounded time window. An analyst needs to know a surge is developing *now*, within an actionable timeframe. No reviewed study defines or predicts a composite engagement-and-sentiment surge within a fixed short-term window.

2. **Single-signal approaches** — Each research strand demonstrates one feature category's value (Szabo: temporal; Bandari: content; Bollen: sentiment; Cheng: structural), yet few combine signals into an integrated predictive framework. The literature suggests multiple signal types interact during trend formation [2][7], but empirical integration remains limited.

3. **Domain transfer problem** — Reviewed studies draw on general social media (YouTube, Digg, Facebook, Twitter) rather than finance-specific discussion platforms. Financial discussions have distinctive characteristics — event-driven reactions, domain-specific language, speculative behaviour — that may invalidate assumptions from general popularity research. Bollen et al. [4] address financial context but predict market outcomes rather than social media dynamics themselves.

4. **Temporal evaluation weakness** — A recurring pattern across the reviewed literature is the use of random or unspecified train-test splits for time-series prediction tasks. Szabo and Huberman [1], Bandari et al. [3], and Cheng et al. [5] all evaluate without strict temporal partitioning, risking information leakage from future observations into training data. Tashman [14] demonstrated that rolling-origin evaluation — where the forecasting origin advances forward through time — produces more reliable accuracy estimates for temporal prediction tasks than fixed holdout splits. Bergmeir and Benítez [15] showed empirically that random cross-validation overestimates predictive accuracy for time-dependent data, recommending blocked or expanding-window schemes that preserve temporal ordering. Despite these methodological advances being well-established in the forecasting literature, they remain largely unadopted in social media prediction studies. This inflates reported performance and leaves unresolved whether models generalise to genuinely unseen future periods — a critical requirement for any system intended for real-time deployment.

This project aims to address all four gaps by defining a composite binary surge target within a fixed 24-hour window, combining temporal, activity-frequency, sentiment, and textual features, applying the framework specifically to stock-related social media discussions, and evaluating with expanding-window temporal cross-validation that ensures no future data leaks into training. Whether this integration yields meaningful predictive performance remains an empirical question that the evaluation (Section 5) examines.

---

## 3. Design

### 3.1 System Architecture

<!-- Pipeline stages: loading → preprocessing → feature engineering → labelling → training → evaluation -->
<!-- Data flow diagram: input sources → intermediate outputs → final artifacts -->

### 3.2 Data Ingestion and Datasets

#### 3.2.1 Dataset Selection Rationale

<!-- Why Reddit as a data source (public, threaded, subreddit-specific, ticker-rich) -->
<!-- Why not other platforms: Twitter/X ephemeral stream lacks persistent threading; StockTwits smaller user base and less organic discussion; Reddit combines persistent threaded posts with large active communities -->
<!-- Why these two subreddits specifically: -->
<!--   r/pennystocks — sparse niche community (80,212 records), low-cap focus, tests model under data scarcity -->
<!--   r/wallstreetbets — high-volume mainstream forum (1,293,981 records), tests scalability and signal extraction from noise -->
<!-- Dual-dataset design: opposite ends of data density spectrum to test generalisability -->
<!-- Time range covered, record structure (columns/fields available) -->

<!-- Dataset summary table:
| Property              | r/pennystocks         | r/wallstreetbets       |
|-----------------------|-----------------------|------------------------|
| Records              | 80,212                | 1,293,981              |
| Date range           | [start] – [end]       | [start] – [end]        |
| Avg posts/day        | [value]               | [value]                |
| Fields used          | timestamp, title, selftext, score, num_comments | same |
| Surge-positive rate  | ~[X]%                 | ~[X]%                  |
-->

<!-- Class distribution note: surge-positive rate is approximately 15%, creating moderate class imbalance that informs model selection and threshold tuning decisions in Section 3.4 -->

#### 3.2.2 Data Collection Method

<!-- Source: pre-collected CSV exports from academic/archival Reddit datasets -->
<!-- Specific source: [name exact source — e.g., Pushshift/Arctic Shift archive, specific Kaggle dataset, or direct Reddit API dump] -->
<!-- No live API scraping — static snapshot ensures reproducibility -->
<!-- Fields retained: timestamp, title, selftext, subreddit, score, num_comments, etc. -->
<!-- Any filtering applied at collection time (date range, post type) -->

<!-- Data quality notes: -->
<!-- Known issues in raw data: deleted/removed posts (showing as [removed] or [deleted]), missing selftext fields, duplicate records -->
<!-- These are addressed in preprocessing (Section 4.2); noted here for transparency about raw data state -->

#### 3.2.3 Ethical Considerations

<!-- Public data: Reddit posts are publicly accessible; no private or deleted content used -->
<!-- Anonymity: no attempt to identify or profile individual users; no individual users singled out in results or examples (aggregated analysis only) -->
<!-- No personally identifiable information (PII) retained or processed -->
<!-- Purpose: academic research only; no trading decisions were made based on model outputs -->
<!-- Ethics approval: formal ethics approval was not required for analysis of publicly available aggregated data under university guidelines — [confirm and state explicitly] -->
<!-- Compliance with university ethics guidelines and Reddit's terms of service -->
<!-- Data storage: local only, not redistributed beyond project submission -->

#### 3.2.4 Dataset Limitations

<!-- Survivorship bias: deleted or moderated posts are not captured in the archival dataset; the analysed data represents only posts that remained publicly visible at collection time -->
<!-- Snapshot timing: engagement metrics (score, num_comments) are frozen at collection time and may not reflect final values — this is why the project uses timestamp-derived features rather than engagement scores -->
<!-- Completeness: potential gaps due to Reddit API rate limits or archival service downtime during collection period -->
<!-- Single-platform scope: findings may not generalise to other financial discussion platforms with different user bases and moderation norms -->
<!-- Temporal coverage: results are bound to the specific time period captured; market regime changes or platform policy shifts outside this window may alter surge dynamics -->

### 3.3 Technology Choices

<!-- Python, scikit-learn, XGBoost, VADER, pandas — why each was chosen over alternatives -->

#### Sentiment Tool Selection

Hutto and Gilbert [12] developed VADER specifically for social media text, incorporating rules for punctuation emphasis, capitalisation, degree modifiers, and negation. VADER outperformed individual human raters on tweet classification (F1=0.96) and generalises across contexts better than purely lexicon-based alternatives. Its design makes it suitable for Reddit posts, which share social media conventions (informal language, emoticons, emphasis through capitalisation). However, VADER's lexicon was constructed from general social media — it has no financial domain tuning, meaning that terms with specialised financial meaning (e.g., "short," "calls," "puts") may be scored incorrectly or as neutral.

Araci [13] addressed this limitation with FinBERT, a BERT-based language model further pre-trained on financial corpora and fine-tuned for financial sentiment classification. FinBERT achieves state-of-the-art results on financial sentiment datasets by capturing contextual meaning that lexicon-based tools miss. However, transformer models carry significant computational cost — inference on hundreds of thousands of records is substantially slower than VADER's rule-based approach.

VADER was chosen as the primary sentiment tool for this project for its speed and social media design, with the acknowledged trade-off that financial domain specificity is limited. The configurable sentiment component architecture allows future upgrade to FinBERT without pipeline restructuring.

### 3.4 Method Design

#### 3.4.1 Feature Design

<!-- Why backward-looking features (leakage prevention argument) -->
<!-- 9 features with formal definitions -->

#### 3.4.2 Surge Definition

<!-- Why a composite surge metric rather than raw volume threshold -->

A surge is defined using a composite metric combining z-score normalised posting volume growth and sentiment change:

> *composite = (w₁ × z_volume) + (w₂ × z_sentiment)*

where z-scores are computed using training-partition statistics only (preventing leakage), and a record is labelled surge (1) if composite exceeds threshold *τ*. The target uses posting volume (timestamp-derived record counts) rather than engagement scores (which are future-contaminated snapshot values). Default configuration: w₁ = w₂ = 0.5, τ = 1.5 standard deviations.

A two-phase experimental approach validates the composite design: Phase 1 uses volume-only (w₂ = 0) as baseline; Phase 2 uses equal composite (w₂ = 0.5) to test whether sentiment adds predictive value.

<!-- Formula: S = w1 * z_volume + w2 * z_sentiment -->
<!-- Threshold τ selection via sensitivity sweep -->

#### 3.4.3 Model Selection Strategy

<!-- Why three model families (linear, ensemble, boosting) for comparison -->
<!-- Why AUC-ROC as primary selection criterion given class imbalance -->

#### 3.4.4 Temporal Validation Design

<!-- Why expanding-window temporal CV rather than k-fold or random splits -->
<!-- k=4 folds, 3 splits structure -->
<!-- Temporal train/test split (80/20) -->

### 3.5 Reproducibility Design

<!-- Fixed seeds, serialised models, config JSON — why these matter -->

---

## 4. Implementation

### 4.1 Code Organisation

<!-- Module structure: src/surge_pipeline/ layout -->

### 4.2 Data Loading and Preprocessing

<!-- Text cleaning, ticker extraction (regex + stopword filtering + known-ticker validation) -->
<!-- Sentiment analysis: VADER compound scoring, deduplication optimisation -->

### 4.3 Feature Engineering

<!-- 9 features with temporal windowing -->
<!-- Implementation specifics and edge cases -->

### 4.4 Surge Labelling

<!-- Composite metric implementation, configurable threshold -->

### 4.5 Model Training

<!-- Multi-model training with hyperparameter search -->
<!-- LR: 10 configs, RF: 36 configs, XGB: ≤50 configs -->
<!-- Expanding-window split logic, fold construction -->
<!-- GridSearchCV with custom scorer, final retraining procedure -->

### 4.6 Evaluation Pipeline

<!-- Statistical tests implementation -->
<!-- Bootstrap CI, McNemar's test, baseline comparisons -->

### 4.7 Challenges and Decisions

<!-- Deviations from original design (reference decision log) -->
<!-- Performance bottlenecks and optimisations applied -->
<!-- Edge cases discovered during development -->

### 4.8 Implementation Progress

<!-- All pipeline stages functional, end-to-end run producing artefacts -->

---

## 5. Evaluation

### 5.1 Evaluation Against Project Objectives

<!-- For each objective: state the goal, present the measured outcome, give a verdict (met / partially met / not met) -->

#### 5.1.1 Objective 1: Predict Posting-Volume Surges

<!-- Did the models beat random and single-feature baselines? -->
<!-- What tier was achieved (minimum 0.60 / target 0.70 / stretch 0.80)? -->

#### 5.1.2 Objective 2: Compare Multiple ML Approaches

<!-- Did the multi-model comparison reveal meaningful differences? -->
<!-- Were differences statistically significant (McNemar's test)? -->

#### 5.1.3 Objective 3: Demonstrate Temporal Validity

<!-- Did expanding-window CV and temporal train/test split prevent data leakage? -->
<!-- Evidence the model generalises to unseen time periods? -->

#### 5.1.4 Objective 4: Build a Reproducible Pipeline

<!-- Can results be recreated from config JSON and fixed seeds? -->
<!-- Are all artifacts traceable? -->

### 5.2 Results

<!-- Present raw results BEFORE analysis. Readers need to see the evidence before the argument. -->

#### 5.2.1 Model Performance Metrics

<!-- Per-model metrics table: Precision, Recall, F1, AUC-ROC with 95% bootstrap CIs -->
<!-- Success tier mapping table (model → tier achieved) -->

#### 5.2.2 Model Comparison

<!-- ROC curves (combined overlay showing all models + random baseline) -->
<!-- McNemar's pairwise significance table (test statistic, p-value, significance after Bonferroni correction) -->

#### 5.2.3 Baseline Comparisons

<!-- Baseline comparison table (random baseline AUC, best single-feature baseline AUC, improvement margin) -->

#### 5.2.4 Error Analysis

<!-- Confusion matrices per model (with actual counts) -->
<!-- Threshold sensitivity curve (metrics vs. classification threshold) -->
<!-- Feature importance bar chart (from Random Forest) -->

### 5.3 Critical Analysis

#### 5.3.1 Why Did the Best Model Outperform Others?

<!-- Feature importance differences between models -->
<!-- Decision boundary complexity (linear vs. tree-based) -->
<!-- Sensitivity to class imbalance -->

#### 5.3.2 What Worked Well?

<!-- Temporal CV preventing overly optimistic estimates -->
<!-- Composite surge metric capturing multi-dimensional signal -->
<!-- Backward-looking features avoiding look-ahead bias -->

#### 5.3.3 What Did Not Work?

<!-- False positive patterns — what types of records are misclassified? -->
<!-- Features with low importance — were they worth including? -->
<!-- VADER limitations on financial/Reddit slang -->

#### 5.3.4 Unexpected Outcomes

<!-- Any model performing surprisingly well or poorly? -->
<!-- Threshold sensitivity — did small τ changes cause large performance shifts? -->
<!-- Class imbalance impact — precision vs. recall trade-off -->

### 5.4 Limitations and Proposed Improvements

<!-- Ticker extraction false positives → known-ticker validation list -->
<!-- VADER ceiling for financial text → FinBERT -->
<!-- Single subreddit scope → multi-subreddit expansion -->
<!-- Class imbalance (~15% positive) → SMOTE, cost-sensitive learning -->
<!-- Static feature window (24h) → multi-scale windows -->

### 5.5 Originality and Contribution

<!-- Composite surge metric — not found in prior literature for penny stock forums -->
<!-- Expanding-window temporal CV — addresses common leakage mistake -->
<!-- Multi-model comparison with statistical significance testing -->
<!-- Frame as incremental advances with clear academic value -->

---

## 6. Conclusion

### 6.1 Current Achievements

<!-- Functional end-to-end pipeline from raw Reddit data to trained classifiers -->
<!-- Multi-model comparison with statistical significance testing -->
<!-- Reproducible results via fixed seeds, serialised configs, and automated pipeline -->
<!-- Evaluation framework with bootstrap CIs, McNemar's test, and baseline comparisons -->

### 6.2 Key Findings

<!-- Best model performance and which tier was achieved -->
<!-- Which features contributed most to prediction -->
<!-- Whether posting-volume surges are predictable from backward-looking features (answer to research question) -->
<!-- Relationship between findings and existing literature -->

### 6.3 Remaining Work

<!-- Sentiment model upgrade (VADER → FinBERT) -->
<!-- Ticker validation refinement -->
<!-- Class imbalance handling -->
<!-- Multi-subreddit expansion -->

### 6.4 Future Developments

<!-- Real-time inference system with streaming Reddit data -->
<!-- Graph-based diffusion features (cross-ticker mention networks) -->
<!-- Multi-scale temporal windows (6h, 24h, 72h) -->
<!-- Integration with market data for downstream trading signal validation -->

---

## References

[1] G. Szabo and B. A. Huberman, "Predicting the popularity of online content," *Communications of the ACM*, vol. 53, no. 8, pp. 80–88, 2010.

[2] K. Lerman and T. Hogg, "Using a model of social dynamics to predict popularity of news," in *Proc. 19th International Conference on World Wide Web (WWW '10)*, pp. 621–630, 2010.

[3] R. Bandari, S. Asur, and B. A. Huberman, "The pulse of news in social media: Forecasting popularity," in *Proc. 6th International AAAI Conference on Weblogs and Social Media (ICWSM '12)*, pp. 26–33, 2012.

[4] J. Bollen, H. Mao, and X. Zeng, "Twitter mood predicts the stock market," *Journal of Computational Science*, vol. 2, no. 1, pp. 1–8, 2011.

[5] J. Cheng, L. Adamic, P. A. Dow, J. M. Kleinberg, and J. Leskovec, "Can cascades be predicted?," in *Proc. 23rd International Conference on World Wide Web (WWW '14)*, pp. 925–936, 2014.

[6] F. Wang and B. A. Huberman, "Quantifying long-term scientific impact," *Science*, vol. 342, no. 6154, pp. 127–132, 2013.

[7] S. Kong, Q. Mei, L. Feng, F. Ye, and Z. Zhao, "Predicting bursts and popularity of hashtags in real-time," in *Proc. 37th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pp. 927–930, 2014.

[8] C. Yuan and W. Li, "Forecasting the development trend of early-stage information diffusion based on empirical data," *Physica A: Statistical Mechanics and its Applications*, vol. 524, pp. 157–167, 2019.

[9] C. Long, B. Lucey, and L. Yarovaya, "I just like the stock: the role of Reddit sentiment in the GameStop share rally," *The Financial Review*, vol. 58, no. 1, pp. 19–37, 2023.

[10] M. Costola, M. Iacopini, and C. Santagiustina, "Self-induced consensus of Reddit users to characterise the GameStop short squeeze," *Scientific Reports*, vol. 12, art. 13780, 2022.

[11] A. Mancini, A. Desiderio, B. Marafino, and A. Navigli, "Detecting pump and dump stock market manipulation from online forums," *Digital Finance*, vol. 6, pp. 365–393, 2024.

[12] C. J. Hutto and E. Gilbert, "VADER: A parsimonious rule-based model for sentiment analysis of social media text," in *Proc. 8th International AAAI Conference on Weblogs and Social Media (ICWSM '14)*, pp. 216–225, 2014.

[13] D. Araci, "FinBERT: Financial sentiment analysis with pre-trained language models," *arXiv preprint arXiv:1908.10063*, 2019.

[14] L. J. Tashman, "Out-of-sample tests of forecasting accuracy: An analysis and review," *International Journal of Forecasting*, vol. 16, no. 4, pp. 437–450, 2000.

[15] C. Bergmeir and J. M. Benítez, "On the use of cross-validation for time series predictor evaluation," *Information Sciences*, vol. 191, pp. 192–213, 2012.

[16] M. Fernández-Delgado, E. Cernadas, S. Barro, and D. Amorim, "Do we need hundreds of classifiers to solve real world classification problems?," *Journal of Machine Learning Research*, vol. 15, no. 1, pp. 3133–3181, 2014.
