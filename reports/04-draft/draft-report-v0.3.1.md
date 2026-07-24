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

This project follows the **CM3005 Data Science** project template, *Predictive Modelling of Social Media Trend Emergence*. It builds a machine learning system that predicts whether a stock ticker's Reddit discussion is about to surge — using only backward-looking features available at observation time. Three classifiers — Logistic Regression, Random Forest, and XGBoost — are trained and compared on this binary task.

The project has three objectives:

- Develop a predictive model using early-stage discussion features (temporal patterns, activity frequency, sentiment) to forecast per-ticker surges before they occur
- Compare multiple ML approaches to determine whether model complexity improves prediction over simpler baselines
- Validate that predictions generalise to unseen future time periods through temporal evaluation protocols that prevent data leakage — a common methodological weakness in social media prediction studies

### 1.2 Problem Statement and Motivation

Stock-related discussions on Reddit can go from quiet to frenzied within hours. A ticker attracting two posts yesterday might appear in fifty today, triggered by earnings surprises, speculative momentum, or coordinated retail interest. These surges develop too quickly for manual monitoring, particularly across forums where thousands of tickers are discussed daily.

This is primarily a research problem: can the onset of a social media surge be detected from the discussion patterns that precede it? Answering this question also has practical relevance for financial analysts seeking early warning of emerging narratives, surveillance teams watching for manipulation, and quantitative researchers studying how attention propagates through online communities.

Existing research predicts overall content popularity [1][3] or models sentiment-to-market correlations [4], but these address different problems. Popularity prediction forecasts *eventual* reach rather than detecting rapid *onset*; sentiment-market studies predict price movements rather than social media dynamics. The reviewed literature does not appear to address predicting the onset of a composite engagement-and-sentiment surge within a bounded short-term window for individual tickers — the gap this project aims to fill (see Section 2.5).

The 2021 GameStop episode illustrated the stakes: rapidly escalating discussion translated into market impact within days [11][12], with no automated system flagging the surge early. This project explores whether such surges are predictable from the discussion patterns that precede them.

### 1.3 Prediction Scope and Surge Definition

A **surge** is a statistically significant increase in both posting volume and sentiment intensity for a specific ticker within a 24-hour window, measured by a composite metric combining normalised volume growth with sentiment change magnitude. The target derives from posting volume (timestamp-based record counts) rather than engagement scores like upvotes, which are future-contaminated snapshot values that would introduce look-ahead bias. Z-scores use training-partition statistics only, preventing leakage. The formal definition, weighting, and threshold selection are detailed in Section 3.3.2.

### 1.4 Scope

**In scope:** Two pre-collected Reddit datasets representing opposite ends of the data density spectrum — r/pennystocks (80,212 records), a sparse niche community, and r/wallstreetbets (1,293,981 records), a high-volume mainstream forum. This dual-dataset design tests whether the methodology generalises across community sizes or whether data density is a binding constraint. Also in scope: feature engineering from text and timestamps, binary classification, a reproducible pipeline with seeded randomness, and cross-dataset transfer evaluation.

**Out of scope:** Real-time ingestion, production deployment, trading signal generation, multi-class targets, cross-platform fusion.

The system achieves AUC-ROC of 0.889 on the high-density dataset and 0.754 on the sparse dataset, demonstrating that surges are predictable from observation-time features but that data density significantly affects performance.

---

## 2. Literature Review

### 2.1. Early Popularity Prediction

Szabo and Huberman [1] demonstrated strong log-linear correlations between early and later popularity on YouTube and Digg, showing that simple regression on early view counts can predict future attention with high accuracy. However, their model assumes a stationary growth process and relies on content that has already accumulated measurable engagement. This limits applicability to *pre-engagement* prediction — the model cannot make forecasts at or near the time of posting, which is precisely the regime of interest for early surge detection. Furthermore, their evaluation was limited to platforms with specific ranking algorithms (Digg's front-page mechanism), raising questions about generalisability to finance-focused forums where content discovery differs fundamentally.

Lerman and Hogg [2] modelled the interplay between social network structure and content discovery, highlighting that popularity depends on behavioural dynamics beyond simple cumulative counts. Their agent-based approach provided mechanistic insight but required detailed knowledge of platform-specific network topology — data rarely available for financial discussion platforms. The model also assumed homogeneous user behaviour, which is unrealistic in stock forums where institutional participants, retail traders, and bots exhibit very different engagement patterns.

### 2.2. Machine Learning and Content-Based Prediction

Bandari et al. [3] advanced the field by demonstrating that content metadata (source, category, subjectivity, named entities) could predict popularity *before* engagement accumulates, achieving ~84% classification accuracy. This was a methodologically important shift toward pre-publication prediction. However, the study used coarse popularity bins rather than continuous or binary surge targets, and the feature set was designed for news articles rather than user-generated financial discussion. Their reliance on manually engineered features also limits transferability — features like "news source reputation" have no direct analogue in anonymous forum posts. The 84% accuracy figure, while frequently cited, should also be interpreted cautiously: it was measured on a four-class classification task with uneven class sizes, meaning that majority-class baselines already achieve substantial accuracy.

Chen and Guestrin [9] introduced XGBoost, a scalable gradient boosting framework that has become a dominant method for structured/tabular classification tasks. Its regularisation mechanisms (L1 and L2 on leaf weights), built-in handling of sparse data, and efficient parallelised tree construction make it well-suited to imbalanced classification problems with heterogeneous feature types — precisely the characteristics of surge prediction. XGBoost consistently outperforms Random Forest and Logistic Regression on tabular benchmarks [9], motivating its inclusion as the most complex model in this project's three-classifier comparison. The `scale_pos_weight` parameter provides native support for class imbalance without requiring external resampling, which is methodologically preferable when training data is temporally ordered and synthetic sample generation could violate temporal assumptions.

### 2.3. NLP and Sentiment Analysis for Finance

Bollen et al. [4] demonstrated that aggregate Twitter mood (particularly the "Calm" dimension) predicted Dow Jones movements with ~87.6% directional accuracy. This was influential in establishing sentiment as a predictive signal for finance. However, the study has significant methodological limitations that subsequent literature has noted: the evaluation period was short (approximately one month of trading days), no out-of-sample validation was reported, and the causal mechanism is unclear — external events may simultaneously drive both social media mood and market outcomes without one causing the other. The lexicon-based mood measurement tools (OpinionFinder and GPOMS) also lack domain specificity for financial language, where terms like "short," "bearish," or "moon" carry specialised meaning that general-purpose sentiment tools misclassify.

Hutto and Gilbert [10] developed VADER (Valence Aware Dictionary and sEntiment Reasoner), a rule-based sentiment tool specifically designed for social media text. VADER handles informal language features common on Reddit — capitalisation for emphasis, emoticons, slang intensifiers, and negation — achieving F1=0.96 on social media benchmarks, substantially outperforming general-purpose lexicons. This project uses VADER rather than TextBlob or GPOMS because its social media orientation better matches the Reddit domain. However, VADER still lacks finance-specific terms (e.g., "diamond hands," "YOLO," "short squeeze"), which is acknowledged in the risk register and motivates the configurable sentiment component weight — allowing the system to down-weight sentiment when its signal-to-noise ratio is low.

### 2.4. Reddit Financial Communities and Retail Investor Behaviour

The January 2021 GameStop episode demonstrated how rapidly escalating Reddit discussion can translate into real market impact. Hasso et al. [11] analysed brokerage accounts during the GameStop frenzy and found that participants were predominantly existing high-risk retail traders rather than first-time investors, suggesting that social media surges amplify pre-existing speculative behaviour rather than creating entirely new market participants. Betzer and Harries [12] provided empirical evidence linking r/wallstreetbets posting volume to abnormal GameStop trading volume, establishing a direct quantitative relationship between Reddit activity metrics and market outcomes.

Bradley et al. [13] studied r/wallstreetbets investment recommendations more broadly, finding that pre-GameStop posts exhibited genuine stock-picking skill (positive abnormal returns), but that the community's culture shifted post-January 2021, leading to deteriorating recommendation quality. This temporal behavioural shift is directly relevant to this project's use of temporal validation — models trained on one period may not generalise to another due to community evolution, motivating the expanding-window cross-validation design.

These studies establish that Reddit financial communities generate measurable market-relevant signals, but all analyse the *consequences* of surges retrospectively rather than predicting their *onset*. This distinction is central to the present project's contribution: rather than asking "did Reddit activity move the stock price?", it asks "can we detect that Reddit activity is about to surge?"

Reddit's financial communities span a broad density spectrum. r/wallstreetbets (13+ million subscribers) produces thousands of posts daily with heavy ticker concentration in large-cap equities, while r/pennystocks serves a niche community discussing low-capitalisation stocks with sparse, dispersed discussion across thousands of tickers. This density contrast has direct methodological implications: predictive models require sufficient per-ticker history to estimate temporal features reliably, and communities with extreme ticker dispersion may fall below viable data thresholds. No prior study has explicitly tested whether surge prediction methodology transfers across communities of different densities — a gap this project addresses through its dual-dataset evaluation design.

### 2.5. Information Diffusion and Cascade Prediction

Cheng et al. [5] achieved ~79.5% accuracy (AUC 0.877) predicting whether Facebook photo cascades would double in size, using only early resharing observations. The methodological rigour was strong: large sample size (millions of cascades), temporal features derived from propagation speed, and structural virality metrics. However, the study focused exclusively on image resharing on Facebook — a platform with explicit social graph structure and algorithmic content distribution that differs markedly from text-based financial forums. The cascade framework also assumes discrete, traceable sharing events, whereas engagement on discussion platforms (upvotes, comments) often lacks explicit propagation chains. The concept of "early propagation speed" nevertheless informs this project's `time_since_previous` feature as a proxy for activity acceleration.

Wang and Huberman [6] and Kong et al. [7] characterised popularity as following identifiable temporal lifecycles (emergence → growth → peak → decline). While these frameworks provide useful conceptual grounding, both studies are primarily descriptive rather than predictive — they identify patterns retrospectively but do not offer methods for real-time forecasting. Yuan and Li [8] extended this by suggesting that early-stage signals may predict later evolution, but their work focused on emergency information diffusion rather than financial contexts, and the temporal granularity (days to weeks) is coarser than the 24-hour window relevant to stock discussion surges.

### 2.6. Temporal Validation and Class Imbalance

Bergmeir and Benítez [14] demonstrated empirically that standard k-fold cross-validation overestimates predictive accuracy on time-dependent data by allowing future observations to inform training. They recommended blocked or expanding-window cross-validation schemes that preserve temporal ordering, finding that random splits could produce substantially inflated performance estimates. This finding directly motivates this project's expanding-window temporal cross-validation design (k=4 folds), where each fold's training set is strictly earlier than its validation set — preventing the optimistic bias that standard cross-validation introduces for temporally structured social media data.

Chawla et al. [15] introduced SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in binary classification, demonstrating that synthetic oversampling of the minority class improves classifier sensitivity without requiring additional real data. However, SMOTE assumes that linear interpolation between minority samples produces valid synthetic instances — an assumption that breaks down for temporally ordered data where adjacent samples have causal relationships. For this reason, this project uses class-weighted loss functions (`class_weight='balanced'` for Logistic Regression and Random Forest; `scale_pos_weight` for XGBoost) rather than resampling, avoiding the generation of synthetic temporal records that could introduce spurious patterns.

### 2.7. Synthesis and Identified Research Gap

Table 1 summarises the reviewed literature against five dimensions relevant to this project.

| Study | Platform | Prediction Target | Pre-engagement? | Finance-specific? | Temporal validation? |
|-------|----------|-------------------|-----------------|--------------------|--------------------|
| Szabo & Huberman [1] | YouTube, Digg | Future view count | No | No | No |
| Lerman & Hogg [2] | Digg | Story popularity | No | No | No |
| Bandari et al. [3] | News articles | Popularity bin | Yes | No | No |
| Bollen et al. [4] | Twitter | Market direction | N/A | Yes | No |
| Cheng et al. [5] | Facebook | Cascade doubling | Partial | No | No |
| Chen & Guestrin [9] | (method paper) | — | — | — | — |
| Hutto & Gilbert [10] | Social media | (tool paper) | — | — | — |
| Hasso et al. [11] | Reddit/Brokerage | Participation | N/A | Yes | N/A |
| Bradley et al. [13] | Reddit (WSB) | Return prediction | N/A | Yes | Partial |
| Bergmeir & Benítez [14] | (method paper) | — | — | — | Yes |

*Table 1: Literature comparison across dimensions relevant to surge prediction.*

The literature establishes three findings: (a) early behavioural signals contain predictive information about future online attention [1][5]; (b) multiple feature types (temporal, content, sentiment, structural) each contribute explanatory power [2][3][4]; and (c) popularity follows identifiable temporal dynamics that can theoretically be detected early [7][8]. Recent Reddit-specific research [11][12][13] confirms that these communities generate quantifiable, market-relevant discussion patterns.

Three critical gaps remain:

1. **Prediction target mismatch** — Most studies predict *eventual outcomes* (final popularity, total cascade size, market direction) rather than detecting the *onset* of rapid growth within a bounded time window. No reviewed study defines or predicts a composite volume-and-sentiment surge within a fixed short-term window.

2. **Single-signal approaches** — Each research strand demonstrates one feature category's value (Szabo: temporal; Bandari: content; Bollen: sentiment; Cheng: structural), yet few combine signals into an integrated predictive framework. The literature suggests multiple signal types interact during trend formation [2][7], but empirical integration remains limited.

3. **Domain and density transfer** — Reviewed studies draw on general social media (YouTube, Digg, Facebook, Twitter) rather than finance-specific platforms. Reddit financial research [11][12][13] analyses surge *consequences* but does not predict surge *onset*. Furthermore, no study tests whether prediction methodology transfers across communities of different data densities — a critical practical question given the heterogeneity of online financial forums.

This project addresses all three gaps: Section 3.3.2 defines the composite binary surge target within a fixed 24-hour window (gap 1); Section 3.3.1 describes the multi-signal feature set combining temporal, activity-frequency, sentiment, and textual features (gap 2); and the dual-dataset evaluation on r/wallstreetbets and r/pennystocks with cross-dataset transfer testing directly addresses the density transfer problem (gap 3).

---

## 3. Design

### 3.1 System Architecture

<!-- Pipeline stages: loading → preprocessing → feature engineering → labelling → training → evaluation -->
<!-- Data flow diagram: input sources → intermediate outputs → final artifacts -->

### 3.2 Technology Choices

<!-- Python, scikit-learn, XGBoost, VADER, pandas — why each was chosen over alternatives -->

### 3.3 Method Design

#### 3.3.1 Feature Design

<!-- Why backward-looking features (leakage prevention argument) -->
<!-- 9 features with formal definitions -->

#### 3.3.2 Surge Definition

<!-- Why a composite surge metric rather than raw volume threshold -->

A surge is defined using a composite metric combining z-score normalised posting volume growth and sentiment change:

> *composite = (w₁ × z_volume) + (w₂ × z_sentiment)*

where z-scores are computed using training-partition statistics only (preventing leakage), and a record is labelled surge (1) if composite exceeds threshold *τ*. The target uses posting volume (timestamp-derived record counts) rather than engagement scores (which are future-contaminated snapshot values). Default configuration: w₁ = w₂ = 0.5, τ = 1.5 standard deviations.

A two-phase experimental approach validates the composite design: Phase 1 uses volume-only (w₂ = 0) as baseline; Phase 2 uses equal composite (w₂ = 0.5) to test whether sentiment adds predictive value.

<!-- Formula: S = w1 * z_volume + w2 * z_sentiment -->
<!-- Threshold τ selection via sensitivity sweep -->

#### 3.3.3 Model Selection Strategy

<!-- Why three model families (linear, ensemble, boosting) for comparison -->
<!-- Why AUC-ROC as primary selection criterion given class imbalance -->

#### 3.3.4 Temporal Validation Design

<!-- Why expanding-window temporal CV rather than k-fold or random splits -->
<!-- k=4 folds, 3 splits structure -->
<!-- Temporal train/test split (80/20) -->

### 3.4 Reproducibility Design

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

[1] G. Szabo and B. A. Huberman. 2010. Predicting the popularity of online content. *Communications of the ACM* 53, 8 (August 2010), 80–88. DOI: 10.1145/1787234.1787254

[2] K. Lerman and T. Hogg. 2010. Using a model of social dynamics to predict popularity of news. In *Proceedings of the 19th International Conference on World Wide Web (WWW '10)*. ACM, New York, NY, 621–630. DOI: 10.1145/1772690.1772754

[3] R. Bandari, S. Asur, and B. A. Huberman. 2012. The pulse of news in social media: Forecasting popularity. In *Proceedings of the 6th International AAAI Conference on Weblogs and Social Media (ICWSM '12)*. AAAI Press, 26–33.

[4] J. Bollen, H. Mao, and X. Zeng. 2011. Twitter mood predicts the stock market. *Journal of Computational Science* 2, 1 (March 2011), 1–8. DOI: 10.1016/j.jocs.2010.12.007

[5] J. Cheng, L. Adamic, P. A. Dow, J. M. Kleinberg, and J. Leskovec. 2014. Can cascades be predicted? In *Proceedings of the 23rd International Conference on World Wide Web (WWW '14)*. ACM, New York, NY, 925–936. DOI: 10.1145/2566486.2567997

[6] F. Wang and B. A. Huberman. 2012. Popularity evolution of online content. Unpublished manuscript. arXiv:1212.4043.

[7] S. Kong, L. Mei, F. Feng, and Z. Ye. 2014. Predicting lifespans of popular tweets in microblog. In *Proceedings of the 37th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '14)*. ACM, New York, NY, 1103–1106. DOI: 10.1145/2600428.2609550

[8] C. Yuan and J. Li. 2019. Research on the prediction model of the diffusion of emergencies in social media. *Information Discovery and Delivery* 47, 4 (November 2019), 203–212. DOI: 10.1108/IDD-05-2019-0039

[9] T. Chen and C. Guestrin. 2016. XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*. ACM, New York, NY, 785–794. DOI: 10.1145/2939672.2939785

[10] C. J. Hutto and E. Gilbert. 2014. VADER: A parsimonious rule-based model for sentiment analysis of social media text. In *Proceedings of the 8th International AAAI Conference on Weblogs and Social Media (ICWSM '14)*. AAAI Press, Ann Arbor, MI.

[11] T. Hasso, D. Müller, M. Pelster, and S. Warkulat. 2022. Who participated in the GameStop frenzy? Evidence from brokerage accounts. *Finance Research Letters* 45 (March 2022), 102140. DOI: 10.1016/j.frl.2021.102140

[12] A. Betzer and J. P. Harries. 2022. How online discussion board activity affects stock trading: The case of GameStop. *Financial Markets and Portfolio Management* 36, 4 (December 2022), 443–472. DOI: 10.1007/s11408-022-00407-w

[13] D. Bradley, J. Hanousek Jr., R. Jame, and Z. Xiao. 2024. Place your bets? The market consequences of investment research on Reddit's WallStreetBets. *Journal of Financial Economics* 152 (February 2024), 103756. DOI: 10.1016/j.jfineco.2023.103756

[14] C. Bergmeir and J. M. Benítez. 2012. On the use of cross-validation for time series predictor evaluation. *Information Sciences* 191 (May 2012), 192–213. DOI: 10.1016/j.ins.2011.12.028

[15] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer. 2002. SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research* 16 (June 2002), 321–357. DOI: 10.1613/jair.953
