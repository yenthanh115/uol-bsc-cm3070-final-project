# Predicting Posting-Volume Surges on Reddit Financial Communities Using Machine Learning

## Abstract

Social media discussions in online financial communities, such as Reddit, can shift from quiet to frenzied within hours, yet detecting these **posting-volume surges** before they fully develop has received little research attention, partly because most prior approaches inadvertently use future information, a problem known as **data leakage**. This project mainly asked a question: can surges be predicted using only features that are genuinely available at observation time? <!-- why not just use only volume growth or sentiment change, why use combined, this project not only provide solution directly but also an experiment of how to find the appropriate solution for this question --> To answer this, a composite surge metric was built from normalised volume growth and sentiment change, and **eleven features** were drawn from timestamps and text content, deliberately excluding <!--too specific but do not explain why exlcuding it --> engagement scores that only settle after a post has already gained traction. Three classifiers (Logistic Regression, Random Forest, and XGBoost) were trained with expanding-window **temporal cross-validation** and tested on held-out future data from two subreddits at opposite ends of the density spectrum: the niche r/pennystocks (80,212 records) and the high-traffic r/wallstreetbets (1,293,981 records). On the larger dataset, XGBoost achieved AUC-ROC of 0.861 and Random Forest reached 0.854, both clearing the stretch performance tier; on the sparser community, Random Forest attained 0.746 at the target tier. All pairwise differences proved statistically significant (McNemar's test, p < 0.017 after Bonferroni correction), and cross-dataset transfer produced AUC of 0.694, useful but clearly requiring community-specific recalibration. Taken together, these findings show that **machine learning** can anticipate surges from backward-looking signals alone, that data density is the main bottleneck for **binary classification** accuracy, and that the leakage-free methodology developed here transfers readily to other timestamped  platforms.

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

Prior work in this area tends to focus on related but distinct problems: forecasting eventual content reach rather than detecting rapid onset, or predicting price movements rather than social media dynamics themselves. In the reviewed literature, predicting the onset of a volume-and-sentiment surge for individual tickers within a short-term window remains largely unaddressed (see Section 2.6).

This project explores whether such surges are predictable from the discussion patterns that precede them.

### 1.3 Prediction Scope and Surge Definition

The original project template uses the term "trend emergence," but trends can be gradual and sustained, making them difficult to label objectively within a fixed time window. This project narrows the scope to **surges**: statistically significant short-term increases in both **posting volume** and **sentiment intensity** for a specific ticker within **a 24-hour window**, measured by a composite metric combining normalised volume growth with sentiment change magnitude. Surges are discrete, quantifiable events that lend themselves to binary classification, making them a more tractable operationalisation of the broader "trend" concept. A surge represents the earliest observable stage of a trend, so predicting surges is equivalent to detecting trends at their point of emergence.

The target derives from posting volume (timestamp-based record counts) rather than engagement scores like upvotes, which are future-contaminated snapshot values that would introduce look-ahead bias. Z-scores use training-partition statistics only, preventing leakage. The formal definition, weighting, and threshold selection are detailed in Section 3.3.

### 1.4 Scope

**In scope:** Two pre-collected Reddit datasets representing opposite ends of the data density spectrum — r/pennystocks (80,212 records), a sparse niche community, and r/wallstreetbets (1,293,981 records), a high-volume mainstream forum. This dual-dataset design tests whether the methodology generalises across community sizes or whether data density is a binding constraint. Also in scope: feature engineering from text and timestamps, binary classification, a reproducible pipeline with seeded randomness, and cross-dataset transfer evaluation.

**Out of scope:** Real-time ingestion, production deployment, trading signal generation, multi-class targets, cross-platform fusion.

The system achieves AUC-ROC of 0.861 on the high-density dataset and 0.746 on the sparse dataset, demonstrating that surges are predictable from observation-time features but that data density significantly affects performance.

### 1.5 Project Timeline

<figure align="center">
  <img src="figures/01-gantt-chart-v0.2.png" alt="Project Timeline" width="1000">
  <figcaption>Figure 2: Project Timeline (Gantt Chart).</figcaption>
</figure>

---

## 2. Literature Review

<!--
## 1. Establish the Current State of Knowledge
Goal: Demonstrate a deep understanding of the key concepts, theories, and historical timeline of your research topic. 

* What are the foundational theories or models governing this field?
* Who are the seminal authors and leading researchers on this topic?
* How has the academic consensus on this topic evolved over time?
* What are the standardized definitions and terms used by experts?

## 2. Identify Gaps and Weaknesses in Extant Research
Goal: Pinpoint what previous research has missed, ignored, or failed to resolve to justify why your own study is necessary. 

* What blind spots or unexamined variables exist in the current literature?
* What are the persistent flaws, limitations, or biases in past studies?
* Where do different studies conflict, disagree, or present contradictory results?
* Is the existing research outdated or lacking application to a new population/context?

## 3. Evaluate Methodologies and Research Designs
Goal: Analyze how previous scientists gathered data so you can adopt effective methods and avoid common technical pitfalls.  

* What data collection methods (qualitative, quantitative, or mixed) are most prevalent?
* What sampling techniques, tools, or data metrics did prior researchers use?
* What structural constraints or ethical obstacles did previous authors face?
* How will your chosen methodological approach address the limitations of prior setups? 

## 4. Synthesize and Connect Prior Findings
Goal: Move beyond mere summary by grouping separate papers into thematic clusters to show the bigger picture. 

* What overarching themes, trends, or sub-topics connect these separate sources?
* How does study A support, extend, or directly refute the findings of study B?
* What major conceptual frameworks emerge when these papers are viewed collectively?
* How do local or niche findings translate to a broader global environment? 

## 5. Provide a Rationale for Your Own Study
Goal: Explicitly link your reading to your own research hypotheses, questions, or project goals.

* How does the existing literature directly inform your current research questions?
* In what explicit ways will your study fill the literature gaps you discovered?
* How will you use past findings as a benchmark to validate your final results? 

-->

### 2.1. The Predictability of Online Attention

The foundational question underlying this project, *can future surges in social media activity be predicted?*, was first addressed through research on *online popularity prediction*. This field established that online attention is not random: content that attracts early engagement tends to attract more, following patterns that are statistically detectable.

Szabo and Huberman [1] produced the seminal result, demonstrating strong log-linear correlations between early and later popularity on YouTube and Digg. Their regression model showed that a content item's view count at time *t* predicts its eventual popularity with high accuracy. This established the core principle: **early behavioural signals carry predictive information about future attention**. However, the model assumes a stationary growth process and requires content to have already accumulated measurable engagement before prediction becomes possible. It cannot forecast at or near the time of posting.

Lerman and Hogg [2] extended this understanding by modelling the interaction between social network structure and content discovery, demonstrating that popularity depends on behavioural dynamics beyond simple cumulative counts. Their agent-based approach revealed that network position and user browsing patterns mediate how content gains visibility. Wang and Huberman [6] and Kong et al. [7] further characterised online attention as following identifiable temporal lifecycles (emergence, growth, peak, and decline) suggesting that content at different lifecycle stages exhibits different observable signatures.

The academic consensus that emerged from this first wave of research can be summarised as: *online attention is predictable from early signals, follows lifecycle dynamics, and is mediated by platform-specific network effects*. However, these models all require content to have already gained some traction before prediction is possible, and they target *eventual* popularity rather than the *onset* of rapid growth.

```mermaid
graph LR
    A[Emergence<br/><i>Few posts, low signal</i>] --> B[Growth<br/><i>Accelerating activity</i>]
    B --> C[Peak<br/><i>Maximum attention</i>]
    C --> D[Decline<br/><i>Activity fading</i>]

    A -.- E[/"🎯 This project's<br/>prediction point"/]
    B -.- F[/"Traditional models<br/>require data here"/]

    style A fill:#e1f5fe,stroke:#0288d1
    style B fill:#fff9c4,stroke:#f9a825
    style C fill:#ffcdd2,stroke:#c62828
    style D fill:#f5f5f5,stroke:#9e9e9e
    style E fill:#c8e6c9,stroke:#2e7d32
    style F fill:#fff3e0,stroke:#e65100
```
*Figure 1: Online attention lifecycle model [6][7]. Traditional popularity prediction requires content to have reached the growth phase before forecasting is possible. This project targets the emergence phase, predicting a surge before substantial engagement has accumulated.*

### 2.2. The Shift Toward Pre-Engagement Prediction

A second wave of research addressed the limitation that early popularity models require existing engagement data. Bandari et al. [3] demonstrated that content metadata (source, category, subjectivity, named entities) could predict popularity *before* any engagement accumulates, achieving ~84% classification accuracy on news articles. This was a methodologically significant shift: for the first time, prediction could occur at or before publication rather than requiring a waiting period.

Cheng et al. [5] achieved ~79.5% accuracy (AUC 0.877) predicting whether Facebook photo cascades would double in size, using temporal features derived from early propagation speed and structural virality metrics. Their work showed that the *rate* of early spread, not just its magnitude, carries predictive signal about whether content will continue growing. Yuan and Li [8] extended this principle to information diffusion more broadly, suggesting that early-stage propagation patterns contain sufficient signal to forecast later trajectory.

This body of work established a second consensus: **prediction is possible before substantial engagement accumulates**, provided features capture content characteristics or early propagation dynamics. However, the prediction targets remained *eventual outcomes* (final popularity, total cascade size) rather than *rapid onset* within a bounded window. An analyst monitoring a financial forum needs to know whether discussion will surge in the *next 24 hours*, not whether it will eventually become popular. This is a distinction no reviewed study addresses.

*Table 1: Evolution of online attention prediction, from post-engagement to pre-engagement approaches.*

| Study | Year | Platform | Prediction Target | Requires Existing Engagement? | Accuracy |
|-------|------|----------|-------------------|-------------------------------|----------|
| Szabo & Huberman [1] | 2010 | YouTube, Digg | Future view count | Yes (needs early views) | r² > 0.9 |
| Lerman & Hogg [2] | 2010 | Digg | Story popularity | Yes (needs network data) | N/A (model) |
| Bandari et al. [3] | 2012 | News articles | Popularity bin | **No** (content metadata only) | ~84% |
| Cheng et al. [5] | 2014 | Facebook | Cascade doubling | Partial (early reshares) | 79.5% (AUC 0.877) |
| Yuan & Li [8] | 2019 | Weibo | Diffusion trajectory | Partial (early propagation) | N/A (descriptive) |

### 2.3. Sentiment as a Predictive Signal in Finance

Parallel to the popularity prediction literature, research in computational finance established that collective sentiment extracted from social media text carries measurable predictive information. Bollen et al. [4] demonstrated that aggregate Twitter mood, particularly the "Calm" dimension measured by GPOMS, predicted Dow Jones movements with ~87.6% directional accuracy. Despite significant methodological limitations (short evaluation period, no out-of-sample validation, unclear causal mechanism), this study was influential in establishing that **textual sentiment from social media has predictive value for financial outcomes**.

The tools used for sentiment extraction have evolved alongside this finding. General-purpose lexicons like OpinionFinder lack domain specificity for financial language, where terms like "short," "bearish," or "moon" carry specialised meaning. Hutto and Gilbert [12] developed VADER specifically for social media text, incorporating rules for punctuation emphasis, capitalisation, degree modifiers, and negation, and achieving F1=0.96 on social media benchmarks. Araci [13] later introduced FinBERT, a transformer model fine-tuned on financial corpora, capturing contextual meaning that rule-based tools miss. This evolution from general lexicons → social-media-specific rules → domain-specific deep learning represents the field's recognition that sentiment tools must match their application domain.

For this project, the key takeaway is that sentiment *change*, not just absolute sentiment, may serve as a leading indicator of surges: if a ticker's discussion becomes markedly more emotional before volume escalates, sentiment shift could provide early warning signal. This motivates including sentiment change magnitude in the composite surge metric.

*Table 2: Evolution of sentiment analysis tools relevant to financial social media.*

| Tool | Type | Domain | Strengths | Limitations for This Project |
|------|------|--------|-----------|------------------------------|
| OpinionFinder as used in [4] | Lexicon | General | Early adoption, widely cited | No social media conventions, no financial terms |
| VADER [12] | Rule-based | Social media | Handles capitalisation, emoticons, negation; F1=0.96 | No financial domain tuning ("short," "moon" misscored) |
| FinBERT [13] | Transformer | Financial text | Context-aware, domain-specific | Computationally expensive for 1M+ records |

### 2.4. Financial Discussion on Reddit

The preceding research established general principles on platforms like YouTube, Digg, Facebook, and Twitter. A more recent strand examines whether these principles apply to Reddit's financial communities specifically, and what unique characteristics these communities introduce.

Penny stocks (low-capitalisation equities typically trading below $5 per share) occupy a distinctive position: their low liquidity and limited analyst coverage mean that social media discussion can constitute a disproportionate share of available information [9][10]. Reddit's subreddit structure creates focused communities with shared norms and persistent threads, unlike Twitter's ephemeral broadcast model.

Long et al. [9] demonstrated that r/WallStreetBets posting volume correlated with abnormal trading volume and returns for discussed stocks, with effects concentrated in small-cap equities. Their analysis showed that increased Reddit attention *preceded* trading activity in their sample, suggesting that discussion patterns carry predictive signal rather than merely reflecting market events. Costola et al. [10] examined the GameStop episode specifically, finding that consensus formation within r/WallStreetBets followed measurable patterns in posting frequency and sentiment alignment *before* reaching critical mass. A small number of committed users drove broader engagement through detectable temporal signatures.

Mancini et al. [11] applied this principle to pump-and-dump detection, building predictive models from the language and timing of forum posts associated with manipulated stocks. Their work confirmed that text-based features from financial discussion forums achieve classification performance significantly above random baselines for predicting anomalous stock activity.

These studies collectively establish that **Reddit financial communities generate measurable, predictive signals**, but all predict *market outcomes* (returns, trading volume, manipulation) rather than *social media dynamics* themselves. The question of whether the discussion itself will escalate, whether a ticker's posting volume is about to surge, remains unaddressed. This is the specific prediction target of the present project.

### 2.5. Methodological Weaknesses in Prior Work

Beyond the substantive gaps identified above, a critical methodological pattern cuts across the reviewed literature: a recurring absence of rigorous temporal evaluation protocols across the studies reviewed here.

Szabo and Huberman [1] evaluate on data drawn from the same time period as training. Bandari et al. [3] use random train-test splits rather than temporal partitions, meaning models may be tested on articles published *before* some training data, a form of information leakage. Cheng et al. [5] randomly sample cascades for evaluation without preserving temporal ordering. Long et al. [9] and Costola et al. [10] analyse correlations across their full datasets without testing whether patterns discovered in earlier periods generalise to later ones.

This matters because Tashman [14] demonstrated that rolling-origin evaluation, where the forecasting origin advances forward through time, produces more reliable accuracy estimates for temporal prediction tasks than fixed or random splits. Bergmeir and Benítez [15] showed empirically that random cross-validation *overestimates* predictive accuracy for time-dependent data, recommending blocked or expanding-window schemes that preserve temporal ordering. Despite these methodological advances being well-established in the forecasting literature, they remain largely unadopted in social media prediction research.

The consequence is that reported performance figures across the reviewed studies may be inflated by temporal leakage, and it remains unresolved whether models would generalise to genuinely unseen future periods. For any system intended for real-world deployment, including surge detection, this is a critical deficiency. Fernández-Delgado et al. [16], in their large-scale classifier comparison, similarly noted that evaluation methodology substantially affects reported performance rankings, reinforcing that *how* a model is evaluated matters as much as *which* model is selected.

*Table 3: Temporal evaluation practices across reviewed studies.*

| Study | Evaluation Method | Temporal Ordering Preserved? | Leakage Risk |
|-------|-------------------|------------------------------|--------------|
| Szabo & Huberman [1] | Same-period evaluation | No | High |
| Bandari et al. [3] | Random train-test split | No | High |
| Bollen et al. [4] | Fixed holdout (1 month) | Partial | Medium |
| Cheng et al. [5] | Random cascade sampling | No | High |
| Long et al. [9] | Full-dataset correlation | No | High |
| Costola et al. [10] | Full-dataset analysis | No | High |
| Mancini et al. [11] | Chronological split | Partial | Medium |

### 2.6. Research Gap and Project Position

The literature reviewed above establishes four cumulative findings:

- Early behavioural signals predict future online attention [1][5]
- Prediction is possible before engagement accumulates, using content and propagation features [3][5][8]
- Sentiment extracted from social media carries predictive value in financial contexts [4][12]
- Reddit financial communities generate measurable signals that precede market activity [9][10][11]

Four gaps remain unaddressed:

1. **Prediction target:** All reviewed studies predict *eventual outcomes* (final popularity, cascade size, market returns) rather than detecting the *onset* of rapid growth within a bounded time window. No study defines or predicts a composite volume-and-sentiment surge within a fixed short-term window.

2. **Signal integration:** Each research strand demonstrates one feature category's value in isolation (temporal [1], content [3], sentiment [4], structural [5]), but empirical integration of multiple signal types into a unified predictive framework remains limited, despite evidence that they interact during trend formation [2][7].

3. **Domain specificity:** General social media prediction research [1][3][5] does not account for the distinctive characteristics of financial discussion (event-driven reactions, domain-specific language, speculative behaviour). Conversely, Reddit financial research [9][10][11] predicts *market* consequences of surges rather than predicting whether surges *will occur*.

4. **Temporal validity:** The use of random or unspecified evaluation splits across the literature [1][3][5][9] means reported results may not reflect real-world predictive performance. Rigorous temporal evaluation methods exist [14][15] but remain unadopted in this domain.

This project addresses these gaps directly. The composite surge metric (Section 3.3) defines a binary onset target within a 24-hour window (gap 1). The feature set (Section 3.4) combines temporal, activity-frequency, sentiment, and textual signals (gap 2). The pipeline is applied to Reddit financial communities using two subreddits at opposite ends of the data density spectrum (gap 3). Expanding-window temporal cross-validation ensures that no future information leaks into training (gap 4). Whether this integration yields meaningful predictive performance is the empirical question examined in Section 5.

---

## 3. Design
<!-- 
SIDE NOTE (DELETE LATER)
- why use Reddit not X or other platform
- why not use API
- why use 2 datasets, why pick pennystocks and wsb
- why use combined/composite metric
- why pick 3 model: LR, RF, XGB
- why use validation-fold
- why not k=5 or k=10 but k=4?
- what are stretch tier vs target tier
- have we set goal for this project (musthave tier, target tier, and stretch tier)
- discuss about data drift ? aware of it and provide solution 
-->

<!--
## 1. Establish Methodological Rationalization
Goal: Justify why your chosen approach is the most effective vehicle for your specific research.

* Why did you choose this specific paradigm (quantitative, qualitative, or mixed-methods)?
* How does this structural design directly answer your primary research questions?
* What alternative designs did you reject, and why were they less suitable?
* How does this design align with the theoretical framework built in your literature review?

## 2. Define Boundaries and the Sampling Strategy
Goal: Clearly outline who or what you are studying, and how you selected your subjects.

* What is your exact target population or unit of analysis (e.g., individuals, texts, organizations)?
* What specific sampling method did you use (e.g., random, purposive, snowball sampling)?
* What were your explicit inclusion and exclusion criteria for participants or data?
* What is your final sample size, and why is it statistically or conceptually sufficient?

## 3. Detail the Operational Procedures
Goal: Provide a step-by-step chronological recipe so another researcher can replicate your study exactly.

* What exact materials, hardware, software, or standardized instruments did you utilize?
* What were the chronological steps taken to set up and execute the study?
* How did you control for extraneous variables or potential sources of bias?
* If you conducted a pilot study, what adjustments did you make based on those trial runs?

## 4. Outline Data Collection Metrics
Goal: Explain exactly how information was gathered and measured during the execution phase.

* What specific types of data were collected (e.g., test scores, interview transcripts, digital logs)?
* How did you ensure the validity (accuracy) and reliability (consistency) of your measurement tools?
* What specific roles did the researchers play during the data gathering process?
* When, where, and over what exact timeframe was the data collected?

## 5. Formulate the Data Analysis Plan
Goal: Explain how you will transform raw data into meaningful answers before you actually present findings.

* What specific statistical tests (quantitative) or coding frameworks (qualitative) will you apply?
* Which software packages (e.g., SPSS, R, NVivo) will be used to process the data?
* How will you handle missing, corrupted, or incomplete data points?
* How do these specific analysis techniques map back to your original hypotheses?

## 6. Address Ethical and Quality Controls
Goal: Prove that your study protects participants and adheres to strict professional standards.

* What institutional review boards (IRB) or ethical committees approved this project?
* How did you secure informed consent and protect participant anonymity or data privacy?
* What steps were taken to minimize physical, psychological, or social risks to subjects?
* How did you address potential researcher bias or conflicts of interest?

------------------------------
## Quick Framework: The 4 D’s of Research Design
When writing or reviewing your design section, ensure it satisfies these four criteria:

* Define: Clearly state the parameters, variables, and populations involved.
* Do: Detail the precise actions taken during the experiment or fieldwork.
* Defend: Explain the logical reasons behind every technical choice you made.
* Disclose: Report all limitations, ethical steps, and structural constraints openly.

-->
### 3.1 System Architecture

The prediction system is implemented as a linear staged pipeline, where each stage consumes the output of the previous one and produces a well-defined intermediate artifact. This staged design was chosen over a monolithic approach for two reasons: it allows each stage to be tested and validated independently, and it enables re-running downstream stages (e.g., retraining with different hyperparameters) without recomputing expensive upstream operations (e.g., sentiment scoring of 1.3M records).

The pipeline comprises six stages:

1. **Data Loading and Preprocessing** — CSV ingestion, text cleaning, regex-based ticker extraction with stopword filtering, and record explosion (one row per record-ticker pair)
2. **Temporal Windowing** — Per-ticker forward/backward 24-hour posting counts using vectorised binary search (forward counts are used for target labelling only; features use backward counts exclusively)
3. **Sentiment Computation** — VADER compound scoring per record, with title-fallback for missing selftext
4. **Target Labelling** — Temporal 80/20 split, z-score normalisation using training-partition statistics only, composite metric computation, and binary thresholding
5. **Feature Engineering** — Eleven backward-only features derived from timestamps and text (detailed in Section 3.4)
6. **Model Training and Evaluation** — Expanding-window temporal cross-validation, hyperparameter tuning, test-set evaluation, and statistical comparison

Each stage writes its output to disk (CSV or joblib-serialised objects), creating an audit trail from raw data to final predictions. The pipeline is invoked via CLI entry points with configuration parameters passed as arguments, enabling reproducible execution with different settings (e.g., threshold sweeps, dataset switching) without code modification.

```mermaid
graph TD
    A[Raw CSV<br/><i>Reddit submissions</i>] --> B[Preprocessing<br/><i>Clean, extract tickers, explode</i>]
    B --> C[Temporal Windowing<br/><i>Forward/backward 24h counts</i>]
    C --> D[Sentiment<br/><i>VADER compound scores</i>]
    D --> E[Target Labelling<br/><i>Z-score → composite → binary</i>]
    E --> F[Feature Engineering<br/><i>11 backward-only features</i>]
    F --> G[Model Training<br/><i>Expanding-window CV</i>]
    G --> H[Evaluation<br/><i>Test set + statistical tests</i>]

    style A fill:#f5f5f5,stroke:#9e9e9e
    style E fill:#e1f5fe,stroke:#0288d1
    style G fill:#c8e6c9,stroke:#2e7d32
    style H fill:#fff9c4,stroke:#f9a825
```
*Figure 2: Pipeline architecture. Shading indicates the three critical design points: target labelling (leakage prevention), model training (temporal validation), and evaluation (statistical rigour).*

A key architectural constraint is that **no stage may access information from the future relative to the observation time of any record**. This constraint propagates through the pipeline: sentiment is scored from the record's own text (not future replies), features use only backward-looking windows, z-scores use training-partition statistics, and validation folds are strictly ordered in time. The design ensures that any prediction the system makes could, in principle, have been made at the moment the post was created — a necessary condition for any predictive system operating on temporal data.


### 3.2 Data Selection and Characteristics


<!-- Why Reddit as a data source (public, threaded, subreddit-specific, ticker-rich) -->
<!-- Why not other platforms: Twitter/X ephemeral stream lacks persistent threading; StockTwits smaller user base and less organic discussion; Reddit combines persistent threaded posts with large active communities -->
<!-- Why these two subreddits specifically: -->
<!--   r/pennystocks — sparse niche community (80,212 records), low-cap focus, tests model under data scarcity -->
<!--   r/wallstreetbets — high-volume mainstream forum (1,293,981 records), tests scalability and signal extraction from noise -->
<!-- Dual-dataset design: opposite ends of data density spectrum to test generalisability -->
<!-- Time range covered, record structure (columns/fields available) -->

<!-- Survivorship bias: deleted or moderated posts are not captured in the archival dataset; the analysed data represents only posts that remained publicly visible at collection time -->
<!-- Snapshot timing: engagement metrics (score, num_comments) are frozen at collection time and may not reflect final values — this is why the project uses timestamp-derived features rather than engagement scores -->
<!-- Completeness: potential gaps due to Reddit API rate limits or archival service downtime during collection period -->
<!-- Single-platform scope: findings may not generalise to other financial discussion platforms with different user bases and moderation norms -->
<!-- Temporal coverage: results are bound to the specific time period captured; market regime changes or platform policy shifts outside this window may alter surge dynamics -->

<!-- Source: pre-collected CSV exports from academic/archival Reddit datasets -->
<!-- Specific source: [https://www.kaggle.com/datasets/leukipp/reddit-finance-data, specific Kaggle dataset, search through Reddit API] -->
<!-- No live API scraping — static snapshot ensures reproducibility -->
<!-- Fields retained: timestamp, title, selftext, subreddit, score, num_comments, etc. -->
<!-- Any filtering applied at collection time (date range, post type) -->

<!-- Data quality notes: -->
<!-- Known issues in raw data: deleted/removed posts (showing as [removed] or [deleted]), missing selftext fields, duplicate records -->
<!-- These are addressed in preprocessing (Section 4.2); noted here for transparency about raw data state -->


<!-- Public data: Reddit posts are publicly accessible; no private or deleted content used -->
<!-- Anonymity: no attempt to identify or profile individual users; no individual users singled out in results or examples (aggregated analysis only) -->
<!-- No personally identifiable information (PII) retained or processed -->
<!-- Purpose: academic research only; no trading decisions were made based on model outputs -->
<!-- Ethics approval: formal ethics approval was not required for analysis of publicly available aggregated data under university guidelines — [confirm and state explicitly] -->
<!-- Compliance with university ethics guidelines and Reddit's terms of service -->
<!-- Data storage: local only, not redistributed beyond project submission -->

Reddit was selected as the data source for **three reasons**: its subreddit structure creates topically focused communities where stock discussion is concentrated and retrievable; posts are publicly accessible and archived, enabling reproducible research without API rate constraints; and its threaded format produces timestamped submissions with text content suitable for both temporal and sentiment feature extraction. Alternative platforms were considered and rejected — Twitter/X's ephemeral stream lacks persistent threading, StockTwits has a smaller user base with less organic discussion diversity, and proprietary trading forums are not publicly accessible.

Two subreddits were selected to represent opposite ends of the data density spectrum:

- **r/pennystocks** — A niche community focused on low-capitalisation equities. Its sparse ticker distribution (thousands of tickers, most with very few posts) tests whether the methodology degrades gracefully under data scarcity.
- **r/wallstreetbets** — A high-volume mainstream forum with concentrated ticker discussion. Its density (hundreds of posts per day on popular tickers) tests whether the pipeline scales and whether signal can be extracted from a noisier, higher-volume environment.

This dual-dataset design directly addresses literature gap 3 (domain specificity) by applying the same pipeline to two communities with fundamentally different characteristics, and enables cross-dataset transfer evaluation — testing whether models trained on one community generalise to the other.

*Table 4: Dataset characteristics.*

| Property | r/pennystocks | r/wallstreetbets |
|----------|---------------|------------------|
| Raw records | 304,524 | 1,293,981 |
| Date range | 2021-01-01 to 2021-12-31 | 2021-01-01 to 2021-12-31 |
| After ticker extraction (exploded) | 80,212 | 577,872 |
| Usable records (post-exclusion) | 24,827 | 457,072 |
| Exclusion rate | 69.0% | 20.9% |
| Train / Test split | 21,549 / 3,278 | 388,149 / 68,923 |
| Test surges | 31 | 2,582 |
| Test surge rate | 0.95% | 3.75% |
| Test imbalance ratio | 105:1 | 26:1 |

Both datasets are static CSV exports from the Reddit Finance Data collection on Kaggle [17], which was compiled by querying the Reddit API for submissions matching finance-related criteria across multiple subreddits. The files used cover the full calendar year 2021. This period was chosen because it spans the January GameStop episode through the subsequent normalisation, capturing both peak surge activity and quieter periods. It provides temporal variety for the expanding-window cross-validation design. No live API scraping was performed; the static snapshot ensures that any researcher with the same input files can reproduce identical results.

Each record contains: a Unix timestamp (`created`), post title, optional selftext body, and engagement fields (`score`, `num_comments`) that are *not* used as features but are retained for transparency. The pipeline extracts ticker symbols from title and selftext using regex with stopword filtering, then explodes multi-ticker records into one row per record-ticker pair — the unit of analysis for feature engineering and prediction.

**Ethical considerations.** All data consists of publicly posted Reddit submissions; no private, deleted, or moderated content is included in the archived dataset. No attempt is made to identify or profile individual users — analysis is aggregated at the ticker level. No personally identifiable information is retained or processed. The project is academic research only; no trading decisions were made from model outputs. Formal ethics approval was not required under university guidelines for analysis of publicly available aggregated data.

**Known limitations.** The archival dataset exhibits survivorship bias: posts deleted or removed by moderators before the archive snapshot are not captured. Engagement metrics (`score`, `num_comments`) are frozen at collection time and may not reflect final values — this is precisely why the project uses timestamp-derived posting volume rather than engagement scores as the prediction target. Potential gaps exist from Reddit API rate limits during the original archival collection. Results are bound to the 2021 time period; market regime shifts or platform policy changes outside this window may alter surge dynamics.

### 3.3 Surge Definition (Target Variable)

Defining "surge" as a binary label is the central design challenge. A naive approach, such as flagging any ticker that crosses a fixed posting-count threshold, fails because tickers have wildly different baselines. A ticker that normally attracts 2 posts per day behaves very differently from one that attracts 200; a fixed count would permanently label high-volume tickers as "surging" while missing genuine spikes in quieter discussions. What matters is not how many posts appear, but whether the current activity is *statistically unusual* for that ticker's recent history.

The solution is a composite metric that normalises volume growth relative to the training distribution and combines it with sentiment change:

> *composite = (w₁ × z_volume) + (w₂ × z_sentiment)*

For each record mentioning ticker *X* at time *t*, the pipeline:

1. Counts *X*-mentioning posts in a forward window (*t*, *t*+24h] and a backward window (*t*−24h, *t*]
2. Computes volume growth: (forward_count / max(backward_count, 1)) − 1
3. Computes sentiment shift: |mean(forward_sentiments) − current_sentiment|
4. Z-score normalises both components using training-partition statistics only (μ and σ frozen from the 80% temporal split)
5. Combines the weighted z-scores into the composite
6. Labels surge=1 if composite > threshold *τ*, else surge=0

Records with too few posts in their forward window are excluded because there is simply not enough data to judge whether a "surge" has occurred.

**Why include sentiment at all?** The literature suggests surges involve heightened emotional tone alongside increased activity [4][10]. Volume alone would miss cases where a small community becomes markedly more agitated without posting more frequently. The composite captures both dimensions while remaining configurable: setting w₂=0 reduces to volume-only, enabling direct comparison of whether sentiment adds predictive value (the Phase 1 vs Phase 2 experiment described below).

**Why z-scores rather than raw percentages?** A ticker going from 1 to 5 posts and one going from 100 to 500 posts both show 400% growth, but the first case might be random noise while the second represents a genuine community-wide event. Z-scoring against the training distribution identifies growth that is *statistically unusual*, applying the same standard regardless of a ticker's typical activity level.

**Preventing leakage in the labels themselves.** The z-score parameters are computed exclusively from the training partition and frozen before the test partition is labelled. Without this step, the mean and standard deviation would incorporate test-period information, subtly contaminating the target variable. This is a form of data leakage that would inflate evaluation metrics.

**Choosing the threshold τ.** The threshold controls how "extreme" an event must be to count as a surge. Higher values produce rarer, more dramatic surges but worsen class imbalance:

*Table 5: Threshold sensitivity on r/wallstreetbets (457,072 usable records).*

| τ | Surge Count | Surge Rate | Imbalance Ratio |
|---|-------------|------------|-----------------|
| 0.5 | 80,455 | 17.6% | 4.7:1 |
| **1.0** | **22,384** | **4.9%** | **19.4:1** |
| 1.5 | 6,602 | 1.4% | 68.2:1 |
| 2.0 | 2,873 | 0.6% | 158:1 |
| 2.5 | 1,348 | 0.3% | 338:1 |

τ=1.0 was selected for the primary evaluation. It produces a ~5% surge rate, which is rare enough to be meaningful but common enough (22,384 events across the dataset, 2,582 in the test set) for statistically reliable model evaluation. A secondary run at τ=1.5 on r/pennystocks tests behaviour under more extreme imbalance.

**Two-phase validation of the composite design.** To determine whether the sentiment component genuinely improves prediction or merely adds noise, the experiment is run twice: Phase 1 with w₂=0 (volume-only labels) and Phase 2 with w₁=w₂=0.5 (equal composite). If Phase 2 outperforms Phase 1, sentiment contributes meaningful signal; if not, the simpler volume-only definition suffices.

### 3.4 Feature Engineering

All features satisfy the backward-looking constraint from Section 3.1: only information available at or before observation time *t* is used. The most tempting predictors, Reddit score and comment count, are excluded for exactly this reason. They look predictive because they *are* the surge; including them would be circular.

With that constraint in mind, the eleven features are organised into four groups:

*Table 6: Feature definitions. All features are computed at observation time t using only backward-looking or concurrent information.*

| Feature | Category | Type | Definition |
|---------|----------|------|------------|
| `sentiment_score` | Content | Continuous [−1, 1] | VADER compound sentiment of the post text |
| `word_count` | Content | Discrete ≥ 0 | Number of words in selftext (0 if absent) |
| `title_length` | Content | Discrete ≥ 0 | Character count of the post title |
| `num_tickers_mentioned` | Content | Discrete ≥ 1 | Number of distinct tickers extracted from the post |
| `hour_of_day` | Temporal | Discrete [0–23] | Hour of post creation (UTC) |
| `day_of_week` | Temporal | Discrete [0–6] | Day of post creation (Monday=0) |
| `time_since_previous` | Activity | Continuous ≥ 0 | Seconds since the previous post mentioning the same ticker |
| `ticker_post_rate_24h` | Activity | Continuous ≥ 0 | Number of same-ticker posts in the preceding 24 hours |
| `ticker_post_acceleration` | Activity | Continuous | Posting rate ratio: count in preceding 12h divided by count in the 12h before that |
| `word_count_x_hour` | Interaction | Continuous | word_count × hour_of_day |
| `accel_x_time_since_prev` | Interaction | Continuous | ticker_post_acceleration × time_since_previous |

**Content features** describe what is being said. Sentiment provides a proxy for emotional intensity [4]. Word count and title length reflect how much effort a poster invested (longer posts tend to be substantive analysis rather than one-line reactions). Ticker count distinguishes focused single-stock discussion from broad market commentary.

**Temporal features** describe when the post appears. Hour and day of week encode cyclical patterns tied to market hours and weekend effects. Surges may cluster around market open or after-hours earnings releases, making timing a useful contextual signal.

**Activity features** describe how the ticker's discussion has been behaving recently. These draw most directly on the popularity prediction literature [1][5]: if posts about a ticker are arriving faster than usual, a surge may already be forming. The `time_since_previous` feature adapts Cheng et al.'s "early propagation speed" concept to the discussion-forum setting, while `ticker_post_acceleration` measures whether the rate itself is increasing or decreasing.

**Interaction features** were added after experiment B2 showed that manually constructed feature combinations improved Random Forest AUC by +1.4 percentage points on the pennystocks dataset. Even tree-based models, which can in principle discover interactions through splits, benefited from having cross-feature products available directly. The `word_count_x_hour` interaction captures a specific pattern observed during EDA: long posts written during pre-market hours (the typical "DD" analysis posts) appear disproportionately before surges.

**What was left out.** Reddit's `score` (upvotes minus downvotes) and `num_comments` are available in the raw data but deliberately excluded. These fields accumulate *after* a post is created and reflect the very engagement dynamics the model is trying to predict. Using them would be equivalent to telling the model the answer, and any performance gains would not generalise to real-time prediction where these values are not yet available.

### 3.5 Model Selection

Rather than picking a single model and optimising it, this project compares three classifier families to answer a specific question: does model complexity actually help for surge prediction? If a simple linear model performs nearly as well as a gradient boosting ensemble, the surge signal is straightforward and the features do most of the work. If complex models pull clearly ahead, the data contains non-linear patterns that simpler approaches miss.

The three families were chosen to span the complexity spectrum:

- **Logistic Regression (LR)** is the interpretable baseline. It fits a linear decision boundary with elastic net regularisation (combined L1/L2), making it the simplest model in this comparison. If LR performs well, the surge signal is approximately linearly separable in the feature space.
- **Random Forest (RF)** represents bagged ensembles. It captures non-linear relationships through tree splits, handles noisy features gracefully, and provides built-in importance estimates. Fernández-Delgado et al. [16] found that random forests achieved the highest overall accuracy across 121 benchmark datasets.
- **XGBoost** represents sequential boosting. Each tree corrects the mistakes of the previous ensemble, with L1/L2 regularisation on leaf weights to prevent overfitting. Gradient boosting methods, the family XGBoost belongs to, consistently rank among the top performers on structured tabular data [16].

Together, these three cover linear, bagged ensemble, and boosted ensemble approaches. The comparison reveals whether surge prediction benefits from increasing model capacity or whether the features themselves carry the signal.

**Why AUC-ROC as the primary metric.** With surge rates between 1% and 5%, accuracy tells us almost nothing. A model that predicts "no surge" for every record achieves 95–99% accuracy while being completely useless. AUC-ROC measures how well the model *ranks* surge-likely records above non-surge records, regardless of where the classification threshold is set. This makes it the right metric when the optimal operating point is not known in advance. Precision, recall, and F1 are reported as secondary metrics at both the default (0.5) and validation-tuned thresholds.

**Handling class imbalance.** Resampling techniques like SMOTE generate synthetic minority-class samples by interpolating between existing ones. For temporally ordered data this is problematic: a synthetic record created between two time points has no meaningful timestamp and could introduce spurious temporal patterns. Instead, the pipeline uses cost-sensitive learning. Logistic Regression and Random Forest use `class_weight='balanced'`, which scales the loss inversely proportional to class frequency. XGBoost uses `scale_pos_weight` set to the negative-to-positive ratio. Both approaches penalise minority-class errors more heavily during training without manufacturing artificial data points.

### 3.6 Temporal Validation Design

Standard k-fold cross-validation randomly assigns records to folds, which means a model might train on posts from October and validate on posts from March. For temporal prediction tasks, this is a form of cheating: the model has seen future patterns before being asked to predict them. Bergmeir and Benítez [15] showed that this inflates accuracy estimates, sometimes substantially. Since this project's central claim is that surges can be predicted from *past* information alone, the validation protocol must respect time ordering throughout.

The pipeline uses a two-level temporal strategy:

**Level 1: Train/test split (80/20 by timestamp).** All records are sorted chronologically. The first 80% form the training partition; the final 20% form the held-out test set. The test set is never seen during model selection, hyperparameter tuning, or threshold calibration. It is used exactly once to produce the final reported metrics.

**Level 2: Expanding-window cross-validation within training (k=4 folds).** Within the training partition, hyperparameters are selected using an expanding-window scheme with four folds:

```mermaid
gantt
    title Expanding-Window Temporal CV (k=4)
    dateFormat X
    axisFormat %s

    section Fold 1
    Train     :done, 0, 25
    Val       :active, 25, 50

    section Fold 2
    Train     :done, 0, 50
    Val       :active, 50, 75

    section Fold 3
    Train     :done, 0, 75
    Val       :active, 75, 100
```
*Figure 3: Expanding-window cross-validation structure. Each fold trains on all data up to a cutoff point and validates on the next temporal block. The training window grows with each fold, mimicking how a deployed model would accumulate more history over time.*

In each fold, the training window includes all records from the start up to a split point, and the validation window is the next chronological block. This means:

- No validation record is ever earlier than any training record (temporal ordering preserved)
- The training set grows with each fold (mimicking deployment, where more history accumulates over time)
- Each fold tests generalisation to a genuinely unseen future period

**Why k=4 rather than k=5 or k=10?** The choice is constrained by the data. Each validation fold must contain enough surge events to produce a stable AUC estimate. With a 5% surge rate and the training partition spanning roughly 9.5 months, four folds produce validation blocks of approximately 2.5 months each, yielding hundreds of positive cases per fold on the WSB dataset. More folds would produce shorter validation windows with fewer surges, increasing variance in the fold-level AUC estimates.

**Threshold tuning on validation folds.** The classification threshold (the probability cutoff above which the model predicts "surge") is not set to the default 0.5. Instead, after hyperparameter selection, the threshold that maximises F1 on the validation folds is identified and applied to the test set. This avoids optimising the threshold on test data, which would leak test-set information into the decision rule.


### 3.7 Evaluation Framework

The evaluation must answer four questions: (1) Do the models predict surges better than trivial strategies? (2) Do the models differ meaningfully from each other? (3) How confident can we be in the reported metrics? (4) Does the methodology transfer across communities?

#### Success Tiers

A model that cannot discriminate surges from non-surges has AUC-ROC of 0.5. But how much better than 0.5 counts as "useful"? Without a directly comparable prior study (the literature review identifies this as a gap), the project defines three performance tiers based on conventional interpretations of AUC in the machine learning literature:

| Tier | AUC-ROC | Interpretation |
|------|---------|----------------|
| Minimum | > 0.60 | Weak but above-chance discrimination; the features contain *some* predictive signal |
| Target | > 0.70 | Moderate discrimination; practically useful for ranking records by surge likelihood |
| Stretch | > 0.80 | Strong discrimination; the model reliably separates surges from non-surges |

These thresholds are conservative. The cascade prediction literature reports higher values (Cheng et al. [5] achieved AUC 0.877), but those studies used engagement-based features and non-temporal evaluation protocols that likely inflate results. The tiers here reflect what is achievable under the strict backward-looking constraint this project imposes.

#### Baseline Comparisons

Each baseline isolates a specific question about the source of predictive performance:

- **Random baseline (AUC = 0.5)** — Does the model beat chance? If not, the features carry no signal.
- **Single-feature baselines** — For each of the eleven features, a single-feature Logistic Regression is trained and its AUC recorded. This determines whether the multi-feature combination adds value over the best individual predictor. If the full model barely exceeds the best single feature, the additional complexity is unjustified.

#### Statistical Robustness

Reporting a single AUC number without uncertainty is misleading — it could be unstable due to the particular test-set composition. Three mechanisms quantify reliability:

**Bootstrap confidence intervals (1,000 iterations).** The test set is resampled with replacement 1,000 times, and AUC-ROC is computed on each resample. The 2.5th and 97.5th percentiles form the 95% confidence interval. Bootstrap was chosen over parametric alternatives because AUC has no simple closed-form variance estimator under class imbalance, and the bootstrap makes no distributional assumptions.

**McNemar's test for pairwise model comparison.** When two models are trained on the same data and evaluated on the same test set, their predictions are *paired* — each record receives a prediction from both. McNemar's test examines the 2×2 table of concordant and discordant predictions (records where one model is correct and the other is not). This is more appropriate than a paired t-test (which requires continuous outputs) or an independent test (which ignores the paired structure). With three model pairs (LR vs RF, LR vs XGB, RF vs XGB), the family-wise error rate is controlled with Bonferroni correction (α = 0.05 / 3 = 0.017). Bonferroni was chosen over less conservative corrections (e.g., Holm) because with only three comparisons the power loss is negligible and the interpretation is simpler.

**Training stability.** All runs use a single fixed seed (42) to ensure exact reproducibility. Bootstrap confidence intervals on the test set quantify evaluation uncertainty, though they do not capture variance from training randomness (a limitation discussed in Section 5.4).

#### Metrics and Thresholds

Models are evaluated using four metrics on the held-out test set:

*Table 7: Evaluation metrics.*

| Metric | Role |
|--------|------|
| AUC-ROC | Primary; threshold-independent ranking quality |
| Precision | Proportion of predicted surges that are real |
| Recall | Proportion of actual surges detected |
| F1-Score | Harmonic mean of precision and recall |

Precision, recall, and F1 depend on the classification threshold. These are reported at two operating points: the default threshold of 0.5, and the validation-tuned threshold identified during cross-validation (Section 3.6). Comparing the two reveals how much threshold tuning matters — if default-threshold F1 is near zero but tuned-threshold F1 is reasonable, the model has discriminative ability that only becomes apparent at the right operating point.

#### Cross-Dataset Transfer Protocol

To test whether the pipeline captures general surge dynamics or merely overfits to community-specific patterns, models trained on r/wallstreetbets are evaluated directly on r/pennystocks (and vice versa) without retraining. This is a stringent test: the two communities differ in posting density, ticker distribution, user demographics, and discussion norms.

The transfer evaluation computes full metrics (AUC-ROC, precision, recall, F1) at both the default and source-community-tuned thresholds, but **AUC-ROC is the primary comparison metric** since the optimal operating point almost certainly differs between communities — a threshold tuned on WSB's 3.75% surge rate is unlikely to be appropriate for pennystocks' 0.95% rate. If transfer AUC exceeds the minimum tier (0.60), the underlying surge patterns share structure across communities. If it falls below, community-specific calibration is necessary — a meaningful finding either way, as it reveals whether "surge" is a universal phenomenon or a community-specific one.

#### Sensitivity Analysis

Two parameter sweeps characterise how robust the results are to design choices:

- **Threshold sensitivity (τ ∈ {0.5, 1.0, 1.5, 2.0, 2.5})** — How does the surge definition affect model performance? If results collapse at slightly different τ values, the methodology is fragile. If performance degrades gracefully, the approach is robust to the specific threshold chosen.
- **Weight sensitivity (w₂ ∈ {0, 0.25, 0.5, 0.75, 1.0})** — Does sentiment actually help? Setting w₂=0 produces volume-only labels (Phase 1); increasing w₂ adds sentiment influence. This directly tests whether the composite design outperforms the simpler alternative, providing the evidence needed to justify (or reject) including sentiment in the surge definition.

### 3.8 Reproducibility and Configuration

Reproducibility is a first-class design constraint rather than an afterthought. Every pipeline run must produce identical outputs given identical inputs and configuration, and every output artefact must be traceable back to the exact parameters that generated it. This requirement is motivated by two concerns: scientific validity (any reported result must be independently verifiable) and practical iteration (when sweeping thresholds or weights across dozens of runs, an analyst must know which configuration produced which outcome).

#### Deterministic Execution via Fixed Seeds

All stochastic operations in the pipeline are governed by a single random seed (default: 42) applied at the start of each run. The pipeline seeds Python's built-in `random` module and NumPy's `np.random` at the start of execution, before any computation begins. This seed propagates through all downstream operations: scikit-learn's `RandomForestClassifier` and `LogisticRegression` receive `random_state=random_seed` as a constructor argument, and XGBoost receives it via the hyperparameter grid. Because the pipeline processes records in deterministic order (sorted by timestamp for temporal operations, stable-sorted for ties), the combination of fixed seeds and deterministic ordering guarantees bit-for-bit identical outputs across runs on the same machine.

#### Configuration as Data

All pipeline parameters are held in a single `PipelineConfig` dataclass that supports round-trip JSON serialisation (serialise via `to_json()` / `save_json()`, deserialise via `from_json()` / `load_json()`):

```json
{
  "file_path": "input/raw/r_wallstreetbets_submissions_reddit.csv",
  "output_dir": "output/processed",
  "temporal_split_ratio": 0.8,
  "surge_method": "forward_growth",
  "min_window_count": 1,
  "threshold_tau": 1.5,
  "sentiment_model": "vader",
  "weight_volume": 0.75,
  "weight_sentiment": 0.25,
  "thresholds": [0.5, 1.0, 1.5, 2.0, 2.5],
  "random_seed": 42
}
```

Every pipeline run writes this configuration to the output directory alongside its results (`YYYY-MM-DD_HH-MM_pipeline_config.json`), creating a permanent link between any output artefact and the exact parameters that produced it. Configuration can flow in two directions: parameters are specified individually via CLI arguments for exploratory runs, or a saved JSON file is loaded wholesale via `--config path/to/config.json` to reproduce a previous run exactly.

This design means that reproducing any historical result requires only two things: the original input CSV and the saved configuration JSON. The command `surge-label --config output/processed/2026-07-18_19-09_pipeline_config.json` will recreate the exact labelling output from that run.

#### CLI Design

The pipeline exposes five named console scripts (defined in `pyproject.toml`), separating concerns so that expensive upstream stages need not be repeated when only downstream parameters change. The labelling script (`surge-label`) accepts all configuration parameters individually or via `--config`; the training script (`surge-train`) consumes the labelled CSV output and runs feature engineering through evaluation. Each script also accepts `--verbose` for debug-level logging, `--log-file auto` for persisted log output, and `--notes` for free-text annotation of the experiment. The full list of entry points and their parameters is detailed in Section 4.1 (Table 8).

#### Experiment Log

An append-only JSONL file (`output/experiment_log.jsonl`) records metadata for every pipeline run. Each line is a self-contained JSON object:

```json
{
  "run_id": "2026-07-18_19-09",
  "pipeline": "labelling",
  "timestamp": "2026-07-18T19:09:42",
  "git_sha": "a3f7c2d",
  "config": { "...all parameters..." },
  "outputs": ["2026-07-18_19-09_labelled.csv", "2026-07-18_19-09_pipeline_config.json"],
  "summary": { "total_records": 457072, "surge_rate": 0.049, "duration_s": 312.4 },
  "notes": "WSB full run, composite weights 0.75/0.25"
}
```

The JSONL format was chosen for three properties: it is append-safe (a crash mid-write cannot corrupt existing entries), Git-friendly (each run adds exactly one line, producing clean diffs), and queryable via `pandas.read_json("experiment_log.jsonl", lines=True)` for longitudinal analysis across experiments. The Git SHA links each run to the exact code version that produced it, enabling reconstruction of the full dependency environment from the repository state at that commit.

#### Model Serialisation

Trained models are persisted via joblib, chosen over Python's built-in pickle for its efficient handling of NumPy arrays within scikit-learn estimators and over ONNX for its simplicity in a research (non-deployment) context. Each model file is timestamped and stored alongside its evaluation metrics. The StandardScaler fitted on training data is serialised jointly with the model to ensure that any future inference applies identical feature normalisation.

#### Artefact Traceability

The combination of these mechanisms creates a complete audit trail from raw data to final predictions:

```
Input CSV → [surge-label + config.json] → Labelled CSV
         → [surge-train + labelled CSV]  → Trained models (joblib)
                                          → Evaluation metrics (JSON)
                                          → Figures (PNG)
                                          → experiment_log.jsonl (append)
```

Every artefact in the `output/` directory carries a timestamp prefix that links it to the corresponding experiment log entry and configuration JSON, making it possible to reconstruct the full provenance chain for any reported result.

#### Scope of the Reproducibility Guarantee

The reproducibility mechanisms described above guarantee identical results *on the same machine with the same library versions*. Three factors limit stronger guarantees:

- **NumPy version sensitivity.** The legacy `np.random.seed` API uses global state whose implementation may change across major NumPy releases. The project requires `numpy>=1.24` as a minimum bound but does not hard-pin the version; the Git SHA in the experiment log allows the exact dependency state to be recovered from `pyproject.toml` at that commit.
- **Platform-dependent floating point.** Different CPU architectures (x86 vs ARM) and compiler optimisations may produce subtly different floating-point results, particularly in aggregation operations over large arrays. Bit-for-bit cross-platform reproducibility is not guaranteed.
- **No containerised environment.** The project does not provide a Docker image or pinned lockfile freezing the full transitive dependency tree. For a student research project this is an acceptable trade-off; a production system would require stricter environment isolation.

These limitations do not affect the validity of the reported results (which were all produced on a single machine with a fixed environment), but they are noted for transparency about the scope of the reproducibility claim.

---




---

## 4. Implementation

### 4.1 Code Organisation

The pipeline is packaged as a standard Python library (`surge-pipeline`, built with setuptools) that depends on pandas, scikit-learn, XGBoost, vaderSentiment, and NumPy (version bounds specified in `pyproject.toml`). All source code sits under `src/`, split into a core library package and a handful of CLI scripts that drive it:

```
src/
├── surge_pipeline/              # Core library (15 modules, ~4,250 LOC)
│   ├── __init__.py              # Public API exports (PipelineConfig, NormalisationParams, etc.)
│   ├── config.py                # PipelineConfig dataclass + JSON serialisation
│   ├── loader.py                # CSV ingestion, ticker extraction, record explosion
│   ├── windowing.py             # Per-ticker forward/backward 24h counts (searchsorted)
│   ├── sentiment.py             # VADER compound scoring with title-fallback
│   ├── labelling.py             # Temporal split, z-scores, composite metric, thresholding
│   ├── normalisation.py         # Z-score parameter persistence (train-only stats)
│   ├── features.py              # 11 backward-only ML features
│   ├── training.py              # Multi-model training with expanding-window CV
│   ├── training_models.py       # Data classes (TrainedModel, CVResult, etc.)
│   ├── evaluation.py            # Metrics, bootstrap CI, McNemar's, tier validation
│   ├── evaluation_models.py     # Evaluation result data classes
│   ├── evaluation_figures.py    # Confusion matrices, ROC curves, importance plots
│   ├── experiment_log.py        # Append-only JSONL experiment tracker
│   ├── timestamps.py            # Timestamp conversion utilities
│   ├── cli_logging.py           # Tee-style console + file logging
│   └── pipeline.py              # Orchestrator: chains stages, manages outputs
├── eda/
│   └── eda_pipeline.py          # Exploratory data analysis with figure generation
├── tests/                       # 10 test modules (pytest)
│   ├── test_loader.py
│   ├── test_windowing.py
│   ├── test_sentiment.py
│   ├── test_labelling.py
│   ├── test_features.py
│   ├── test_training.py
│   ├── test_evaluation_significance.py
│   ├── test_evaluation_figures.py
│   ├── test_config.py
│   └── test_pipeline_integration.py
├── run_labeling.py              # CLI: full labelling pipeline (stages 1–4 + threshold sweep)
├── run_training.py              # CLI: model training + evaluation (stages 5–6)
├── run_cross_validation.py      # CLI: cross-dataset transfer evaluation
├── generate_figures.py          # CLI: standalone figure generation from saved artefacts
└── generate_prediction_examples.py  # CLI: sample predictions for report examples
```

Installing the package in editable mode (`pip install -e .`) registers five console commands, so any experiment can be kicked off from the terminal without navigating into the source tree:

*Table 8: CLI entry points.*

| Command | Script | Purpose |
|---------|--------|---------|
| `surge-label` | `run_labeling:main` | Run the labelling pipeline (load → window → sentiment → label → threshold sweep) |
| `surge-train` | `run_training:main` | Train all three models and produce the full evaluation report |
| `surge-cross-val` | `run_cross_validation:main` | Test whether a model trained on one subreddit transfers to the other |
| `surge-figures` | `generate_figures:main` | Regenerate publication figures from saved evaluation artefacts |
| `surge-examples` | `generate_prediction_examples:main` | Produce worked prediction examples for manual inspection |

**How the modules map to the pipeline stages.** Each stage described in Section 3.1 corresponds to one or two library modules. The mapping is deliberate: isolating each stage in its own module means a change to, say, the sentiment backend does not touch the windowing logic, and any stage can be unit-tested in isolation.

| Pipeline Stage | Module(s) | Key Function |
|----------------|-----------|--------------|
| 1. Data Loading & Preprocessing | `loader.py` | `load_data()`, `extract_tickers()` |
| 2. Temporal Windowing | `windowing.py` | `compute_windowed_counts()` |
| 3. Sentiment Computation | `sentiment.py` | `compute_sentiment()` |
| 4. Target Labelling | `labelling.py`, `normalisation.py` | `apply_labelling()`, `sweep_thresholds()` |
| 5. Feature Engineering | `features.py` | `compute_features()`, `get_feature_matrix()` |
| 6. Training & Evaluation | `training.py`, `evaluation.py` | `train_models()`, `evaluate_model()` |

The `pipeline.py` orchestrator wires stages 1–4 together, seeds the random number generators, logs how many records survive each stage, and finishes with a threshold sweep for sensitivity analysis. Stages 5–6 live in a separate script (`run_training.py`) that picks up the labelled CSV produced by stage 4. Splitting the work this way has a practical benefit: relabelling the data with a different threshold or sentiment weight does not force a full retraining run, and retraining with new hyperparameters does not require re-scoring sentiment across 1.3 million records.

**Configuration and reproducibility.** Section 3.8 describes the reproducibility infrastructure in detail. The implementation consequence is straightforward: a single `PipelineConfig` dataclass holds every tuneable parameter, and a fixed seed (default 42) is applied to Python's `random` and NumPy at pipeline start. Scikit-learn estimators receive the same seed via their `random_state` constructor argument. Every output file carries a timestamp prefix (`YYYY-MM-DD_HH-MM`) that ties it back to the matching experiment log entry, so any result can be traced to the exact configuration that produced it.

**Testing.** Ten pytest modules cover every pipeline stage with unit and integration tests. They exercise edge cases (empty DataFrames, tickers with a single post, missing selftext), verify numerical correctness (windowing counts checked against brute-force reference implementations), and confirm end-to-end behaviour on synthetic data. The full suite runs without the large production datasets.

### 4.2 Data Loading and Preprocessing

The loader (`loader.py`) takes a raw Reddit CSV and turns it into the unit of analysis — one row per (record, ticker) pair, sorted by time — in four steps.

**Loading.** The pipeline reads the CSV with pandas and records a SHA-256 file hash for provenance tracking (Section 3.8). During development or CI runs where the real dataset is absent, a synthetic data generator stands in so downstream stages can still be exercised.

**Text cleaning.** Reddit posts often contain `[deleted]` or `[removed]` placeholders left behind by moderation or user deletion. These get replaced with empty strings so they don't pollute downstream text processing. Null `title` and `selftext` fields are filled with empty strings the same way. No deduplication step is needed: each row in the archival dataset already corresponds to a unique Reddit submission ID, and the explosion step that follows only splits rows — it never creates new ones.

**Ticker extraction.** Figuring out which stocks a post is actually discussing is harder than it looks. The extraction logic applies two regex patterns in priority order:

1. **Dollar-sign pattern** (`$AMC`, `$TSLA`) — highest confidence, since the dollar prefix is an explicit ticker marker in financial communities.
2. **Uppercase word pattern** (standalone 2–5 character uppercase words) — a broader net that catches tickers mentioned without the dollar sign.

Both patterns are filtered against a curated stopword set of 297 terms, split into eight named categories for easy auditing: common English words (149 terms), Reddit slang and trading verbs (55), finance abbreviations (32), datetime/timezone terms (26), geography (9), market venue names (8), technology buzzwords (7), and currencies (6). Defining the categories as separate Python sets and combining them via union makes it straightforward to add new entries as false positives are discovered.

A heuristic stopword approach was chosen over a known-ticker validation list for two reasons. First, penny stock tickers change frequently as companies list and delist; a static ticker list from 2021 would miss new listings that appear within the dataset's time span. Second, the stopword approach fails gracefully: the worst case is a false-positive ticker adding noise to one record, whereas a missing-ticker list would silently drop posts about stocks it doesn't know about. The trade-offs of this design choice are revisited in Section 5.4.

**Timestamp parsing and sorting.** The raw CSV stores timestamps either as Unix epoch seconds (`created_utc`) or as datetime strings (`created`). Both formats are normalised to timezone-aware `datetime64[ns, UTC]`, and the entire DataFrame is sorted chronologically. This ordering is a hard precondition for the binary-search windowing logic in the next stage and is preserved throughout all subsequent processing.

**Record explosion.** A single post can mention several tickers at once (e.g., "comparing $AMC vs $GME"). These multi-ticker records are split into separate rows — one per ticker — via pandas' `explode()`. After explosion, each row represents one (record, ticker) pair, which is the grain at which features, labels, and predictions operate. Records that yield zero tickers after extraction are dropped entirely:

*Table 9: Loader-stage attrition by dataset (ticker extraction only; further windowing-stage exclusions are reported in Table 4).*

| Step | r/pennystocks | r/wallstreetbets |
|------|---------------|------------------|
| Raw records loaded | 304,524 | 1,293,981 |
| Excluded (no tickers found) | 224,312 (73.7%) | 716,109 (55.3%) |
| After explosion (record-ticker pairs) | 80,212 | 577,872 |

The high no-ticker rate on r/pennystocks reflects how the community actually talks: many posts are general market commentary, memes, or questions that never name a specific stock. This is a genuine property of the subreddit, not a limitation of the extraction logic. The surviving 80,212 records pass to the windowing stage, where a further 69.0% are excluded for having too few posts in their forward window to determine whether a surge occurred (Table 4).

### 4.3 Feature Engineering

Section 3.4 covers *what* the eleven features are and *why* each earned its place. This section is about the *how* — algorithmic choices, edge cases handled, and the one constraint that governs every line of code: nothing may peek into the future.

**Reused columns.** Two features come for free because earlier stages already did the work: `sentiment_score` is just the `sentiment_polarity` column renamed, and `ticker_post_rate_24h` is a direct copy of `backward_count` from windowing. Reusing these avoids recomputation, but the real benefit is avoiding the risk of a subtly different definition creeping in. Two more — `hour_of_day` and `day_of_week` — are trivial extractions from `created_utc` with no per-ticker logic involved.

**Time since previous post.** This measures how many hours have passed since the last post about the same ticker. The implementation groups records by ticker (they're already in chronological order from the loader) and takes the timestamp difference between consecutive entries in each group. The first post ever about a given ticker has no predecessor, so it gets a sentinel value of −1 rather than zero — the model needs to tell apart "nothing was said before" from "someone posted one second ago." Tree-based models handle this sentinel naturally (they just split on it); Logistic Regression's StandardScaler normalises it alongside the real values, which works because the −1 cluster sits well away from the positive-valued population.

**Ticker post acceleration.** The most involved feature to compute. It answers a simple question: is discussion about this ticker picking up speed or dying down? The approach splits the backward 24-hour window into two halves and counts posts in each:

- *Recent half*: posts in (t−12h, t], not counting the current record
- *Older half*: posts in (t−24h, t−12h]

Acceleration is `count_recent / max(count_older, 1)`. Above 1.0 means activity is ramping up; below 1.0 means it's fading. The `max(..., 1)` prevents division by zero when nothing appeared in the older half — silence becomes a baseline of 1, so acceleration just equals the recent count.

The counting relies on NumPy's `searchsorted` over sorted per-ticker timestamp arrays, the same O(n log n) binary-search trick used in the windowing stage. Even on the 577,872-row wallstreetbets dataset this stays fast.

**Word count and title length.** Both are straightforward whitespace-split token counts. `word_count` combines title and selftext; `title_length` uses title alone. Posts without body text (link posts, one-line reactions) get their word count from the title only. These features are deliberately coarse — character-level counts or TF-IDF vectors would balloon dimensionality without a clear hypothesis about what the extra detail would buy.

**Number of tickers mentioned.** Since the dataset is exploded (one row per ticker per post), a post mentioning three stocks shows up as three separate rows. This feature recovers the original post's breadth: group by record `id`, count distinct tickers. A single-stock post scores 1; a comparison covering $AMC, $GME, and $BB scores 3 on all three of its rows.

**Interaction features.** Two hand-crafted products round out the set:

- `word_count_x_hour` = word_count × hour_of_day — gives the model an explicit signal for the interplay between content length and posting time, so it doesn't have to discover the combination through multi-level splits alone.
- `accel_x_time_since_prev` = ticker_post_acceleration × time_since_previous — captures sudden acceleration after a stretch of silence. For first-occurrence records (where `time_since_previous` = −1), the product is zeroed out rather than letting the sentinel produce a meaningless negative value.

**What comes out.** Once all eleven columns are in place, the module logs the feature matrix shape and per-feature summary statistics (mean, std, min, max) on non-excluded records. A helper function (`get_feature_matrix()`) then slices out just those eleven columns as a clean DataFrame ready for model training, dropping all the intermediate and metadata columns. On the wallstreetbets dataset the whole computation finishes in under 10 seconds — the searchsorted calls dominate, but per-ticker grouping keeps each individual array small enough to stay fast.

### 4.4 Surge Labelling

Section 3.3 covers the *why* behind the composite surge metric. This section is about the *how* — where the leakage-prevention boundary lives in code, what sequence the steps follow, and what happens when things go wrong.

**The temporal split.** The labelling module's first job is to stamp every record as `train` or `test`. It converts timestamps to epoch seconds, finds the 80th percentile, and draws the line there — everything at or before that point goes to training, the rest to test. Since records are already in chronological order, this puts roughly 80% of them in training and 20% in test. The split is deliberately plain: one cut on sorted time, no stratification, no shuffling. Any randomness here would undermine the temporal ordering the whole pipeline relies on.

**Computing z-score parameters.** With the split done, the module calculates mean (μ) and population standard deviation (σ, `ddof=0`) for both the volume growth ratio from windowing and the absolute sentiment shift from stage 3 — using *only* included training records. Population std is a deliberate choice: the training partition is the reference distribution everything else gets measured against, not a sample of some larger unknown population. "Included" matters too: records the windowing stage flagged as excluded (too few posts in their forward window) get skipped here, since their volume growth values come from insufficient data and would skew the statistics. The four parameters that come out (μ_vol, σ_vol, μ_sent, σ_sent) are frozen and written to JSON as part of the pipeline summary, so any later inference run can normalise fresh data the exact same way without needing the training set.

**Normalisation.** Every non-excluded record — training *and* test — gets z-scored using those frozen training statistics. This is the leakage-prevention boundary in action: the test partition's raw values are measured against a distribution it never contributed to. If σ happens to be zero (possible on very sparse tickers where every training record has identical volume growth), z-scores are set to 0.0 rather than blowing up to infinity. Excluded records get NaN across all derived columns, so they propagate cleanly as missing data and can't be mistaken for legitimate zeros downstream.

**The composite metric.** With z-scores in hand, computing the composite is straightforward:

> *composite = (w₁ × z_volume) + (w₂ × z_sentiment)*

The weights live in `PipelineConfig` — `weight_volume` and `weight_sentiment`, both defaulting to 0.5. Setting `weight_sentiment = 0` gives the Phase 1 (volume-only) variant for the ablation comparison described in Section 3.3. The module logs which mode it's running in, so experiment records are self-documenting.

**Thresholding.** The binary label comes from a single comparison: `surge_label = 1 if composite > τ, else 0`. Excluded records get NaN instead. Class distribution stats (surge count, no-surge count, rate, imbalance ratio) are computed and logged per partition, so you can immediately see whether the chosen τ gives enough positive examples to train on.

**Threshold sweep.** Rather than re-running the pipeline five times to test different τ values, `sweep_thresholds()` does it in one pass. It loops over the configured list (default: [0.5, 1.0, 1.5, 2.0, 2.5]), creates a fresh DataFrame copy and a tweaked config for each, and produces a complete set of class distribution metrics per τ. This is what powers the sensitivity analysis in Section 3.7 — one invocation, all five variants, written out together.

**Edge cases handled in code:**

- *Empty DataFrame* — returns immediately with empty columns and zeroed statistics (shows up in unit tests).
- *All records excluded* — logs a warning and sets every label to NaN (can happen on extremely sparse tickers at high `min_window_count` settings).
- *σ = 0* — z-scores default to 0.0 with a logged warning rather than crashing.
- *Weights summing to ≠ 1* — perfectly fine by design. The composite is a weighted sum, not a weighted average, and since τ is always picked empirically through the threshold sweep, the absolute scale doesn't matter. A different weight sum just shifts where τ needs to land to produce a given surge rate.

**Output.** The module appends five columns to the DataFrame: `partition`, `z_volume`, `z_sentiment`, `composite`, and `surge_label`. The result is saved as `{timestamp}_labelled_dataset.csv` in the output directory (following the timestamp-prefix convention from Section 4.1), with the `NormalisationStats` written to the accompanying pipeline summary JSON. The training script picks up the CSV by path; downstream inference loads the normalisation params from JSON to apply identical scaling to new data.

### 4.5 Model Training

Section 3.5 picks the three model families and Section 3.6 lays out the temporal validation design. This section is about how all of that actually runs — the grids that get searched, the mechanics of the expanding-window splits, and the train → select → retrain flow from start to finish.

**Data preparation.** The training script loads the labelled CSV from Section 4.4, keeps only records where `partition == "train"` and `excluded == False`, pulls the eleven feature columns into a NumPy matrix, and grabs the binary `surge_label` as the target vector. It also computes the class imbalance ratio (negatives / positives) — XGBoost's `scale_pos_weight` grid needs this number.

**Expanding-window fold construction.** The training partition gets sliced into four chronological blocks of roughly equal size. From those four blocks, the module builds three expanding-window splits:

- Split 1: train on block 1, validate on block 2
- Split 2: train on blocks 1–2, validate on block 3
- Split 3: train on blocks 1–3, validate on block 4

Once built, a verification step checks that `max(train_timestamp) < min(val_timestamp)` for every split. If that invariant fails, the pipeline raises an error rather than quietly training on future data.

**Hyperparameter grids.** Each model searches a grid sized to be thorough without being wasteful:

*Table 10: Hyperparameter search spaces.*

| Model | Parameters Searched | Grid Size |
|-------|--------------------|-----------| 
| Logistic Regression | C ∈ {0.01, 0.1, 1, 10, 100}, l1_ratio ∈ {0, 1} | 10 configs |
| Random Forest | n_estimators ∈ {50, 100, 200}, max_depth ∈ {3, 5, 10, None}, min_samples_leaf ∈ {1, 2, 5} | 36 configs |
| XGBoost | n_estimators ∈ {50, 100, 200}, max_depth ∈ {3, 5, 7}, learning_rate ∈ {0.01, 0.1, 0.3}, scale_pos_weight ∈ {1, ratio/2, ratio} | ≤50 configs |

For Logistic Regression, `penalty='elasticnet'` with `l1_ratio=0` gives pure L2 (Ridge) and `l1_ratio=1` gives pure L1 (Lasso) — the two regularisation extremes rather than intermediate blends, keeping the grid compact. All LR models use `class_weight='balanced'` and the `saga` solver (the only one that supports the elastic net penalty). Random Forest likewise uses `class_weight='balanced'`. XGBoost takes a different approach to imbalance: it searches three `scale_pos_weight` values — no reweighting (1.0), moderate (imbalance_ratio / 2), and full (imbalance_ratio) — and fixes `eval_metric='logloss'` for convergence monitoring. The XGBoost grid is capped at 50 configurations so training stays manageable.

**The training loop.** For each model type, the pipeline walks through every hyperparameter configuration and runs all three expanding-window splits on each:

1. Fit a `StandardScaler` on that split's training portion and transform both train and validation features.
2. Build the model with the current hyperparameters and the fixed seed.
3. Fit and score: AUC-ROC on the validation fold's predicted probabilities.
4. If a validation fold has only one class (possible in early folds of sparse datasets), fall back to AUC = 0.5 rather than crashing.

The configuration with the highest mean AUC across the three folds wins. The scaler is re-fit fresh for each split — the validation fold's feature distribution never leaks into training-fold scaling. Same logic as the z-score normalisation in Section 4.4, just applied at the feature level.

**Final retraining.** With the best hyperparameters locked in, the winning model gets retrained on the *entire* training partition — all four blocks, not just the first three — with a single StandardScaler fit. This squeezes every available training record into the final model before it ever touches the held-out test set. The scaler is saved alongside the model; any future prediction has to go through this exact scaler.

**Threshold tuning.** After retraining, the final model predicts probabilities on the last validation fold (block 4) using the final scaler. There's a methodological trade-off here worth being upfront about: the final model was trained on all four blocks *including* block 4, so these threshold-tuning predictions are technically in-sample. The alternative — holding out a separate slice just for threshold selection — would shrink the training data. The in-sample threshold is accepted because (a) a threshold is a simple operating-point choice, not a learned parameter that could overfit in the usual sense, (b) the test set stays completely untouched, and (c) evaluation reports metrics at both the tuned and default (0.5) thresholds so readers can judge the difference themselves. The tuned threshold gets stored with the serialised model.

**Model serialisation.** Each trained model is saved as a joblib file bundling the fitted estimator, the StandardScaler, the winning hyperparameters, the optimal threshold, and metadata (timestamp, phase, seed). Filename pattern: `{model_name}_{phase}_{seed}_{timestamp}.joblib`. A `latest_models.json` manifest tracks which files belong to the most recent run, so the evaluation script can find them without path guessing.

**Timing.** Most of the wall-clock time goes to Random Forest — 36 configs, each growing parallel trees via `n_jobs=-1`. XGBoost is faster per configuration but searches up to 50 combinations. Logistic Regression breezes through its 10 configs. On the wallstreetbets training set (388,149 records), a full three-model run typically wraps up within 15–25 minutes on a multi-core machine.

### 4.6 Evaluation Pipeline

Section 3.7 lays out what gets measured and why. This section is about the machinery that actually produces those numbers — scoring the test set, running the statistical tests, and assembling the final output. The whole evaluation runs as part of the `surge-train` script: once training finishes, the same invocation carries straight through to evaluation without needing a second command.

**Test-set prediction.** Each trained model is applied to the held-out test partition (`partition == "test"`, `excluded == False`). The model's saved StandardScaler handles feature transformation — the test set never sees its own statistics. What comes out is a vector of predicted probabilities per model; binary predictions are derived later by thresholding at either the default 0.5 or the tuned operating point, depending on what's needed downstream.

**Per-model metrics.** From probabilities and default-threshold predictions, the module computes precision, recall, F1, and AUC-ROC using scikit-learn's standard functions. Confusion matrices get explicit `labels=[0, 1]` to keep cell ordering stable even when a model predicts only one class — something that actually happened during early experiments with extreme imbalance. If the test set has only one class present, AUC defaults to 0.5 with a logged warning rather than blowing up.

**Bootstrap confidence intervals.** The test set is resampled with replacement 1,000 times and all four metrics are recomputed on each draw. The 2.5th and 97.5th percentiles give the 95% CI. If a bootstrap sample ends up with just one class (rare, but possible when positive counts are small), that iteration contributes AUC = 0.5 rather than being thrown out. The whole thing is seeded so it reproduces identically.

**McNemar's pairwise test.** For each model pair, the module counts discordant predictions — records where one model got it right and the other didn't — and builds a 2×2 contingency table. The predictions come from sklearn's default 0.5 threshold (via `model.predict()`), keeping the comparison fair rather than letting per-model threshold tuning tilt the results. Small discordant counts (<25) get an exact binomial test; larger ones use the chi-squared approximation. With three pairs to compare, the significance threshold is Bonferroni-corrected to α = 0.017.

**Baseline comparisons.** A single-feature Logistic Regression (with `class_weight='balanced'`) is trained and evaluated for each of the eleven features individually. This gives eleven single-feature AUC values; the best one becomes the bar the full models need to clear. The random baseline is 0.5 by definition — beating that just means the features carry *some* signal.

**Success tier classification.** Each model's AUC gets checked against the three tier boundaries (minimum > 0.60, target > 0.70, stretch > 0.80). The best model's tier determines the overall project verdict.

**Final summary.** Everything above — metrics, McNemar results, baselines, tier assignments, recommended config — gets packed into a `FinalSummary` dataclass and written out as a timestamped JSON file. This single artefact is what Section 5 draws on when presenting results.

### 4.7 Challenges and Decisions

Development didn't follow the design document in a straight line. Several problems surfaced during implementation that forced course corrections — some small, some fundamental.

**Timestamp unit mismatch (July 8–9).** The most disruptive bug was a unit conversion error in the temporal split logic. An early run produced a `split_timestamp` roughly 1000× too small, which cascaded into 89% record exclusion (most forward windows looked empty) and left just 7 surges in the test set — nowhere near enough for meaningful evaluation. The fix itself was simple (consistent epoch-second handling throughout), but the episode ate a full debugging session and motivated adding the temporal-ordering verification check described in Section 4.5.

**Data sparsity on r/pennystocks.** The original development dataset (r/pennystocks, 80,212 exploded records) produced exclusion rates between 69% and 89% depending on the `min_window_count` setting — the parameter that controls how many forward-window posts a record needs before its label counts as trustworthy. Higher values give cleaner labels but throw away more data. At `min_window_count=3`, test sets shrank to as few as 7 positive examples. AUC estimates from 7 surges are dominated by noise — one misranked record swings the metric by ±0.07. This was the main reason for bringing in r/wallstreetbets as a second dataset. With 457,072 usable records and 2,582 test surges, evaluation numbers finally became stable.

**Sentiment as a computational bottleneck.** VADER is fast per-record (~0.1ms), but applied across 1.3M raw wallstreetbets records it ate ~870 of the pipeline's ~900 second runtime. (Sentiment runs as stage 3, between windowing and labelling; it doesn't have its own implementation subsection because the logic is straightforward — the challenge was purely one of scale.) Two optimisations brought things under control: skipping records the windowing stage had already flagged as excluded (they don't need scores for training anyway) and caching the VADER analyzer at module level instead of creating a new one per call.

**Precision/recall collapse at default threshold.** On the wallstreetbets dataset at τ=1.5 (1.44% surge rate, 102:1 imbalance), XGBoost hit AUC 0.888 but predicted zero surges at the 0.5 threshold — precision and recall both exactly 0.0. The model was ranking surges correctly, but its probability outputs clustered far below 0.5 because of the extreme class prior. This wasn't a model failure; it was a calibration problem. Lowering τ to 1.0 (pushing the surge rate to ~5%) and adding `scale_pos_weight` to the XGBoost grid fixed it, and motivated the validation-fold threshold tuning described in Section 4.5.

**Feature dropping hurt more than noise removal helped (Experiment B1).** Removing the two weakest single-feature AUC predictors (`sentiment_score` at 0.47, `ticker_post_rate_24h` at 0.56) was supposed to reduce noise. Instead it dropped Random Forest AUC from 0.740 to 0.655 — an 8.5 percentage point loss. The takeaway: features with poor individual discrimination can still contribute when working alongside others. Both were restored and remain in the final eleven.

**Interaction features (Experiment B2).** Adding two hand-crafted interaction terms (`word_count_x_hour`, `accel_x_time_since_prev`) lifted Random Forest AUC by +1.4 percentage points on the pennystocks dataset. Modest but consistent, and since the features cost almost nothing to compute, they stayed.

**TextBlob → VADER switch.** The initial sentiment backend (TextBlob) was swapped for VADER after observing that TextBlob scored Reddit-style emphatic text (capitalisation, exclamation marks, slang intensifiers) as near-neutral. VADER's social-media-aware rules gave more dispersed polarity distributions and slightly better downstream AUC. The switchable backend architecture meant this was a config change, not a rewrite.

### 4.8 Implementation Status

All six pipeline stages are fully implemented and produce complete artefacts end-to-end:

| Stage | Status | Key Output |
|-------|--------|------------|
| 1. Data Loading | Complete | Exploded DataFrame (80K / 577K rows) |
| 2. Temporal Windowing | Complete | Forward/backward counts per ticker |
| 3. Sentiment | Complete | VADER polarity + forward-window means |
| 4. Target Labelling | Complete | Binary surge labels + threshold sweep |
| 5. Feature Engineering | Complete | 11-feature matrix |
| 6. Training & Evaluation | Complete | 3 trained models + full evaluation JSON |

Both datasets (r/pennystocks and r/wallstreetbets) run through the complete pipeline with reproducible results. Cross-dataset transfer evaluation, bootstrap confidence intervals, and McNemar's significance tests all work correctly. The experiment log holds 20+ runs documenting the progression from initial prototype to final reported numbers. All code in the core `surge_pipeline/` package passes the 10-module pytest suite, mypy type checking, and ruff linting without errors; the auxiliary `eda/` module carries one unused-variable warning that doesn't affect pipeline operation.

With the implementation complete, the next section presents and analyses the results it produces.

---

## 5. Evaluation

### 5.1 Evaluation Against Project Objectives

This section revisits the three objectives from Section 1.1 and measures each against the experimental results. A fourth item — reproducibility — is assessed separately as a design achievement that underpins the validity of the reported numbers.

#### 5.1.1 Objective 1: Predict Posting-Volume Surges

The central question was whether backward-looking features alone carry enough signal to forecast surges before they happen. The answer depends on how much data the model has to work with.

On r/wallstreetbets (68,923 test records, 3.75% surge rate), both tree-based models cleared the stretch tier: XGBoost reached AUC-ROC 0.892 [95% CI: 0.881–0.902] and Random Forest 0.880 [0.869–0.890]. These figures comfortably exceed the random baseline (0.50) and also beat the best single-feature predictor on this dataset (`word_count` alone scores 0.805). Logistic Regression achieved 0.707, which clears the target tier but falls short of that single-feature baseline — a point worth noting, since it means the linear model struggles to combine features as effectively as it could simply count words.

On the sparser r/pennystocks (3,278 test records, 0.95% surge rate), Random Forest achieved 0.753 [0.673–0.824], meeting the target tier. Here the best single feature (`hour_of_day`) manages only 0.591, so the multi-feature combination is clearly essential.

In short: surges are predictable from observation-time features. The harder constraint is data density, not methodology.

#### 5.1.2 Objective 2: Compare Multiple ML Approaches

The comparison was designed to answer a practical question: does model complexity pay off, or does a simple logistic regression do nearly as well?

On WSB, complexity pays clearly. The ranking follows the expected gradient: XGBoost (0.892) > Random Forest (0.880) > Logistic Regression (0.707), and all three pairwise differences are statistically significant (McNemar's test, p < 0.001 after Bonferroni correction at α = 0.017). The 17.3 percentage point gap between LR and XGBoost is large enough to be operationally meaningful, not just statistically detectable.

On pennystocks, the picture is more nuanced. Random Forest leads (0.753), followed by XGBoost (0.734) and LR (0.680). The inversion between XGBoost and Random Forest — where the more complex boosting model underperforms the bagged ensemble — is a finding in itself, and Section 5.3.1 analyses why this happens on sparse data. All three pairwise comparisons remain significant (p < 0.001), so the ordering is not a sampling fluke.

The takeaway: model complexity helps when data is abundant, but the advantage is not guaranteed under scarcity. Both findings required running the comparison to discover.

#### 5.1.3 Objective 3: Demonstrate Temporal Validity

The expanding-window protocol was designed to answer a specific worry: do these models genuinely predict the future, or do they merely memorise patterns from training data that happen to recur in nearby time periods?

The strongest evidence comes from the test-set results themselves. The held-out 20% — the final months of 2021 data, never seen during training, validation, or threshold selection — still produced AUC 0.892 on WSB and 0.753 on pennystocks. These are not inflated numbers from random cross-validation; they represent performance against genuinely unseen future data. Cross-dataset transfer (Section 5.2.5) provides further evidence: models trained on one community's entire timeline still discriminate surges in the other community's held-out future period.

The temporal protocol also reveals how much standard validation overstates performance. Random Forest's validation-fold F1 on WSB was 0.911; on the actual test set it dropped to 0.145. This gap is not a failure — it is the methodology doing its job. Without the strict temporal holdout, the 0.911 figure is what would have been reported, and any downstream use of the model would have been badly disappointed.

No future information leaked into training at any stage: z-score parameters are frozen from training-partition statistics, features use only backward-looking windows, and the temporal ordering was verified programmatically before every training run.

#### 5.1.4 Additional Achievement: Reproducible Pipeline

Reproducibility was not listed as a formal objective in Section 1.1, but it deserves separate assessment because it underpins confidence in everything reported above.

The pipeline is exactly reproducible: given the same input CSV, configuration JSON, and random seed, it produces bit-for-bit identical output. This was verified by running the same A1 configuration (seed=42) on July 13 and July 19 — both produced a best-model AUC of 0.753 (experiment log entries `2026-07-13_06-14` and `2026-07-19_13-14`).

Separately from exact reproducibility, the results are robust to the specific seed choice. Running the pennystocks pipeline with five different seeds (42, 123, 456, 789, 2024) produced best-model AUC values between 0.734 and 0.753, a range of just 0.019. The conclusions do not depend on getting lucky with the default seed.

The experiment log captures 30+ runs with full configuration JSONs, Git commit SHAs, and timestamped output paths. Any reported number can be traced back through this chain to the exact code and configuration that produced it.

### 5.2 Results

What follows are the raw experimental outcomes — numbers first, interpretation in Section 5.3. All metrics come from the held-out test partition (final 20% chronologically), which was never seen during training or threshold selection.

#### 5.2.1 Model Performance

*Table 11: Test-set performance at default threshold (0.5). 95% bootstrap CIs from 1,000 resamples.*

| Dataset | Model | AUC-ROC [95% CI] | Precision | Recall | F1 | Tier |
|---------|-------|-------------------|-----------|--------|-----|------|
| WSB (68,923 records, 668 surges, 0.97% rate) | LR | 0.707 [0.684–0.729] | 0.013 | 0.801 | 0.026 | Target |
| | RF | 0.880 [0.869–0.890] | 0.095 | 0.311 | 0.145 | Stretch |
| | XGB | 0.892 [0.881–0.902] | 0.043 | 0.819 | 0.081 | Stretch |
| Pennystocks (3,278 records, 31 surges, 0.95% rate) | LR | 0.680 [0.588–0.778] | 0.013 | 0.645 | 0.025 | Minimum |
| | RF | 0.753 [0.673–0.824] | 0.068 | 0.194 | 0.101 | Target |
| | XGB | 0.734 [0.641–0.821] | 0.000 | 0.000 | 0.000 | Target |

The AUC numbers tell one story; the precision and recall columns tell another. With sub-1% surge rates in both test sets, the default 0.5 probability threshold produces near-zero precision across the board. The models rank surges correctly (that is what AUC measures) but their raw probability outputs sit far below 0.5 because the learned prior is overwhelmingly "not a surge." XGBoost on pennystocks takes this to the extreme: it predicts zero surges at the default threshold despite ranking them correctly enough to achieve AUC 0.734.

Threshold tuning brings the precision–recall trade-off to a more practical operating point:

*Table 12: Metrics at validation-tuned thresholds.*

| Dataset | Model | Tuned Threshold | Precision | Recall | F1 | F1 Δ |
|---------|-------|-----------------|-----------|--------|-----|------|
| WSB | LR | 0.81 | 0.058 | 0.280 | 0.097 | +0.071 |
| | RF | 0.88 | 0.180 | 0.051 | 0.079 | −0.066 |
| | XGB | 0.85 | 0.217 | 0.235 | 0.226 | +0.145 |
| Pennystocks | LR | 0.67 | 0.085 | 0.194 | 0.118 | +0.093 |
| | RF | 0.79 | 0.200 | 0.097 | 0.130 | +0.030 |
| | XGB | 0.16 | 0.114 | 0.129 | 0.121 | +0.121 |

After tuning, XGBoost on WSB reaches F1 = 0.226 — roughly one in five flags is a real surge, and about a quarter of surges get caught. XGBoost on pennystocks needs a threshold of 0.16 to start predicting any surges at all. Random Forest on WSB is the one case where tuning hurts: the selected threshold overcorrects toward precision and catches too few actual surges.

To give a concrete sense of the error budget, here are the confusion matrices for the best model on each dataset at its tuned threshold:

*Table 13: Confusion matrix for the best model at tuned threshold.*

| Dataset | Model | Threshold | TP | FP | FN | TN |
|---------|-------|-----------|-----|------|------|-------|
| WSB | XGBoost | 0.85 | 157 | 565 | 511 | 67,690 |
| Pennystocks | Random Forest | 0.79 | 3 | 12 | 28 | 3,235 |

On WSB, XGBoost correctly identifies 157 of 668 surges while generating 565 false alarms out of 68,255 non-surge records — a false positive rate under 1%, despite the low precision. On pennystocks, the numbers are too small for confident conclusions (3 true positives out of 31 actual surges).

#### 5.2.2 Statistical Validation

*Table 14: McNemar's pairwise significance tests (default 0.5 threshold, Bonferroni-adjusted α = 0.017).*

| Dataset | Model Pair | χ² | p-value | Significant? |
|---------|------------|-----|---------|--------------|
| WSB | LR vs RF | 36,664 | < 0.001 | Yes |
| WSB | LR vs XGB | 24,246 | < 0.001 | Yes |
| WSB | RF vs XGB | 9,208 | < 0.001 | Yes |
| Pennystocks | LR vs RF | 1,406 | < 0.001 | Yes |
| Pennystocks | LR vs XGB | 1,486 | < 0.001 | Yes |
| Pennystocks | RF vs XGB | 66 | < 0.001 | Yes |

Every pairwise comparison is significant. A note on interpreting the magnitudes: the WSB χ² values are enormous (tens of thousands) partly because the sample size is large — 68,923 paired predictions generate many opportunities for disagreement. The models also disagree in a structurally consistent way: at the default threshold, LR and XGBoost flag thousands of records (high recall, low precision) while Random Forest flags very few (low recall, higher precision). These opposing strategies produce large discordant cell counts and correspondingly large test statistics.

*Table 15: Multi-feature models vs. baselines (AUC-ROC).*

| Dataset | Random Baseline | Best Single Feature | Best Model | Δ over Single Feature |
|---------|-----------------|---------------------|------------|----------------------|
| WSB | 0.500 | 0.805 (word_count) | 0.892 (XGB) | +0.087 |
| Pennystocks | 0.500 | 0.591 (hour_of_day) | 0.753 (RF) | +0.162 |

The surprise here is how well `word_count` performs alone on WSB (AUC 0.805). This is analysed in Section 5.3. On pennystocks, no individual feature gets above 0.591, so the multi-feature combination clearly earns its complexity.

#### 5.2.3 Cross-Dataset Transfer

*Table 16: Cross-dataset transfer AUC-ROC (no retraining).*

| Direction | LR | RF | XGBoost |
|-----------|------|------|---------|
| WSB-trained → Pennystocks test | 0.652 | 0.676 | 0.684 |
| Pennystocks-trained → WSB test | 0.753 | 0.842 | 0.871 |

The asymmetry is striking. A model trained on the small pennystocks dataset transfers to WSB at AUC 0.871 — just 2.1 points below natively-trained performance. Going the other direction, WSB models manage only 0.684 on pennystocks, clearing the minimum tier but falling short of target. Section 5.3 discusses why this asymmetry occurs.

#### 5.2.4 Feature Importance

*Table 17: Top-5 permutation importances (10 repeats, scoring=roc_auc) for tree-based models.*

| Rank | WSB — Random Forest | WSB — XGBoost | Pennystocks — RF | Pennystocks — XGB |
|------|--------------------:|-------------:|----------------:|------------------:|
| 1 | sentiment (+0.146) | sentiment (+0.203) | sentiment (+0.112) | sentiment (+0.166) |
| 2 | post_rate_24h (+0.045) | post_rate_24h (+0.067) | time_since_prev (+0.076) | time_since_prev (+0.054) |
| 3 | word_count (+0.010) | word_count (+0.027) | word_count (+0.031) | post_rate_24h (+0.017) |
| 4 | time_since_prev (+0.005) | time_since_prev (+0.003) | title_length (+0.015) | word_count (+0.009) |
| 5 | accel_x_time (+0.005) | day_of_week (+0.002) | num_tickers (+0.014) | num_tickers (+0.005) |

The pattern is consistent: `sentiment_score` is the single most important feature for both tree-based models on both datasets, contributing between +0.112 and +0.203 when permuted. The activity features (`time_since_previous`, `ticker_post_rate_24h`) fill out the top three. Everything else contributes small or negligible amounts — particularly `ticker_post_acceleration` and the interaction terms, which never exceed +0.005.

#### 5.2.5 Sentiment Contribution (Phase 1 vs Phase 2)

The cleanest test of whether sentiment helps is to hold the model constant and compare performance under volume-only labels (Phase 1, w₂=0) vs composite labels (Phase 2, w₁=w₂=0.5):

*Table 18: Volume-only (w₂=0) vs composite (w₁=w₂=0.5) — XGBoost AUC-ROC.*

| Dataset | Phase 1 (volume only) | Phase 2 (composite) | Δ AUC |
|---------|-----------------------|--------------------:|------:|
| WSB | 0.710 | 0.892 | +0.182 |
| Pennystocks | 0.685 | 0.734 | +0.049 |

Adding sentiment to the surge definition improves XGBoost by +18.2 points on WSB and +4.9 on pennystocks. A caveat: changing the weight also changes the surge rate (volume-only produces 0.53% surges on WSB vs 1.44% for the composite), so the two configurations define different prediction tasks. The improvement reflects both richer signal and a slightly more predictable target.

A broader weight sweep on WSB reinforces this finding:

*Table 19: Weight sensitivity — XGBoost AUC-ROC on r/wallstreetbets (τ=1.5).*

| w₂ | w₁ | AUC-ROC | Surge Rate | Tier |
|----|-----|---------|------------|------|
| 0.00 | 1.00 | 0.710 | 0.53% | Target |
| 0.25 | 0.75 | 0.708 | 1.01% | Target |
| 0.50 | 0.50 | 0.892 | 1.44% | Stretch |
| 0.75 | 0.25 | 0.861 | 4.30% | Stretch |
| 1.00 | 0.00 | 0.872 | 9.62% | Stretch |

There is a sharp discontinuity between w₂=0.25 and w₂=0.50 (+0.184 AUC). Part of this jump comes from the surge rate increasing (higher rates make easier targets), but the magnitude is disproportionate to a 0.43 percentage-point rate change — suggesting the sentiment component adds genuine predictive structure. All configurations with w₂ ≥ 0.50 reach the stretch tier; volume-dominant configurations stay at target.

### 5.3 Critical Analysis

#### 5.3.1 Why Did Model Rankings Differ Between Datasets?

The model ordering flipped between datasets: XGBoost dominated on WSB (0.892 vs RF's 0.880) while Random Forest led on pennystocks (0.753 vs XGBoost's 0.734). Three factors explain this:

**Training set size.** XGBoost's sequential boosting builds each tree to correct the previous ensemble's residuals. This error-correction mechanism requires sufficient training examples to distinguish genuine residual patterns from noise. WSB provides 388,149 training records with 5,649 surges; pennystocks offers 21,549 training records with only ~606 surges. With fewer positive examples, XGBoost's later boosting rounds may chase noise rather than signal — a form of overfitting that Random Forest's bagging approach is more resistant to.

**Feature interaction discovery.** Permutation importance shows that both tree-based models rely primarily on `sentiment_score` and `time_since_previous`. However, Random Forest's built-in (Gini) importances reveal a more distributed feature usage (day_of_week: 0.166, ticker_post_rate_24h: 0.205) than permutation importance suggests. On the sparse dataset, this broader feature utilisation may provide more robust predictions when any single feature's signal is unreliable.

**Probability calibration and class imbalance.** XGBoost on pennystocks predicted zero surges at the default threshold (all probabilities below 0.5), despite achieving AUC 0.734 — proving it *ranks* correctly but outputs miscalibrated probabilities. Random Forest, with its vote-fraction probability estimates, produced a less extreme calibration failure (predicting 6 surges at default threshold). The 105:1 imbalance ratio on pennystocks pushes learned priors so far toward the negative class that XGBoost's logistic output saturates near zero.

#### 5.3.2 What Worked Well

**The composite surge metric captured a genuine multi-dimensional phenomenon.** The Phase 1 vs Phase 2 comparison (Table 21) demonstrates that volume growth and sentiment change jointly define surges more meaningfully than volume alone. The +18.2 percentage point AUC improvement on WSB when adding sentiment to the target definition is the single largest performance gain observed across all experiments — larger than the difference between model families, threshold choices, or feature sets.

**Temporal cross-validation produced conservative, reliable estimates.** The gap between validation-fold performance and test-set performance is revealing: Random Forest on WSB achieved val_F1 = 0.911 during threshold tuning but test F1 = 0.145. This large gap is *expected* rather than alarming — it reflects the intentional conservatism of the temporal holdout. The validation fold is closer in time to the training data than the test set is, so the test set represents a harder generalisation challenge. The methodology's value lies precisely in producing this harder test.

**Backward-looking features avoided the circularity trap.** By excluding Reddit engagement scores (score, num_comments), the pipeline sacrifices easy discriminative power but gains genuine predictive validity. The resulting models can, in principle, score a post at the moment of its creation — before any community response has occurred. This constraint makes the AUC-ROC values achieved (0.753–0.892) more meaningful than higher values achievable with engagement features would be.

**Cross-dataset transfer confirmed structural generality.** The pennystocks-trained XGBoost transferring to WSB at AUC 0.871 (vs 0.892 natively) suggests the model learned generalisable surge patterns rather than community-specific quirks. The 2.1 percentage point transfer cost is remarkably small.

#### 5.3.3 What Did Not Work

**Precision remained poor at all operating points.** Even at the best tuned threshold (XGBoost on WSB at 0.85), precision reached only 0.217. This means approximately four out of five "surge" predictions are false alarms. For a surveillance system, this false positive rate would generate alert fatigue. The root cause is the severe class imbalance: with only 3.75% of records being surges, even a model with strong ranking ability produces many false positives when forced into binary decisions. AUC-ROC appropriately captures the model's discriminative ability, but the operational precision gap is a genuine limitation for deployment scenarios.

**Several features contributed near-zero importance.** `ticker_post_acceleration` showed negative or negligible permutation importance across all models on both datasets (−0.002 to +0.002). The interaction terms (`word_count_x_hour`, `accel_x_time_since_prev`) similarly contributed ≤0.005 in permutation importance on WSB despite the earlier experiment (B2) showing a +1.4 point gain on pennystocks. These features neither helped nor substantially hurt, but their presence adds complexity without measurable return at the final evaluation stage.

**VADER's sentiment scoring showed limited discrimination.** As a single feature, `sentiment_score` achieves only AUC 0.559–0.587 across datasets — barely above chance. Yet it dominates permutation importance (+0.146 to +0.203) in the full models. This apparent contradiction resolves when considering that permutation importance measures a feature's contribution *in the context of other features*: sentiment becomes discriminative when combined with activity-rate information, even though it carries weak signal alone. The limitation is that VADER's rule-based scoring cannot capture the sarcasm, irony, and domain-specific sentiment (e.g., "🚀🚀🚀" as extreme bullishness) that characterise Reddit financial discussion.

**Transfer from high-density to low-density underperformed.** WSB-trained models transferred to pennystocks at only AUC 0.684, below the target tier. The high-density community produces different feature distributions — particularly `word_count` (the best single-feature predictor on WSB at 0.805, but only 0.573 on pennystocks) — meaning patterns learned from WSB's verbose due-diligence posts do not map well to pennystocks' shorter, more speculative discussions.

#### 5.3.4 Unexpected Outcomes

**`word_count` as the best single predictor on WSB.** The single-feature baseline analysis revealed that word count alone achieved AUC 0.805 on WSB — a strong predictor by any standard, and unexpectedly more discriminative than activity features like posting rate (0.683) or time since previous (0.650). This suggests that on WSB specifically, surges are preceded by substantive long-form posts (the community's "DD" or due-diligence analysis posts), which trigger subsequent discussion cascades. This pattern was not anticipated from the literature.

**Threshold sensitivity revealed a non-linear response.** The weight sensitivity sweep (Table 22) showed a sharp discontinuity between w₂=0.25 (AUC 0.708) and w₂=0.50 (AUC 0.892), rather than the gradual improvement one might expect from adding more signal. This suggests that below a certain sentiment weight, the composite metric defines surges in a way that is fundamentally harder to predict from the available features — as if volume-only surges are more random, while sentiment-enriched surges have more structured precursors.

**Pennystocks models transferred upward better than WSB models transferred downward.** Conventional wisdom suggests that models trained on more data generalise better. The opposite occurred here: the pennystocks-trained XGBoost (trained on 21,549 records) achieved 0.871 on WSB's test set, while the WSB-trained XGBoost (trained on 388,149 records) managed only 0.684 on pennystocks' test set. The explanation lies not in data volume but in distributional mismatch — a model trained under scarcity learns conservative patterns that happen to hold universally, while a model trained in abundance may overfit to community-specific characteristics (like WSB's distinctive word_count distribution).

**The validation-test gap exposed practical deployment concerns.** Random Forest's validation-fold F1 of 0.911 vs test F1 of 0.145 on WSB represents an 84% performance drop. While the temporal holdout *correctly* reveals this gap (and would be invisible under random cross-validation), the magnitude suggests that surge dynamics shift meaningfully across the 2-month gap between the last validation fold and the test set. This is evidence of temporal non-stationarity — surge patterns in mid-2021 differ from those in late-2021, likely due to shifting market regimes and community demographics over time.

### 5.4 Limitations and Proposed Improvements

#### 5.4.1 Data Limitations

**Small positive class on r/pennystocks.** With only 31 surges in the test set, evaluation metrics are inherently noisy. Bootstrap CIs span ±0.08–0.09 in AUC, meaning the true model performance could plausibly range from "barely above chance" (lower CI bound of 0.588 for LR) to "strong" (upper CI bound of 0.824 for RF). The confidence intervals overlap substantially between models, limiting the strength of conclusions drawn from pennystocks alone. This is a fundamental property of the data: at τ=1.5, only 2.8% of records are surges, and with 3,278 test records, 31 positives are the mathematical consequence. Lowering τ to 1.0 increases surges to 8.2% (144 test surges), improving estimate stability at the cost of defining surges more loosely.

**Single time period (2021).** All results are bound to a single calendar year that included an extraordinary market event (the GameStop squeeze in January 2021). Surge dynamics during this period may not be representative of normal market conditions. The model may have learned patterns specific to the speculative mania of early 2021 rather than stable, recurring surge precursors. Evaluating on a different year would reveal how much performance depends on market regime.

**Survivorship bias in the archival data.** Posts deleted by moderators or users before the archive snapshot are absent from the dataset. If moderators disproportionately remove manipulation-related posts (which might be the most predictive of surges), the model is trained on a censored view of the community. The magnitude of this bias is unknown.

**Training randomness not fully characterised.** Seed stability testing across 5 seeds (range: 0.019 in best-model AUC) provides some evidence of robustness, but does not constitute a full sensitivity analysis of training variance. The fixed-seed approach means the reported metrics represent one draw from the distribution of possible outcomes. Bootstrap CIs capture *evaluation* variance but not *training* variance — the uncertainty from different random forest tree selections, XGBoost initialisation, and logistic regression solver convergence.

#### 5.4.2 Methodological Limitations

**VADER's domain mismatch.** VADER was designed for general social media and handles capitalisation, emoticons, and degree modifiers well. However, it lacks financial domain knowledge: "short" (bearish signal) is scored neutrally, "moon" (extreme bullish signal) receives no special treatment, and sarcasm — pervasive on Reddit — is entirely invisible to rule-based scoring. FinBERT [13] would address the domain gap at significant computational cost (~50× slower per record). Given that `sentiment_score` is the single most important feature for tree-based models (+0.146 to +0.203 permutation importance), even modest improvements in sentiment scoring could propagate into substantially better predictions.

**Ticker extraction heuristics.** The stopword-based extraction approach (Section 4.2) trades precision for recall: it captures tickers not on any predefined list but admits false positives. Common false positives include abbreviations (CEO, IPO, DD) that survive the 297-term stopword set, and short words in all caps used for emphasis ("HOLD," "SELL"). Each false positive dilutes the per-ticker activity counts with irrelevant noise. A hybrid approach combining the current regex with validation against a live ticker list (from SEC EDGAR or financial APIs) would reduce noise while preserving coverage of newly listed stocks.

**Fixed 24-hour window.** The pipeline uses a single temporal scale (24 hours forward, 24 hours backward) for windowing, surge definition, and activity features. This may miss multi-scale dynamics: some surges build over 6 hours while others develop across 72 hours. A multi-scale window approach (6h, 12h, 24h, 72h) would allow the model to detect surges at different temporal resolutions, though it would multiply feature count and introduce additional design choices.

**Cost-sensitive learning vs. resampling.** The pipeline uses `class_weight='balanced'` and `scale_pos_weight` rather than synthetic oversampling (SMOTE). While this avoids the temporal-consistency problems of SMOTE (Section 3.5), it may not fully compensate for extreme imbalance ratios like 105:1. Threshold-based approaches to imbalance handling (treating it as a post-hoc calibration problem rather than a training-time problem) might be more effective, as the threshold tuning results suggest.

#### 5.4.3 Proposed Improvements

| Improvement | Expected Impact | Effort | Priority |
|-------------|----------------|--------|----------|
| FinBERT for sentiment | Higher discrimination from sentiment feature | Medium (GPU required, ~50× slower) | High |
| Multi-scale windows (6h, 12h, 24h, 72h) | Capture surges at different speeds | Medium (feature engineering) | High |
| Known-ticker validation list | Reduce noise from false positive tickers | Low (SEC EDGAR API integration) | Medium |
| Additional time periods (2020, 2022) | Test temporal stability beyond 2021 | Low (data acquisition) | Medium |
| Probability calibration (Platt/isotonic) | Fix precision collapse without threshold hacks | Low (post-hoc calibration layer) | Medium |
| Additional subreddits (r/stocks, r/investing) | Test density gradient more granularly | Medium (pipeline re-run) | Low |
| Graph-based features (cross-ticker mentions) | Capture community network effects | High (new feature paradigm) | Low |

### 5.5 Originality and Contribution

This project makes four specific contributions to the intersection of social media prediction and computational finance:

**1. A leakage-free surge prediction framework.** The reviewed literature (Section 2.5) reveals pervasive temporal leakage in social media prediction studies — random train-test splits, same-period evaluation, and features derived from future engagement. This project demonstrates a complete methodology that prevents leakage at every stage: target labels use training-partition z-scores only, features are strictly backward-looking, and evaluation uses a held-out future period. The resulting performance figures (AUC 0.753–0.892) are conservative but trustworthy — they represent what a deployed system could actually achieve, not inflated estimates from leaked information. This contribution is methodological rather than algorithmic: the same framework applies to any temporal prediction task.

**2. Composite surge metric combining volume and sentiment.** No reviewed study defines a composite binary target that integrates normalised posting-volume growth with sentiment change magnitude within a fixed time window. The closest prior work (Costola et al. [10]) examines consensus formation qualitatively rather than defining a quantifiable, reproducible metric. The Phase 1 vs Phase 2 comparison (Table 21) provides direct evidence that this composite design captures a richer phenomenon than volume alone (+18.2% AUC on WSB), validating the additional complexity. The metric is fully parameterised (threshold τ, weights w₁/w₂), making it adaptable to different communities and operational requirements.

**3. Empirical evidence that data density is the binding constraint.** The dual-dataset design produces a controlled comparison: same pipeline, same features, same models, different data density. The performance gap (0.753 on pennystocks vs 0.892 on WSB) and the asymmetric transfer results (sparse→dense works well; dense→sparse does not) establish that data density — not model choice, feature engineering, or methodology — is the primary determinant of prediction quality. This finding has practical implications: practitioners should invest in data collection (more subreddits, longer time periods, finer granularity) before investing in model complexity.

**4. Statistical rigour in model comparison.** Reporting a single AUC number without uncertainty quantification is standard practice in much of the applied ML literature. This project provides bootstrap confidence intervals, pairwise McNemar's tests with Bonferroni correction, single-feature baselines, cross-dataset transfer, seed stability analysis, and a systematic weight sensitivity sweep. The result is that every claim about model performance is supported by quantified evidence, and every model difference is tested for statistical significance. This level of evaluation rigour is not novel in statistics but remains uncommon in applied social media prediction research.

These contributions are incremental rather than revolutionary — they extend existing methods (temporal evaluation from forecasting literature [14][15], composite metrics from multi-criteria decision-making, permutation importance from interpretable ML) into an underexplored application domain. Their value lies in the careful integration of established techniques into a coherent framework that produces reliable, interpretable results for a prediction problem that previous research has not directly addressed.

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

[1] Gabor Szabo and Bernardo A. Huberman. 2010. Predicting the popularity of online content. *Commun. ACM* 53, 8 (August 2010), 80–88. https://doi.org/10.1145/1787234.1787254

[2] Kristina Lerman and Tad Hogg. 2010. Using a model of social dynamics to predict popularity of news. In *Proceedings of the 19th International Conference on World Wide Web (WWW '10)*. ACM, New York, NY, USA, 621–630. https://doi.org/10.1145/1772690.1772754

[3] Roja Bandari, Sitaram Asur, and Bernardo A. Huberman. 2012. The pulse of news in social media: Forecasting popularity. In *Proceedings of the 6th International AAAI Conference on Weblogs and Social Media (ICWSM '12)*. AAAI Press, 26–33.

[4] Johan Bollen, Huina Mao, and Xiaojun Zeng. 2011. Twitter mood predicts the stock market. *J. Comput. Sci.* 2, 1 (March 2011), 1–8. https://doi.org/10.1016/j.jocs.2010.12.007

[5] Justin Cheng, Lada Adamic, P. Alex Dow, Jon M. Kleinberg, and Jure Leskovec. 2014. Can cascades be predicted? In *Proceedings of the 23rd International Conference on World Wide Web (WWW '14)*. ACM, New York, NY, USA, 925–936. https://doi.org/10.1145/2566486.2567997

[6] Fang Wang and Bernardo A. Huberman. 2013. Quantifying long-term scientific impact. *Science* 342, 6154 (October 2013), 127–132. https://doi.org/10.1126/science.1237825

[7] Shoubin Kong, Qiaozhu Mei, Ling Feng, Fei Ye, and Zhe Zhao. 2014. Predicting bursts and popularity of hashtags in real-time. In *Proceedings of the 37th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '14)*. ACM, New York, NY, USA, 927–930. https://doi.org/10.1145/2600428.2609476

[8] Chao Yuan and Wentao Li. 2019. Forecasting the development trend of early-stage information diffusion based on empirical data. *Physica A* 524 (June 2019), 157–167. https://doi.org/10.1016/j.physa.2019.04.053

[9] Cathy Long, Brian Lucey, and Larisa Yarovaya. 2023. I just like the stock: The role of Reddit sentiment in the GameStop share rally. *Financ. Rev.* 58, 1 (February 2023), 19–37. https://doi.org/10.1111/fire.12328

[10] Michele Costola, Matteo Iacopini, and Carlo R. M. A. Santagiustina. 2022. Self-induced consensus of Reddit users to characterise the GameStop short squeeze. *Sci. Rep.* 12, 1 (August 2022), Article 13780. https://doi.org/10.1038/s41598-022-17925-2

[11] Adriano Mancini, Aldo Desiderio, Brendan Marafino, and Alessandro Navigli. 2024. Detecting pump and dump stock market manipulation from online forums. *Digital Finance* 6 (2024), 365–393. https://doi.org/10.1007/s42521-024-00113-6

[12] Clayton J. Hutto and Eric Gilbert. 2014. VADER: A parsimonious rule-based model for sentiment analysis of social media text. In *Proceedings of the 8th International AAAI Conference on Weblogs and Social Media (ICWSM '14)*. AAAI Press, Ann Arbor, MI, 216–225.

[13] Dogu Araci. 2019. FinBERT: Financial sentiment analysis with pre-trained language models. Retrieved from https://arxiv.org/abs/1908.10063

[14] Leonard J. Tashman. 2000. Out-of-sample tests of forecasting accuracy: An analysis and review. *Int. J. Forecast.* 16, 4 (October–December 2000), 437–450. https://doi.org/10.1016/S0169-2070(00)00065-0

[15] Christoph Bergmeir and José M. Benítez. 2012. On the use of cross-validation for time series predictor evaluation. *Inf. Sci.* 191 (May 2012), 192–213. https://doi.org/10.1016/j.ins.2011.12.028

[16] Manuel Fernández-Delgado, Eva Cernadas, Senén Barro, and Dinani Amorim. 2014. Do we need hundreds of classifiers to solve real world classification problems? *J. Mach. Learn. Res.* 15, 1 (January 2014), 3133–3181.

[17] Leukipp. 2021. Reddit Finance Data. Kaggle. Retrieved from https://www.kaggle.com/datasets/leukipp/reddit-finance-data
