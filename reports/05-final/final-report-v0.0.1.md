---
title: "Predicting Volume and Sentiment Surges in Reddit Financial Communities Using Machine Learning"
# pandoc-crossref configuration (auto-numbering for tables, figures, sections)
figureTitle: "Figure"
tableTitle: "Table"
listingTitle: "Listing"
figPrefix: "Figure"
tblPrefix: "Table"
lstPrefix: "Listing"
secPrefix: "Section"
numberSections: true
linkReferences: true
nameInLink: true
---

```{=typst}
// Page setup — ACM-style margins and font
#set page(margin: 2.54cm)
#set text(font: "Times New Roman", size: 11pt)
#set par(justify: true, leading: 0.55em, first-line-indent: 1em)

// Heading styles
#show heading.where(level: 1): it => {
  set text(size: 14.4pt, weight: "bold")
  set align(left)
  v(0.5cm)
  it
  v(0.3cm)
}

#show heading.where(level: 2): set text(size: 12pt, weight: "bold")
#show heading.where(level: 3): set text(size: 11pt, weight: "bold", style: "italic")

// Caption handling: pandoc-crossref already writes "Table N:" / "Figure N:" into
// the caption text, so suppress Typst's own automatic figure prefix to avoid
// doubling (e.g. "Table 1: Table 1: ...").
#show figure.caption: it => it.body
#show figure: set figure(supplement: none, numbering: none)

// Table styling
#show figure.where(kind: table): set block(breakable: true)

#show table: set table(
  stroke: (x, y) => (
    top: if y == 0 { 0.4pt + rgb("#eeeeee") } else { none },
    bottom: if y == 0 { 1pt + rgb("#cccccc") } else { 0.4pt + rgb("#eeeeee") },
    left: if x > 0 { 0.4pt + rgb("#f0f0f0") } else { none },
    right: if x < 1 { 0.4pt + rgb("#f0f0f0") } else { none }
  ),
  fill: (col, row) => if row == 0 { rgb("#fafafa") } else { none },
  inset: 7pt,
  align: (col, row) => left + horizon
)

// Code block styling
#show raw.where(block: true): it => block(
  stroke: 0.5pt + rgb("#cccccc"),
  radius: 4pt,
  inset: 10pt,
  fill: rgb("#fcfcfc"),
  width: 100%,
  it
)
```

# Abstract {.unnumbered}

Social media discussions in financial communities can shift from quiet to frenzied within hours. This project develops a screening pipeline that predicts whether discussion around an individual stock ticker will surge during the following 24 hours, using only information available when each post is observed. The predictions are intended for market surveillance and compliance teams prioritising unusual activity for investigation, quantitative researchers selecting tickers for deeper analysis, and platform moderators allocating monitoring capacity. The pipeline combines backward-looking activity, textual, temporal, and sentiment features with a composite target based on future volume growth and sentiment change, then evaluates Logistic Regression, Random Forest, and XGBoost using expanding-window validation and a chronological held-out test period. This design tests whether the predictions remain useful when applied to genuinely later data rather than allowing future observations into training. XGBoost achieved the strongest ranking performance on the high-volume `r/wallstreetbets` community (AUC-ROC 0.892), while Random Forest performed best on the sparser `r/pennystocks` community (0.753), indicating that data density constrains performance more than model complexity. On `r/wallstreetbets`, the best operating point produced 21.7% precision and 23.5% recall: approximately one in five flagged cases was a surge, but most surges were missed. This trade-off can support selective human review, particularly for quantitative research, but is insufficient for broad surveillance or moderation coverage and does not justify autonomous action. Cross-community transfer remained above chance (AUC 0.684 from `r/wallstreetbets` to `r/pennystocks`) but requires community-specific recalibration. The main contribution is therefore a temporally valid ranking and screening framework, with performance strong enough to prioritise attention but limited by class imbalance, threshold calibration, sparse-community uncertainty, and the use of a single 2021 observation period.

---

# Introduction

## Project Concept and Objectives

Following the **CM3005 Data Science** project template, *Predictive Modelling of Social Media Trend Emergence*, this project implements a machine learning system that predicts whether a stock ticker's Reddit discussion is about to surge, using only backward-looking features available at observation time. Three classifiers (Logistic Regression, Random Forest, and XGBoost) are trained and compared on this task.

The project has three objectives:

- Build a predictive model from early-stage discussion features (temporal patterns, activity frequency, sentiment) that can forecast per-ticker surges before they happen
- Compare multiple ML approaches to find out whether more complex models actually improve prediction over simpler baselines
- Confirm that predictions hold up on unseen future time periods by using temporal evaluation protocols that prevent data leakage, a common methodological weakness in social media prediction studies

## Problem Statement and Motivation {#sec:problem-motivation}

Stock-related discussions on Reddit can go from quiet to frenzied within hours. A ticker attracting two posts yesterday might appear in fifty today, triggered by earnings surprises, speculative momentum, or coordinated retail interest. These surges develop too quickly for manual monitoring, particularly across forums where thousands of tickers are discussed daily.

The central research question is: *can a social media surge be predicted before it happens by looking at early discussion patterns?* The answer holds practical value for three user groups. **Market surveillance and compliance teams** would use predictions to prioritise which tickers warrant investigation for unusual or potentially coordinated activity among thousands of candidates [1]. **Quantitative researchers** would use them to select a small set of tickers for deeper financial and textual analysis. **Platform moderation teams** would use them to allocate monitoring and moderation capacity before discussion spikes. Across all three, the system operates as a *prioritisation tool*, reducing a large surveillance universe to a manageable shortlist for human review, rather than an autonomous decision-maker [2].

These use cases do not share an identical error tolerance. For surveillance and moderation, missing 70% of surges would be unacceptable because the purpose is broad coverage and missed events may carry greater cost than additional reviews. For quantitative research, a lower-recall shortlist can be acceptable when the captured cases are sufficiently relevant to justify follow-up analysis. Across the applications, false positives are tolerable when they produce a reviewable queue: as an operational starting point, precision of approximately 20% means that about one in five flagged cases merits attention, provided the daily volume remains within analyst capacity. This project therefore sets a stricter screening aspiration of recall $\ge 0.50$ and precision $\ge 0.10$ at the chosen threshold, with 20–30 flags per day, while treating these as application criteria rather than claims that the model already satisfies them. [@sec:operational-criteria] formalises these requirements, and [@sec:operational-precision] evaluates the best model against them.

Standard rule-based heuristics, such as flagging a ticker when volume exceeds $+2\sigma$, cannot capture non-linear interactions across diverse data streams. This project proposes a learning-based approach that integrates temporal, textual, and sentiment features into a unified predictive framework. Prior research focuses on related but different problems, forecasting eventual content reach or predicting price movements, rather than predicting sudden short-term spikes in discussion volume for specific tickers (see [@sec:research-gap]).

## Prediction Scope and Surge Definition

The original project template uses the term "trend emergence," but trends can be gradual and sustained, making them difficult to label objectively. This project focuses on surges: statistically significant, short-term spikes in both **posting volume** and **sentiment intensity** for a ticker **within a 24-hour window**, identified using a composite metric combining normalised volume growth with sentiment shift magnitude (see [@sec:sentiment-contribution]). Because surges are discrete and quantifiable, they can be framed as a binary classification problem.

The target is based on timestamped post volume rather than engagement metrics like upvotes, as post-hoc scores introduce look-ahead bias. Z-scores are calculated using training-set statistics alone to prevent data leakage. [@sec:surge-definition] outlines the formal definitions, weighting, and threshold choices.

The underlying hypothesis is that backward-looking temporal and textual features carry sufficient signal to discriminate surges from baseline activity, and that predictive performance scales with data density rather than model complexity.

## Scope

This study evaluates binary surge classification across two archival 2021 Reddit datasets: `r/pennystocks` and `r/wallstreetbets` (hereafter **WSB**). Using strictly backward-looking features, three classifier families: Logistic Regression (**LR**), Random Forest (**RF**), and XGBoost (**XGB**) are trained to predict 24-hour ticker surges. Model performance is assessed using an 80/20 chronological holdout split alongside 4-fold expanding-window cross-validation, supported by bootstrap confidence intervals, McNemar's pairwise tests, and single-feature baselines. Cross-dataset transfer experiments assess model generalisability, using a fully deterministic pipeline with fixed seeds for end-to-end reproducibility.

Excluded from scope: real-time data ingestion and production deployment; individual user behaviours, comment networks, and cross-platform channels; multi-class or regression targets; trading signals, financial advice, or causal claims regarding market impact.

## Report Structure

Section 2 reviews the literature on online attention prediction, financial sentiment, and Reddit-specific research. Section 3 details the design: surge definition, feature engineering, model selection, and temporal validation. Section 4 describes implementation. Section 5 presents results, statistical validation, critical analysis, and limitations. Section 6 concludes with key findings and future directions.

---

# Literature Review

## The Predictability of Online Attention

Predicting trends broadly involves analyzing time-series data, applying deep learning sequence models, tracking how information spreads through networks, and detecting unscheduled events automatically. This review focuses specifically on supervised tabular classification using hand-crafted features to predict binary outcomes, distinguishing it from sequence modeling, graph methods, learned representations, and continuous trajectory forecasting. The seventeen sources reviewed here (Section 7) were selected because they directly inform the three decisions this project makes: *what to predict* (popularity and surge onset literature), *what signals to use* (sentiment and content features), and *how to evaluate rigorously* (temporal validation methods). A fourth strand, Reddit-specific financial research, confirms that this platform contains distinct, predictable signals.

The foundational question underlying this project, *can future surges in social media activity be predicted?*, was first addressed through research on **online popularity prediction**. This literature demonstrated that online attention is not random: *content that attracts early engagement tends to attract more, following patterns that are statistically detectable*.

Szabo and Huberman [5] produced the seminal result in this domain, demonstrating strong linear correlations between early and later popularity on YouTube and Digg. Their regression model showed that a content item's view count at time *t* predicts its eventual popularity with high accuracy. This established the core principle: early behavioural signals carry predictive information about future attention. However, because the model assumes a stationary growth process, it requires content to have already accumulated measurable engagement before prediction becomes possible. It cannot forecast at or near the time of posting.

Lerman and Hogg [6] extended this understanding by modelling the interaction between social network structure and content discovery, demonstrating that popularity depends on behavioural dynamics beyond simple cumulative counts. Their agent-based approach revealed that network position and user browsing patterns mediate how content gains visibility. Wang and Huberman [7] and Kong et al. [8] further characterised online attention as following identifiable temporal lifecycles (emergence, growth, peak, and decline) suggesting that content at different lifecycle stages exhibits different observable signatures. Kong et al. [8] specifically addressed *burst* detection for hashtags in real-time, but their approach operates at the hashtag level with contemporaneous features. It detects bursts as they happen rather than predicting them before onset, and works at a platform-wide granularity rather than per-entity (ticker) level.

The academic consensus that emerged from this first wave of research can be summarised as: *online attention is predictable from early signals, follows lifecycle dynamics, and is mediated by platform-specific network effects*. However, these models all require content to have already gained some traction before prediction is possible, and they target *eventual* popularity rather than the *onset* of rapid growth. Furthermore, a key tension exists within these findings: Szabo and Huberman [5] show that early popularity strongly predicts final outcome, yet Cheng et al. [9] later found that cascade prediction accuracy plateaus after the initial phase. This suggests that predictability diminishes once content leaves the emergence stage, which is precisely the window this project targets.

![Online attention lifecycle model [7][8]. Traditional popularity prediction requires content to have reached the growth phase before forecasting is possible. This project targets the emergence phase, predicting a surge before substantial engagement has accumulated.](figures/fig1-online-attention-lifecycle-model.png){#fig:lifecycle}

## The Shift Toward Pre-Engagement Prediction

A second wave of research addressed the limitation that early popularity models require existing engagement data. Bandari et al. [10] demonstrated that content metadata such as source, category, subjectivity, named entities could predict popularity prior to engagement accumulates, achieving approximately 84% classification accuracy on news articles.This represented a key methodological shift: prediction could occur at or before publication rather than requiring an observation period.

Cheng et al. [9] achieved approximately 79.5% accuracy (AUC = 0.877) predicting whether Facebook photo cascades would double in size, using temporal features derived from early propagation speed and structural virality metrics. Their findings showed that the *rate* of initial spread, rather than its magnitude, carries predictive signal for sustained growth. Yuan and Li [11] extended this principle across information diffusion contexts, showing that early-stage propagation patterns contain sufficient signal to forecast long-term diffusion trajectory.

This body of work established a second consensus: prediction is achievable before substantial engagement accumulates, provided features capture content characteristics or early propagation dynamics. However, the prediction targets remained eventual outcomes, such as final popularity, total cascade size rather than *rapid onset* within a bounded time window. For example, an analyst monitoring a financial forum requires knowledge of whether discussion will surge within the sibsequent 24 hours, rather than whether it will eventually become popular. This temporal distinction represents a key gap unaddressed.

| Study | Year | Platform | Prediction Target | Requires Existing Engagement? | Accuracy |
|-----------|------|----------|------------|------------|----------|
| Szabo & Huberman [5] | 2010 | YouTube, Digg | Future view count | Yes (needs early views) | $r^{2}$ > 0.9 |
| Lerman & Hogg [6] | 2010 | Digg | Story popularity | Yes (needs network data) | N/A (model) |
| Bandari et al. [10] | 2012 | News articles | Popularity bin | **No** (content metadata only) | ~84% |
| Cheng et al. [9] | 2014 | Facebook | Cascade doubling | Partial (early reshares) | 79.5% (AUC = 0.877) |
| Yuan & Li [11] | 2019 | Weibo | Diffusion trajectory | Partial (early propagation) | N/A (descriptive) |

: Evolution of online attention prediction, from post-engagement to pre-engagement approaches. {#tbl:attention-evolution}

## Sentiment as a Predictive Signal in Finance

Alongside popularity research, computational finance studies established that collective social media sentiment carries measurable predictive information. Bollen et al. [12] demonstrated that aggregate Twitter mood, particularly the "Calm" dimension measured by the Google-Profile of Mood States (GPOMS), predicted Dow Jones movements with roughly 87.6% directional accuracy. Although limited by a short evaluation window, missing out-of-sample testing, and an unclear causal mechanism, their study proved pivotal in establishing that **social media textual sentiment can inform financial forecasting**.

The tools used for sentiment extraction have evolved alongside this finding. General-purpose lexicons like OpinionFinder lack domain specificity for financial language, where terms like "short," "bearish," or "moon" carry specialised meaning. To better capture online discourse, Hutto and Gilbert [13] developed VADER specifically for social media text, incorporating rules for punctuation emphasis, capitalisation, degree modifiers, and negation, and achieving F1=0.96 on social media benchmarks. Araci [14] later introduced FinBERT, a transformer model fine-tuned on financial corpora, capturing contextual meaning that rule-based tools miss. This progression from general lexicons, to social-media rules, to domain-specific deep learning highlights the field's consensus that sentiment analysis tools must be tailored to their specific domain

For this project, the key takeaway is that sentiment *change*, rather than absolute sentiment value, may serve as a leading indicator of activity surges: if a ticker's discussion becomes markedly more emotional before volume escalates, sentiment shift could provide early warning signal. This motivates including sentiment change magnitude in the composite surge metric.

| Tool | Type | Domain | Strengths | Limitations for This Project |
|----------|----------|--------|---------------|--------------|
| OpinionFinder as used in [12] | Lexicon | General | Early adoption, widely cited | No social media conventions, no financial terms |
| VADER [13] | Rule-based | Social media | Handles capitalisation, emoticons, negation; F1=0.96 | No financial domain tuning ("short," "moon" misscored) |
| FinBERT [14] | Transformer | Financial text | Context-aware, domain-specific | Computationally expensive for 1M+ records |

: Evolution of sentiment analysis tools relevant to financial social media. {#tbl:sentiment-tools}

## Financial Discussion on Reddit

While the preceding research established foundational principles on platforms like YouTube, Digg, Facebook, and Twitter, recent studies examine whether these dynamics hold within Reddit's financial communities and how their unique structural features alter information flow.

Penny stocks (low-capitalisation equities trading below $5) occupy a distinctive position because their low liquidity and limited analyst coverage mean that social media discussion can constitute a disproportionate share of available information [15][16]. Unlike Twitter's ephemeral broadcast environment, Reddit's subreddit structure creates concentrated communities with persistent threads and shared behavioral norms.

Long et al. [15] demonstrated that `WSB` posting volume correlated with abnormal trading volume and returns for discussed stocks, with effects concentrated in small-cap equities. Their analysis showed that increased Reddit attention *preceded* trading activity in their sample, suggesting that discussion patterns carry predictive signal rather than merely reflecting market events. Costola et al. [16] examined the GameStop episode specifically, finding that consensus formation within `WSB` followed measurable patterns in posting frequency and sentiment alignment *before* reaching critical mass. A small number of committed users drove broader engagement through detectable temporal signatures. Extending this to security manipulation, Mancini et al. [17] constructed predictive models using the textual and temporal properties of forum posts. Their work confirmed that forum-derived text features significantly outperform chance baselines in forecasting anomalous stock activity.

These studies confirm that Reddit financial communities produce predictive signals, yet prior work exclusively targets market outcomes such as returns, volume, manipulation rather than platform dynamics. Notably, these findings align with general literature: Costola et al.'s consensus patterns [16] reflect Lerman and Hogg's network discovery dynamics [6], while Long et al.'s temporal precedence [15] echoes Cheng et al.'s early speed metrics [9]. However, predicting whether discussion itself will rapidly escalate, forecasting an imminent surge in posting volume, remains an open challenge. Addressing this gap is the central focus of this paper.

## Methodological Weaknesses in Prior Work {#sec:methodological-weaknesses}

Beyond the substantive gaps identified above, a critical methodological pattern cuts across the literature: a recurring absence of rigorous temporal evaluation protocols.

Szabo and Huberman [5] evaluate on data drawn from the same time period as training. Bandari et al. [10] use random train-test splits rather than temporal partitions, allowing models to be tested on articles published before some training data, a form of information leakage. Cheng et al. [9] randomly sample cascades for evaluation without preserving temporal ordering. Long et al. [15] and Costola et al. [16] analyse correlations across their full datasets without testing whether historical patterns generalise to later periods.

This methodological oversight is significant because Tashman [18] demonstrated that rolling-origin evaluation, where the forecasting origin advances forward through time, produces far more reliable accuracy estimates for temporal prediction tasks than fixed or random splits. Furthermore, Bergmeir and Benítez [19] showed empirically that random cross-validation overestimates predictive accuracy on time-dependent data, and recommended blocked or expanding-window schemes that preserve temporal ordering. Despite established best practices in time-series forecasting, they remain largely unadopted in social media prediction research.

Consequently, reported performance figures across the reviewed studies may be inflated by temporal leakage, and it remains uncertain whether models would generalise to genuinely unseen future periods. For any system intended for real-world deployment, including surge detection, this is a critical deficiency. As Fernández-Delgado et al. [20] noted in their large-scale classifier benchmark, evaluation methodology substantially affects reported performance rankings, reinforcing that how a model is evaluated matters as much as which model is selected.

| Study | Evaluation Method | Temporal Ordering Preserved? | Leakage Risk |
|---------------|----------------|-------------|--------------|
| Szabo & Huberman [5] | Same-period evaluation | No | High |
| Bandari et al. [10] | Random train-test split | No | High |
| Bollen et al. [12] | Fixed holdout (1 month) | Partial | Medium |
| Cheng et al. [9] | Random cascade sampling | No | High |
| Long et al. [15] | Full-dataset correlation | No | High |
| Costola et al. [16] | Full-dataset analysis | No | High |
| Mancini et al. [17] | Chronological split | Partial | Medium |

: Temporal evaluation practices across reviewed studies. {#tbl:temporal-practices}

## Research Gap and Project Position {#sec:research-gap}

The literature reviewed above establishes four cumulative findings:

- Early behavioural signals predict future online attention [5][9]
- Prediction is possible before engagement accumulates, using content and propagation features [10][9][11]
- Sentiment extracted from social media carries predictive value in financial contexts [12][13]
- Reddit financial communities generate measurable signals that precede market activity [15][16][17]

Four key gaps remain unaddressed:

1. **Prediction target:** All reviewed studies predict *eventual outcomes* (e.g., total popularity, cascade size, or market returns) rather than detecting the *onset* of rapid growth within a bounded time frame. Existing literature lacks a formulation for defining or predicting a composite volume-and-sentiment surge within a fixed short-term window for individual entities.

2. **Signal integration:** Each research strand demonstrates one feature category's value in isolation such as temporal [5], content [10], sentiment [12], or structural [9], but empirical integration of multiple signal types into a unified predictive framework remains limited, despite evidence that they interact during trend formation [6][8].

3. **Domain specificity:** General social media prediction research [5][10][9] neglects the distinct dynamics of financial discussions such as event-driven reactions, domain-specific language, and speculative behaviour. Conversely, Reddit financial research [15][16][17] predicts market consequences of surges rather than predicting whether surges *will occur*.

4. **Temporal validity:** The use of random or unspecified evaluation splits across the literature [5][10][9][15] means reported results may not reflect real-world predictive performance. Rigorous temporal evaluation methods exist [18][19] but remain unadopted in this domain.

This project addresses these gaps directly. First, the composite surge metric ([@sec:surge-definition]) defines a binary onset target evaluated within a strict 24-hour window (Gap 1). Second, the feature set ([@sec:feature-engineering]) combines temporal, activity-frequency, sentiment, and textual signals (Gap 2). The pipeline is applied to Reddit financial communities using two subreddits at opposite ends of the data density spectrum (Gap 3) to evaluate domain-specific applicability. Finally, model evaluation strictly utilizes expanding-window temporal cross-validation ensures that no future information leaks into training (Gap 4). Whether this integration yields meaningful predictive performance is the empirical question examined in Section 5.

---

# Design

## System Architecture

The prediction system is a six-stage linear pipeline. Each stage consumes the previous stage's output and writes intermediate artefacts to disk, enabling independent re-execution without recomputing upstream operations.

1. **Data Loading and Preprocessing**: Ingests raw CSV data, performs text cleaning, regex-based ticker extraction, and explodes multi-entity records into unique record–ticker pairs
2. **Temporal Windowing**: Computes per-ticker forward and backward 24-hour posting volume via vectorized binary search
3. **Sentiment Computation**: Calculates VADER compound scores per record with title-fallback when selftext is absent
4. **Target Labelling**: Establishes temporal train/test split (80/20), computes z-score parameters from training statistics only, and derives binary targets via composite thresholding
5. **Feature Engineering**: Extracts eleven backward-looking features ([@sec:feature-engineering])
6. **Model Training and Evaluation**: Expanding-window cross-validation, hyperparameter tuning, holdout evaluation, and statistical testing

![Pipeline architecture. Shading indicates critical design points: target labelling (leakage prevention), model training (temporal validation), and evaluation (statistical rigour).](figures/fig2-data-pipeline-v0.1.png){#fig:pipeline}

The foundational constraint is that **no stage may access future information relative to any record's observation time**. Features use backward-looking windows exclusively, z-scores are frozen from training statistics, and validation folds are strictly time-ordered.

## Data Selection and Characteristics

### Platform Selection

The prediction task imposes five requirements on the data source: (1) public availability without authentication barriers, enabling third-party reproduction; (2) per-record timestamps at sub-hourly granularity, supporting 24-hour windowing; (3) per-entity (ticker) attribution, allowing surge computation at the stock level rather than the aggregate level; (4) sufficient post volume to construct meaningful per-ticker time series; and (5) archival availability as a static snapshot, ensuring identical data across experimental runs.

| Criterion | Reddit | Twitter/X |
|-----------|--------|-----------|
| Public bulk archive | Yes (Kaggle, Pushshift) | No (API restricted post-2023) |
| Timestamp granularity | Unix-second | Unix-second |
| Per-ticker attribution | Explicit ($TICKER convention) | Implicit (cashtags, noisy) |
| Volume (2021, finance) | ~1.3M submissions across 9 subreddits | Higher volume but inaccessible in bulk |
| Reproducibility | Static CSV, byte-identical across runs | Rate-limited streaming; results vary by collection window |

: Platform comparison against selection criteria. {#tbl:platform-comparison}

Twitter/X was the strongest alternative on volume and timestamp granularity but became infeasible after the 2023 API policy changes eliminated affordable bulk access for academic research. Reddit satisfies all five criteria simultaneously: publicly archived with per-second timestamps, explicit ticker conventions in financial subreddits, sufficient volume for statistical learning, and available as static Kaggle exports that guarantee byte-identical reproduction.

Specifically, this project uses the Reddit Finance Data collection on Kaggle [21], a curated static export covering nine financial subreddits. The archive spans the 2021 calendar year, the only period available in this dataset. This period coincidentally includes the January GameStop meme-stock episode, providing genuine high-magnitude surges for the model to learn from alongside months of more typical activity. While the time period was not selected for this reason, the presence of both extreme and steady-state regimes within a single year strengthens evaluation. The trade-off is that engagement fields (score, num_comments) represent final snapshot values rather than point-in-time observations, addressed by excluding them from features entirely ([@sec:feature-engineering]).

### Subreddit Selection

The Reddit Finance Data archive on Kaggle [21] was selected over alternative acquisition routes (direct Reddit API, Pushshift dumps) because it uniquely combines completeness, accessibility, and reproducibility. The direct API enforces rate limits that make historical bulk collection impractical and non-reproducible across researchers. Pushshift provides comprehensive archives but access became unreliable after mid-2023 policy changes and requires multi-terabyte processing infrastructure. The Kaggle archive offers a single downloadable, version-controlled CSV export covering nine financial subreddits for the full 2021 calendar year (~1.38M submissions), with per-second timestamps, post text, and engagement fields producing byte-identical data across runs.

From the nine available subreddits, two were selected to represent opposite extremes of posting density. Three criteria guided the choice: (a) sufficient ticker diversity for per-ticker windowing, (b) adequate post volume after ticker extraction to support temporal cross-validation, and (c) availability of selftext for sentiment computation.

| Subreddit | Raw Records | Ticker Diversity | Suitability |
|-----------|-------------|------------------|-------------|
| `WSB` | ~1,294,000 | High (multi-ticker) | Dense mainstream community; selected |
| `r/pennystocks` | ~305,000 | High (2,912 tickers; lowest missing-selftext rate) | Sparse niche community; selected |
| `r/stocks` | ~200,000 | Low (longer-form, fewer ticker mentions) | Rejected: >90% exclusion after extraction |
| `r/investing` | ~150,000 | Low (portfolio/strategy focus) | Rejected: same issue as r/stocks |
| `r/GME` | ~273,000 | Single ticker | Rejected: per-ticker design becomes trivial |

: Candidate subreddit evaluation. {#tbl:subreddit-eval}

`WSB` provides the high-density condition: 577,872 exploded record–ticker pairs with stable per-ticker statistics. `r/pennystocks` provides the low-density condition: 80,212 pairs, testing methodology viability under sparsity. The remaining subreddits were excluded because their posting norms produced extraction-stage exclusion exceeding 90% or because single-ticker focus eliminates the cross-stock dimension. Additional subreddits are acknowledged as future work ([@sec:proposed-improvements]).

- `r/pennystocks`: sparse niche community (80,212 exploded records) focused on low-capitalisation equities, testing methodology under data scarcity.
- `WSB`: high-volume mainstream forum (577,872 exploded records), testing scalability and signal isolation within high-noise environments.

This dual-dataset strategy addresses Gap 3 (domain specificity) and enables cross-dataset transfer evaluation.

| Property | r/pennystocks | WSB |
|----------|---------------|------------------|
| Raw records | 304,524 | 1,293,981 |
| Date range | 2021-01-01 to 2021-12-31 | 2021-01-01 to 2021-12-31 |
| After ticker extraction (exploded) | 80,212 | 577,872 |
| Usable records (post-exclusion) | 24,827 | 457,072 |
| Train / Test split | 21,549 / 3,278 | 388,149 / 68,923 |
| Test surges | 31 | 668 |
| Test imbalance ratio | 105:1 | 102:1 |

: Dataset characteristics. {#tbl:dataset-characteristics}

**Ethics:** All data consists of publicly posted forum submissions; analysis is aggregated at ticker level with no individual user identification. **Known limitations:** Survivorship bias (deleted posts absent), frozen engagement metrics, and results bound to 2021.

## Surge Definition (Target Variable) {#sec:surge-definition}

A fixed posting-count threshold fails because tickers have different baselines. The solution is a composite metric that normalises volume growth relative to the training distribution and combines it with sentiment change. For each record mentioning ticker *X* at time *t*, the pipeline:

1. Counts posts mentioning ticker $X$ within a backward window $[t - 24\text{h}, t)$ and a forward window $(t, t+24h]$
2. Computes volume growth: $$\Delta V = \frac{C_{\text{fwd}}}{\max(C_{\text{bwd}}, 1)} - 1$$where $C_{\text{fwd}}$ and $C_{\text{bwd}}$ are forward and backward post counts respectively.
3. Computes sentiment shift magnitude: $$\Delta S = \vert{}\bar{S}_{\text{fwd}} - s_t\vert{}$$where $\bar{S}_{\text{fwd}}$ is the mean VADER score across forward-window posts and $s_t$ is the current post's score.
4. Standardizes both using training-partition parameters exclusively: $$Z(\Delta V) = \frac{\Delta V - \mu_{\Delta V,\text{train}}}{\sigma_{\Delta V,\text{train}}}, \quad Z(\Delta S) = \frac{\Delta S - \mu_{\Delta S,\text{train}}}{\sigma_{\Delta S,\text{train}}}$$
5. Combines into a composite score and applies binary thresholding:$$\text{Composite} = w_1 \cdot Z(\Delta V) + w_2 \cdot Z(\Delta S)$$A record is labelled surge ($y = 1$) if $\text{Composite} > \tau$.

Computing $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ strictly from the training partition prevents test-set distribution information from leaking into target labels.

**Observation window:** A 24-hour window aligns with daily trading cycles. Shorter windows (6h) yield sparse per-ticker counts and unstable statistics; longer windows (72h) blur surge onset with sustained activity. [@sec:proposed-improvements] explores multi-scale alternatives.

**Sentiment weighting:** Incorporating $\Delta S$ captures scenarios where discussion grows polarised without immediate volume spikes [4, 10]. A volume-only definition ($w_2 = 0$) degrades AUC on `WSB` from 0.892 to 0.710 ([@sec:sentiment-contribution]).

| $\tau$ | Surge Count | Surge Rate | Imbalance Ratio |
|---|-------------|------------|-----------------|
| $0.5$ | 80,455 | 17.6% | 4.7:1 |
| $1.0$ | 22,384 | 4.9% | 19.4:1 |
| $\mathbf{1.5}$ | **6,602** | **1.4%** | **68.2:1** |
| $2.0$ | 2,873 | 0.6% | 158:1 |
| $2.5$ | 1,348 | 0.3% | 338:1 |

: Threshold sensitivity on WSB. {#tbl:threshold-sensitivity}

**Threshold selection:** $\tau = 1.5$ isolates true statistical anomalies (1.4% surge rate) while retaining sufficient positive instances (668 test surges on `WSB`) for reliable estimation. A secondary evaluation at $\tau = 1.0$ provides sensitivity analysis.

**Two-phase validation:** Phase 1 ($w_1 = 1.0, w_2 = 0.0$) evaluates a volume-only target; Phase 2 ($w_1 = 0.5, w_2 = 0.5$) evaluates the composite. Comparing phases isolates sentiment's empirical contribution. A full weight sweep ($w_2 \in \{0.0, 0.25, 0.50, 0.75, 1.00\}$) is reported in [@sec:sentiment-contribution].

## Feature Engineering {#sec:feature-engineering}

All eleven features satisfy a strict backward-looking constraint: each is derived exclusively from information available at or before observation timestamp $t$. Post-hoc engagement metrics (Reddit score, num_comments) are excluded because they accumulate after publication and would introduce lookahead bias.

| | Feature | Category | Definition |
|-|-------------|--------|----------------|
|1| `sentiment_score` | Content | VADER compound sentiment of the post text |
|2| `word_count` | Content | Words in selftext (0 if absent) |
|3| `title_length` | Content | Character count of title |
|4| `num_tickers_mentioned` | Content | Distinct tickers extracted from the post |
|5| `hour_of_day` | Temporal | Hour of post creation (UTC) |
|6| `day_of_week` | Temporal | Day of post creation (Monday=0) |
|7| `time_since_previous` | Activity | Seconds since previous same-ticker post |
|8| `ticker_post_rate_24h` | Activity | Same-ticker posts in preceding 24 hours |
|9| `ticker_post_acceleration` | Activity | Rate ratio: count in preceding 12h ÷ count in 12h before that |
|10| `word_count_x_hour` | Interaction | word_count × hour_of_day |
|11| `accel_x_time_since_prev` | Interaction | ticker_post_acceleration × time_since_previous |

: Feature definitions. All features use backward-looking or concurrent information only. {#tbl:feature-definitions}

Features are organised into four categories: **content** (textual characteristics and sentiment), **temporal** (cyclical market-aligned patterns), **activity** (discussion momentum drawing on popularity prediction literature [1, 5]), and **interaction** (cross-feature dynamics that yielded +1.4pp AUC lift in ablation on `r/pennystocks`).

## Model Selection

The prediction objective is a supervised binary classification task ($y \in \{0, 1\}$): whether a ticker experiences a composite surge within the subsequent 24 hours. Three classifier families spanning the complexity spectrum evaluate whether architectural sophistication improves prediction:

- **Logistic Regression (LR):** Interpretable linear baseline with Elastic Net regularisation ($L_1 + L_2$). Strong performance would indicate approximate linear separability of the surge feature space.
- **Random Forest (RF):** Bagged decision tree ensemble capturing non-linear relationships through tree splits. Fernández-Delgado et al. [20] demonstrated consistent top-tier performance across tabular benchmarks.
- **XGBoost:** Gradient-boosted decision trees where each tree corrects prior ensemble errors, with $L_1/L_2$ leaf-weight regularisation. Gradient boosting consistently achieves state-of-the-art results on structured tabular data.

**Primary metric (AUC-ROC):** With surge rates of 1–5%, accuracy is uninformative, a naive "no surge" predictor achieves 95–99%. AUC-ROC measures ranking quality across all thresholds. Precision, Recall, $F_1$, and PR-AUC are reported as secondary metrics at default and validation-optimized thresholds.

**Handling class imbalance:** SMOTE is unsuitable for temporal data because synthetic instances lack meaningful timestamps and risk local data leakage. Instead, imbalance is addressed via cost-sensitive learning: `class_weight='balanced'` for LR and RF, and `scale_pos_weight` (negative-to-positive ratio) for XGBoost.

## Temporal Validation Design

Standard $k$-fold cross-validation violates chronological ordering by permitting models to train on future observations while validating on past ones, systematically overestimating performance [19]. The evaluation pipeline employs a two-level temporal partitioning scheme:

**Level 1: Train/test split (80/20 by timestamp).** Records are sorted by observation timestamp. The earliest 80% constitute the training partition; the final 20% form the held-out test set, evaluated exactly once.

**Level 2: Expanding-window CV within training ($k=4$).** The training partition is divided into four chronological blocks producing three validation splits. Each fold trains on all preceding blocks and validates on the next, ensuring every validation instance occurs strictly after all training instances.

![Expanding-window CV. The training partition is divided into four temporal blocks, producing three validation splits. Each fold trains on all data up to a cutoff and validates on the next block, mimicking deployment where more history accumulates over time.](figures/fig3-expanding-window-cv.png){#fig:expanding-cv}

A fold count of $k = 4$ ensures sufficient positive surge instances per validation window for stable AUC estimation while maintaining adequate initial training depth. Following hyperparameter optimization, $F_1$-optimized decision thresholds are locked on validation folds and applied unchanged to the test set, ensuring uncontaminated final evaluation. Temporal non-stationarity (shifting community behaviour across 2021) is mitigated by the expanding-window design but remains a structural risk; empirical evidence is detailed in [@sec:temporal-stability].

## Evaluation Framework

The empirical evaluation addresses four questions: (1) Do models predict surges significantly better than trivial baselines? (2) Do ensemble methods yield statistically significant gains over linear baselines? (3) How confident are the metric estimates? (4) Does the learned representation generalise across communities?

| Tier | AUC-ROC | Interpretation |
|------|---------|----------------|
| Minimum | > 0.60 | Weak but above-chance discrimination |
| Target | > 0.70 | Moderate; practically useful for ranking |
| Stretch | > 0.80 | Strong; reliably separates surges from non-surges |

: Success tiers. {#tbl:success-tiers}

**Baselines:** Performance is benchmarked against a random baseline ($\text{AUC} = 0.50$) and eleven univariate Logistic Regression models (one per feature), determining whether multi-feature integration outperforms the single best feature.

**Statistical robustness:** 1,000 bootstrap resamples of the test set construct 95% confidence intervals. McNemar's test assesses pairwise significance between models, with Bonferroni-corrected $\alpha = 0.017$ ($0.05 / 3$ comparisons).

| Metric | Role |
|--------|------|
| AUC-ROC | Primary; threshold-independent ranking quality |
| Precision | Proportion of predicted surges that are real |
| Recall | Proportion of actual surges detected |
| F1-Score | Harmonic mean of precision and recall |

: Evaluation metrics. {#tbl:eval-metrics}

**Cross-dataset transfer:** Models trained on one subreddit are evaluated on the other's test set without retraining. AUC-ROC serves as the transfer metric since it is invariant to operating-point shifts. Transfer AUC > 0.60 indicates shared cross-community surge structure.

**Sensitivity analysis:** Two sweeps stress-test robustness: threshold sensitivity ($\tau \in \{0.5, 1.0, 1.5, 2.0, 2.5\}$) and sentiment weight sensitivity ($w_2 \in \{0.0, 0.25, 0.50, 0.75, 1.00\}$).

## Operational Context and Acceptance Criteria {#sec:operational-criteria}

The system is designed as a daily screening tool, not an autonomous decision-maker. Each cycle, the model ranks all active tickers by predicted surge probability and surfaces the top-$k$ for human review. This mirrors established practice in fraud detection and medical screening, where classifiers serve as prioritisation layers [2][23].

**Deriving acceptance thresholds from workflow constraints.** A typical surveillance analyst can review 20–30 flagged tickers per day. On `WSB`, where 500–800 unique tickers appear daily, the system must reduce this universe by at least an order of magnitude.

| Criterion | Requirement | Rationale |
|-----------|-------------|-------------------|
| Ranking quality (AUC-ROC) | ≥ 0.80 | True surges must appear near the top of the ranked list [2] |
| Recall at operating threshold | ≥ 0.50 | Catch at least half of genuine surges [22] |
| Precision at operating threshold | ≥ 0.10 | No more than ~9 false alarms per true positive [3] |
| Daily alert volume | 20–30 flags | Matches analyst throughput |

: Operational acceptance criteria derived from user scenarios ([@sec:problem-motivation]). {#tbl:acceptance-criteria}

**Asymmetric error costs.** False negatives (missed surges) carry higher consequential cost than false positives (unnecessary reviews). A compliance team missing a pump-and-dump campaign faces regulatory risk; investigating a benign ticker costs only analyst-hours. This asymmetry motivates recall-oriented thresholds and cost-sensitive learning [4][22].

**Scope of prediction.** A high surge probability means discussion is statistically likely to escalate within 24 hours. It does not imply price movement, manipulation, or any recommended action. The system flags candidates for investigation; humans determine causality and response [23].

**Limitations.** These criteria assume human-in-the-loop review. Fully automated action would require precision ≥ 0.80 and formal probability calibration, neither of which the current system achieves. [@sec:operational-precision] evaluates the best model against these criteria.

## Development Plan

[@tbl:timeline] outlines the main project phases, activities, and expected deliverables.

| Id | Phase | Key Activities | Deliverables |
|----|-------|----------------|--------------|
| Phase 1 | Business Understanding & Scoping | Define problem, users, research question, scope and success criteria | Project definition and requirements |
| Phase 2 | Literature Review | Review trend prediction, engagement prediction and sentiment analysis research | Literature review and research gap |
| Phase 3 | Data Understanding | Dataset investigation, exploratory analysis and quality assessment | Dataset profile and EDA results |
| Phase 4 | System Design | Design prediction pipeline, feature set, surge definition and evaluation strategy | System architecture and design specification |
| Phase 5 | Prototype Development | Implement baseline pipeline and Logistic Regression model | Feasibility prototype |
| Phase 6 | Full Pipeline Implementation | Preprocessing, feature engineering and advanced models | Complete predictive system |
| Phase 7 | Evaluation & Analysis | Model comparison, cross-validation, error analysis and feature importance analysis | Evaluation results |
| Phase 8 | Refinement | Improve models and address identified weaknesses | Refined prototype |
| Phase 9 | Final Report & Demonstration | Final documentation and demonstration video | Final report and video |

: Project timeline and deliverables. {#tbl:timeline}

The plan is derived from the CRISP-DM data-mining process model, whose stages (business understanding, data understanding, modelling, evaluation) map directly onto Phases 1–9; this grounding ensures the schedule follows an established methodology rather than an ad-hoc ordering. The phases are sequenced by dependency: scoping and the literature review (Phases 1–2) fix the research question and success criteria that the system design (Phase 4) must satisfy, and data understanding (Phase 3) constrains the surge definition and feature set before any modelling begins. A feasibility prototype (Phase 5) is scheduled ahead of full implementation specifically to de-risk the approach, validating the leakage-free labelling and a single baseline model before committing effort to the complete pipeline. The process is iterative rather than strictly linear: evaluation (Phase 7) feeds refinement (Phase 8), which loops back through implementation and re-evaluation as weaknesses such as threshold miscalibration and data sparsity are identified and addressed. This design allocates the most time to the implementation and evaluation phases, where the project's technical risk is concentrated.

![Project Timeline (Gantt Chart).](figures/fig4-gantt-chart-v0.2.png){#fig:gantt}

---

# Implementation

## Code Organisation

The pipeline is packaged as a standard Python 3.10+ library (`surge-pipeline`, built with setuptools) depending on `pandas` ($\ge 2.0$), `scikit-learn` ($\ge 1.3$), `XGBoost` ($\ge 2.0$), `vaderSentiment` ($\ge 3.3.2$), and `NumPy` ($\ge 1.24$). Exact pinned versions are recorded in `requirements.txt` and logged with each experiment run for full reproducibility. 

All source code resides under `src/`, split into a core library modules and executable CLI scripts:

```default {#lst:source-org caption="Source code organisation. The \`surge_pipeline/\` package contains one module per pipeline stage, enforcing separation of concerns. Each module has a corresponding test file. CLI entry points orchestrate multi-stage runs without embedding logic themselves."}
src/
├── surge_pipeline/              # Core library (15 modules, ~4,250 LOC)
│   ├── config.py                # Configuration dataclass + JSON I/O
│   ├── loader.py                # CSV ingestion, ticker extraction, explosion
│   ├── windowing.py             # Per-ticker 24h counts (searchsorted)
│   ├── sentiment.py             # VADER scoring with title-fallback
│   ├── labelling.py             # Temporal split, z-scores, thresholding
│   ├── normalisation.py         # Z-score parameter persistence
│   ├── features.py              # 11 backward-only features
│   ├── training.py              # Expanding-window CV + grid search
│   ├── evaluation.py            # Metrics, bootstrap CI, McNemar's
│   ├── evaluation_figures.py    # ROC curves, confusion matrices, plots
│   ├── experiment_log.py        # Append-only JSONL tracker
│   └── pipeline.py              # Orchestrator: chains all stages
├── tests/                       # 10 test modules (pytest)
├── run_labeling.py              # CLI: full labelling pipeline (stages 1–4)
├── run_training.py              # CLI: model training + evaluation (stages 5–6)
└── run_cross_validation.py      # CLI: cross-dataset transfer evaluation
```

Each pipeline stage maps  directly to one or two library modules. This modular separation ensures that changes to one stage (e.g., swapping out the sentiment backend) cannot touch another's logic, and any stage can be unit-tested in isolation.

Executable commands are exposed via entry-point CLI scripts to streamline individual stages and end-to-end runs.

| Command | Purpose |
|---------|-----------------------------|
| `surge-label` | Run the labelling pipeline (load -> window -> sentiment -> label -> threshold sweep) |
| `surge-train` | Train all three models and produce the full evaluation report |
| `surge-cross-val` | Test whether a model trained on one subreddit transfers to the other |
| `surge-figures` | Regenerate publication figures from saved evaluation artefacts |

: CLI entry points. {#tbl:cli-entry-points}

Pipeline behavior is controlled centrally via a `PipelineConfig` dataclass, which holds every tuneable parameter and can be overridden via configuration files. To guarantee determinism across runs, a fixed global seed (default 42) is systematically set across Python's native random module, NumPy, and all scikit-learn estimators.

## Data Loading and Preprocessing

The data loader module  (`loader.py`) ingest raw Reddit submission exports and transforms them into the core unit of analysis (one row per record-ticker pair, sorted chronologically) in four steps:

**Step 1: Text Cleaning**

Moderation placeholders (such as `[deleted]`, `[removed]`) and `null` are replaced with empty strings. Each row in the raw archival dataset maps to a unique Reddit submission ID, so no deduplication is needed. [@lst:text-cleaning] shows the implementation:

```python {#lst:text-cleaning caption="Text cleaning (from loader.py). Moderation-redacted content and null values are normalised to empty strings before downstream extraction, ensuring regex patterns operate on consistent input without raising exceptions on missing data."}
# Clean selftext: replace [deleted], [removed], NaN with empty string
df["selftext"] = df["selftext"].fillna("")
df["selftext"] = df["selftext"].replace({"[deleted]": "", "[removed]": ""})

# Clean title: fill NaN with empty string
df["title"] = df["title"].fillna("")
```

**Step 2: Ticker Extraction** 

Tickers are extracted from submission text using two prioritized regex patterns: (1) Dollar-sign tickers (e.g., `\$([A-Za-z]{1,5})` for `$AMC`, `$TSLA`), which carry the highest confidence since the dollar prefix is an explicit marker in financial communities; (2) Standalone 2–5 character uppercase words `(\b[A-Z]{2,5}\b)`, which cast a broader net. Extracted matches are filtered against a curated stopword lexicon of 297 terms across eight categories (common English, Reddit slang, finance abbreviations, etc.), developed through iterative error analysis on early runs. A stopword filter was selected over a closed universe of exchange-listed tickers because penny stocks and emerging tickers rotate frequently; the worst-case failure mode is a false-positive adding minor noise to one record, whereas an outdated master ticker list would silently drop posts about unknown stocks. [@lst:ticker-extraction] shows the extraction logic:

```python {#lst:ticker-extraction caption="Ticker extraction with dual regex priority cascade and stopword filtering (from loader.py). The dollar-sign pattern captures explicit financial references with high precision; the uppercase pattern broadens recall at the cost of precision, mitigated by a 297-term stopword lexicon spanning eight categories."}
# Extract tickers from combined title + selftext
combined_text = f"{title} {selftext}"

# 1. Dollar-sign pattern (highest priority always included)
dollar_matches = _DOLLAR_SIGN_PATTERN.findall(combined_text)
for match in dollar_matches:
    ticker = match.upper()
    if ticker not in TICKER_STOPWORDS:
        tickers.add(ticker)

# 2. Uppercase word pattern (filtered against stopwords)
word_matches = _UPPERCASE_WORD_PATTERN.findall(combined_text)
for match in word_matches:
    ticker = match.upper()
    if ticker not in TICKER_STOPWORDS:
        tickers.add(ticker)
```

**Step 3: Timestamp Normalisation**

Raw timestamps (Unix epoch integers or ISO datetime strings) are normalised to standard `datetime64[ns, UTC]` and sorted. This chronological ordering is a hard precondition for the binary-search windowing that follows. [@lst:timestamp-norm] shows the implementation:

```python {#lst:timestamp-norm caption="Timestamp normalisation and chronological sorting (from loader.py). The loader accepts two timestamp formats, Unix epoch integers (common in Reddit API exports) and ISO datetime strings (common in Kaggle archives), unifying both into timezone-aware datetime64[ns, UTC]. The sort establishes the chronological invariant required by all downstream stages."}
# Parse timestamps: handle both epoch-second and datetime-string formats
if "created_utc" in df.columns:
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
elif "created" in df.columns:
    df["created_utc"] = pd.to_datetime(df["created"], utc=True)
    df = df.drop(columns=["created"])
else:
    raise ValueError(
        "Dataset must contain either 'created_utc' (epoch) or "
        "'created' (datetime string) column."
    )

df = df.sort_values("created_utc").reset_index(drop=True)
```

**Step 4: Ticker Explosion & Filtering** 

Multi-ticker posts (e.g., "comparing `$AMC` vs `$GME`") are exploded into separate record–ticker rows using pandas.explode(). Records that yield zero valid tickers after filtering are dropped from the pipeline. [@lst:ticker-explosion] shows the implementation:

```python {#lst:ticker-explosion caption="Ticker explosion and filtering (from loader.py). The comma-separated ticker string is split into a list and exploded via pandas.explode(), converting one multi-ticker post into multiple rows, one per (record, ticker) pair. This transforms the unit of analysis from post to post-about-a-specific-ticker, enabling per-ticker temporal windowing in subsequent stages."}
# Exclude records with missing/empty tickers
df["tickers"] = df["tickers"].astype(str).str.strip()
df["tickers"] = df["tickers"].replace({"": np.nan, "nan": np.nan, "None": np.nan})
mask_has_tickers = df["tickers"].notna()
df = df[mask_has_tickers].reset_index(drop=True)

# Explode multi-ticker records: one row per (record_id, ticker) pair
df["tickers"] = df["tickers"].str.split(",")
df = df.explode("tickers", ignore_index=True)

# Clean individual ticker values
df["tickers"] = df["tickers"].str.strip().str.upper()
df = df[df["tickers"].str.len() > 0].reset_index(drop=True)
df = df.rename(columns={"tickers": "ticker"})
```

| Step | r/pennystocks | WSB |
|------|---------------|------------------|
| Raw records loaded | 304,524 | 1,293,981 |
| Excluded (no tickers found) | 224,312 (73.7%) | 716,109 (55.3%) |
| After explosion (record-ticker pairs) | 80,212 | 577,872 |

: Loader-stage attrition. {#tbl:loader-attrition}

## Feature Engineering (Implementation)

Eleven features feed the classifiers. The governing constraint is that every feature must be computable from data *at or before* the current record's timestamp. Nothing may peek into the future.

| # | Feature | Category | Computation Method |
|---|---------|------|---------------------------|
| 1 | `ticker_post` \ `_rate_24h` | Activity | Reuses `backward_count` produced by `windowing.compute_windowed_counts()`. For each record mentioning ticker $X$ at time $t$, counts all other posts mentioning $X$ with timestamps in the half-open interval $(t - 24\text{h},\; t)$. Counting is performed via `np.searchsorted` on the chronologically sorted per-ticker timestamp array, yielding $O(n \log n)$ complexity per ticker group. The self-post is excluded by using `side='left'` at the right boundary. |
| 2 | `time_since_` \ `previous_post` | Activity | Computed by `features._compute_time_` \ `since_previous()`. For each record at time $t$, identifies the immediately preceding post mentioning the same ticker by iterating through the chronologically sorted per-ticker group (`groupby('ticker')`). Computes elapsed hours: $(t - t_{\text{prev}}) / 3600$. Returns $-1$ for the first occurrence of a ticker (no prior history). Uses epoch-second conversion for numeric subtraction. |
| 3 | `ticker_post_` \ `acceleration` | Activity | Computed by `features._compute_ticker_` \ `post_acceleration()`. Splits the backward 24 h window into two 12 h halves: recent $(t - 12\text{h},\; t]$ and older $(t - 24\text{h},\; t - 12\text{h}]$. Counts posts in each half using `np.searchsorted` (4 boundary lookups per record, $O(n \log n)$ per ticker group). Computes ratio: `count_recent / max(count_older, 1)`. Values $> 1.0$ indicate accelerating discussion; values $< 1.0$ indicate deceleration. The `max(..., 1)` denominator guard prevents division by zero when no posts exist in the older half. Boundary semantics: `side='right'` for inclusive-left boundaries, `side='left'` for exclusive-right. |
| 4 | `sentiment_` \ `score` | Content | Reuses `sentiment_polarity` produced by `sentiment.compute_sentiment()` via `_compute_polarity_vader()`. VADER's `polarity_scores()` is applied to the post's selftext; if selftext is empty or absent, the title is used as fallback. The compound score ranges from $-1$ (most negative) to $+1$ (most positive). Computed strictly from the record's own text at creation time, no forward window information. |
| 5 | `word_count` | Content | Computed inline in `features.compute_features()`. Concatenates `title + " " + selftext`, splits on whitespace (`str.split().str.len()`), counts resulting tokens. Empty/null selftext is replaced with empty string before concatenation. Measures post effort/depth as a proxy for informational content. |
| 6 | `title_length` | Content | Computed inline in `features.compute_features()`. Splits title on whitespace (`str.split().str.len()`) and counts tokens. Captures headline effort independently of body length. Null titles treated as empty string (0 tokens). |
| 7 | `num_tickers_` \ `mentioned` | Content | Computed by `features._compute_num_` \ `tickers_mentioned()`. Groups the exploded DataFrame by original post `id` and counts distinct ticker values per group using `groupby('id')['ticker']` \ `.transform('nunique')`. A post mentioning 3 tickers will have value 3 in all its exploded rows. Captures whether a post is ticker-specific or broad market commentary. |
| 8 | `hour_of_` \ `day` | Temporal | Computed inline in `features.compute_` \ `features()`. Extracts UTC hour (0–23) from `created_utc` via `pd.to_datetime(..., utc=True)` \ `.dt.hour`. Captures intraday cyclicality aligned with US market hours (pre-market activity typically spikes 13:00–14:00 UTC). |
| 9 | `day_of_week` | Temporal | Computed inline in `features.compute_` \ `features()`. Extracts day-of-week index (Monday=0, Sunday=6) from `created_utc` via `.dt.dayofweek`. Captures weekly periodicity: weekday posts cluster near market sessions; weekend posts are predominantly speculative. |
| 10 | `word_count_` \ `x_hour` | Interaction | Computed inline in `features.compute_` \ `features()` via element-wise multiplication: `word_count × hour_of_` \ `day`. Encodes the hypothesis that long analytical posts at peak trading hours (high word count × high hour value in UTC afternoon) are stronger surge precursors than either signal alone. Gives tree models an explicit split surface without requiring deep multi-level branching. |
| 11 | `accel_x_time_` \ `since_prev` | Interaction | Computed inline in `features.compute_` \ `features()`. Multiplicative interaction: `ticker_post_acceleration × time_` \ `since_previous`. Captures the pattern of sudden acceleration after prolonged silence, a ticker dormant for many hours that suddenly attracts rapid posting. For first-occurrence records (`time_since_previous = -1`), the value is clamped to 0 via `np.where(tsp < 0, 0, tsp)` to avoid spurious negative products. Ablation (Experiment B2) confirmed +1.4 pp AUC lift from including both interaction terms. |

: Feature engineering detail. Each row specifies what the feature captures and how it is computed, including the responsible function. {#tbl:feature-detail}

The most algorithmically involved feature is `ticker_post_acceleration`. It splits the backward 24-hour window into two 12-hour halves (a recent half covering $(t - 12\text{h}, t]$ and an older half covering $(t - 24\text{h}, t - 12\text{h}]$ and then computes the ratio: $$\text{ticker\_post\_acceleration} = \frac{\text{count}_{\text{recent}}}{\max(\text{count}_{\text{older}}, 1)}$$

Values above 1.0 indicate accelerating discussion volume. By leveraging pre-sorted timestamp arrays per ticker group, the implementation uses NumPy's searchsorted to perform interval counting in $O(n \log n)$ time:

```python {#lst:acceleration caption="Ticker post acceleration via split-window binary search (from features.py). The backward 24-hour window is bisected into recent and older halves. Four searchsorted calls per ticker group compute counts in each half; the ratio detects whether posting is accelerating (>1.0) or decelerating (<1.0). The max(..., 1) guard prevents division by zero when the older half is empty."}
# Count posts in recent half (t-12h, t] excluding self
recent_left = np.searchsorted(times, times - 12H_SECONDS, side="right")
recent_right = np.searchsorted(times, times, side="left")
count_recent = recent_right - recent_left

# Count posts in older half (t-24h, t-12h]
older_left = np.searchsorted(times, times - 24H_SECONDS, side="right")
older_right = np.searchsorted(times, times - 12H_SECONDS, side="right")
count_older = older_right - older_left

acceleration = count_recent / np.maximum(count_older, 1)
```

The two interaction terms (`word_count_x_hour` and `accel_x_time_since_prev`) provide models with an explicit signal for combined dynamic, such as a sudden surge in post volume following a period of silence, without requiring multi-level decision tree splits to discover the interaction. An ablation study confirmed a consistent +1.4pp AUC lift from including these terms.

## Surge Labelling

The labelling module converts raw windowing and sentiment outputs into binary surge/no-surge labels while enforcing strict temporal isolation.

**Temporal Split** 

Records are partitioned chronologically at the 80th percentile of timestamps (sorted by time without shuffling).  All records occurring at or before the cutpoint are assigned to the training set; the rest to test.

**Z-score Normalisation** 

The mean ($\mu$) and standard deviation ($\sigma$) for both the volume growth ratio and sentiment shift are computed exclusively from training set records. These frozen parameters are then applied to standardise both partitions: $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

Measuring the test set against training-derived distributions prevents future data leakage into historical baselines.

**Composite Metric and Thresholding.** 

The surge composite score combines the standardized metrics:

$$\text{composite} = (w_{\text{volume}} \cdot z_{\text{volume}}) + (w_{\text{sentiment}} \cdot z_{\text{sentiment}})$$

A record is labelled as a surge ($y = 1$) if its composite score exceeds threshold $\tau$, and non-surge ($y = 0$) otherwise. 

```python {#lst:surge-labelling caption="Surge labelling pipeline (from labelling.py). Steps 1-2 establish the leakage-prevention mechanism: mean and standard deviation are estimated exclusively from the training partition, then applied unchanged to all records including the test set. Steps 3-4 combine the normalised volume growth and sentiment shift into a weighted composite and threshold it into a binary target. Because test-set records are normalised against a distribution they never contributed to, the target labels encode no future information. Population standard deviation (ddof=0) is used because the training partition constitutes the entire reference population for normalisation, not a sample drawn from a larger one."}
# Step 1: Temporal split at 80th percentile timestamp
epoch_seconds = to_epoch_seconds(df["created_utc"])
split_ts = np.percentile(epoch_seconds, ratio * 100)
partitions = np.where(epoch_seconds <= split_ts, "train", "test")

# Step 2: Compute μ/σ from training partition ONLY
train_volume = df.loc[train_mask, "posting_volume_growth"].values.astype(np.float64)
train_sentiment = np.abs(
    df.loc[train_mask, "sentiment_change"].values.astype(np.float64)
)
mu_vol = float(np.mean(train_volume))
sigma_vol = float(np.std(train_volume, ddof=0))   # population std
mu_sent = float(np.mean(train_sentiment))
sigma_sent = float(np.std(train_sentiment, ddof=0))

# Step 3: Z-score normalise ALL records using frozen training stats
z_volume = (all_volume - mu_vol) / sigma_vol
z_sentiment = (all_sentiment - mu_sent) / sigma_sent

# Step 4: Composite metric and binary labelling
composite = (w1 * z_volume) + (w2 * z_sentiment)
surge_label = np.where(composite > tau, 1.0, 0.0)
```


Records are excluded as unlabellable under two conditions:
1. The 24-hour forward window contains fewer than two same-ticker posts (preventing division-by-zero or meaningless growth ratios).
2. The record's forward window extends beyond the dataset's final timestamp boundary.

These filtering rules account for the dataset attrition from 577,872 exploded records to 457,072 usable records on `WSB` ([@tbl:dataset-characteristics]). For sensitivity analysis, sweep_thresholds() evaluates $\tau \in \{0.5, 1.0, 1.5, 2.0, 2.5\}$ in a single vectorised pass. Setting $w_{\text{sentiment}} = 0$ yields the volume-only variant evaluated in the Phase 1 ablation.

## Model Training

**Expanding-window Cross-validation** 

The training partition is divided into four chronological blocks to construct three expanding validation splits:
1. Split 1: Train on Block 1; validate on Block 2
2. Split 2: Train on Blocks 1–2; validate on Block 3
3. Split 3: Train on Blocks 1–3; validate on Block 4

An explicit temporal check enforces $\max(t_{\text{train}}) < \min(t_{\text{val}})$ in every split to prevent lookahead bias. [@lst:expanding-splits] shows the split construction and temporal validation framework:

```python {#lst:expanding-splits caption="Expanding-window split construction and temporal verification (from training.py). The expanding window concatenates all preceding folds as training data, validating on the immediately subsequent fold. The hard assertion max_train > min_val triggers a ValueError if any split violates chronological ordering, making lookahead leakage a crash rather than a silent corruption."}
def get_expanding_window_splits(folds):
    """Convert sequential folds into expanding-window train/val splits."""
    splits = []
    for i in range(1, len(folds)):
        train_idx = np.concatenate(folds[:i])
        val_idx = folds[i]
        splits.append((train_idx, val_idx))
    return splits

def _verify_temporal_ordering(df, folds, splits):
    """Verify that all splits respect chronological ordering."""
    epoch = to_epoch_seconds(df["created_utc"])
    for i, (train_idx, val_idx) in enumerate(splits):
        max_train = epoch[train_idx].max()
        min_val = epoch[val_idx].min()
        if max_train > min_val:
            raise ValueError(
                f"Temporal ordering violated in split {i}: "
                f"max(train_ts)={max_train} > min(val_ts)={min_val}"
            )
```

Hyperparameters are tuned via grid search across each model class (see [@tbl:hyperparams] for complete search spaces).

| Model | Parameters Searched | Grid Size |
|-------------|--------------------|--------| 
| Logistic Regression | $C \in \{0.01, 0.1, 1, 10, 100\}$, $\text{l1\_ratio} \in \{0, 1\}$ | 10 |
| Random Forest | $\text{n\_estimators} \in \{50, 100, 200\}, \text{max\_depth} \in \{3, 5, 10, \text{None}\}, \text{min\_samples\_leaf} \in \{1, 2, 5\}$ | 36 |
| XGBoost | $\text{n\_estimators} \in \{50, 100, 200\}$, $\text{max\_depth} \in \{3, 5, 7\}$, $\text{learning\_rate} \in \{0.01, 0.1, 0.3\}$, $\text{scale\_pos\_weight} \in \{1, \text{ratio}/2$, $\text{ratio}\}$ | $\le50$ |

: Hyperparameter search spaces. {#tbl:hyperparams}

To eliminate validation leakage, a `StandardScaler` is fitted exclusively on the training fold of each split before transforming the validation fold. [@lst:grid-search] shows the inner training loop:

```python {#lst:grid-search caption="Grid search with per-fold scaler isolation (from training.py). Each fold fits a fresh StandardScaler on training indices only, then transforms the validation fold using those frozen statistics. This prevents mean/variance leakage across the temporal boundary. The best configuration is retrained on the entire training partition before test-set evaluation, maximising the data available to the final model."}
for params in param_grid:
    fold_aucs = []
    for train_idx, val_idx in splits:
        # Fit scaler on training fold ONLY, prevents validation leakage
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X_train_full[train_idx])
        X_fold_val = scaler.transform(X_train_full[val_idx])
        y_fold_train = y_train_full[train_idx]
        y_fold_val = y_train_full[val_idx]

        model = make_model_fn(params, random_seed)
        model.fit(X_fold_train, y_fold_train)

        y_prob = model.predict_proba(X_fold_val)[:, 1]
        auc = roc_auc_score(y_fold_val, y_prob)
        fold_aucs.append(auc)

    mean_auc = np.mean(fold_aucs)
    if mean_auc > best_mean_auc:
        best_mean_auc = mean_auc
        best_params = params

# Retrain winner on full training partition
final_scaler = StandardScaler()
X_train_scaled = final_scaler.fit_transform(X_train_full)
final_model = make_model_fn(best_params, random_seed)
final_model.fit(X_train_scaled, y_train_full)
```

**Model Instantiation and Class Imbalance Handling**

Each model family is constructed via a factory function that injects the class-imbalance strategy directly into the loss function. [@lst:model-factory] shows the three model constructors:

```python {#lst:model-factory caption="Model factory functions (from training.py). All three models handle class imbalance through cost-sensitive learning rather than synthetic oversampling. Logistic Regression and Random Forest use class_weight balanced (sklearn automatically computes inverse frequency weights). XGBoost uses scale_pos_weight, grid-searched over {1, ratio/2, ratio} where ratio = n_negative / n_positive (typically 19:1 to 105:1 in this dataset). This avoids SMOTE incompatibility with temporal data, where synthetic records lack meaningful timestamps."}
def _make_lr(params, random_seed):
    return LogisticRegression(
        C=params["C"],
        l1_ratio=params["l1_ratio"],
        solver="saga",              # supports Elastic Net (L1 + L2)
        class_weight="balanced",    # inversely weight classes by frequency
        random_state=random_seed,
        max_iter=5000,
    )

def _make_rf(params, random_seed):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight="balanced",    # per-tree class reweighting
        random_state=random_seed,
        n_jobs=-1,
    )

def _make_xgb(params, random_seed):
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        scale_pos_weight=params["scale_pos_weight"],  # neg/pos ratio
        random_state=random_seed,
        eval_metric="logloss",
        verbosity=0,
    )
```

The base ratio $r$ for `scale_pos_weight` is computed dynamically from the training partition's actual class distribution:$$r = \frac{N_{\text{negative}}}{N_{\text{positive}}}$$

```python {#lst:imbalance-ratio caption="Dynamic imbalance ratio computation (from training.py). The negative-to-positive ratio is calculated from the actual training partition class distribution, then used to construct a three-level grid for XGBoost scale_pos_weight: no reweighting (1.0), moderate (ratio/2), and full (ratio). This data-driven approach adapts automatically to different surge thresholds and datasets without manual tuning."}
n_positive = int(np.sum(y_train_full == 1))
n_negative = int(np.sum(y_train_full == 0))
imbalance_ratio = float(n_negative) / max(n_positive, 1)

# Grid searches over: no reweighting, moderate, and full reweighting
weight_values = sorted(set([1.0, imbalance_ratio / 2, imbalance_ratio]))
```

The hyperparameter configuration yielding the highest mean validation Area Under the ROC Curve (AUC) across all three splits is selected as the winning model. This optimal configuration is then retrained on the entire 80% training partition, using a freshly fitted `StandardScaler`, prior to generating final predictions on the held-out test set.

## Implementation Decisions Driven by Empirical Findings

Iterative development uncovered several dataset and pipeline edge cases, driving key architectural decisions:

**Timestamp Unit Mismatch & Temporal Integrity** 

An early timestamp conversion error during temporal splitting caused an unintended 89% record exclusion, leaving only 7 positive surge instances in the test set. Beyond fixing the unit bug, this failure mode prompted the inclusion of explicit automated assertions, such as enforcing $\max(t_{\text{train}}) < \min(t_{\text{val}})$, directly inside the expanding-window training pipeline to guarantee data integrity.

**Data Sparsity and Dataset Scaling** 

Initial experiments on `r/pennystocks` (~80,000 records) produced as few as 7 positive test examples at higher threshold settings ($\tau \ge 2.0$), rendering AUC estimates highly sensitive to noise. Scaling up data ingestion to `WSB` (577,872 records produced 668 test surges at $\tau = 1.5$) provided stable metric estimation and enabled robust cross-dataset transfer experiments.

**Class Imbalance and Decision Boundary Calibration** 

At $\tau = 1.5$, extreme class imbalance (1.44% surge rate; 102:1 ratio) led XGBoost to achieve a high AUC of 0.888 while predicting zero positive surges at the standard 0.5 decision threshold. This was identified as a probability calibration issue rather than a structural model failure. Lowering $\tau = 1.0$ (~5% surge rate) and adding XGBoost's scale_pos_weight parameter to the hyperparameter search grid successfully restored probability alignment and recall.

## Implementation Status

All six core pipeline stages are fully implemented and execute end-to-end to generate reproducible artifacts. [@tbl:impl-status] summarizes the implementation status and outputs for each stage.

| Stage | Status | Key Output |
|-------|--------|------------|
| 1. Data Loading | Complete | Exploded DataFrame (~80,000 / 577,872 records) |
| 2. Temporal Windowing | Complete | Forward/backward counts per ticker |
| 3. Sentiment | Complete | VADER polarity + forward-window means |
| 4. Target Labelling | Complete | Binary surge labels + threshold sweep |
| 5. Feature Engineering | Complete | 11-feature matrix |
| 6. Training & Evaluation | Complete | 3 trained models + full evaluation JSON |

: Pipeline implementation stages, status, and corresponding primary outputs. {#tbl:impl-status}

Both the `r/pennystocks` and `WSB` datasets process completely through the pipeline with deterministic results. Execution runtime (from target labelling through final evaluation) is approximately 8 minutes for `r/pennystocks` and 19 minutes for `WSB` on a standard laptop CPU, with VADER sentiment computation accounting for the majority of compute time.

Determinism was verified empirically: running configuration A1 (seed 42) on July 13 and July 19 produced identical AUC values (0.753) and byte-identical execution logs. Results are robust to seed choice across five seeds (42, 123, 456, 789, 2024) on `r/pennystocks`, with AUC scores spanning 0.734 to 0.753 (a 0.019 margin). All 30+ experimental runs are fully trackable via logged configuration JSONs, Git commit SHAs, and timestamped output paths.

Advanced pipeline features, including cross-dataset transfer evaluation, 1,000-sample bootstrap confidence intervals, and McNemar's pairwise significance tests, are fully operational. 

## Testing Strategy

Pipeline stability, software health, and correctness claims are maintained through a 10-module `pytest` suite, strict static type checking (`mypy`), and automated linting (`ruff`), all passing with zero errors. The test suite mirrors the library's module structure, using analytically hand-computed edge cases to verify mathematical and temporal invariants across each pipeline stage.

| Test Module | Pipeline Stage | Key Invariants Tested |
|-------------|---------------|----------------------|
| `test_loader.py` | Data Loading | Ticker regex extracts known patterns; stopword filter blocks false positives; explosion produces correct row count |
| `test_windowing.py` | Temporal Windowing | Forward/backward counts match hand-computed values; boundary inclusion/exclusion semantics; per-ticker independence |
| `test_sentiment.py` | Sentiment | VADER scores match reference; title-fallback activates when selftext is empty |
| `test_labelling.py` | Target Labelling | Z-scores use training-only μ/σ; composite formula correctness; threshold labelling at known values |
| `test_features.py` | Feature Engineering | No feature accesses future data; `score`/`num_comments` excluded; `time_since_previous` is backward-only |
| `test_training.py` | Model Training | Temporal folds are chronologically ordered; expanding-window splits respect `max(train) < min(val)`; grid sizes $\le50$ |
| `test_evaluation_significance.py` | Evaluation | McNemar's test produces correct $\chi^2$ for known contingency tables; bootstrap CIs have expected coverage |
| `test_config.py` | Configuration | JSON serialisation round-trip preserves all parameters |
| `test_pipeline_integration.py` | End-to-end | Full pipeline produces identical output across two runs (determinism) |
| `test_evaluation_figures.py` | Figures | Figure generation completes without error on synthetic data |

: Test modules and the invariants they verify. {#tbl:test-modules}

**Temporal Leakage Prevention**

The most critical test module (`test_labelling.py`) verifies that normalization statistics do not leak future information into historical records. The test constructs synthetic data with distinct training ($\mu = 5.0, \sigma = 5.0$) and test ($\mu = 17.5, \sigma = 2.5$) distributions, then asserts that test-set $z$-scores are derived exclusively using the training parameters:

```python {#lst:leakage-test caption="Leakage-prevention test (from test_labelling.py). The test constructs data with known training and test distributions, then asserts that test-set z-scores are computed using training parameters (mean=5, sd=5) rather than test-set parameters (mean=17.5, sd=2.5). A negative assertion confirms the wrong computation does not occur."}
class TestZScoreNormalisation:
    def test_test_set_uses_training_stats_not_own(self, base_timestamp):
        """Critical: test-set records are normalised with TRAINING statistics."""
        # Training [0..7]: values = [0,0,0,0,10,10,10,10] -> μ=5, σ=5
        # Test [8..9]: values = [15, 20]
        volumes = [0,0,0,0, 10,10,10,10, 15, 20]
        ...
        result = apply_labelling(df, config)

        # Test z-scores MUST use training stats: z(15)=(15-5)/5=2.0
        assert result.df["z_volume"].iloc[11] == approx(2.0)
        assert result.df["z_volume"].iloc[15] == approx(3.0)

        # If they wrongly used test-only stats (μ=17.5, σ=2.5):
        # z(15) would be -1.0, verify this is NOT the case
        assert result.df["z_volume"].iloc[11] != approx(-1.0)
```

**Feature Time-Boundary Contracts**

A second critical test suite (`test_features.py`) enforces strict backward-only feature calculation. It programmatically validates that:
- Inter-arrival features (e.g., time_since_previous) depend strictly on preceding timestamps $t' \le t$.
- Post-hoc engagement signals (e.g., upvotes or comments accumulated after publication time $t$) are structurally excluded from the feature design matrix.

```python {#lst:feature-contract caption="Backward-only feature contract tests (from test_features.py). The first test verifies that time_since_previous computes inter-arrival time using only preceding records. The second test asserts that post-hoc engagement metrics (which accumulate after publication) are structurally excluded from the feature set."}
class TestNoFutureLeakage:
    def test_time_since_previous_is_backward_only(self, base_timestamp):
        """time_since_previous should only look at records before t."""
        # AAPL at t=0h, t=6h, t=12h
        result = compute_features(df)

        # Record 0: first occurrence ? -1 (no prior history)
        assert result["time_since_previous"].iloc[0] == -1.0
        # Record 1: 6 hours since record 0 (looks backward only)
        assert result["time_since_previous"].iloc[5] == approx(6.0)
        # Record 2: 6 hours since record 1 (not 12h since record 0)
        assert result["time_since_previous"].iloc[6] == approx(6.0)

    def test_features_exclude_score_and_num_comments(self, labelled_df):
        """Post-hoc engagement metrics must NOT appear in features."""
        assert "score" not in FEATURE_COLUMNS
        assert "num_comments" not in FEATURE_COLUMNS
```

These unit tests execute on synthetic datasets and run automatically prior to every experiment. They provide continuous assurance that refactoring or parameter changes cannot silently introduce temporal leakage.

---

# Evaluation

## Evaluation Against Project Objectives {#sec:eval-objectives}

### Objective 1: Predict Volume and Sentiment Surges

Predictability depends directly on data density. On `WSB` (68,923 test records, 0.97% surge rate), both tree-based models cleared the stretch tier: XGBoost 0.892 [95% CI: 0.881–0.902], Random Forest 0.880 [0.869–0.890]. Both exceed the best single feature (`word_count` alone: 0.805). On sparser `r/pennystocks` (3,278 test records, 0.95% surge rate), Random Forest achieved 0.753 [0.673–0.824], meeting target; the best single feature (`hour_of_day`) manages only 0.591, confirming multi-feature combination is essential. The binding constraint is data density, not methodology.

### Objective 2: Compare Multiple ML Approaches

On **WSB**: XGBoost (0.892) > RF (0.880) > LR (0.707). All pairwise differences statistically significant (McNemar's, $p < 0.001$ after Bonferroni correction). On **pennystocks**: RF (0.753) > XGBoost (0.734) > LR (0.680), the performance inversion where bagging outperforms boosting under scarcity is analysed in [@sec:complexity-density]. Architectural complexity helps when data is abundant but offers no guarantee under scarcity.

### Objective 3: Demonstrate Temporal Validity

The held-out 20% (final months of 2021, never seen during training) produced AUC 0.892 on `WSB` and 0.753 on pennystocks, genuine unseen-future performance. The temporal protocol reveals how standard validation overstates results: RF's validation-fold F1 was 0.911; on the test set it dropped to 0.145. This gap is the methodology working as intended, without strict temporal holdout, the inflated figure would have been reported. Leakage was prevented at every stage: z-scores frozen from training statistics, features backward-looking only, chronological ordering enforced programmatically.

## Results

All performance metrics are reported on the chronologically held-out test partition (the final 20% of data), which was isolated from training and threshold selection.

### Model Performance

On `WSB`, XGBoost achieved AUC-ROC 0.892 [0.881–0.902] and Random Forest 0.880 [0.869–0.890], both clearing the stretch tier. On `r/pennystocks`, Random Forest led at 0.753 [0.673–0.824], meeting target.

| Dataset | Model | AUC-ROC [95% CI] | Precision | Recall | F1 | Tier |
|---------|-------|-------------------|-----------|--------|-----|------|
| WSB (68,923 records, 668 surges, 0.97% rate) | LR | 0.707 [0.684–0.729] | 0.013 | 0.801 | 0.026 | Target |
| | RF | 0.880 [0.869–0.890] | 0.095 | 0.311 | 0.145 | Stretch |
| | XGB | 0.892 [0.881–0.902] | 0.043 | 0.819 | 0.081 | Stretch |
| pennystocks (3,278 records, 31 surges, 0.95% rate) | LR | 0.680 [0.588–0.778] | 0.013 | 0.645 | 0.025 | Minimum |
| | RF | 0.753 [0.673–0.824] | 0.068 | 0.194 | 0.101 | Target |
| | XGB | 0.734 [0.641–0.821] | 0.000 | 0.000 | 0.000 | Target |

: Test-set performance at default classification threshold (0.5). 95% bootstrap CIs from 1,000 resamples. {#tbl:perf-default}

AUC scores are strong but precision is near zero everywhere at the default 0.5 threshold. With sub-1% surge rates, probability outputs cluster far below 0.5, models rank surges correctly but the threshold is too conservative. This is a calibration problem, not a discrimination failure. On pennystocks, RF and XGB confidence intervals overlap substantially; McNemar's test ([@sec:statistical-validation]) nonetheless confirms significant prediction differences.

Decision thresholds were optimized on the last validation fold via F1-maximization:

| Dataset | Model | Tuned Threshold | Precision | Recall | F1 | F1 $\Delta$ |
|---------|-------|-----------------|-----------|--------|-----|------|
| WSB | LR | 0.81 | 0.058 | 0.280 | 0.097 | +0.071 |
| | RF | 0.88 | 0.180 | 0.051 | 0.079 |-0.066 |
| | XGB | 0.85 | 0.217 | 0.235 | 0.226 | +0.145 |
| pennystocks | LR | 0.67 | 0.085 | 0.194 | 0.118 | +0.093 |
| | RF | 0.79 | 0.200 | 0.097 | 0.130 | +0.030 |
| | XGB | 0.16 | 0.114 | 0.129 | 0.121 | +0.121 |

: Metrics at validation-tuned thresholds. {#tbl:perf-tuned}

After tuning, XGBoost on `WSB` achieves the best $F_1 = 0.226$ at threshold 0.85. Notably, XGBoost on pennystocks requires threshold 0.16 to produce any positive predictions, while RF on `WSB` selects 0.88 that sacrifices recall for precision (net $\Delta F_1 = -0.066$).

| Dataset | Model | Threshold | TP | FP | FN | TN |
|---------|-------|-----------|-----|------|------|-------|
| WSB | XGBoost | 0.85 | 157 | 565 | 511 | 67,690 |
| pennystocks | Random Forest | 0.79 | 3 | 12 | 28 | 3,235 |

: Confusion matrix for the best model at tuned threshold. {#tbl:confusion}

At the best operating point on WSB, XGBoost identifies 157 of 668 surges with 565 false alarms (~1 true positive per 4.6 flags), manageable for human review but unsuitable for automation.

![Confusion matrix for XGBoost at tuned threshold (0.85) on the WSB held-out test set. The model correctly identifies 157 surges (TP) while generating 565 false alarms (FP), with 511 missed surges (FN). The extreme class imbalance (67,690 TN) visually confirms why precision remains low despite strong ranking ability.](figures/10_confusion_matrix_xgboost.png){#fig:confusion-matrix}

![Combined ROC curves for Logistic Regression, Random Forest, and XGBoost on the WSB held-out test set. The diagonal represents a random classifier (AUC = 0.5).](figures/fig9-roc_curves_combined.png){#fig:roc-curves}

### Statistical Validation {#sec:statistical-validation}

All pairwise comparisons show statistically significant differences ($p < 0.001$). 

| Dataset | Model Pair | $\chi^2$ | p-value | Significant? |
|---------|------------|-----|---------|--------------|
| WSB | LR vs RF | 36,664 | < 0.001 | Yes |
| WSB | LR vs XGB | 24,246 | < 0.001 | Yes |
| WSB | RF vs XGB | 9,208 | < 0.001 | Yes |
| pennystocks | LR vs RF | 1,406 | < 0.001 | Yes |
| pennystocks | LR vs XGB | 1,486 | < 0.001 | Yes |
| pennystocks | RF vs XGB | 66 | < 0.001 | Yes |

: McNemar's pairwise significance tests (default 0.5 threshold, Bonferroni-adjusted α = 0.017). {#tbl:mcnemar}

The large $\chi^2$ values on the `WSB` test partition reflect both the substantial sample size ($N = 68,923$) and distinct error profiles across models, such as XGBoost predicting strictly negative instances at the 0.5 threshold while Random Forest makes selective positive predictions.

To evaluate the utility of combining multiple signals, model performance was benchmarked against the single strongest predictive feature ([@sec:operational-criteria]). On `WSB`, the top single-feature heuristic achieves an AUC of 0.805; combining features in the full models yields a 0.087 gain in AUC, demonstrating that multi-feature integration successfully captures complex signal interactions.

| Dataset | Random Baseline | Best Single Feature | Best Model | $\Delta$ over Single Feature |
|---------|-----------------|---------------------|------------|----------------------|
| WSB | 0.500 | 0.805 (word_count) | 0.892 (XGB) | +0.087 |
| pennystocks | 0.500 | 0.591 (hour_of_day) | 0.753 (RF) | +0.162 |

: Multi-feature models vs. baselines (AUC-ROC). {#tbl:baselines}

The benefit of multi-feature modeling is even more pronounced on r/pennystocks, where the strongest individual feature achieves an AUC of just 0.591. Here, multi-feature models improve performance by +0.162 AUC, turning an otherwise weak signal into a viable predictive

### Cross-Dataset Transfer

| Direction | LR | RF | XGBoost |
|-----------|------|------|---------|
| WSB-trained → pennystocks test | 0.652 | 0.676 | 0.684 |
| pennystocks-trained → WSB test | 0.753 | 0.842 | 0.871 |

: Cross-dataset transfer AUC-ROC (no retraining). {#tbl:transfer}

Transfer is strongly asymmetric. A pennystocks-trained XGBoost transfers to `WSB` at AUC 0.871 (only 0.021 below native), well above the stretch threshold. Conversely, WSB-trained XGBoost achieves only 0.684 on pennystocks (0.069 below native RF), though still exceeding minimum. XGBoost achieves superior transfer in both directions. Notably, the smaller training set (21,549 records) transfers more effectively than the larger one (388,149 records), highlighting that community signal structure matters more than dataset size. [@sec:cross-community] analyses this asymmetry.

### Feature Importance

| Rank | WSB – RF | WSB – XGB | pennystocks – RF | pennystocks – XGB |
|------|--------------------:|-------------:|----------------:|------------------:|
| 1 | sentiment (+0.146) | sentiment (+0.203) | sentiment (+0.113) | sentiment (+0.165) |
| 2 | post_rate_24h (+0.045) | post_rate_24h (+0.067) | time_since_prev (+0.086) | time_since_prev (+0.069) |
| 3 | word_count (+0.010) | word_count (+0.027) | word_count (+0.029) | post_rate_24h (+0.015) |
| 4 | time_since_prev (+0.005) | time_since_prev (+0.003) | post_rate_24h (+0.023) | word_count (+0.008) |
| 5 | accel_x_time (+0.005) | day_of_week (+0.002) | word_count_x_hour (+0.017) | num_tickers (+0.006) |

: Top-5 permutation importances (10 repeats, scoring=roc_auc) for tree-based models. {#tbl:feature-importance}

`sentiment_score` dominates everywhere (+0.113 to +0.203). On WSB, activity features fill ranks 2–3; on pennystocks, `time_since_previous` rises to second (+0.086), reflecting reliance on temporal gaps when volume is low. Interaction terms remain negligible on WSB (≤+0.005) but `word_count_x_hour` reaches +0.017 on pennystocks RF.

![Permutation feature importance comparison across models and datasets. Sentiment score consistently dominates, while the relative ordering of activity and temporal features shifts between high-density (WSB) and sparse (pennystocks) communities.](figures/feature_importance_comparison.png){#fig:feature-importance}

### Sentiment Contribution (Phase 1 vs Phase 2) {#sec:sentiment-contribution}

| Dataset | Phase 1 (volume only) | Phase 2 (composite) | $\Delta$ AUC |
|---------|-----------------------|--------------------:|------:|
| WSB | 0.710 | 0.892 | +0.182 |
| pennystocks | 0.685 | 0.734 | +0.049 |

: Volume-only ($w_{2}=0$) vs composite ($w_{1}=w_{2}=0.5$), XGBoost AUC-ROC. {#tbl:phase-comparison}

Adjusting the target weight also alters the base surge rate (0.53% to 1.44% on `WSB`), so the gain reflects both richer signal and slightly higher class balance. The full weight sweep:

| $w_{2}$ | $w_{1}$ | AUC-ROC | Surge Rate | Tier |
|----|-----|---------|------------|------|
| 0.00 | 1.00 | 0.710 | 0.53% | Target |
| 0.25 | 0.75 | 0.708 | 1.01% | Target |
| 0.50 | 0.50 | 0.892 | 1.44% | Stretch |
| 0.75 | 0.25 | 0.861 | 4.30% | Stretch |
| 1.00 | 0.00 | 0.872 | 9.62% | Stretch |

: Weight sensitivity, XGBoost AUC-ROC on WSB ($\tau = 1.5$). {#tbl:weight-sensitivity}

The +0.184 AUC gain between $w_2 = 0.25$ and $w_2 = 0.50$ is disproportionate to the 0.43pp surge-rate increase, providing strong evidence of genuine predictive structure from sentiment. The relationship is non-monotonic: pure sentiment ($w_2 = 1.0$, AUC 0.872) outperforms $w_2 = 0.75$ (0.861), indicating volume contributions are most effective at equal balance.

## Critical Analysis

### Model Complexity vs Data Density {#sec:complexity-density}

XGBoost leads on `WSB` (0.892 vs. RF's 0.880), whereas Random Forest leads on `r/pennystocks` (0.753 vs. XGBoost's 0.734). Three factors explain this inversion:

First, XGBoost's sequential boosting requires sufficient positive examples to distinguish signal from noise. `WSB` provides 5,934 training surges versus 667 in pennystocks; with fewer positives, later boosting iterations fit noise, a risk bagging mitigates by averaging independent trees.

Second, Random Forest distributes splits more evenly across features. On pennystocks, Gini importances span six features above 0.06 with none exceeding 0.21, offering robustness when individual signals are unreliable.

Third, XGBoost's probability calibration degrades under extreme imbalance. At 31:1 training ratio in pennystocks (105:1 in test), its logistic output saturates near zero, yielding zero positive predictions at threshold 0.5. RF's vote-fraction probabilities produce less extreme skew.

For communities with fewer than ~5,000 positive training instances, Random Forest provides more robust performance than gradient boosting.

### The Sentiment Signal {#sec:sentiment-signal}

As a standalone feature, `sentiment_score` achieves modest AUC (0.559–0.587), yet dominates permutation importance (+0.146 to +0.203). This discrepancy arises because permutation importance measures contribution *in context of other features*. Sentiment becomes highly discriminative when conditioned on activity rates: accelerating volume combined with elevated emotional tone is a stronger surge precursor than either signal alone.

This also explains Phase 1 vs Phase 2 divergence. With sentiment excluded from the target ($w_2 = 0$), volume-only surges exhibit higher variance and prove harder to forecast. Incorporating sentiment ($w_2 \ge 0.50$) yields surges with more structured precursors. While part of the +0.182 AUC gain stems from altered task difficulty, its disproportionate magnitude evidences genuine predictive structure.

The performance ceiling is constrained by VADER's limitations: it fails on domain-specific semantics ("short" as bearish, "moon" as bullish) and sarcasm. FinBERT [14] represents a promising improvement avenue.

### Cross-Community Transfer and Generalisability {#sec:cross-community}

The pennystocks-trained model (21,549 records) transfers upward to `WSB` at AUC 0.871, whereas the WSB-trained model (388,149 records) achieves only 0.684 downward. This contradicts the assumption that larger datasets yield better transferability.

Distributional mismatch explains the asymmetry. On `WSB`, `word_count` alone achieves AUC 0.805, reflecting community culture where surges are preceded by lengthy "due diligence" posts. On pennystocks, `word_count` yields only 0.573. WSB-trained models heavily leverage this community-specific artifact, causing performance drops when transferred. Conversely, pennystocks-trained models cannot rely on a single dominant feature; they learn broader representations combining sentiment and activity metrics that generalise across domains.

Design guideline: models intended for multi-platform deployment should be trained on the most signal-constrained community to force feature generalisation, or fine-tuned on target-community data.

### Operational Precision and False Alarm Rate {#sec:operational-precision}

[@sec:operational-criteria] defined acceptance criteria: AUC-ROC ≥ 0.80, recall ≥ 0.50, precision ≥ 0.10, with 20–30 daily flags. At the best operating point (XGBoost, threshold 0.85 on `WSB`):

| Criterion | Requirement | Achieved | Met? |
|-----------|-------------|----------|------|
| AUC-ROC | ≥ 0.80 | 0.892 | Yes |
| Recall | ≥ 0.50 | 0.235 | No |
| Precision | ≥ 0.10 | 0.217 | Yes |
| Daily alert volume | 20–30 | ~10.5 flags/day* | Partial |

: Best model evaluated against operational acceptance criteria ([@sec:operational-criteria]). {#tbl:acceptance-eval}

\* Computed from 722 total flags (157 TP + 565 FP) over the 68-day test period – 10.6 flags per day.

The system meets ranking quality and precision requirements but catches only 23.5% of surges rather than the targeted 50%. Lowering the threshold to achieve recall ≥ 0.50 (at default 0.50: recall = 0.819) produces ~584 flags/day, operationally unusable. No single threshold simultaneously satisfies all three criteria at 102:1 class imbalance.

![Classification threshold sensitivity for XGBoost on WSB. As the decision threshold varies, precision and recall trade off sharply. No single threshold simultaneously achieves recall ≥ 0.50 and precision ≥ 0.10, illustrating the fundamental constraint imposed by the 102:1 class imbalance.](figures/12_threshold_sensitivity_xgboost.png){#fig:threshold-sensitivity}

**Implications by user scenario.** *Compliance teams*: insufficient as standalone surveillance, but complementary to rule-based volume alerts. *Quantitative researchers*: well-suited, ~2 genuine surges surfaced daily among 10 flags. *Platform moderators*: rank-based deployment (top-$k$ tickers daily) avoids the threshold problem entirely.

**Why ranking matters more than binary decisions.** The model ranks surges effectively (AUC = 0.892) but struggles to produce calibrated binary predictions at any single threshold, a mathematical consequence of the base rate, not a model failure [22]. Saito and Rehmsmeier [3] demonstrated that under severe imbalance, precision-recall analysis reveals limitations that ROC curves mask. PR-AUC of approximately 0.14 (14× above random) confirms this: strong discrimination across the full score range, but false alarms dominate at any threshold permissive enough for reasonable recall.

The operationally correct deployment is rank-based: sort tickers by predicted probability, review the top-$k$ daily, and accept that some surges fall outside the review window [2]. A fully automated system would require precision ≥ 0.80 and formal probability calibration [23], neither achieved here ([@sec:proposed-improvements]).

### Temporal Stability and Distribution Shift {#sec:temporal-stability}

The validation-test gap for Random Forest (val_F1 = 0.911 at tuned threshold vs test F1 = 0.210 at that same threshold) suggests temporal non-stationarity, where surge dynamics shifted as the post-GameStop wave subsided. The gap is even starker at the default threshold (test F1 = 0.145). Importantly, AUC remains high (0.880) on the test set, indicating that ranking ability transfers intact; the F1 collapse reflects threshold miscalibration under distribution shift rather than wholesale model failure. This distinction matters: overfitting would degrade both AUC and F1, whereas a shift in class balance or surge characteristics affects only the calibrated threshold. A deployed system would need periodic retraining or adaptive threshold selection. However, cross-dataset transfer at 0.871 suggests core patterns are stable enough to cross community boundaries; instability is concentrated in threshold calibration rather than underlying ranking.

## Limitations and Proposed Improvements

### Limitations

**Sample size on pennystocks**: 31 test surges yield bootstrap CIs spanning ±0.08–0.09 in AUC with overlap between models, so conclusions from pennystocks alone are tentative.

**Single calendar year** (2021) including the GameStop episode: models may have learned regime-specific patterns. Running on 2020 or 2022 data would test generality.

**Structural correlation between target and top feature**: `sentiment_score` dominates importance, but sentiment change is part of the composite target. The feature uses *current* sentiment while the target uses *forward-window* shift, not leakage, but a circularity that likely inflates sentiment's apparent importance. Phase 1 provides a partial control: with $w_2 = 0$, XGBoost still achieves AUC 0.710 ([@tbl:phase-comparison]).

**Smaller concerns**: survivorship bias (deleted posts absent); fixed temporal split (~June 2021) makes test difficulty regime-dependent; training variance only partially characterised (0.019 AUC range across 5 seeds).

**Class imbalance as deployment constraint**: The 102:1 ratio fundamentally constrains what the system can deliver. Even a perfect ranker (AUC = 1.0) faces a precision ceiling when forced into binary decisions at sub-1% prevalence. Surge prediction at this level must operate as either (a) a ranking layer with human top-$k$ review, or (b) a first-stage filter combined with a higher-precision second stage [3][22]. Neither metric alone captures operational value; both must be reported with deployment context. For communities with higher surge prevalence (e.g., $\tau = 1.0$ producing ~5% surge rate), precision constraints relax substantially, suggesting the "right" surge definition should be co-designed with end-users based on alert-handling capacity.

### Proposed Improvements {#sec:proposed-improvements}

| Improvement | Motivation (from results) | Effort | Priority |
|-------------|---------------------------|--------|----------|
| FinBERT for sentiment | Top feature (+0.203) but VADER misses financial semantics ([@sec:sentiment-signal]) | Medium | High |
| Multi-scale windows (6h, 12h, 24h, 72h) | Fixed 24h window may miss faster/slower surges | Medium | High |
| Probability calibration (Platt/isotonic) | Precision collapse at default threshold is a calibration problem | Low | High |
| Known-ticker validation list | Ticker heuristics admit false positives diluting activity counts | Low | Medium |
| Additional time periods (2020, 2022) | Single-year limitation | Medium | Medium |
| Additional subreddits (r/stocks, r/investing) | Tests density gradient more granularly | Medium | Low |

: Proposed improvements, ranked by priority. {#tbl:improvements}

---

# Conclusion

## Current Achievements

The core question driving this project addresses a critical gap in the existing literature: can volume and sentiment surges in Reddit financial communities be predicted using only information available at the exact moment of post creation? To prevent the future-engagement feature leakage common in prior work, such as reliance on post-hoc upvote or comment counts, the proposed pipeline enforces strict temporal ordering across surge definition, feature extraction, and model evaluation.

The resulting framework evaluates raw Reddit data through statistically validated classifiers across two communities, three models, and over thirty experimental runs. Ultimately, this work provides three core contributions: a rigorous leakage-free forecasting methodology, a composite surge metric, and empirical evidence identifying data density as the primary constraint on predictive performance.

## Originality and Contribution

This project makes three contributions.

First, a **leakage-free methodology applied where neglected**. The literature review ([@sec:methodological-weaknesses], [@tbl:temporal-practices]) shows temporal leakage is the norm in social media prediction studies. This project applies established temporal evaluation principles [18][19] end-to-end, showing that performance of AUC = 0.753–0.892 is both achievable and trustworthy. The framework is reusable for any timestamped prediction problem.

Second, a **composite surge metric** integrating normalised volume growth with sentiment change, fully parameterised by threshold and weights. The Phase 1 vs Phase 2 experiment ([@tbl:phase-comparison]) confirms it captures a richer phenomenon than volume alone (+0.182 AUC on WSB).

Third, **empirical evidence that data density is the binding constraint**. Same pipeline, same models, different community size: the gap between datasets (0.753 vs 0.892) and asymmetric transfer (sparse→dense at 0.871; dense→sparse at 0.684) demonstrate this clearly.

Direct comparison with published baselines is not possible, as no reviewed study predicts surges on the same datasets with the same temporal protocol. These are incremental contributions, combining established techniques into a coherent framework for a problem prior work has not directly addressed, with each claim grounded in quantified evidence.

## Key Findings

The short answer to the central research question is yes: surges can be predicted from backward-looking signals alone. How well depends almost entirely on how much data the community generates. On WSB, XGBoost and Random Forest both reached the stretch tier (AUC = 0.892 and 0.880). On sparser r/pennystocks, Random Forest managed 0.753, clearing target but with wide confidence intervals over only 31 test surges.

Three findings were not obvious going in:

**Having more data matters more than having a better model.** The gap between datasets (13–14 AUC points) is larger than the gap between any two models on the same dataset. For surge detection on smaller communities, invest in data collection before model tuning.

**The advantage of more complex models is not guaranteed.** On WSB, gradient boosting earns its complexity. On pennystocks the picture flips: Random Forest beats XGBoost. With only a few hundred positive examples, boosting's sequential corrections fit noise, while averaging independent trees is more forgiving.

**Sentiment does more than add a useful feature; it changes what a surge is.** When sentiment is stripped from the target definition, XGBoost's AUC on `WSB` drops 18 points. Sentiment is the most important feature by permutation importance, yet poor alone. The signal is interactive: accelerating discussion combined with rising emotional intensity is a meaningful pattern, but either one alone is not.

Cross-dataset transfer revealed an asymmetry. Training on pennystocks and testing on `WSB` yields AUC 0.871, nearly matching the native result, but training on `WSB` and testing on pennystocks yields only 0.684. `WSB` models lean heavily on word count, long analytical posts precede surges there, and that pattern does not exist in the other community. Models trained on sparse data spread reliance across weaker signals and learn something closer to universal.

For teams monitoring financial communities, the practical takeaway is to invest in data coverage before model sophistication.

## Limitations and Future Work

All three objectives were met ([@sec:eval-objectives]), but four limitations bound the conclusions.

**The pennystocks evaluation rests on thin ground.** Thirty-one test surges produce confidence intervals wide enough that model rankings could shift with a different test period.

**2021 may not be representative.** The GameStop episode makes this one of the most unusual periods of retail speculation in memory. Running on 2020 or 2022 data would matter more for credibility than any model improvement.

**VADER does not speak Reddit finance.** It misreads domain-specific terms ("short" as negative, "moon" as neutral). FinBERT [14] would handle this, though applying it across 1.3M records requires GPU infrastructure.

**The models rank surges well but flag them poorly.** At extreme imbalance, probability outputs compress so much that useful thresholds sit near 0.16. Post-training calibration (Platt scaling or isotonic regression) would make scores interpretable without changing discriminative power.

Two directions would extend the methodology:

**Multi-scale temporal windows.** Adding parallel windows at 6h, 12h, and 72h would let the model match its horizon to surge timescale and reveal whether findings are partly an artefact of the 24-hour alignment with daily posting rhythms.

**Live prediction testing.** Everything here is retrospective. Connecting the pipeline to a live stream, where predictions are recorded before outcomes are known, would produce evidence that retrospective evaluation cannot. In deployment, the system would ingest posts in a rolling stream, recompute features hourly, and surface top-$k$ tickers for human review. Building that system is beyond current scope, but nothing in the architecture prevents it.

---

# References {.unnumbered}

[1] FINRA. 2025. Social Media-Influenced Investing. Financial Industry Regulatory Authority. Retrieved from https://www.finra.org/rules-guidance/key-topics/fintech/report/social-media-influenced-investing

[2] Tom Fawcett. 2006. An introduction to ROC analysis. *Pattern Recognition Letters* 27, 8 (June 2006), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010

[3] Takaya Saito and Marc Rehmsmeier. 2015. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE* 10, 3 (March 2015), e0118432. https://doi.org/10.1371/journal.pone.0118432

[4] Charles Elkan. 2001. The foundations of cost-sensitive learning. In *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI '01)*. Morgan Kaufmann, Seattle, WA, 973–978.

[5] Gabor Szabo and Bernardo A. Huberman. 2010. Predicting the popularity of online content. *Commun. ACM* 53, 8 (August 2010), 80–88. https://doi.org/10.1145/1787234.1787254

[6] Kristina Lerman and Tad Hogg. 2010. Using a model of social dynamics to predict popularity of news. In *Proceedings of the 19th International Conference on World Wide Web (WWW '10)*. ACM, New York, NY, USA, 621–630. https://doi.org/10.1145/1772690.1772754

[7] Fang Wang and Bernardo A. Huberman. 2013. Quantifying long-term scientific impact. *Science* 342, 6154 (October 2013), 127–132. https://doi.org/10.1126/science.1237825

[8] Shoubin Kong, Qiaozhu Mei, Ling Feng, Fei Ye, and Zhe Zhao. 2014. Predicting bursts and popularity of hashtags in real-time. In *Proceedings of the 37th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '14)*. ACM, New York, NY, USA, 927–930. https://doi.org/10.1145/2600428.2609476

[9] Justin Cheng, Lada Adamic, P. Alex Dow, Jon M. Kleinberg, and Jure Leskovec. 2014. Can cascades be predicted? In *Proceedings of the 23rd International Conference on World Wide Web (WWW '14)*. ACM, New York, NY, USA, 925–936. https://doi.org/10.1145/2566486.2567997

[10] Roja Bandari, Sitaram Asur, and Bernardo A. Huberman. 2012. The pulse of news in social media: Forecasting popularity. In *Proceedings of the 6th International AAAI Conference on Weblogs and Social Media (ICWSM '12)*. AAAI Press, 26–33.

[11] Chao Yuan and Wentao Li. 2019. Forecasting the development trend of early-stage information diffusion based on empirical data. *Physica A* 524 (June 2019), 157–167. https://doi.org/10.1016/j.physa.2019.04.053

[12] Johan Bollen, Huina Mao, and Xiaojun Zeng. 2011. Twitter mood predicts the stock market. *J. Comput. Sci.* 2, 1 (March 2011), 1–8. https://doi.org/10.1016/j.jocs.2010.12.007

[13] Clayton J. Hutto and Eric Gilbert. 2014. VADER: A parsimonious rule-based model for sentiment analysis of social media text. In *Proceedings of the 8th International AAAI Conference on Weblogs and Social Media (ICWSM '14)*. AAAI Press, Ann Arbor, MI, 216–225.

[14] Dogu Araci. 2019. FinBERT: Financial sentiment analysis with pre-trained language models. Retrieved from https://arxiv.org/abs/1908.10063

[15] Cathy Long, Brian Lucey, and Larisa Yarovaya. 2023. I just like the stock: The role of Reddit sentiment in the GameStop share rally. *Financ. Rev.* 58, 1 (February 2023), 19–37. https://doi.org/10.1111/fire.12328

[16] Michele Costola, Matteo Iacopini, and Carlo R. M. A. Santagiustina. 2022. Self-induced consensus of Reddit users to characterise the GameStop short squeeze. *Sci. Rep.* 12, 1 (August 2022), Article 13780. https://doi.org/10.1038/s41598-022-17925-2

[17] Adriano Mancini, Aldo Desiderio, Brendan Marafino, and Alessandro Navigli. 2024. Detecting pump and dump stock market manipulation from online forums. *Digital Finance* 6 (2024), 365–393. https://doi.org/10.1007/s42521-024-00113-6

[18] Leonard J. Tashman. 2000. Out-of-sample tests of forecasting accuracy: An analysis and review. *Int. J. Forecast.* 16, 4 (October–December 2000), 437–450. https://doi.org/10.1016/S0169-2070(00)00065-0

[19] Christoph Bergmeir and José M. Benítez. 2012. On the use of cross-validation for time series predictor evaluation. *Inf. Sci.* 191 (May 2012), 192–213. https://doi.org/10.1016/j.ins.2011.12.028

[20] Manuel Fernández-Delgado, Eva Cernadas, Senén Barro, and Dinani Amorim. 2014. Do we need hundreds of classifiers to solve real world classification problems? *J. Mach. Learn. Res.* 15, 1 (January 2014), 3133–3181.

[21] Leukipp. 2021. Reddit Finance Data. Kaggle. Retrieved from https://www.kaggle.com/datasets/leukipp/reddit-finance-data

[22] Haibo He and Edwardo A. Garcia. 2009. Learning from imbalanced data. *IEEE Trans. Knowl. Data Eng.* 21, 9 (September 2009), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

[23] Andrew J. Vickers and Elena B. Elkin. 2006. Decision curve analysis: A novel method for evaluating prediction models. *Medical Decision Making* 26, 6 (November 2006), 565–574. https://doi.org/10.1177/0272989X06295361