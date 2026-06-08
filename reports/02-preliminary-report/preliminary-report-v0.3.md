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

Financial discussions on social media platforms often experience sudden increases in public attention and emotional intensity. Discussions surrounding specific stocks can rapidly attract large numbers of comments, interactions, and strong sentiment, particularly following news events, earnings announcements, rumours, or speculative activity.

This problem affects financial analysts, researchers, organisations, and market observers who monitor social media discussions to understand emerging investor sentiment and public attention. In large social media environments, thousands of stock-related discussions occur every day, making it difficult to identify which discussions are likely to experience substantial growth before that growth becomes obvious.

Existing research frequently focuses on predicting overall popularity or analysing already-popular content, providing less emphasis on forecasting whether a stock-related discussion is about to experience a significant surge within a clearly defined future time window.

### 1.4 Motivation

- Discussions experiencing rapid engagement and sentiment growth often attract broader public attention
- May influence information diffusion, investor behaviour, and market perception
- Early detection enables proactive monitoring rather than reactive analysis
- Addresses a gap in the literature around short-term surge prediction with composite targets

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

---

## 3. Proposed Methodology

### 3.1 Data Pipeline Stages

1. **Data Loading** — Read static dataset from disk (CSV)
2. **Preprocessing** — Deduplicate, parse timestamps, normalise text, remove nulls
3. **Feature Engineering** — Compute sentiment, temporal, engagement-rate, and text features
4. **Target Labelling** — Compute composite surge target using 24-hour prediction window
5. **Model Training** — Train LR, RF, XGBoost with temporal train-test split
6. **Evaluation** — Compute metrics, generate confusion matrices and ROC curves

### 3.2 Tools and Technologies

---

## 4. Risk Register

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

## 5. Project Plan and Timeline

---

## 6. Initial Literature Review Summary

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


---

## 7. Success Criteria


