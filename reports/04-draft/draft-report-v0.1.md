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
-->
---

## 1. Introduction

### 1.1 Project Motivation

<!-- Why predicting posting-volume surges on r/pennystocks matters for market participants and researchers -->

### 1.2 Project Aims

<!-- Binary classification of ticker-level 24h posting-volume surges using backward-looking features -->

### 1.3 Project Concept

<!-- Composite surge metric, temporal CV methodology, multi-model comparison approach -->

### 1.4 Scope and Research Question

<!-- Can posting-volume surges on r/pennystocks be predicted from backward-looking features? -->

---

## 2. Literature Review

### 2.1 Social Media and Financial Markets

<!-- Relationship between social media activity and market behaviour -->

### 2.2 Sentiment Analysis in Finance

<!-- NLP approaches for financial text, VADER, FinBERT, limitations -->

### 2.3 Penny Stocks and Retail Investor Communities

<!-- Characteristics of penny stock discussion, Reddit as data source -->

### 2.4 Predictive Modelling for Social Media Signals

<!-- Prior work on predicting volume/activity surges, temporal approaches -->

### 2.5 Research Gap

<!-- What is missing in the literature that this project addresses -->

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

<!-- ACM citation format -->
