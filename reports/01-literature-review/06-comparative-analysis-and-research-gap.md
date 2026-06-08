# 6. Comparative Analysis and Research Gap

## 6.1 Comparing the Main Predictive Approaches

The literature reviewed shows that social media prediction has developed through several complementary approaches, each focusing on a different dimension of online behaviour. Early popularity prediction studies emphasised temporal engagement signals, showing that early views, votes, or reposts can help forecast later outcomes. Machine learning research extended this by incorporating content and metadata features, suggesting that popularity may depend not only on user reactions but also on how information is framed and presented. Sentiment analysis studies added emotional and semantic signals, highlighting the predictive value of public mood and textual meaning. Diffusion and cascade research introduced a network-oriented perspective, showing that early propagation patterns and resharing behaviour can indicate future spread. Together, these studies suggest that social media behaviour is not random but shaped by multiple measurable signals.

| Study | Signal | Method | Strength | Limitation |
| -------- | -------- | -------- | -------- | -------- |
| Szabo & Huberman | Temporal | Regression | Simple | Popularity focus |
| Lerman | Engagement | Social model | Behavioural insight | Platform specific |
| Bandari | Content | ML | Early prediction | Weak sentiment |
| Bollen | Sentiment | NLP | Finance relevance | Correlation only |
| Cheng | Diffusion | Cascade model | Strong early signal | Needs network data |

## 6.2 Strengths and Limitations Across the Literature

Each research strand contributes useful insights but also has clear limitations. Early popularity prediction offers interpretable and efficient methods, yet it mainly addresses eventual popularity after engagement has already begun. Content-based machine learning improves early prediction by using features available before strong interaction occurs, but early studies often relied on manually engineered variables and limited contextual understanding. Sentiment analysis is especially relevant in finance-related settings because emotional tone may shape public reaction, but lexicon-based methods struggle with sarcasm, ambiguity, and domain-specific language. Diffusion research provides strong insight into how trends spread through networks, yet it often depends on detailed relationship data and computationally intensive modelling. As a result, no single approach fully captures early trend emergence.

## 6.3 Key Research Gaps

Three main gaps emerge from the literature. First, most studies focus on eventual outcomes such as popularity, cascade size, or market movement rather than the earliest transition from ordinary discussion to emerging trend [1]–[5]. Second, many studies rely on a single category of features, even though trend formation is likely shaped by interactions between temporal, behavioural, semantic, and structural signals. Third, relatively little attention has been given to finance-specific trend emergence, despite the importance of sentiment, speculation, and rapid reactions to external events in financial discussions. These gaps suggest that models developed for general social media popularity may not fully capture the dynamics of emerging financial trends.

## 6.4 Implications for the Proposed Project

These gaps provide a clear rationale for the proposed project. Rather than predicting final popularity, the project focuses on detecting the early stage of trend emergence in finance-related social media discussions. The literature suggests that this is most likely to be achieved through an integrated framework that combines temporal activity, engagement behaviour, and sentiment-based features. Such an approach builds on the strengths of earlier studies while remaining more practical than highly complex diffusion models. The project therefore aims to contribute a focused and achievable predictive framework for identifying early signals of emerging financial discussions before widespread popularity becomes visible.
