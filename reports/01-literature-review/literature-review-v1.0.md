# 1. Introduction

## 1.1 Background and Relevance of Social Media Trend Detection

Social media platforms are major spaces for the rapid spread of information, opinions, and public discussion. Through reposts, comments, likes, and hashtags, topics can attract attention quickly and provide real-time signals of changing public interest. Detecting these trends early has practical value in areas such as marketing, journalism, public opinion monitoring, and finance. In financial contexts especially, sudden shifts in online discussion and sentiment may shape investor attention, influence expectations, and contribute to short-term market reactions.

## 1.2 Trend Emergence as a Research Problem

Despite its importance, identifying trend emergence at an early stage remains difficult. Social media activity is highly dynamic, noisy, and often short-lived. Many discussions produce brief spikes in attention before fading, while others develop into sustained trends with broader influence. In this review, trend emergence refers to the point at which a topic begins to move beyond ordinary background discussion and shows signs of sustained growth in attention, engagement, or diffusion. This differs from popularity prediction, which mainly estimates the eventual level of attention that content will receive.

<figure align="center">
  <img src="figures/01-trend-lifecycle.png" alt="Trend Lifecycle" width="500">
  <figcaption>Figure 1: Trend Lifecycle.</figcaption>
</figure>

Recent research also suggests that online trends should be understood as part of a broader lifecycle rather than as isolated popularity outcomes. Wang and Huberman [6] show that collective attention follows identifiable temporal dynamics, while Kong et al. [7] describe popularity as evolving through stages such as emergence, growth, peak, and decline. Yuan and Li [8] further indicate that early stages of popularity evolution may contain signals that appear before large-scale diffusion. From this perspective, trend emergence is the earliest phase of a broader process, making it distinct from studies that focus mainly on final popularity or later-stage spread.

## 1.3 Scope of the Review and Research Gap
This literature review examines key methodological approaches relevant to this problem, including early popularity prediction, machine learning and content-based forecasting, sentiment analysis, and information diffusion research. 

<figure align="center">
  <img src="figures/02-methodologies.png" alt="Early Trend Detection Methodologies" width="500">
  <figcaption>Figure 2: Early Trend Detection Methodologies.</figcaption>
</figure>
 
Together, these studies show that online behaviour can be predicted from temporal, behavioural, semantic, and structural signals. However, much of the literature focuses on predicting eventual popularity rather than detecting the earliest stage of trend formation. Many studies also rely on a single category of features, and relatively little attention has been given to finance-specific trend emergence. This review therefore evaluates the strengths and limitations of existing approaches and identifies the research gaps that support a predictive modelling project focused on the early detection of finance-related social media trends.

# 2. Early Popularity Prediction Foundations

## 2.1 Early Popularity Prediction as a Foundation

One of the earliest foundations of research relevant to social media trend prediction is early popularity prediction. Although it is not identical to trend emergence detection, it established the important idea that early user behaviour may contain signals about future online outcomes. This provides a useful starting point for understanding how initial activity can be analysed to anticipate later growth.

## 2.2 Temporal Patterns and Early Engagement Signals

A key contribution in this area came from Szabo and Huberman [1], who examined whether measurements collected shortly after publication could predict future popularity on platforms such as YouTube and Digg. Their results showed very strong correlations between early and later popularity, suggesting that online attention growth is not entirely random. The study also demonstrated that relatively simple statistical models could generate accurate forecasts by using early engagement data within short observation windows. This was important because it showed that predictive modelling could rely on measurable behavioural patterns rather than purely retrospective description.

## 2.3 Behavioural and Social Interaction Dynamics

Alongside temporal analysis, Lerman and Hogg [2] highlighted the role of behavioural and social dynamics. Their work suggested that popularity depends not only on the accumulation of views or votes, but also on how users discover, share, and respond to content within a social platform. This perspective is relevant to trend emergence because it emphasises that collective attention develops through interaction between users, content, and platform structures. It also supports the idea that predictive models may benefit from combining multiple behavioural indicators.

## 2.4 Contributions and Limitations for Trend Emergence Research

Together, these studies made two important contributions: they demonstrated that early online behaviour contains predictive signals, and they helped shift research toward forecasting rather than description. However, both studies mainly focused on eventual popularity rather than the earliest stage of trend emergence. They also relied largely on temporal and engagement-based indicators, giving limited attention to sentiment, semantic meaning, or domain-specific context. These limitations are important in finance-related environments, where public reaction and emotional tone may strongly influence whether a discussion develops into an emerging trend.

# 3. Machine Learning and Content-Based Prediction

## 3.1 Expanding Prediction Beyond Early Engagement

As social media prediction research developed, methods moved beyond simple temporal popularity measures toward machine learning approaches that used a wider range of predictive features. This shift was important because online attention is shaped not only by early engagement, but also by characteristics of the content itself. As a result, prediction increasingly came to rely on combining behavioural signals with structured information about source, topic, and language.

## 3.2 Content and Metadata as Predictive Features

A key example of this development is Bandari et al. [3], who examined whether the popularity of news stories could be predicted using content and metadata features before substantial user engagement had occurred. Their study included variables such as publication source, category, subjectivity, and named entities. The results showed that these features could classify articles into popularity categories with approximately 84% accuracy, although predicting exact popularity values remained more difficult. This was a significant contribution because it suggested that predictive signals may exist even before strong diffusion begins.

## 3.3 Methodological Contribution of Early Machine Learning

The study also marked an important methodological transition from simple correlation-based forecasting to more flexible machine learning pipelines based on feature extraction, model training, and classification. It broadened understanding of online attention by showing that popularity may depend not only on how users respond, but also on how information is framed and presented. This supports the idea that predictive models should include multiple feature types rather than relying on a single indicator.

## 3.4 Relevance and Limitations for Trend Emergence Research

Despite these advances, early machine learning approaches still focused mainly on popularity outcomes rather than the earliest stage of trend emergence. They also depended heavily on manually engineered features and offered limited insight into sentiment, contextual meaning, or how discussions spread through networks. These limitations are especially important in finance-related environments, where emotional tone and public reaction may strongly influence whether a discussion develops into an emerging trend.

# 4. NLP and Sentiment Analysis in Social Media Prediction

## 4.1 From Behavioural Signals to Textual Meaning

As social media platforms became important spaces for opinion sharing and public discussion, researchers began exploring whether the linguistic and emotional content of online posts could improve prediction. This led to the growing use of natural language processing (NLP) and sentiment analysis in social media research. Unlike earlier approaches focused mainly on engagement counts or temporal growth, sentiment-based methods attempt to capture what users are expressing and how they feel.

## 4.2 Sentiment and Collective Mood as Predictive Signals

A key study in this area is Bollen et al. [4], which examined whether public mood measured from Twitter could predict movements in the Dow Jones Industrial Average. Using nearly ten million tweets, the authors analysed multiple emotional dimensions, including Calm, Alert, Sure, Vital, Kind, and Happy, rather than relying only on simple positive–negative sentiment. Their results showed that some mood states, especially Calm, were associated with market movements several days in advance. When these signals were used in a neural-network model, the study reported approximately 87.6% directional prediction accuracy.

## 4.3 Contribution to Predictive Modelling Research

This study was important because it showed that social media content may contain predictive signals beyond simple popularity or engagement measures. It helped establish sentiment analysis as a major strand of predictive modelling research and demonstrated the value of transforming unstructured text into quantitative features. The paper also suggested that richer emotional representations may be more useful than simple polarity measures, which is especially relevant in finance-related environments where optimism, fear, and uncertainty often shape discussion dynamics.

## 4.4 Relevance and Limitations for Trend Emergence Research

Despite its influence, the study has clear limitations. It focused on market prediction rather than direct trend emergence, and its lexicon-based methods had limited ability to capture sarcasm, context, or financial jargon. The findings also do not establish causation, since external events may influence both social media mood and market outcomes. Nevertheless, the study remains highly relevant because it supports the inclusion of sentiment-based features as part of a broader framework for detecting early signals in finance-related social media discussions.

# # 5. Information Diffusion and Cascade Prediction

## 5.1 From Popularity to Diffusion Processes

As social media research developed, scholars increasingly recognised that online growth is shaped not only by individual engagement or content features, but also by how information spreads through networks of users. This led to growing interest in information diffusion and cascade prediction, which examine how posts, topics, or discussions propagate over time. These approaches are especially relevant to trend emergence because trends often develop through expanding participation and repeated sharing rather than through isolated popularity events.

## 5.2 Early Cascade Signals and Predictive Potential

An influential study in this area is Cheng et al. [5], Can Cascades Be Predicted? The authors investigated whether large Facebook cascades could be predicted using early observations of sharing behaviour. In this context, a cascade refers to the spread of information from user to user through reposting or resharing. By analysing millions of cascades, the study showed that meaningful predictive signals can emerge early in the diffusion process. After observing only the first few reshares, the model achieved approximately 79.5% accuracy and an AUC of 0.877 when predicting whether a cascade would continue growing substantially.

## 5.3 Contribution to Trend Emergence Research

This study was important because it shifted attention from simple popularity counts to the structural dynamics of information spread. It showed that temporal propagation patterns, especially the speed of early resharing, can be stronger predictors than content features alone. The concept of structural virality further suggested that how information spreads may matter as much as how much attention it initially receives. This insight is highly relevant to trend emergence, where rapid expansion across connected users may signal the transition from ordinary discussion to broader collective attention.

## 5.4 Relevance and Limitations for the Current Project

Despite its value, the study has limitations. Its findings were shaped by Facebook’s platform structure and focused mainly on photo-sharing cascades, which may not generalise directly to finance-related discussions on platforms such as Twitter or Reddit. Diffusion-based models may also require detailed network data and high computational resources. Nevertheless, the study supports the inclusion of temporal growth indicators and diffusion concepts within a more practical predictive framework for early trend detection.

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

# 7. Conclusion

This literature review examined four major strands of research relevant to social media prediction: early popularity prediction, machine learning and content-based prediction, sentiment analysis, and information diffusion modelling. Together, these studies show that online behaviour is not entirely unpredictable and that meaningful signals about future developments can often be detected during the early stages of social media activity. Across these approaches, researchers have identified temporal, behavioural, semantic, and structural patterns that help explain how online attention develops.

At the same time, the review highlights important limitations in the existing literature. Much of the research focuses on predicting eventual popularity, cascade size, or market movement rather than identifying the earliest stage of trend emergence. Many studies also rely on a single category of features, even though trend formation is likely to be influenced by multiple interacting factors. In addition, relatively limited attention has been given to finance-specific trend emergence, despite the importance of sentiment, speculation, and rapid reactions to external events in financial discussions.

These gaps provide a clear rationale for the proposed project. Rather than predicting final popularity, the project focuses on detecting the early transition from ordinary discussion to emerging trend in finance-related social media data. Guided by the literature, the proposed approach integrates temporal activity, engagement behaviour, and sentiment-based features within a practical predictive framework. This offers a focused and achievable contribution to the early detection of finance-related social media trends.

# 8. References

[1] Gabor Szabo and Bernardo A. Huberman. 2010. Predicting the popularity of online content. Communications of the ACM 53, 8 (2010), 80–88. https://doi.org/10.1145/1787234.1787254

[2] Kristina Lerman and Tad Hogg. 2010. Using a model of social dynamics to predict popularity of news. In Proceedings of the 19th International Conference on World Wide Web (WWW '10). ACM, New York, NY, USA, 621–630. https://doi.org/10.1145/1772690.1772758

[3] Roozbeh Bandari, Sitaram Asur, and Bernardo A. Huberman. 2012. The pulse of news in social media: Forecasting popularity. In Proceedings of the International AAAI Conference on Web and Social Media, Vol. 6, No. 1. AAAI Press, 26–33.

[4] Johan Bollen, Huina Mao, and Xiao-Jun Zeng. 2011. Twitter mood predicts the stock market. Journal of Computational Science 2, 1 (2011), 1–8. https://doi.org/10.1016/j.jocs.2010.12.007

[5] Justin Cheng, Lada Adamic, P. Alex Dow, Jon Kleinberg, and Jure Leskovec. 2014. Can cascades be predicted? In Proceedings of the 23rd International Conference on World Wide Web (WWW '14). ACM, New York, NY, USA, 925–936. https://doi.org/10.1145/2566486.2567997

[6] Chunyan Wang and Bernardo A. Huberman. 2012. Long trend dynamics in social media. EPJ Data Science 1, 1 (2012), Article 2. https://doi.org/10.1140/epjds2

[7] Qingqing Kong, Wei Mao, Guangxing Chen, and Daniel Zeng. 2018. Exploring trends and patterns of popularity stage evolution in social media. IEEE Transactions on Systems, Man, and Cybernetics: Systems 48, 12 (2018), 2408–2420. https://doi.org/10.1109/TSMC.2017.2719279

[8] Dongyuan Yuan and Yan Li. 2025. Discovering and early predicting popularity evolution patterns of social media emergency information. Aslib Journal of Information Management 77, 1 (2025), 115–137. https://doi.org/10.1108/AJIM-06-2024-0288
