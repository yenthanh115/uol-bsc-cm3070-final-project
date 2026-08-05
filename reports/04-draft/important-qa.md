**Why binary classification?** A ticker either surges within the next 24 hours or it doesn't, with no meaningful in-between. That makes binary classification the natural fit. Regression would ask "how big?" when the real question is "did it happen?" Multi-class buckets (small/medium/large) would require arbitrary cut-points and make the class-imbalance problem worse. Anomaly detection is unsupervised, so it can't use the labelled surge history; it would also flag every rare surge as anomalous regardless of whether it has any distinguishing structure.

**Could a simple threshold rule work just as well?**

*Table 15: Multi-feature models vs. baselines (AUC-ROC).*

| Dataset | Random Baseline | Best Single Feature | Best Model | Δ over Single Feature |
|---------|-----------------|---------------------|------------|----------------------|
| WSB | 0.500 | 0.805 (word_count) | 0.892 (XGB) | +0.087 |
| Pennystocks | 0.500 | 0.591 (hour_of_day) | 0.753 (RF) | +0.162 |

The best single-feature predictor is equivalent to a threshold rule on one signal. On WSB it reaches 0.805, which is already strong, but the multi-feature models add another 8.7 AUC points by combining signals that no single rule can integrate. On pennystocks the gap is even wider (+0.162), where the best individual feature barely clears 0.591 and multi-feature combination is what makes the task solvable at all.