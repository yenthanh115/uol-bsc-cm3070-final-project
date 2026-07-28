### Risk Register

| # | Risk | Likelihood | Impact | Mitigation | Status |
|---|------|-----------|--------|------------|--------|
| 1 | Class imbalance (few surge events) | High | High | Balanced class weighting, AUC-ROC evaluation, report precision-recall curves | Open |
| 2 | TextBlob sentiment limitations | Medium | Medium | Document limitations; configurable component weight; FinBERT as future alternative | Open |
| 3 | Data quality issues | Medium | Medium | Robust preprocessing with logging; document exclusion criteria | Open |
| 4 | Temporal data leakage | Medium | High | Strict temporal split; backward-only features; engagement scores excluded | Mitigated |
| 5 | Overfitting on small effective dataset | Medium | High | Temporal CV, L1/L2 regularisation, report train-test gaps | Open |
| 9 | Per-ticker sparsity | High | High | Min window count (N≥3); report exclusion rate; test with larger datasets | Open |
| 10 | Snapshot engagement as features | High | Critical | Eliminated — features use only timestamps and text | Mitigated |
| 11 | Circular labelling from snapshot scores | High | Critical | Target uses posting volume growth from timestamps only | Mitigated |
| 12 | Temporal concept drift | Medium | Medium | Report train/test rate differences; acknowledge as limitation | Open |