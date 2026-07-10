
## Current Performance Problem

| Issue | Detail |
|-------|--------|
| 89% exclusion | Only 8,577 of 80,212 records are usable (tickers too sparse for 24h windowing) |
| 7 test surges | Statistically meaningless for precision/recall evaluation |
| Best model AUC = 0.77 | Target tier, but single-feature `word_count` (0.81) beats all multi-feature models |
| Sentiment useless | AUC = 0.47 (below random) |

## Improvement Plan

### Track A: Fix the data sparsity (highest impact)

**A1 — Lower `min_window_count` to 1**

The current threshold (`forward_count < 3`) is too aggressive for a long-tail dataset. With `min_window_count=1`:
- Included records: 24,827 (3× more data)
- More test surges → reliable evaluation
- Tradeoff: growth ratio `forward/backward` is noisier with count=1, but models handle noise better than data starvation

**A2 — Use a larger/denser dataset**

The r/pennystocks dataset has 11,739 tickers with median 2 posts each — it's fundamentally sparse. Better options:

| Dataset | Why better |
|---------|-----------|
| r/wallstreetbets (2021) | 10–100× denser per ticker (GME, AMC era). Widely available on Kaggle/pushshift. |
| Combined multi-subreddit | Merge r/pennystocks + r/wallstreetbets + r/stocks for ticker coverage |
| Longer time range | If more r/pennystocks data exists (2020–2022), the same tickers accumulate more posts |

The loader already handles the format (`created` column, datetime strings). Any Reddit submission CSV with the same schema works drop-in.

**A3 — Switch from forward-window exclusion to backward-only windowing**

Instead of requiring future posts to exist (which excludes end-of-timeline records), define surge based on **whether the backward count exceeds a threshold relative to the ticker's historical average**. This makes every record with sufficient backward history usable — no forward-looking data needed for labelling.

### Track B: Improve feature signal

**B1 — Drop or downweight weak features**

Current single-feature AUCs show sentiment (0.47) and ticker_post_rate (0.56) hurt performance. Options:
- Train with only top-5 features (word_count, hour_of_day, num_tickers, title_length, ticker_acceleration)
- Add L1 regularisation to automatically zero out weak features (LR already uses elasticnet)

**B2 — Add interaction features**

- `word_count × hour_of_day` (long posts at peak hours)
- `ticker_post_acceleration × time_since_previous` (rapid acceleration after silence)

**B3 — Tune the surge threshold τ**

With τ=1.5, surge rate is 2.2%. Lowering to τ=1.0 gives more positives (from the sensitivity sweep), improving statistical power while maintaining meaningful class separation.

### Track C: Try a different raw dataset (quick experiment)

**C1 — Download r/wallstreetbets 2021 from Kaggle**

Several well-known datasets exist:
- "Reddit WallStreetBets Posts" on Kaggle (~1M posts, 2020–2021)
- Arctic Shift / Pushshift archives

The pipeline needs no code changes — just point `--file-path` at the new CSV (as long as it has `id`, `created`/`created_utc`, `title`, `selftext` columns).

---

## Recommended Priority Order

| Priority | Action | Effort | Expected Impact |
|----------|--------|--------|-----------------|
| 1 | Lower min_window_count to 1 | 5 min | 3× more data, reliable test metrics |
| 2 | Lower τ to 1.0 | 5 min | More test surges, better class balance |
| 3 | Drop sentiment feature | 15 min | Remove noise, may lift AUC |
| 4 | Try r/wallstreetbets dataset | 30 min | Fundamentally denser data, likely strong results |
| 5 | Backward-only surge definition | 2–3 hours | Removes forward-looking dependency entirely |

Want me to start with priorities 1–3 (quick config/code changes) or go straight to trying a different dataset?