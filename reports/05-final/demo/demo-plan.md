# Demo Plan — Surge-Labelling Pipeline for Reddit Trend Prediction

**Project template:** CM3005 Data Science — *Project Idea: Predictive Modelling of Social Media Trend Emergence*
**Target length:** 5 minutes max
**Audience focus:** the end user — how someone actually runs and interacts with the pipeline
**Format:** staged walkthrough. Each stage has an *Objective*, *Commands* (where needed), and a *Transcript* (what you say out loud, in natural spoken language).

---

## Setup before you hit record (do NOT count against the 5 minutes)

Get these ready so the demo flows without waiting:

- Virtual environment already created and activated (`.venv`).
- Dependencies installed (`pip install -r requirements.txt`) and TextBlob corpora downloaded (`python -m textblob.download_corpora`).
- **Install the package so the console scripts are on PATH:** `pip install -e .` from the repo root. This registers the commands `surge-label`, `surge-train`, `surge-cross-val`, `surge-figures`, and `surge-examples`.
- A terminal open **at the repo root** (console scripts run from anywhere, so paths below are repo-root-relative).
- A file explorer or editor open on the repo root so you can show `input/`, `output/`, and `src/`.
- Optional: pre-run the pipeline once so cached figures/outputs exist as a fallback if a live run is slow.
- Optional sanity check: `surge-label --help` should print usage — confirms the install worked before you go live.

Time budget at a glance:

| Stage | Content | Target time |
|-------|---------|-------------|
| 1 | Intro + what problem it solves | 0:40 |
| 2 | One-time setup (show, don't run) | 0:20 |
| 3 | Project structure tour | 0:30 |
| 4 | Run the labelling pipeline (input → labelled data) | 1:20 |
| 5 | Train + evaluate the model | 1:15 |
| 6 | Read the results as a user (predictions + reproducibility) | 0:55 |
| — | **Total** | **~5:00** |

---

## Stage 1 — What this project does (0:40)

**Objective:** Name the project template up front, then in one breath set up the problem and the user's goal so everything after has context.

**Commands:** none.

**Transcript:**
> "Every so often a stock blows up on Reddit before it moves on the market — think of the GameStop frenzy. What if you could catch that wave *early*? That's what this project does: it spots emerging surge trends in Reddit stock chatter, on r/pennystocks and r/wallstreetbets.
>
> It's built on the CM3005 Data Science template, 'Predictive Modelling of Social Media Trend Emergence' — the idea of predicting when a topic takes off before it actually does. And the nice part for you as a user: you never touch the internals. You point one command at a CSV of Reddit posts, it handles the loading, the time windows, the sentiment scoring, the labelling — and out comes a model that calls surge versus no-surge. Let me show you how it's set up."

---

## Stage 2 — One-time setup (0:20)

**Objective:** Show how the tool gets installed so the console commands aren't magic. Everything here is **pre-done before recording** — you show the commands on screen and narrate them, you don't run the heavy steps live.

**Commands (show on screen, already executed):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m textblob.download_corpora
pip install -e .          # registers the surge-* console commands
```

**Transcript:**
> "Setup is standard and done once: a virtual environment, the dependencies, the TextBlob data, and the project installed in editable mode. That last step registers the pipeline as console commands — so this is exactly what a user runs after a clean install."

**Command (run live — instant):**
```powershell
surge-label --help
```

**Transcript:**
> "It's on the path and reports its options — install confirmed. Now the pipeline."

---

## Stage 3 — Project structure tour (0:30)

**Objective:** Show the user the mental model: `input/` is read-only, `output/` is generated, `src/` has the command-line entry points. Keep it fast — a scroll, not a lecture.

**Commands (optional, from repo root):**
```powershell
# Quick visual of the top level
Get-ChildItem
```

**Transcript:**
> "The structure enforces a clear split: `input` is read-only, so every result traces back to untouched source data, and `output` holds what the pipeline produces — labelled data, models, metrics, figures — each run timestamped so nothing gets overwritten.
>
> There are four commands: `surge-label` produces the labelled dataset, `surge-train` trains and evaluates, `surge-cross-val` tests generalisation to the other subreddit, and `surge-figures` regenerates the plots. The demo uses the first two."

---

## Stage 4 — Run the labelling pipeline (1:20)

**Objective:** Show the core interaction — pointing the tool at raw data and getting a labelled dataset out. This is the heart of the "how a user interacts" story.

**Commands (from repo root):**
```powershell
# Turn raw Reddit posts into a labelled dataset
surge-label --file-path input/raw/r_pennystocks_submissions_reddit.csv --output-dir output/processed --verbose
```

If a full run is too slow for the clock, run the sensitivity sweep instead — it's quick and visual:
```powershell
surge-label --file-path input/raw/r_pennystocks_submissions_reddit.csv --sweep-only
```

**Transcript:**
> "This is the core command. Given the raw CSV, it runs four stages: load the posts, extract ticker mentions, aggregate them into time windows, and score sentiment with TextBlob — then label each window surge or no-surge against a threshold, tau, default 1.5.
>
> The output is the labelled dataset, plus the class distribution, a threshold-sensitivity table, and the exact config used — so the run is reproducible and auditable. The threshold is a parameter, not a code change: `--threshold-tau 2.0` is stricter, and the sensitivity table is what makes that choice evidence-based."

> **Note for the presenter:** `surge-label` is the installed console command for the labelling pipeline. Same tool, cleaner invocation — no `python` prefix and no folder-relative paths.

---

## Stage 5 — Train and evaluate the model (1:15)

**Objective:** Show the second user command: feeding the labelled dataset into training and reading the headline metric.

**Commands (from repo root):**
```powershell
# Train Logistic Regression / Random Forest / XGBoost with temporal cross-validation
surge-train --data-path output/processed/labelled_dataset.csv --output-dir output/evaluation
```

**Transcript:**
> "The second command trains three models — logistic regression, random forest, and XGBoost — and evaluates them. Validation is temporal: always train on earlier data, test on later. This is a forecasting problem, so a random split would let the model see the future and inflate the score. Temporal validation is the honest test.
>
> On ROC-AUC the random forest reaches about 0.75 — that meets the project's target tier, and it beats both random performance and the strongest single feature alone, so it's genuinely combining signal. The figures — confusion matrix, ROC curve, threshold sensitivity — are generated automatically. Two commands, raw posts to an evaluated model."

> **Note for the presenter:** surge is a rare event, so precision/recall are low and AUC is the honest headline metric. If asked, frame it as "ranking quality on a heavily imbalanced problem," not "it predicts every surge."

---

## Stage 6 — Read the results as a user (0:55)

**Objective:** Close the loop — show the human-readable outputs that make the model's decisions understandable, plus the reproducibility story.

**Commands (optional — open files rather than run code):**
- Open `output/evaluation/` and show a `*_final_summary.json` (best model + tier achieved).
- Open `output/evaluation/prediction_examples_A2_xgboost.json` to show concrete flagged posts.
- Open `output/experiment_log.jsonl` to show the run history.

**Transcript:**
> "One AUC number isn't enough to trust a model, so the outputs are made to be inspected. The summary gives the best model and its tier. The prediction-examples file lists actual posts the model classified — hits, false positives, and missed surges — each with ticker, title, and sentiment, so the decisions are examinable, not opaque.
>
> Reproducibility is built in too: every run appends one line to an experiment log — timestamp, git commit, config, metrics — so any result traces back to the exact code that produced it. Two commands, and a result you can both verify and explain. Thank you."

---

## Fallback / Q&A cheat sheet (only if time allows)

- **"Does it work on the other subreddit?"** → `surge-cross-val` evaluates saved models on r/wallstreetbets vs r/pennystocks to test generalisation.
- **"Can I reproduce an old run exactly?"** → yes, each run saves its `pipeline_config.json`; pass it back via `--config`.
- **"Where are the charts?"** → `output/figures/evaluation/` — regenerate any time with `surge-figures`, no retraining needed.
- **"Why is precision low?"** → surges are rare (heavy class imbalance); AUC is the fair headline metric.
- **"How are these commands defined?"** → console script entry points in `pyproject.toml` under `[project.scripts]`, installed via `pip install -e .`.

## Console command reference

| Console command | Equivalent script | Purpose |
|-----------------|-------------------|---------|
| `surge-label` | `src/run_labeling.py` | Raw posts → labelled dataset |
| `surge-train` | `src/run_training.py` | Train + evaluate models |
| `surge-cross-val` | `src/run_cross_validation.py` | Cross-subreddit generalisation test |
| `surge-figures` | `src/generate_figures.py` | Rebuild evaluation figures |
| `surge-examples` | `src/generate_prediction_examples.py` | Extract TP/FP/FN prediction examples |
