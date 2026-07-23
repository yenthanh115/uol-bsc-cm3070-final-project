# Running and Testing the Surge-Labelling Pipeline

This document provides instructions for setting up, running, and testing the project so that reviewers and collaborators can reproduce the results.

## Requirements

### Software

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or higher |
| pip | latest recommended |
| Git | any recent version |
| Operating System | Windows 10/11, macOS, or Linux |

The pipeline uses only CPU-based computation (no GPU required). The main libraries are:

- pandas, numpy, scipy (data processing)
- scikit-learn, xgboost (machine learning)
- matplotlib, seaborn (visualisation)
- textblob, vaderSentiment (sentiment analysis)
- pytest, ruff, mypy (development/testing)

### Hardware

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| Disk space | 1 GB (includes venv) | 2 GB |
| CPU | Any modern x86_64 or ARM | Multi-core for faster training |

The input datasets total approximately 260 MB (38 MB for r/pennystocks + 222 MB for r/wallstreetbets). A full labelling + training run on the larger WSB dataset takes roughly 5-10 minutes on a modern machine. The r/pennystocks dataset runs in under 2 minutes.

No internet connection is required after initial setup (all data is included in the repository).

---

## Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd uol-bsc-cm3070-final-project
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   ```

   Activate it:
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - Windows (cmd): `.venv\Scripts\activate.bat`
   - macOS / Linux: `source .venv/bin/activate`

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in editable mode** (enables CLI entry points and correct imports)

   ```bash
   pip install -e .
   ```

5. **Download TextBlob corpora** (one-time, required for sentiment analysis)

   ```bash
   python -m textblob.download_corpora
   ```

---

## Running the Pipeline

All commands assume you are in the project root with the virtual environment activated.

### Step 1: Labelling

Process raw Reddit CSV data into a labelled dataset with temporal features and surge labels.

```bash
python src/run_labeling.py \
    --file-path input/raw/r_pennystocks_submissions_reddit.csv \
    --threshold-tau 1.5 \
    --weight-sentiment 0.5 \
    --random-seed 42 \
    --verbose
```

Key flags:
- `--file-path` — path to the input CSV (either `r_pennystocks_submissions_reddit.csv` or `r_wallstreetbets_submissions_reddit.csv`)
- `--threshold-tau` — surge threshold (default 1.5)
- `--weight-volume` / `--weight-sentiment` — composite score weights (default 0.5 each)
- `--random-seed` — seed for reproducibility (default 42)
- `--log-file auto` — saves a timestamped log to `output/logs/`

Output is written to `output/processed/`.

### Step 2: Training and Evaluation

Train classifiers (Logistic Regression, Random Forest, XGBoost) on the labelled data and evaluate them:

```bash
python src/run_training.py \
    --data-path output/processed/<timestamp>_labelled_dataset.csv \
    --models-dir output/models \
    --seed 42
```

Or use the latest output automatically:

```bash
python src/run_training.py \
    --data-path output/processed/latest_outputs.json \
    --models-dir output/models \
    --seed 42
```

Add `--no-figures` to skip generating evaluation plots and speed up the run.

Results are written to `output/evaluation/`.

### Step 3: Generate Evaluation Figures (optional)

Produce confusion matrices, ROC curves, and threshold sensitivity plots from saved models without retraining:

```bash
python src/generate_figures.py
```

Figures are saved to `output/figures/evaluation/`.

### Step 4: Cross-Dataset Validation (optional)

Test whether models trained on one subreddit generalise to another:

```bash
python src/run_cross_validation.py \
    --model-dir output/models \
    --eval-data output/processed/<labelled_dataset>.csv \
    --partition test
```

---

## Running Tests

The project uses **pytest** for unit and integration testing.

### Run all tests

```bash
python -m pytest src/tests/ -v --tb=short
```

### Run a quick check (stop on first failure)

```bash
python -m pytest src/tests/ -x -q --tb=short
```

### Run a specific test file

```bash
python -m pytest src/tests/test_labelling.py -v
```

### Run tests matching a keyword

```bash
python -m pytest src/tests/ -k "sentiment" -v
```

### Test files overview

| File | Covers |
|------|--------|
| `test_config.py` | Pipeline configuration serialisation |
| `test_loader.py` | CSV loading and ticker extraction |
| `test_windowing.py` | Temporal window count computation |
| `test_sentiment.py` | Sentiment scoring (VADER/TextBlob) |
| `test_labelling.py` | Surge labelling logic and thresholds |
| `test_features.py` | Feature engineering |
| `test_training.py` | Model training and temporal CV |
| `test_evaluation_figures.py` | Figure generation |
| `test_evaluation_significance.py` | Statistical significance testing |
| `test_pipeline_integration.py` | End-to-end pipeline integration |

---

## Linting and Type Checking

### Ruff (linting and formatting)

```bash
# Check for lint errors
ruff check src/

# Auto-fix fixable issues
ruff check src/ --fix

# Check formatting
ruff format --check src/

# Apply formatting
ruff format src/
```

### Mypy (static type checking)

```bash
mypy --config-file pyproject.toml
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: surge_pipeline` | Run `pip install -e .` from the project root |
| TextBlob errors about missing corpora | Run `python -m textblob.download_corpora` |
| Permission error on `.venv\Scripts\Activate.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell |
| Out of memory during training | Use the smaller `r_pennystocks` dataset (38 MB) instead of the WSB dataset (222 MB) |
| Tests fail with import errors | Ensure you activated the virtual environment and installed with `pip install -e .` |

---

## Quick Verification (Minimal Run)

To quickly verify everything works end-to-end with the smaller dataset:

```bash
# 1. Run labelling (approx. 1-2 minutes)
python src/run_labeling.py --file-path input/raw/r_pennystocks_submissions_reddit.csv --random-seed 42 --verbose

# 2. Train models (approx. 1-2 minutes)
python src/run_training.py --data-path output/processed/latest_outputs.json --models-dir output/models --seed 42 --no-figures

# 3. Run tests (approx. 30 seconds)
python -m pytest src/tests/ -x -q --tb=short
```

If all three steps complete without errors, the environment is correctly set up.
