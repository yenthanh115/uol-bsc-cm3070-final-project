# Surge-Labelling Pipeline for Reddit Trend Prediction

A machine learning pipeline that detects emerging "surge" trends in Reddit penny stock discussions. The system processes submission data from r/pennystocks, computes temporal windowed features and sentiment scores, then labels windows as surge/no-surge for downstream classification.

## Project Structure

```
.
├── input/                              # Everything the pipeline READS (immutable)
│   ├── raw/                            # Original Reddit CSV
│   │   └── r_pennystocks_submissions_reddit.csv
│   └── reference/                      # Static reference files (e.g., ticker list)
│
├── output/                             # Everything the pipeline WRITES (reproducible)
│   ├── processed/                      # Pipeline stage outputs
│   │   ├── labelled_dataset.csv
│   │   ├── pipeline_summary.json
│   │   ├── pipeline_config.json
│   │   └── threshold_sensitivity.csv
│   ├── models/                         # Serialised trained models
│   ├── evaluation/                     # Metrics and final summary
│   │   ├── evaluation_metrics.json
│   │   └── final_summary.json
│   ├── figures/                        # ALL generated figures
│   │   ├── eda/                        # EDA figures (01–09)
│   │   └── evaluation/                 # Model evaluation figures (10+)
│   ├── logs/                           # CLI run logs (via --log-file)
│   └── experiment_log.jsonl            # Consolidated log of all runs
│
├── src/
│   ├── surge_pipeline/                 # Core pipeline modules
│   │   ├── config.py                   # PipelineConfig dataclass (JSON-serialisable)
│   │   ├── cli_logging.py             # Tee-style CLI output logging to file
│   │   ├── loader.py                   # Data loading and ticker extraction
│   │   ├── windowing.py                # Temporal windowed count computation
│   │   ├── sentiment.py                # TextBlob sentiment scoring
│   │   ├── labelling.py                # Surge labelling and threshold sweep
│   │   ├── normalisation.py            # Z-score normalisation (train stats)
│   │   ├── features.py                 # Feature engineering for ML
│   │   ├── training.py                 # Logistic Regression with temporal CV
│   │   ├── evaluation.py              # Precision, Recall, F1, ROC-AUC evaluation
│   │   ├── experiment_log.py           # Append-only experiment log (JSONL)
│   │   └── pipeline.py                # Orchestrator (load → window → sentiment → label)
│   ├── eda/
│   │   └── eda_pipeline.py             # Exploratory data analysis with figures
│   ├── tests/                          # Unit tests
│   ├── run_labeling.py                 # CLI: run the labelling pipeline
│   ├── run_training.py                 # CLI: train model and evaluate
│   └── run_cross_validation.py         # CLI: cross-dataset generalisation test
│
├── reports/                            # Academic reports (literature review, design, etc.)
├── admin/                              # Project admin (decision log, journal)
└── README.md
```

## Prerequisites

- Python 3.10 or higher

## Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd uol-bsc-cm3070-final-project
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**

   - Windows (cmd):
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - macOS / Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Download TextBlob corpora** (required for sentiment analysis)

   ```bash
   python -m textblob.download_corpora
   ```

## Usage

All commands below assume you are in the `src/` directory and your virtual environment is activated.

```bash
cd src
```

### Run the Labelling Pipeline

Process raw Reddit data through all stages (load, window, sentiment, label):

```bash
# Using a data file
python run_labeling.py --file-path ../input/raw/r_pennystocks_submissions_reddit.csv --output-dir ../output/processed

# Using a JSON config file
python run_labeling.py --config ../output/processed/pipeline_config.json

# Run threshold sweep only (sensitivity analysis)
python run_labeling.py --file-path ../input/raw/r_pennystocks_submissions_reddit.csv --sweep-only

# Enable verbose logging
python run_labeling.py --file-path ../input/raw/r_pennystocks_submissions_reddit.csv --verbose

# Capture full CLI output to a log file (auto-timestamped)
python run_labeling.py --file-path ../input/raw/r_pennystocks_submissions_reddit.csv --verbose --log-file auto

# Capture to a specific log file
python run_labeling.py --file-path ../input/raw/r_pennystocks_submissions_reddit.csv --log-file my_run.log
```

Key parameters:

| Flag | Default | Description |
|------|---------|-------------|
| `--file-path` | *(empty — uses synthetic data)* | Path to input CSV |
| `--output-dir` | `../output/processed` | Directory for results |
| `--threshold-tau` | `1.5` | Surge threshold (tau) |
| `--temporal-split-ratio` | `0.8` | Train/test split ratio |
| `--random-seed` | `42` | Seed for reproducibility |
| `--sweep-only` | `false` | Only run threshold sweep |
| `--notes` | *(empty)* | Free-text annotation for the experiment log |
| `--log-file` | *(disabled)* | Log file path; use `auto` for timestamped file in `output/logs/` |

### Train and Evaluate the Model

Train a Logistic Regression baseline with temporal cross-validation:

```bash
python run_training.py --data-path ../output/processed/labelled_dataset.csv --output-dir ../output/evaluation

# Skip figure generation
python run_training.py --data-path ../output/processed/labelled_dataset.csv --no-figures

# With log file capture
python run_training.py --data-path ../output/processed/labelled_dataset.csv --log-file auto
```

### Run Exploratory Data Analysis

```bash
python -m eda.eda_pipeline
```

### Run Tests

```bash
# Run all tests
python -m pytest tests/

# Quick run: stop on first failure, minimal output, short tracebacks
python -m pytest tests/ -x -q --tb=short

# Save test results to a file (merges stderr into stdout)
python -m pytest tests/ -x -q --tb=short 2>&1 > test_results.txt
```

Useful pytest flags:

| Flag | Description |
|------|-------------|
| `-x` | Stop immediately on first failure |
| `-q` | Quiet mode (less verbose output) |
| `--tb=short` | Shortened traceback on errors |
| `-v` | Verbose mode (show each test name) |
| `-k "keyword"` | Run only tests matching keyword |

## Configuration

The pipeline is configured via `PipelineConfig` (defined in `src/surge_pipeline/config.py`). You can pass parameters on the command line or provide a JSON config file:

```json
{
  "file_path": "../input/raw/r_pennystocks_submissions_reddit.csv",
  "output_dir": "../output/processed",
  "temporal_split_ratio": 0.8,
  "min_window_count": 3,
  "threshold_tau": 1.5,
  "weight_volume": 0.5,
  "weight_sentiment": 0.5,
  "thresholds": [0.5, 1.0, 1.5, 2.0, 2.5],
  "random_seed": 42
}
```

## Outputs

After a full pipeline run, the following files are produced in `output/processed/`:

- `labelled_dataset.csv` — Full labelled dataset with features and surge labels
- `pipeline_summary.json` — Normalisation parameters, class distributions, stage counts
- `threshold_sensitivity.csv` — Surge rate and viability at each threshold
- `pipeline_config.json` — Exact config used (audit trail)

After training, results are written to `output/evaluation/`:

- `evaluation_metrics.json` — Precision, Recall, F1, ROC-AUC results
- `final_summary.json` — Best model, tier achieved, statistical comparisons

Generated figures are saved under `output/figures/`:

- `output/figures/eda/` — EDA visualisations (posting frequency, ticker distributions, etc.)
- `output/figures/evaluation/` — Model evaluation plots (confusion matrix, ROC curve, threshold sensitivity)

## CLI Log Files

When `--log-file auto` is used, a timestamped log capturing the full CLI output (stdout + stderr + Python logging) is saved to `output/logs/`. This provides a traceable record of each run without needing to scroll back through terminal history.

| `--log-file` value | Behaviour |
|--------------------|-----------|
| *(omitted)* | No log file (console only) |
| `auto` | Auto-named: `output/logs/YYYY-MM-DD_HH-MM-SS_<pipeline>.log` |
| `myrun.log` | Named file placed in `output/logs/` |
| `../path/to/file.log` | Explicit path (parent dirs created automatically) |

Each log file includes a header with the exact command, timestamp, and Python version for reproducibility.

## Experiment Log

Every pipeline and training run automatically appends a record to `output/experiment_log.jsonl`. This provides a single, consolidated history of all experiments without needing to traverse individual output files.

### Format

The file uses [JSON Lines](https://jsonlines.org/) format (one JSON object per line):

```jsonl
{"run_id": "202607111714", "pipeline": "labelling", "timestamp": "2026-07-11T17:14:00", "git_sha": "a1b2c3d", "config": {"threshold_tau": 1.5, "weight_volume": 0.5, ...}, "outputs": ["202607111714_labelled_dataset.csv", ...], "summary": {"surge_rate": 2.81, ...}}
{"run_id": "202607112009", "pipeline": "training", "timestamp": "2026-07-11T20:09:00", "git_sha": "a1b2c3d", "config": {"seed": 42, ...}, "outputs": ["202607112009_evaluation_metrics.json", ...], "summary": {"best_model": "xgboost", "best_auc_roc": 0.72, ...}}
```

### Fields

| Field | Description |
|-------|-------------|
| `run_id` | Timestamp prefix linking to output filenames |
| `pipeline` | Stage identifier: `labelling`, `training`, or `sweep` |
| `timestamp` | ISO 8601 timestamp of the run |
| `git_sha` | Short git commit hash (for reproducibility) |
| `config` | Key hyperparameters used |
| `outputs` | List of generated output file paths |
| `summary` | Key metrics for quick comparison |
| `notes` | Optional free-text annotation (via `--notes` flag) |

### Querying the log

Load the experiment history into a pandas DataFrame for analysis:

```python
import pandas as pd

log = pd.read_json("output/experiment_log.jsonl", lines=True)
log[log["pipeline"] == "training"].sort_values("timestamp")
```

### Annotating runs

Use the `--notes` flag on either runner to tag experiments:

```bash
python run_labeling.py --config config.json --notes "testing tau=2.0 with textblob"
python run_training.py --notes "phase2 with higher sentiment weight"
```

### Cross-Dataset Validation

Test whether trained models generalise to a different subreddit:

```bash
# Evaluate WSB-trained models on r/pennystocks test set
python run_cross_validation.py \
    --model-dir ../output/models \
    --eval-data ../output/processed/labelled_dataset.csv \
    --partition test \
    --notes "Cross-val: WSB models on pennystocks"

# Evaluate on all non-excluded records (train + test)
python run_cross_validation.py \
    --model-dir ../output/models \
    --eval-data ../output/processed/labelled_dataset.csv \
    --notes "Cross-val: WSB models on pennystocks full"
```

Key parameters:

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | `../output/models` | Directory containing saved `.joblib` model files |
| `--eval-data` | *(required)* | Path to the labelled dataset CSV to evaluate on |
| `--partition` | *(all non-excluded)* | Partition to evaluate: `train`, `test`, or omit for all |
| `--output-dir` | `../output/evaluation` | Directory for cross-validation results JSON |
| `--notes` | *(empty)* | Free-text annotation for the experiment log |

The script loads pre-trained models (with their scalers), computes features on the target dataset, and reports AUC, precision, recall, and F1 for each model. Results are saved as a timestamped JSON file in the evaluation output directory.

## License

See [LICENSE](LICENSE) for details.
