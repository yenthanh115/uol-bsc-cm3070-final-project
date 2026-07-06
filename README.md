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
│   └── figures/                        # ALL generated figures
│       ├── eda/                        # EDA figures (01–09)
│       └── evaluation/                 # Model evaluation figures (10+)
│
├── src/
│   ├── surge_pipeline/                 # Core pipeline modules
│   │   ├── config.py                   # PipelineConfig dataclass (JSON-serialisable)
│   │   ├── loader.py                   # Data loading and ticker extraction
│   │   ├── windowing.py                # Temporal windowed count computation
│   │   ├── sentiment.py                # TextBlob sentiment scoring
│   │   ├── labelling.py                # Surge labelling and threshold sweep
│   │   ├── normalisation.py            # Z-score normalisation (train stats)
│   │   ├── features.py                 # Feature engineering for ML
│   │   ├── training.py                 # Logistic Regression with temporal CV
│   │   ├── evaluation.py              # Precision, Recall, F1, ROC-AUC evaluation
│   │   └── pipeline.py                # Orchestrator (load → window → sentiment → label)
│   ├── eda/
│   │   └── eda_pipeline.py             # Exploratory data analysis with figures
│   ├── tests/                          # Unit tests
│   ├── run_pipeline.py                 # CLI: run the labelling pipeline
│   └── run_training.py                 # CLI: train model and evaluate
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
python run_pipeline.py --file-path ../input/raw/r_pennystocks_submissions_reddit.csv --output-dir ../output/processed

# Using a JSON config file
python run_pipeline.py --config ../output/processed/pipeline_config.json

# Run threshold sweep only (sensitivity analysis)
python run_pipeline.py --file-path ../input/raw/r_pennystocks_submissions_reddit.csv --sweep-only

# Enable verbose logging
python run_pipeline.py --file-path ../input/raw/r_pennystocks_submissions_reddit.csv --verbose
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

### Train and Evaluate the Model

Train a Logistic Regression baseline with temporal cross-validation:

```bash
python run_training.py --data-path ../output/processed/labelled_dataset.csv --output-dir ../output/evaluation

# Skip figure generation
python run_training.py --data-path ../output/processed/labelled_dataset.csv --no-figures
```

### Run Exploratory Data Analysis

```bash
python -m eda.eda_pipeline
```

### Run Tests

```bash
cd src
python -m pytest tests/
```

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

## License

See [LICENSE](LICENSE) for details.
