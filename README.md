# Surge-Labelling Pipeline for Reddit Trend Prediction

A machine learning pipeline that detects emerging "surge" trends in Reddit stock discussions. The system processes submission data from r/pennystocks and r/wallstreetbets, computes temporal windowed features and sentiment scores, labels windows as surge/no-surge, then trains and evaluates classifiers (Logistic Regression, Random Forest, XGBoost).

## Project Structure

```
.
├── input/          # Data the pipeline READS (raw Reddit CSVs, reference files)
├── output/         # Everything the pipeline WRITES (processed data, models, figures, logs)
├── src/            # Pipeline package, CLI runners, and tests
│   ├── surge_pipeline/   # Core modules (loading, windowing, sentiment, labelling, training, evaluation)
│   ├── tests/            # Unit and integration tests
│   ├── run_labeling.py           # CLI: run the labelling pipeline
│   ├── run_training.py           # CLI: train models and evaluate
│   ├── run_cross_validation.py   # CLI: cross-dataset generalisation test
│   └── generate_figures.py       # CLI: figure generation from saved models
├── eda/            # Exploratory data analysis notebooks
└── README.md
```

## Requirements

- Python 3.10 or higher, CPU only (no GPU needed)
- ~1 GB disk (includes virtual environment); 8 GB RAM recommended

## Setup

```bash
git clone <repository-url>
cd uol-bsc-cm3070-final-project

python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Windows (cmd):        .venv\Scripts\activate.bat
# macOS / Linux:        source .venv/bin/activate

pip install -r requirements.txt
pip install -e .                    # enables CLI entry points and imports
python -m textblob.download_corpora # one-time, required for sentiment analysis
```

## Dataset

The pipeline reads two Reddit submission CSVs that are **not included in the
repository** (they total ~260 MB and are excluded via `.gitignore`). Download
them from Kaggle and place them in `input/raw/` before running any commands.

| Source (Kaggle) | File in the dataset | Rename/save to |
|---|---|---|
| [`leukipp/pennystocks`](https://www.kaggle.com/datasets/leukipp/pennystocks) | `submissions_reddit.csv` | `input/raw/r_pennystocks_submissions_reddit.csv` |
| [`leukipp/wallstreetbets`](https://www.kaggle.com/datasets/leukipp/wallstreetbets) | `submissions_reddit.csv` | `input/raw/r_wallstreetbets_submissions_reddit.csv` |

### Option A — Kaggle CLI

```bash
pip install kaggle          # already in requirements.txt
# Requires ~/.kaggle/kaggle.json (Kaggle → Account → Create New API Token)

kaggle datasets download -d leukipp/pennystocks -p input/raw --unzip
kaggle datasets download -d leukipp/  -p input/raw --unzip
# then rename each extracted submissions_reddit.csv to the names in the table above
```

### Option B — Manual download
Download each dataset from the Kaggle links above, unzip, and copy/rename submissions_reddit.csv into input/raw/ using the target names in the table.

### Synthetic fallback
Running `surge-label` with no `--file-path` (or a path that doesn't exist) generates a small synthetic dataset automatically, so the pipeline can be run end-to-end without downloading anything. Results are illustrative only.

## Usage

`pip install -e .` installs console commands (`surge-label`, `surge-train`, `surge-figures`, `surge-cross-val`) that run from the project root with the virtual environment activated. Each accepts `--help` for its full flag list.

### 1. Labelling

Process raw Reddit data through all stages (load → window → sentiment → label):

```bash
surge-label --file-path input/raw/r_pennystocks_submissions_reddit.csv
```

Common flags: `--threshold-tau` (surge threshold, default 1.5), `--weight-volume` / `--weight-sentiment` (composite score weights, default 0.5 each), `--random-seed` (default 42), `--sweep-only` (threshold sensitivity only), `--verbose`, `--log-file auto` (timestamped log in `output/logs/`), `--notes "..."` (annotate the experiment log). Output is written to `output/processed/`.

### 2. Training and Evaluation

Train Logistic Regression, Random Forest, and XGBoost with temporal cross-validation, then evaluate (per-model metrics, McNemar's significance tests, baseline comparisons, bootstrap confidence intervals):

```bash
surge-train
```

Add `--no-figures` to skip evaluation plots. Results are written to `output/evaluation/`, models to `output/models/`.

### 3. Generate Evaluation Figures (optional)

Produce confusion matrices, ROC curves, and threshold sensitivity plots from saved models without retraining:

```bash
surge-figures
```

Figures are saved to `output/figures/evaluation/`. Use `--help` for `--data-path`, `--models-dir`, `--phase`, and `--prefix` options.

### 4. Cross-Dataset Validation (optional)

Test whether models trained on one subreddit generalise to another:

```bash
surge-cross-val --model-dir output/models --eval-data output/processed/labelled_dataset.csv --partition test
```

## Configuration

The pipeline is configured via `PipelineConfig` in `src/surge_pipeline/config.py`. Pass parameters on the command line, or provide a JSON config file with `--config`:

```json
{
  "file_path": "../input/raw/r_pennystocks_submissions_reddit.csv",
  "threshold_tau": 1.5,
  "weight_volume": 0.5,
  "weight_sentiment": 0.5,
  "random_seed": 42
}
```

## Outputs

Processed data, metrics, and figures land in `output/` (see each Usage step for the specific subdirectory). Figures are split into `output/figures/eda/` and `output/figures/evaluation/`. Every run also appends one [JSON Lines](https://jsonlines.org/) record (config, git SHA, key metrics) to `output/experiment_log.jsonl`, giving a consolidated history of all experiments; load it with `pandas.read_json(..., lines=True)`.

## Testing and Code Quality

```bash
# From the project root:
python -m pytest src/tests/ -v --tb=short   # run all tests
python -m pytest src/tests/ -x -q            # stop on first failure
python -m pytest src/tests/ -k "sentiment"   # run tests matching a keyword

ruff check src/                              # lint
ruff format src/                             # format
mypy --config-file pyproject.toml            # type check
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: surge_pipeline` | Run `pip install -e .` from the project root |
| TextBlob errors about missing corpora | Run `python -m textblob.download_corpora` |
| Permission error on `Activate.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell |
| Out of memory during training | Use the smaller `r_pennystocks` dataset instead of r/wallstreetbets |
| Tests fail with import errors | Activate the virtual environment and run `pip install -e .` |

## License

See [LICENSE](LICENSE) for details.
