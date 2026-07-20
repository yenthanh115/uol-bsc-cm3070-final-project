#!/bin/bash
set -e
export PYTHONIOENCODING=utf-8

# Machine 1: WSB Core (A2, B1, C1)
# Run from project root: bash admin/m1.sh

echo "=== Machine 1 START: $(date) ==="

# A2
python src/run_labeling.py --file-path input/raw/r_wallstreetbets_submissions_reddit.csv --threshold-tau 1.5 --weight-sentiment 0.5 --random-seed 42 --log-file auto --notes "A2"
A2_DATA=$(python -c "import json; d=json.load(open('output/processed/latest_outputs.json')); print(d['outputs']['labelled_dataset'])")
python src/run_training.py --data-path "$A2_DATA" --models-dir output/models/A2 --seed 42 --weight-sentiment 0.5 --threshold-tau 1.5 --no-figures --log-file auto --notes "A2"

# B1
python src/run_labeling.py --file-path input/raw/r_wallstreetbets_submissions_reddit.csv --threshold-tau 1.5 --weight-sentiment 0.0 --random-seed 42 --log-file auto --notes "B1"
B1_DATA=$(python -c "import json; d=json.load(open('output/processed/latest_outputs.json')); print(d['outputs']['labelled_dataset'])")
python src/run_training.py --data-path "$B1_DATA" --models-dir output/models/B1 --seed 42 --weight-sentiment 0.0 --threshold-tau 1.5 --no-figures --log-file auto --notes "B1"

# C1
python src/run_labeling.py --file-path input/raw/r_wallstreetbets_submissions_reddit.csv --threshold-tau 1.0 --weight-sentiment 0.5 --random-seed 42 --log-file auto --notes "C1"
C1_DATA=$(python -c "import json; d=json.load(open('output/processed/latest_outputs.json')); print(d['outputs']['labelled_dataset'])")
python src/run_training.py --data-path "$C1_DATA" --models-dir output/models/C1 --seed 42 --weight-sentiment 0.5 --threshold-tau 1.0 --no-figures --log-file auto --notes "C1"

echo "=== Machine 1 COMPLETE: $(date) ==="
