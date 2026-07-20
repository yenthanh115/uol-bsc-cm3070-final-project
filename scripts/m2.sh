#!/bin/bash
set -e
export PYTHONIOENCODING=utf-8

# Machine 2: Pennystocks All (A1, B3, C2, F2)
# Run from project root: bash admin/m2.sh

echo "=== Machine 2 START: $(date) ==="

# A1 (baseline — models needed for D2 cross-dataset eval)
python src/run_labeling.py --file-path input/raw/r_pennystocks_submissions_reddit.csv --threshold-tau 1.5 --weight-sentiment 0.5 --random-seed 42 --log-file auto --notes "A1"
A1_DATA=$(python -c "import json; d=json.load(open('output/processed/latest_outputs.json')); print(d['outputs']['labelled_dataset'])")
python src/run_training.py --data-path "$A1_DATA" --models-dir output/models/A1 --seed 42 --weight-sentiment 0.5 --threshold-tau 1.5 --no-figures --log-file auto --notes "A1"

# B3 (volume-only)
python src/run_labeling.py --file-path input/raw/r_pennystocks_submissions_reddit.csv --weight-sentiment 0.0 --threshold-tau 1.5 --random-seed 42 --log-file auto --notes "B3"
B3_DATA=$(python -c "import json; d=json.load(open('output/processed/latest_outputs.json')); print(d['outputs']['labelled_dataset'])")
python src/run_training.py --data-path "$B3_DATA" --models-dir output/models/B3 --seed 42 --weight-sentiment 0.0 --threshold-tau 1.5 --no-figures --log-file auto --notes "B3"

# C2 (tau=1.0)
python src/run_labeling.py --file-path input/raw/r_pennystocks_submissions_reddit.csv --threshold-tau 1.0 --weight-sentiment 0.5 --random-seed 42 --log-file auto --notes "C2"
C2_DATA=$(python -c "import json; d=json.load(open('output/processed/latest_outputs.json')); print(d['outputs']['labelled_dataset'])")
python src/run_training.py --data-path "$C2_DATA" --models-dir output/models/C2 --seed 42 --weight-sentiment 0.5 --threshold-tau 1.0 --no-figures --log-file auto --notes "C2"

# F2 (multi-seed robustness — seeds 123, 456, 789, 2024; seed=42 is already A1)
for EXP_SEED in 123 456 789 2024; do
    NOTES="F2-seed${EXP_SEED}"
    python src/run_labeling.py --file-path input/raw/r_pennystocks_submissions_reddit.csv --random-seed $EXP_SEED --log-file auto --notes "$NOTES"
    DATA=$(python -c "import json; d=json.load(open('output/processed/latest_outputs.json')); print(d['outputs']['labelled_dataset'])")
    python src/run_training.py --data-path "$DATA" --models-dir "output/models/F2_seed${EXP_SEED}" --seed $EXP_SEED --no-figures --log-file auto --notes "$NOTES"
done

echo "=== Machine 2 COMPLETE: $(date) ==="
