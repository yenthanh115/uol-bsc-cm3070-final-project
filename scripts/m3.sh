#!/bin/bash
set -e

# Machine 3: WSB Weight Sensitivity Sweep (G2, G4, G5)
# Run from project root: bash admin/m3.sh

echo "=== Machine 3 START: $(date) ==="

# G2 (w_vol=0.75, w_sent=0.25)
python src/run_labeling.py --file-path input/raw/r_wallstreetbets_submissions_reddit.csv --weight-volume 0.75 --weight-sentiment 0.25 --threshold-tau 1.5 --random-seed 42 --log-file auto --notes "G2"
DATA=$(python -c "import json; d=json.load(open('output/processed/latest_outputs.json')); print(d['outputs']['labelled_dataset'])")
python src/run_training.py --data-path "$DATA" --models-dir output/models/G2 --seed 42 --weight-sentiment 0.25 --threshold-tau 1.5 --no-figures --log-file auto --notes "G2"

# G4 (w_vol=0.25, w_sent=0.75)
python src/run_labeling.py --file-path input/raw/r_wallstreetbets_submissions_reddit.csv --weight-volume 0.25 --weight-sentiment 0.75 --threshold-tau 1.5 --random-seed 42 --log-file auto --notes "G4"
DATA=$(python -c "import json; d=json.load(open('output/processed/latest_outputs.json')); print(d['outputs']['labelled_dataset'])")
python src/run_training.py --data-path "$DATA" --models-dir output/models/G4 --seed 42 --weight-sentiment 0.75 --threshold-tau 1.5 --no-figures --log-file auto --notes "G4"

# G5 (w_vol=0.0, w_sent=1.0)
python src/run_labeling.py --file-path input/raw/r_wallstreetbets_submissions_reddit.csv --weight-volume 0.0 --weight-sentiment 1.0 --threshold-tau 1.5 --random-seed 42 --log-file auto --notes "G5"
DATA=$(python -c "import json; d=json.load(open('output/processed/latest_outputs.json')); print(d['outputs']['labelled_dataset'])")
python src/run_training.py --data-path "$DATA" --models-dir output/models/G5 --seed 42 --weight-sentiment 1.0 --threshold-tau 1.5 --no-figures --log-file auto --notes "G5"

echo "=== Machine 3 COMPLETE: $(date) ==="
