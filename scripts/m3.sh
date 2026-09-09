#!/bin/bash
set -e
export PYTHONIOENCODING=utf-8

# Machine 3: WSB Weight Sensitivity Sweep (G2, G4, G5)
# Run from project root: bash scripts/m3.sh
#
# After each experiment trains, the labelled-dataset path is recorded to
# output/models/<EXP>/dataset_path.txt so scripts/m4_figures.sh can generate
# per-experiment figures against the correct dataset.

echo "=== Machine 3 START: $(date) ==="

# Record the just-produced labelled dataset for a given experiment id.
record_dataset() {
  local exp="$1"
  local data="$2"
  mkdir -p "output/models/${exp}"
  printf '%s\n' "${data}" > "output/models/${exp}/dataset_path.txt"
}

# G2 (w_vol=0.75, w_sent=0.25)
python src/run_labeling.py --file-path input/raw/r_wallstreetbets_submissions_reddit.csv --weight-volume 0.75 --weight-sentiment 0.25 --threshold-tau 1.5 --random-seed 42 --log-file auto --notes "G2"
DATA=$(python -c "import json; d=json.load(open('output/processed/latest_outputs.json')); print(d['outputs']['labelled_dataset'])")
python src/run_training.py --data-path "$DATA" --models-dir output/models/G2 --seed 42 --weight-sentiment 0.25 --threshold-tau 1.5 --no-figures --log-file auto --notes "G2"
record_dataset "G2" "$DATA"

# G4 (w_vol=0.25, w_sent=0.75)
python src/run_labeling.py --file-path input/raw/r_wallstreetbets_submissions_reddit.csv --weight-volume 0.25 --weight-sentiment 0.75 --threshold-tau 1.5 --random-seed 42 --log-file auto --notes "G4"
DATA=$(python -c "import json; d=json.load(open('output/processed/latest_outputs.json')); print(d['outputs']['labelled_dataset'])")
python src/run_training.py --data-path "$DATA" --models-dir output/models/G4 --seed 42 --weight-sentiment 0.75 --threshold-tau 1.5 --no-figures --log-file auto --notes "G4"
record_dataset "G4" "$DATA"

# G5 (w_vol=0.0, w_sent=1.0)
python src/run_labeling.py --file-path input/raw/r_wallstreetbets_submissions_reddit.csv --weight-volume 0.0 --weight-sentiment 1.0 --threshold-tau 1.5 --random-seed 42 --log-file auto --notes "G5"
DATA=$(python -c "import json; d=json.load(open('output/processed/latest_outputs.json')); print(d['outputs']['labelled_dataset'])")
python src/run_training.py --data-path "$DATA" --models-dir output/models/G5 --seed 42 --weight-sentiment 1.0 --threshold-tau 1.5 --no-figures --log-file auto --notes "G5"
record_dataset "G5" "$DATA"

echo "=== Machine 3 COMPLETE: $(date) ==="
