#!/bin/bash
set -e
export PYTHONIOENCODING=utf-8

# Machine 4: Per-experiment figure generation.
# Run from project root: bash scripts/m4_figures.sh
#
# Produces evaluation figures for each trained experiment into a dated,
# per-experiment folder so runs never overwrite each other:
#
#     output/figures/evaluation/${TODAY}_${EXP}/
#         10_confusion_matrix_*.png
#         11_roc_curves_combined.png
#         12_classification_threshold_sensitivity_*.png
#         13_feature_importance_comparison.png
#
# This closes the gap where run_training.py (with figures enabled) writes to a
# single flat output/figures/evaluation/ directory and overwrites across
# experiments. The report references figures under
# output/figures/evaluation/[TODAY]_[EXP_ID]/ (e.g. 20260909_A2), which this
# script is what actually produces.
#
# Prerequisites: models must already be trained by m1/m2/m3.sh, so that
# output/models/${EXP}/ contains the *.joblib model files. The corresponding
# labelled dataset must also be reachable (see EXP_DATA below).

echo "=== Machine 4 (figures) START: $(date) ==="

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# TODAY tags every figure folder for this batch. Override on the command line:
#     TODAY=20260909 bash scripts/m4_figures.sh
TODAY="${TODAY:-$(date +%Y%m%d)}"

MODELS_ROOT="output/models"
FIGURES_ROOT="output/figures/evaluation"

# Experiments to render. Each entry: "EXP_ID:PHASE:SEED".
#   PHASE follows run_training's model-filename convention: phase1 when the
#   target is volume-only (weight_sentiment == 0), phase2 otherwise. Note G5
#   uses weight_sentiment=1.0, so it is phase2, not phase1.
#   SEED matches the seed the models were trained with.
EXPERIMENTS=(
  "A1:phase2:42"      # pennystocks baseline (tau=1.5, w2=0.5)
  "A2:phase2:42"      # WSB baseline (tau=1.5, w2=0.5)  -> figures used in the report
  "B1:phase1:42"      # WSB volume-only (w2=0)
  "B3:phase1:42"      # pennystocks volume-only (w2=0)
  "C1:phase2:42"      # WSB tau=1.0
  "C2:phase2:42"      # pennystocks tau=1.0
  "G2:phase2:42"      # WSB w2=0.25
  "G4:phase2:42"      # WSB w2=0.75
  "G5:phase2:42"      # WSB w2=1.0 (weight_sentiment=1.0 -> phase2)
  "F2_seed123:phase2:123"
  "F2_seed456:phase2:456"
  "F2_seed789:phase2:789"
  "F2_seed2024:phase2:2024"
)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

# Extract the model timestamp embedded in a per-experiment joblib filename.
# Filenames look like: xgboost_phase2_42_2026-07-20_09-18.joblib
# The timestamp is the trailing YYYY-MM-DD_HH-MM before the .joblib extension.
extract_timestamp() {
  local models_dir="$1"
  local phase="$2"
  local seed="$3"
  # Prefer the manifest if the training run wrote one.
  if [ -f "${models_dir}/latest_models.json" ]; then
    python -c "import json,sys; print(json.load(open('${models_dir}/latest_models.json')).get('timestamp',''))" 2>/dev/null && return 0
  fi
  # Otherwise parse it off any model file for this phase/seed.
  local sample
  sample=$(ls "${models_dir}"/*_"${phase}"_"${seed}"_*.joblib 2>/dev/null | head -n 1 || true)
  if [ -z "${sample}" ]; then
    return 1
  fi
  # Strip prefix "<name>_<phase>_<seed>_" and suffix ".joblib".
  basename "${sample}" .joblib | sed -E "s/^.*_${phase}_${seed}_//"
}

# ------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------
for ENTRY in "${EXPERIMENTS[@]}"; do
  IFS=":" read -r EXP PHASE SEED <<< "${ENTRY}"

  MODELS_DIR="${MODELS_ROOT}/${EXP}"
  FIG_DIR="${FIGURES_ROOT}/${TODAY}_${EXP}"

  if [ ! -d "${MODELS_DIR}" ]; then
    echo "  [SKIP] ${EXP}: no models dir at ${MODELS_DIR} (train it with m1/m2/m3 first)."
    continue
  fi

  TS=$(extract_timestamp "${MODELS_DIR}" "${PHASE}" "${SEED}" || true)
  if [ -z "${TS}" ]; then
    echo "  [SKIP] ${EXP}: could not resolve a ${PHASE}/seed=${SEED} model timestamp in ${MODELS_DIR}."
    continue
  fi

  # Resolve this experiment's own labelled dataset. m1/m2/m3.sh record it to
  # dataset_path.txt at training time. This is required for correctness: the
  # generate_figures.py fallback reads output/processed/latest_outputs.json,
  # which only ever points at the *last* labelling run, so every experiment
  # would otherwise be scored against the wrong dataset.
  DATA_FILE=""
  if [ -f "${MODELS_DIR}/dataset_path.txt" ]; then
    DATA_FILE=$(head -n 1 "${MODELS_DIR}/dataset_path.txt")
  fi
  if [ -z "${DATA_FILE}" ] || [ ! -f "${DATA_FILE}" ]; then
    echo "  [SKIP] ${EXP}: no valid dataset_path.txt in ${MODELS_DIR}."
    echo "         (Re-run the matching m1/m2/m3.sh, or write the labelled CSV path there.)"
    continue
  fi

  echo "  [FIG ] ${EXP}: phase=${PHASE} seed=${SEED} ts=${TS} -> ${FIG_DIR}"

  python src/generate_figures.py \
    --models-dir "${MODELS_DIR}" \
    --data-path "${DATA_FILE}" \
    --figures-dir "${FIG_DIR}" \
    --phase "${PHASE}" \
    --seed "${SEED}" \
    --timestamp "${TS}"
done

echo "=== Machine 4 (figures) COMPLETE: $(date) ==="
echo "Figures written under ${FIGURES_ROOT}/${TODAY}_<EXP>/"
