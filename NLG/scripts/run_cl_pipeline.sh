#!/bin/bash
# ============================================================================
# NLG Continual Learning Pipeline
# ============================================================================
# Processes 490 train LoRAs in batches of 50, incrementally computes
# eigenfluxes, and evaluates on train_subset + eval LoRAs.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NLG_DIR="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# Configuration
# ============================================================================
TRAIN_JSON="${NLG_DIR}/train.json"
TRAIN_SUBSET_JSON="${NLG_DIR}/train_subset.json"
EVAL_JSON="${NLG_DIR}/eval.json"
BATCH_SIZE=50
NUM_COMPONENTS=256
BASE_PATH="/mnt/data1/ankit/Lots_of_LoRAs/iid/"
PREFIX="Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-"
OUTPUT_DIR="${1:-${NLG_DIR}/cl_outputs}"

# ============================================================================
# Main
# ============================================================================
echo "=============================================="
echo "NLG Continual Learning Pipeline"
echo "=============================================="
echo "Train JSON: ${TRAIN_JSON}"
echo "Train Subset JSON: ${TRAIN_SUBSET_JSON}"
echo "Eval JSON: ${EVAL_JSON}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Num Components: ${NUM_COMPONENTS}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=============================================="

cd "${NLG_DIR}"

python get_eigenflux.py \
    --train_json "${TRAIN_JSON}" \
    --train_subset_json "${TRAIN_SUBSET_JSON}" \
    --eval_json "${EVAL_JSON}" \
    --batch_size ${BATCH_SIZE} \
    --num_components ${NUM_COMPONENTS} \
    --base_path "${BASE_PATH}" \
    --prefix "${PREFIX}" \
    --output_dir "${OUTPUT_DIR}"

echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
