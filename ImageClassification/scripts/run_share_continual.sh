#!/usr/bin/env bash
# =============================================================================
# run_share_continual.sh — Share ImageClassification Continual Learning Pipeline
# =============================================================================
#
# Implements the full Share continual-learning pipeline for ViT image
# classification on class subsets.
#
# Pipeline overview
# -----------------
# Phase 1  Bootstrap
#          Train LoRA on ALL class subsets (one model per subset, sequentially).
#          This creates the LoRA checkpoints from which the initial EigenFlux
#          adapter is derived.
#
# Phase 2  EigenFlux initialisation
#          Run get_eigenflux.py on the first subset's LoRA checkpoint to create
#          an EigenFlux adapter that captures the subspace of that checkpoint.
#
# Phase 3  Continual EigenFlux training
#          For each subset T-i:
#            a) T-0: Load from Phase-2 EigenFlux init → train → save adapter.
#            b) T-i (i>0): Load from T-(i-1)'s trained adapter → train → save.
#          After training T-i (i>0):
#            – Run weight_update.py to retroactively update T-(i-1)'s adapter
#              so its eigenvector basis spans both tasks' reconstructions.
#
# Usage
# -----
#   ./scripts/run_share_continual.sh [BASE_OUTPUT_DIR] [DATASET] [SUBSET_SIZE]
#
# All positional arguments are optional; defaults are shown below.
#
# Environment variable overrides (all optional)
# ---------------------------------------------
#   DATA_ROOT         – where torchvision downloads datasets (default: ./data)
#   MODEL_NAME        – HuggingFace ViT model ID
#   LORA_EPOCHS       – epochs for the LoRA bootstrap phase
#   EF_EPOCHS         – epochs for each EigenFlux subset
#   LORA_LR / EF_LR   – learning rates
#   BATCH_SIZE
#   LORA_RANK
#   EIGENFLUX_R       – LoRA rank inside the EigenFlux adapter (≤ LORA_RANK)
#   NUM_COMPONENTS    – number of EigenFlux principal components
#   NUM_GS_COMPONENTS – extra Gram-Schmidt components (default: 0)
#   SEED
#   USE_WANDB         – set to "--use_wandb" to enable W&B logging
#   WANDB_PROJECT
# =============================================================================

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

BASE_OUTPUT_DIR="${1:-./share_continual_outputs}"
DATASET="${2:-CIFAR100}"
SUBSET_SIZE="${3:-10}"

DATA_ROOT="${DATA_ROOT:-./data}"
MODEL_NAME="${MODEL_NAME:-google/vit-base-patch16-224}"

LORA_EPOCHS="${LORA_EPOCHS:-40}"
EF_EPOCHS="${EF_EPOCHS:-40}"

LORA_LR="${LORA_LR:-5e-6}"
EF_LR="${EF_LR:-5e-4}"

BATCH_SIZE="${BATCH_SIZE:-128}"
LORA_RANK="${LORA_RANK:-8}"
EIGENFLUX_R="${EIGENFLUX_R:-8}"
NUM_COMPONENTS="${NUM_COMPONENTS:-32}"
NUM_GS_COMPONENTS="${NUM_GS_COMPONENTS:-0}"

SEED="${SEED:-42}"
USE_WANDB="${USE_WANDB:-}"          # pass "--use_wandb" to enable
WANDB_PROJECT="${WANDB_PROJECT:-ViT_Share}"

ADAPTER_NAME="default"

# Directories
LORA_CKPT_DIR="${BASE_OUTPUT_DIR}/lora/${DATASET}/model_checkpoints"
SUBSETS_FILE="${BASE_OUTPUT_DIR}/lora/${DATASET}/sampled_subsets.txt"
EIGENFLUX_INIT_DIR="${BASE_OUTPUT_DIR}/eigenflux_init"
ADAPTERS_DIR="${BASE_OUTPUT_DIR}/adapters"
UPDATED_DIR="${BASE_OUTPUT_DIR}/updated_adapters"
LOGS_DIR="${BASE_OUTPUT_DIR}/logs"

# Navigate to the ImageClassification directory regardless of where we're called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# ============================================================================
# Helpers
# ============================================================================

log() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
    echo ""
}

die() {
    echo "ERROR: $1" >&2
    exit 1
}

mkdir -p "${LORA_CKPT_DIR}" "${EIGENFLUX_INIT_DIR}" \
         "${ADAPTERS_DIR}" "${UPDATED_DIR}" "${LOGS_DIR}"

# ============================================================================
# Phase 1 — LoRA Bootstrap
# ============================================================================
phase1_lora_bootstrap() {
    log "Phase 1: LoRA Bootstrap (all subsets, method=lora)"

    python train_vit.py \
        --method lora \
        --model_name "${MODEL_NAME}" \
        --r "${LORA_RANK}" \
        --dataset "${DATASET}" \
        --data_root "${DATA_ROOT}" \
        --subset_size "${SUBSET_SIZE}" \
        --epochs "${LORA_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LORA_LR}" \
        --save_path "${BASE_OUTPUT_DIR}/lora" \
        ${USE_WANDB} \
        --wandb_project "${WANDB_PROJECT}" \
        2>&1 | tee "${LOGS_DIR}/phase1_lora.log"

    [[ -d "${LORA_CKPT_DIR}" ]] || die "LoRA checkpoint directory not created: ${LORA_CKPT_DIR}"
    [[ -f "${SUBSETS_FILE}" ]] || die "Subsets file not created: ${SUBSETS_FILE}"

    log "Phase 1 complete.  Checkpoints: ${LORA_CKPT_DIR}"
}

# ============================================================================
# Phase 2 — Compute Initial EigenFlux from Subset-1 LoRA
# ============================================================================
phase2_eigenflux_init() {
    log "Phase 2: EigenFlux initialisation from Subset-1 LoRA checkpoint"

    local subset1_ckpt="${LORA_CKPT_DIR}/subset_1_model.pth"
    [[ -f "${subset1_ckpt}" ]] || die "Subset-1 LoRA checkpoint not found: ${subset1_ckpt}"

    python get_eigenflux.py \
        --lora_checkpoint "${subset1_ckpt}" \
        --model_name "${MODEL_NAME}" \
        --eigenflux_r "${EIGENFLUX_R}" \
        --num_eigenvector_components "${NUM_COMPONENTS}" \
        --num_gram_schmidt_components "${NUM_GS_COMPONENTS}" \
        --loading_source_index 0 \
        --adapter_name "${ADAPTER_NAME}" \
        --output_dir "${EIGENFLUX_INIT_DIR}" \
        2>&1 | tee "${LOGS_DIR}/phase2_eigenflux_init.log"

    [[ -d "${EIGENFLUX_INIT_DIR}" ]] || die "EigenFlux init dir not created"
    log "Phase 2 complete.  EigenFlux init: ${EIGENFLUX_INIT_DIR}"
}

# ============================================================================
# Phase 3 — Continual EigenFlux Training
# ============================================================================

# Count the number of subsets from the file.
count_subsets() {
    wc -l < "${SUBSETS_FILE}"
}

# Train a single subset with EigenFlux.
# Args: subset_idx (0-based), load_path, save_path
train_eigenflux_subset() {
    local idx="$1"         # 0-based
    local load_path="$2"
    local save_path="$3"
    local subset_num=$((idx + 1))

    log "Phase 3 [T-${idx}]: Training subset ${subset_num} with EigenFlux"
    echo "  Load : ${load_path}"
    echo "  Save : ${save_path}"

    python train_vit.py \
        --method eigenflux \
        --model_name "${MODEL_NAME}" \
        --eigenflux_r "${EIGENFLUX_R}" \
        --num_components "${NUM_COMPONENTS}" \
        --adapter_name "${ADAPTER_NAME}" \
        --eigenflux_load_path "${load_path}" \
        --eigenflux_save_path "${save_path}" \
        --dataset "${DATASET}" \
        --data_root "${DATA_ROOT}" \
        --subset_size "${SUBSET_SIZE}" \
        --subset_index "${subset_num}" \
        --sampled_subsets_path "${SUBSETS_FILE}" \
        --epochs "${EF_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${EF_LR}" \
        --save_path "${BASE_OUTPUT_DIR}/eigenflux" \
        ${USE_WANDB} \
        --wandb_project "${WANDB_PROJECT}" \
        2>&1 | tee "${LOGS_DIR}/phase3_subset_${subset_num}.log"

    [[ -d "${save_path}" ]] || die "Adapter directory not created: ${save_path}"
}

# Update the previous subset's adapter after current subset is trained.
# Args: prev_idx (0-based), curr_idx (0-based)
update_previous_adapter() {
    local prev_idx="$1"
    local curr_idx="$2"
    local prev_num=$((prev_idx + 1))
    local curr_num=$((curr_idx + 1))

    local prev_path="${ADAPTERS_DIR}/subset_${prev_num}_trained"
    local curr_path="${ADAPTERS_DIR}/subset_${curr_num}_trained"
    local out_path="${UPDATED_DIR}/subset_${prev_num}_updated"

    log "Weight update: updating T-${prev_idx} (subset ${prev_num}) after T-${curr_idx} (subset ${curr_num})"

    python weight_update.py \
        --previous_adapter_path "${prev_path}" \
        --previous_adapter_name "${ADAPTER_NAME}" \
        --current_adapter_path  "${curr_path}" \
        --current_adapter_name  "${ADAPTER_NAME}" \
        --model_name "${MODEL_NAME}" \
        --eigenflux_r "${EIGENFLUX_R}" \
        --num_components "${NUM_COMPONENTS}" \
        --output_dir "${out_path}" \
        2>&1 | tee "${LOGS_DIR}/weight_update_T${curr_idx}_prev_T${prev_idx}.log"

    log "Updated adapter for T-${prev_idx} saved to: ${out_path}"
}

phase3_continual_training() {
    log "Phase 3: Continual EigenFlux Training"

    local num_subsets
    num_subsets=$(count_subsets)
    echo "Training ${num_subsets} subsets in sequence."

    local prev_adapter_path=""

    for idx in $(seq 0 $((num_subsets - 1))); do
        local subset_num=$((idx + 1))
        local save_path="${ADAPTERS_DIR}/subset_${subset_num}_trained"

        if [[ ${idx} -eq 0 ]]; then
            # T-0: initialise from Phase-2 EigenFlux
            local load_path="${EIGENFLUX_INIT_DIR}"
        else
            # T-i (i>0): warm-start from previous task's trained adapter
            local load_path="${prev_adapter_path}"
        fi

        train_eigenflux_subset "${idx}" "${load_path}" "${save_path}"

        # Retroactively update T-(i-1)'s adapter once T-i is trained.
        if [[ ${idx} -ge 1 ]]; then
            update_previous_adapter $((idx - 1)) "${idx}"
        fi

        prev_adapter_path="${save_path}"
        log "Completed T-${idx} (subset ${subset_num}).  Adapter at: ${save_path}"
    done

    log "Phase 3 complete."
}

# ============================================================================
# Summary
# ============================================================================
print_summary() {
    log "Share Continual Learning — Run Complete"
    echo "Dataset      : ${DATASET}"
    echo "Subset size  : ${SUBSET_SIZE}"
    echo "Output root  : ${BASE_OUTPUT_DIR}"
    echo ""
    echo "Key directories"
    echo "  LoRA checkpoints    : ${LORA_CKPT_DIR}"
    echo "  EigenFlux init      : ${EIGENFLUX_INIT_DIR}"
    echo "  Trained adapters    : ${ADAPTERS_DIR}"
    echo "  Updated adapters    : ${UPDATED_DIR}"
    echo "  Logs                : ${LOGS_DIR}"
    echo ""
    echo "Trained adapter layout"
    local num_subsets
    num_subsets=$(count_subsets)
    for i in $(seq 1 "${num_subsets}"); do
        echo "  T-$((i-1)) : ${ADAPTERS_DIR}/subset_${i}_trained"
    done
    echo ""
    echo "Updated adapter layout (backward transfer)"
    for i in $(seq 1 $((num_subsets - 1))); do
        echo "  T-$((i-1)) updated after T-$((num_subsets-1)) : ${UPDATED_DIR}/subset_${i}_updated"
    done
}

# ============================================================================
# Main
# ============================================================================
main() {
    log "Starting Share ImageClassification Continual Learning Pipeline"
    echo "  Dataset       : ${DATASET}"
    echo "  Subset size   : ${SUBSET_SIZE}"
    echo "  LoRA rank     : ${LORA_RANK}"
    echo "  EigenFlux r   : ${EIGENFLUX_R}"
    echo "  Components    : ${NUM_COMPONENTS}"
    echo "  Gram-Schmidt  : ${NUM_GS_COMPONENTS}"
    echo "  LoRA epochs   : ${LORA_EPOCHS}"
    echo "  EF epochs     : ${EF_EPOCHS}"
    echo "  Output root   : ${BASE_OUTPUT_DIR}"

    phase1_lora_bootstrap
    phase2_eigenflux_init
    phase3_continual_training
    print_summary
}

main "$@"
