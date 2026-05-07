#!/usr/bin/env bash
# =============================================================================
# run_share_imagenet_r.sh — Share continual learning on ImageNet-R (200 classes)
# =============================================================================
#
# SD-Lora-CL settings: 20-class increments (10 tasks), rank=10, lr=1e-2,
# 20 epochs per task, constant LR scheduler, weight_decay=2e-4.
#
# Data layout expected on disk:
#   <DATA_ROOT>/imagenet-r/train/<class_folder>/
#   <DATA_ROOT>/imagenet-r/test/<class_folder>/
#
# Usage:
#   ./scripts/run_share_imagenet_r.sh [BASE_OUTPUT_DIR] [DATA_ROOT]
# =============================================================================

set -euo pipefail

BASE_OUTPUT_DIR="${1:-./share_imagenet_r_outputs}"
DATA_ROOT="${2:-./data}"

MODEL_NAME="${MODEL_NAME:-google/vit-base-patch16-224}"
DATASET="imagenet_r"
SUBSET_SIZE="${SUBSET_SIZE:-20}"   # 200 classes / 20 = 10 tasks

LORA_EPOCHS="${LORA_EPOCHS:-20}"
EF_EPOCHS="${EF_EPOCHS:-20}"

LORA_LR="${LORA_LR:-1e-2}"
EF_LR="${EF_LR:-1e-2}"

BATCH_SIZE="${BATCH_SIZE:-128}"
LORA_RANK="${LORA_RANK:-10}"
EIGENFLUX_R="${EIGENFLUX_R:-8}"
NUM_COMPONENTS="${NUM_COMPONENTS:-32}"
NUM_GS_COMPONENTS="${NUM_GS_COMPONENTS:-0}"

USE_WANDB="${USE_WANDB:-}"
WANDB_PROJECT="${WANDB_PROJECT:-ViT_Share_ImageNetR}"
ADAPTER_NAME="default"

LORA_CKPT_DIR="${BASE_OUTPUT_DIR}/lora/${DATASET}/model_checkpoints"
SUBSETS_FILE="${BASE_OUTPUT_DIR}/lora/${DATASET}/sampled_subsets.txt"
EIGENFLUX_INIT_DIR="${BASE_OUTPUT_DIR}/eigenflux_init"
ADAPTERS_DIR="${BASE_OUTPUT_DIR}/adapters"
UPDATED_DIR="${BASE_OUTPUT_DIR}/updated_adapters"
LOGS_DIR="${BASE_OUTPUT_DIR}/logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

log() { echo ""; echo "============================================================================"; echo "$1"; echo "============================================================================"; echo ""; }
die() { echo "ERROR: $1" >&2; exit 1; }

mkdir -p "${LORA_CKPT_DIR}" "${EIGENFLUX_INIT_DIR}" "${ADAPTERS_DIR}" "${UPDATED_DIR}" "${LOGS_DIR}"

phase1_lora_bootstrap() {
    log "Phase 1: LoRA Bootstrap — ImageNet-R (all ${SUBSET_SIZE}-class subsets)"
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
        --weight_decay 2e-4 \
        --save_path "${BASE_OUTPUT_DIR}/lora" \
        ${USE_WANDB} --wandb_project "${WANDB_PROJECT}" \
        2>&1 | tee "${LOGS_DIR}/phase1_lora.log"
    [[ -f "${SUBSETS_FILE}" ]] || die "Subsets file not created: ${SUBSETS_FILE}"
    log "Phase 1 complete."
}

phase2_eigenflux_init() {
    log "Phase 2: EigenFlux init from Subset-1 LoRA"
    local ckpt="${LORA_CKPT_DIR}/subset_1_model.pth"
    [[ -f "${ckpt}" ]] || die "Checkpoint not found: ${ckpt}"
    python get_eigenflux.py \
        --lora_checkpoint "${ckpt}" \
        --model_name "${MODEL_NAME}" \
        --eigenflux_r "${EIGENFLUX_R}" \
        --num_eigenvector_components "${NUM_COMPONENTS}" \
        --num_gram_schmidt_components "${NUM_GS_COMPONENTS}" \
        --loading_source_index 0 \
        --adapter_name "${ADAPTER_NAME}" \
        --output_dir "${EIGENFLUX_INIT_DIR}" \
        2>&1 | tee "${LOGS_DIR}/phase2_eigenflux_init.log"
    log "Phase 2 complete."
}

train_eigenflux_subset() {
    local idx="$1" load_path="$2" save_path="$3"
    local subset_num=$((idx + 1))
    log "Phase 3 [T-${idx}]: Training subset ${subset_num} — ImageNet-R"
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
        --weight_decay 2e-4 \
        --save_path "${BASE_OUTPUT_DIR}/eigenflux" \
        ${USE_WANDB} --wandb_project "${WANDB_PROJECT}" \
        2>&1 | tee "${LOGS_DIR}/phase3_subset_${subset_num}.log"
    [[ -d "${save_path}" ]] || die "Adapter not created: ${save_path}"
}

update_previous_adapter() {
    local prev_idx="$1" curr_idx="$2"
    local prev_num=$((prev_idx + 1)) curr_num=$((curr_idx + 1))
    log "Weight update: T-${prev_idx} (subset ${prev_num}) after T-${curr_idx} (subset ${curr_num})"
    python weight_update.py \
        --previous_adapter_path "${ADAPTERS_DIR}/subset_${prev_num}_trained" \
        --previous_adapter_name "${ADAPTER_NAME}" \
        --current_adapter_path  "${ADAPTERS_DIR}/subset_${curr_num}_trained" \
        --current_adapter_name  "${ADAPTER_NAME}" \
        --model_name "${MODEL_NAME}" \
        --eigenflux_r "${EIGENFLUX_R}" \
        --num_components "${NUM_COMPONENTS}" \
        --output_dir "${UPDATED_DIR}/subset_${prev_num}_updated" \
        2>&1 | tee "${LOGS_DIR}/weight_update_T${curr_idx}_prev_T${prev_idx}.log"
}

phase3_continual_training() {
    log "Phase 3: Continual EigenFlux Training — ImageNet-R"
    local num_subsets
    num_subsets=$(wc -l < "${SUBSETS_FILE}")
    local prev_adapter_path=""
    for idx in $(seq 0 $((num_subsets - 1))); do
        local subset_num=$((idx + 1))
        local save_path="${ADAPTERS_DIR}/subset_${subset_num}_trained"
        local load_path
        if [[ ${idx} -eq 0 ]]; then load_path="${EIGENFLUX_INIT_DIR}"
        else load_path="${prev_adapter_path}"; fi
        train_eigenflux_subset "${idx}" "${load_path}" "${save_path}"
        if [[ ${idx} -ge 1 ]]; then update_previous_adapter $((idx - 1)) "${idx}"; fi
        prev_adapter_path="${save_path}"
    done
    log "Phase 3 complete."
}

main() {
    log "Share ImageNet-R Continual Learning Pipeline"
    echo "  Dataset: ${DATASET}  |  Subset size: ${SUBSET_SIZE}  |  Output: ${BASE_OUTPUT_DIR}"
    phase1_lora_bootstrap
    phase2_eigenflux_init
    phase3_continual_training
    log "Done! Adapters at ${ADAPTERS_DIR}"
}

main "$@"
