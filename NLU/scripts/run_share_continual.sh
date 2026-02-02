#!/bin/bash
# ============================================================================
# Share Continual Learning Script
# ============================================================================
# This script runs the full Share (Continual) learning pipeline on GLUE tasks
# Task order: T-0:COLA -> T-1:MRPC -> T-2:RTE -> T-3:STSB -> T-4:QNLI -> T-5:SST-2
# Weights from T-{i-1} are loaded for T-{i}, enabling continual knowledge transfer
# ============================================================================

set -e  # Exit on any error

# ============================================================================
# Configuration - Modify these paths as needed
# ============================================================================
BASE_OUTPUT_DIR="${1:-./share_continual_outputs}"

# Source LoRA for eigenflux initialization (corresponds to first task in TASK_ORDER)
SOURCE_LORA_PATH="${2:-ankit-vaidya19/cola_lora_r_8}"
SOURCE_LORA_NAME="${3:-cola}"
MODEL_NAME="roberta-base"
EIGENFLUX_R=8
EIGENFLUX_NUM_COMPONENTS=32
EIGENFLUX_NUM_RANK_UPDATES=2
NUM_EIGENVECTOR_COMPONENTS=32
NUM_GRAM_SCHMIDT_COMPONENTS=0
SEED=0

# Task-specific hyperparameters
declare -A TASK_EPOCHS=(
    ["cola"]=80
    ["mrpc"]=30
    ["rte"]=40
    ["stsb"]=20
    ["qnli"]=13
    ["sst2"]=60
)

declare -A TASK_LR=(
    ["cola"]=4e-4
    ["mrpc"]=4e-4
    ["rte"]=5e-4
    ["stsb"]=4e-4
    ["qnli"]=4e-4
    ["sst2"]=5e-4
)

declare -A TASK_BATCH_SIZE=(
    ["cola"]=16
    ["mrpc"]=32
    ["rte"]=32
    ["stsb"]=32
    ["qnli"]=32
    ["sst2"]=16
)

declare -A TASK_USE_RANK_UPDATES=(
    ["cola"]=True
    ["mrpc"]=True
    ["rte"]=True
    ["stsb"]=True
    ["qnli"]=True
    ["sst2"]=True
)

declare -A TASK_NUM_LABELS=(
    ["cola"]=2
    ["mrpc"]=2
    ["rte"]=2
    ["stsb"]=2
    ["qnli"]=2
    ["sst2"]=2
)

# Task order for continual learning
TASK_ORDER=("cola" "mrpc" "rte" "stsb" "qnli" "sst2")

# ============================================================================
# Helper Functions
# ============================================================================
log() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
    echo ""
}

# ============================================================================
# Step 1: Generate EigenFlux from source LoRA
# ============================================================================
generate_eigenflux() {
    local task_name=$1
    local output_dir="${BASE_OUTPUT_DIR}/${task_name}_eigenflux"
    
    log "Generating EigenFlux for ${task_name} from ${SOURCE_LORA_NAME} source LoRA"
    
    python get_eigenflux.py \
        --source_lora_paths "${SOURCE_LORA_PATH}" \
        --source_lora_names "${SOURCE_LORA_NAME}" \
        --target_task_name "${task_name}" \
        --num_labels ${TASK_NUM_LABELS[$task_name]} \
        --model_name_or_path "${MODEL_NAME}" \
        --eigenflux_r ${EIGENFLUX_R} \
        --num_eigenvector_components ${NUM_EIGENVECTOR_COMPONENTS} \
        --num_gram_schmidt_components ${NUM_GRAM_SCHMIDT_COMPONENTS} \
        --loading_source_index 0 \
        --output_dir "${output_dir}" \
        --save_eigenflux_components \
        --save_eigenflux_loadings
    
    echo "${output_dir}"
}

# ============================================================================
# Step 2: Training Function
# ============================================================================
train_task() {
    local task_name=$1
    local load_path=$2
    local save_path=$3
    local task_idx=$4
    
    log "Training T-${task_idx}: ${task_name} (loading from: ${load_path})"
    
    python run_glue.py \
        --model_name_or_path "${MODEL_NAME}" \
        --task_name "${task_name}" \
        --do_train \
        --do_eval \
        --max_seq_length 512 \
        --per_device_train_batch_size ${TASK_BATCH_SIZE[$task_name]} \
        --seed ${SEED} \
        --learning_rate ${TASK_LR[$task_name]} \
        --lr_scheduler_type 'reduce_lr_on_plateau' \
        --weight_decay 0.1 \
        --warmup_ratio 0.06 \
        --num_train_epochs ${TASK_EPOCHS[$task_name]} \
        --evaluation_strategy epoch \
        --apply_eigenflux True \
        --eigenflux_r ${EIGENFLUX_R} \
        --eigenflux_num_components ${EIGENFLUX_NUM_COMPONENTS} \
        --eigenflux_num_rank_updates ${EIGENFLUX_NUM_RANK_UPDATES} \
        --eigenflux_use_rank_updates ${TASK_USE_RANK_UPDATES[$task_name]} \
        --eigenflux_adapter_name "${task_name}" \
        --eigenflux_load_path "${load_path}" \
        --eigenflux_save_path "${save_path}" \
        --output_dir "${BASE_OUTPUT_DIR}/training/${task_name}" \
        --overwrite_output_dir \
        --logging_dir "${BASE_OUTPUT_DIR}/logs/${task_name}" \
        --logging_steps 10 \
        --report_to wandb \
        --run_name "T${task_idx}_${task_name}"
}

# ============================================================================
# Step 3: Evaluation Function (for evaluating previous tasks at current timestep)
# ============================================================================
evaluate_task() {
    local eval_task_name=$1
    local adapter_path=$2
    local current_timestep=$3
    local eval_task_idx=$4
    
    log "Evaluating T-${eval_task_idx}: ${eval_task_name} at timestep T-${current_timestep}"
    
    python run_glue.py \
        --model_name_or_path "${MODEL_NAME}" \
        --task_name "${eval_task_name}" \
        --do_eval \
        --max_seq_length 512 \
        --per_device_eval_batch_size 32 \
        --seed ${SEED} \
        --apply_eigenflux True \
        --eigenflux_r ${EIGENFLUX_R} \
        --eigenflux_num_components ${EIGENFLUX_NUM_COMPONENTS} \
        --eigenflux_num_rank_updates 0 \
        --eigenflux_use_rank_updates False \
        --eigenflux_adapter_name "${eval_task_name}" \
        --eigenflux_load_path "${adapter_path}" \
        --output_dir "${BASE_OUTPUT_DIR}/eval/T${current_timestep}/${eval_task_name}" \
        --overwrite_output_dir \
        --logging_dir "${BASE_OUTPUT_DIR}/logs/eval_T${current_timestep}_${eval_task_name}" \
        --report_to wandb \
        --run_name "eval_T${current_timestep}_task_${eval_task_name}"
}

# ============================================================================
# Step 4: Evaluate all previous tasks at current timestep
# ============================================================================
evaluate_previous_tasks() {
    local current_timestep=$1
    local adapter_path=$2
    
    log "Evaluating all tasks (T-0 to T-${current_timestep}) at timestep T-${current_timestep}"
    
    for eval_idx in $(seq 0 ${current_timestep}); do
        eval_task="${TASK_ORDER[$eval_idx]}"
        evaluate_task "${eval_task}" "${adapter_path}" "${current_timestep}" "${eval_idx}"
    done
}

# ============================================================================
# Step 5: Update previous timestep weights after completing current timestep
# ============================================================================
update_previous_weights() {
    local current_timestep=$1
    local current_adapter_path=$2
    local prev_timestep=$((current_timestep - 1))
    
    if [ ${prev_timestep} -lt 0 ]; then
        log "No previous timestep to update (T-0 is the first)"
        return
    fi
    
    log "Updating T-${prev_timestep} weights after completing T-${current_timestep}"
    
    # Only get the T-{prev_timestep} weight (immediately previous timestep only)
    local prev_task="${TASK_ORDER[$prev_timestep]}"
    local prev_path="${BASE_OUTPUT_DIR}/${prev_task}_eigenflux_trained"
    
    # Current task info
    local current_task="${TASK_ORDER[$current_timestep]}"
    local output_dir="${BASE_OUTPUT_DIR}/T${prev_timestep}_updated"
    
    log "Running weight update: updating ${prev_task} with ${current_task}"
    
    python weight_update.py \
        --previous_weight_paths "${prev_path}" \
        --previous_weight_names "${prev_task}" \
        --current_weight_path "${current_adapter_path}" \
        --current_weight_name "${current_task}" \
        --num_components ${NUM_EIGENVECTOR_COMPONENTS} \
        --output_dir "${output_dir}"
    
    echo "Updated weights saved to: ${output_dir}"
    
    # Run inference on updated T-{prev_timestep} weight
    log "Running inference on updated T-${prev_timestep} weight"
    
    local updated_adapter_path="${output_dir}/${prev_task}_updated.safetensors"
    evaluate_task "${prev_task}" "${updated_adapter_path}" "${current_timestep}_updated" "${prev_timestep}"
}

# ============================================================================
# Main Execution
# ============================================================================
main() {
    log "Starting Share Continual Learning Pipeline"
    echo "Base output directory: ${BASE_OUTPUT_DIR}"
    echo "Source LoRA: ${SOURCE_LORA_PATH} (${SOURCE_LORA_NAME})"
    echo "Task order: ${TASK_ORDER[*]}"
    
    mkdir -p "${BASE_OUTPUT_DIR}/training"
    mkdir -p "${BASE_OUTPUT_DIR}/logs"
    mkdir -p "${BASE_OUTPUT_DIR}/eval"
    
    # Track previous task's adapter path for loading
    local prev_adapter_path=""
    
    for idx in "${!TASK_ORDER[@]}"; do
        task="${TASK_ORDER[$idx]}"
        
        if [ $idx -eq 0 ]; then
            # T-0: Generate EigenFlux from source LoRA and train
            load_path=$(generate_eigenflux "${task}")
        else
            # T-1 to T-5: Load from previous task's trained adapter
            load_path="${prev_adapter_path}"
        fi
        
        # Set save path for this task
        save_path="${BASE_OUTPUT_DIR}/${task}_eigenflux_trained"
        
        # Train the task
        train_task "${task}" "${load_path}" "${save_path}" "${idx}"
        
        # Update prev_adapter_path for next task
        prev_adapter_path="${save_path}"
        
        log "Completed training T-${idx}: ${task}"
        echo "Adapter saved to: ${save_path}"
        
        # Evaluate all previous tasks (T-0 to T-idx) using current adapter
        evaluate_previous_tasks "${idx}" "${save_path}"
        
        log "Completed evaluation at T-${idx}"
        
        # Update previous timestep weights and run inference on them
        if [ ${idx} -ge 1 ]; then
            update_previous_weights "${idx}" "${save_path}"
            log "Completed weight update and inference for T-$((idx - 1)) at T-${idx}"
        fi
    done
    
    log "Share Continual Learning Pipeline Complete!"
    echo ""
    echo "Results Summary:"
    echo "----------------"
    echo "Training outputs:"
    for idx in "${!TASK_ORDER[@]}"; do
        task="${TASK_ORDER[$idx]}"
        echo "  T-${idx} (${task}): ${BASE_OUTPUT_DIR}/training/${task}"
    done
    echo ""
    echo "Evaluation outputs (matrix format):"
    for current_idx in "${!TASK_ORDER[@]}"; do
        echo "  After T-${current_idx} (${TASK_ORDER[$current_idx]}):"
        for eval_idx in $(seq 0 ${current_idx}); do
            eval_task="${TASK_ORDER[$eval_idx]}"
            echo "    - ${eval_task}: ${BASE_OUTPUT_DIR}/eval/T${current_idx}/${eval_task}"
        done
    done
}

# ============================================================================
# Run
# ============================================================================
main "$@"
