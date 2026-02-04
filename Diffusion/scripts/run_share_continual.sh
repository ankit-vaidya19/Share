#!/bin/bash
# ============================================================================
# Share Continual Learning Script for Flux Diffusion Models
# ============================================================================
# For each timestep:
# 1) LoRA reconstruction is done and image generated
# 2) If previous timestep exists, all LoRAs are recalculated and images generated
# ============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================
OUTPUT_DIR="${1:-./cl_outputs}"
MODEL_NAME="${2:-black-forest-labs/FLUX.1-dev}"
NUM_COMPONENTS=32

# LoRA paths and names (modify these)
LORA_PATHS=(
    "path/to/style1_eigenflux"
    "path/to/style2_eigenflux"
    "path/to/style3_eigenflux"
)

LORA_NAMES=(
    "style1"
    "style2"
    "style3"
)

PROMPTS=(
    "A photograph in style1 aesthetic"
    "A photograph in style2 aesthetic"
    "A photograph in style3 aesthetic"
)

# ============================================================================
# Main
# ============================================================================
echo "Starting Continual Learning Pipeline for Flux"
echo "Output directory: ${OUTPUT_DIR}"

python flux.py \
    --lora_paths "${LORA_PATHS[@]}" \
    --lora_names "${LORA_NAMES[@]}" \
    --prompts "${PROMPTS[@]}" \
    --model_name_or_path "${MODEL_NAME}" \
    --num_components ${NUM_COMPONENTS} \
    --output_dir "${OUTPUT_DIR}"

echo "Pipeline complete!"
