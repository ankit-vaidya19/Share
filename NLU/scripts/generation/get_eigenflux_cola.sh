#!/bin/bash
# Generate EigenFlux initialization for COLA from COLA source LoRA

python get_eigenflux.py \
  --source_lora_paths "ankit-vaidya19/cola_lora_r_8" \
  --source_lora_names cola \
  --target_task_name cola \
  --num_labels 2 \
  --model_name_or_path roberta-base \
  --eigenflux_r 8 \
  --num_eigenvector_components 32 \
  --num_gram_schmidt_components 0 \
  --loading_source_index 0 \
  --output_dir ./cola_eigenflux \
  --save_eigenflux_components \
  --save_eigenflux_loadings
