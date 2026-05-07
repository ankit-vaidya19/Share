"""
weight_update.py — Share ImageClassification
=============================================
After completing task T-i, update the EigenFlux adapter of task T-(i-1) so
that its components span the subspace containing BOTH tasks' reconstructions.

This implements the backward-transfer step of the Share continual-learning
algorithm for ViT image classification.

Usage
-----
python weight_update.py \
    --previous_adapter_path  ./adapters/subset_1_trained \
    --previous_adapter_name  subset_1 \
    --current_adapter_path   ./adapters/subset_2_trained \
    --current_adapter_name   subset_2 \
    --model_name             google/vit-base-patch16-224 \
    --eigenflux_r            8 \
    --num_components         32 \
    --output_dir             ./adapters/subset_1_updated
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from transformers import ViTModel
from peft import EigenFluxConfig, get_peft_model

from utils import (
    get_eigenvectors,
    get_all_reconstructions,
    build_combined_lora_dict,
    set_eigenflux_parameters,
    calculate_eigenflux,
)

# ============================================================================
# Core update logic
# ============================================================================


def load_eigenflux_model(
    model_name: str,
    adapter_path: str,
    eigenflux_r: int,
    num_components: int,
    adapter_name: str = "default",
    use_rank_updates: bool = False,
    num_rank_updates: int = 0,
) -> object:
    """
    Instantiate a ViT + EigenFlux PEFT model and load a saved adapter from disk.
    """
    base_vit = ViTModel.from_pretrained(model_name)
    config = EigenFluxConfig(
        r=eigenflux_r,
        num_components=num_components,
        use_rank_updates=use_rank_updates,
        num_rank_updates=num_rank_updates,
        target_modules=["query", "value"],
    )
    model = get_peft_model(base_vit, config, adapter_name)
    model.load_adapter(adapter_path, adapter_name, is_trainable=False)
    model.set_adapter(adapter_name)
    model.eval()
    return model


def update_previous_adapter(
    prev_model,
    curr_model,
    prev_adapter_name: str,
    curr_adapter_name: str,
    num_components: int,
    output_dir: str,
) -> None:
    """
    Recompute eigenvectors from the combined reconstructions of *prev_model*
    and *curr_model*, then update *prev_model*'s EigenFlux parameters to use
    the new, richer basis.

    Steps
    -----
    1. Reconstruct effective LoRA-A/B weights from each adapter.
    2. Combine them into a lora_dict and compute new eigenvectors.
    3. For the PREVIOUS adapter, project its reconstructed weights onto the
       new eigenvectors (compute new components & loadings).
    4. Write the updated adapter to *output_dir*.
    """
    print("=" * 60)
    print("Updating previous EigenFlux adapter")
    print("=" * 60)

    # Step 1: Collect reconstructions from both adapters.
    print("\n[1/4] Reconstructing weights from both adapters …")
    prev_recons = get_all_reconstructions(prev_model, prev_adapter_name)
    curr_recons = get_all_reconstructions(curr_model, curr_adapter_name)

    reconstructions_per_adapter = {
        prev_adapter_name: prev_recons,
        curr_adapter_name: curr_recons,
    }

    # Step 2: Build a combined lora_dict and compute new eigenvectors.
    print("\n[2/4] Computing new eigenvectors from combined reconstructions …")
    combined_lora_dict = build_combined_lora_dict(reconstructions_per_adapter)
    new_eigen_dict = get_eigenvectors(combined_lora_dict, unwind_tensor=False)

    # Step 3: Build EigenFlux state dict for the previous adapter using the
    #         NEW eigenvectors but the previous adapter's reconstructed weights
    #         as the source for loadings.
    print("\n[3/4] Projecting previous adapter onto new eigenvectors …")

    # Flatten previous reconstructions into a lora_sd-style dict for
    # calculate_eigenflux (which expects {layer_key.lora_A/B: tensor}).
    prev_lora_sd = {}
    for module_name, ab in prev_recons.items():
        # recons_A: (in_features, rank) → treated as transposed lora_A weight
        # recons_B: (out_features, rank) → treated as lora_B weight
        key_a = f"{module_name}.lora_A.{prev_adapter_name}.weight"
        key_b = f"{module_name}.lora_B.{prev_adapter_name}.weight"
        prev_lora_sd[key_a] = ab["A"]  # (in_features, rank)
        prev_lora_sd[key_b] = ab["B"]  # (out_features, rank)

    # Remap new_eigen_dict keys so they match the lora_sd key format expected
    # by calculate_eigenflux (layer_key + ".lora_A/B…").
    remapped_eigen_dict = {}
    for module_name in prev_recons.keys():
        key_a = f"{module_name}.lora_A.{prev_adapter_name}.weight"
        key_b = f"{module_name}.lora_B.{prev_adapter_name}.weight"
        # combined_lora_dict keys are: "{module_name}.lora_A" / ".lora_B"
        ck_a = f"{module_name}.lora_A"
        ck_b = f"{module_name}.lora_B"
        if ck_a in new_eigen_dict:
            remapped_eigen_dict[key_a] = new_eigen_dict[ck_a]
        if ck_b in new_eigen_dict:
            remapped_eigen_dict[key_b] = new_eigen_dict[ck_b]

    # Compute updated EigenFlux parameters.
    updated_eigenflux_sd = calculate_eigenflux(
        remapped_eigen_dict,
        prev_lora_sd,
        num_components,
        compute_loadings=True,
    )

    # The keys from calculate_eigenflux look like:
    #   "{module_name}.lora_A.{adapter_name}.eigenflux_A.components"
    # We need them in the form expected by set_eigenflux_parameters:
    #   "{module_name}.eigenflux_A.components"
    # Remap by stripping the ".lora_A/B.<adapter_name>." infix.
    import re

    remapped_eigenflux_sd = {}
    for k, v in updated_eigenflux_sd.items():
        # e.g.  "…query.lora_A.<name>.eigenflux_A.components" is already
        # processed by replace_key to be "…query.eigenflux_A.components"
        # so no further remapping is needed; just pass through.
        remapped_eigenflux_sd[k] = v

    # Step 4: Apply to the previous PEFT model and save.
    print(f"\n[4/4] Saving updated adapter to {output_dir} …")
    set_eigenflux_parameters(prev_model, remapped_eigenflux_sd, prev_adapter_name)
    os.makedirs(output_dir, exist_ok=True)
    prev_model.save_pretrained(output_dir)
    print("Saved.")


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update a previous EigenFlux adapter after learning a new task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--previous_adapter_path",
        type=str,
        required=True,
        help="Path to the PEFT adapter directory for the previous task",
    )
    parser.add_argument(
        "--previous_adapter_name",
        type=str,
        default="default",
        help="Adapter name stored in the previous adapter directory",
    )
    parser.add_argument(
        "--current_adapter_path",
        type=str,
        required=True,
        help="Path to the PEFT adapter directory for the current task",
    )
    parser.add_argument(
        "--current_adapter_name",
        type=str,
        default="default",
        help="Adapter name stored in the current adapter directory",
    )
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--eigenflux_r", type=int, default=8)
    parser.add_argument(
        "--num_components",
        type=int,
        default=32,
        help="Number of EigenFlux components (must match the trained adapters)",
    )
    parser.add_argument(
        "--use_rank_updates",
        action="store_true",
        help="Enable rank-update vectors inside EigenFlux",
    )
    parser.add_argument(
        "--num_rank_updates", type=int, default=2, help="Number of rank-update vectors"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for the updated previous-task adapter",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\nLoading PREVIOUS adapter from: {args.previous_adapter_path}")
    prev_model = load_eigenflux_model(
        args.model_name,
        args.previous_adapter_path,
        args.eigenflux_r,
        args.num_components,
        adapter_name=args.previous_adapter_name,
        use_rank_updates=args.use_rank_updates,
        num_rank_updates=args.num_rank_updates,
    )

    print(f"\nLoading CURRENT adapter from:  {args.current_adapter_path}")
    curr_model = load_eigenflux_model(
        args.model_name,
        args.current_adapter_path,
        args.eigenflux_r,
        args.num_components,
        adapter_name=args.current_adapter_name,
        use_rank_updates=args.use_rank_updates,
        num_rank_updates=args.num_rank_updates,
    )

    update_previous_adapter(
        prev_model,
        curr_model,
        prev_adapter_name=args.previous_adapter_name,
        curr_adapter_name=args.current_adapter_name,
        num_components=args.num_components,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
