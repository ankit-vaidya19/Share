"""
get_eigenflux.py — Share ImageClassification
=============================================
Compute an EigenFlux adapter for a ViT model from one or more LoRA checkpoints
that were produced by the EigenLoRA ViTClassifier training pipeline.

Usage (bootstrap from a single LoRA checkpoint)
------------------------------------------------
python get_eigenflux.py \
    --lora_checkpoint ./checkpoints/lora/CIFAR100/model_checkpoints/subset_1_model.pth \
    --model_name google/vit-base-patch16-224 \
    --eigenflux_r 8 \
    --num_eigenvector_components 32 \
    --num_gram_schmidt_components 0 \
    --adapter_name default \
    --output_dir ./eigenflux_init/subset_1

Usage (from multiple checkpoints – richer eigenvector basis)
------------------------------------------------------------
python get_eigenflux.py \
    --lora_checkpoints ./ckpts/s1.pth ./ckpts/s2.pth \
    --lora_names subset_1 subset_2 \
    --model_name google/vit-base-patch16-224 \
    --eigenflux_r 8 \
    --num_eigenvector_components 32 \
    --num_gram_schmidt_components 16 \
    --output_dir ./eigenflux_init/multi
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from transformers import ViTModel
from peft import EigenFluxConfig, get_peft_model

from utils import (
    combine_loras,
    get_eigenvectors,
    calculate_eigenflux,
    add_gram_schmidt_vectors,
    set_eigenflux_parameters,
)


# ============================================================================
# Helpers
# ============================================================================

def _vit_prefix(key: str) -> str:
    """
    Normalise a key from a ViTClassifier checkpoint by removing the 'vit.'
    prefix so it matches the stand-alone ViT PEFT model's key space.
    """
    if key.startswith("vit."):
        return key[4:]
    return key


def load_lora_weights_from_checkpoint(path: str) -> dict:
    """
    Load a ViTClassifier .pth checkpoint and return only the LoRA weights
    (lora_A / lora_B), with the 'vit.' prefix stripped.
    """
    sd = torch.load(path, map_location="cpu")
    lora_sd = {}
    for k, v in sd.items():
        if "lora_A" in k or "lora_B" in k:
            lora_sd[_vit_prefix(k)] = v
    if not lora_sd:
        raise ValueError(f"No lora_A / lora_B keys found in checkpoint: {path}")
    print(f"  Loaded {len(lora_sd)} LoRA weight tensors from {os.path.basename(path)}")
    return lora_sd


# ============================================================================
# Core pipeline
# ============================================================================

def build_lora_dict(checkpoint_paths: list, checkpoint_names: list) -> dict:
    """
    Aggregate LoRA weights from multiple checkpoints into a
    {layer_key: {adapter_name: tensor}} dict suitable for get_eigenvectors().
    """
    lora_dict: dict = {}
    for path, name in zip(checkpoint_paths, checkpoint_names):
        sd = load_lora_weights_from_checkpoint(path)
        lora_dict = combine_loras(lora_dict, sd, name)
    print(f"Aggregated {len(checkpoint_paths)} LoRA checkpoint(s), "
          f"{len(lora_dict)} layer keys")
    return lora_dict


def compute_and_apply_eigenflux(
    eigenflux_model,
    lora_dict: dict,
    # for loadings, use the first (or only) checkpoint
    source_lora_sd: dict,
    num_components: int,
    adapter_name: str,
    compute_loadings: bool = True,
) -> None:
    """
    Run eigendecomposition on *lora_dict*, compute EigenFlux components /
    loadings from *source_lora_sd*, and apply them to *eigenflux_model*
    in-place via direct module access.
    """
    print("Computing eigenvectors …")
    eigen_dict = get_eigenvectors(lora_dict, unwind_tensor=False)

    print(f"Calculating EigenFlux ({num_components} components) …")
    eigenflux_sd = calculate_eigenflux(
        eigen_dict,
        source_lora_sd,
        num_components,
        compute_loadings=compute_loadings,
    )

    print("Applying EigenFlux parameters to PEFT model …")
    set_eigenflux_parameters(eigenflux_model, eigenflux_sd, adapter_name)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute an EigenFlux adapter from ViT LoRA checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Input: single checkpoint (convenience) ----
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Path to a single .pth LoRA checkpoint (EigenLoRA ViTClassifier format). "
             "Ignored when --lora_checkpoints is provided.",
    )

    # ---- Input: multiple checkpoints ----
    parser.add_argument(
        "--lora_checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Paths to one or more .pth LoRA checkpoints.",
    )
    parser.add_argument(
        "--lora_names",
        type=str,
        nargs="+",
        default=None,
        help="Names for each checkpoint (must match --lora_checkpoints). "
             "Defaults to subset_1, subset_2, … if not provided.",
    )

    # ---- EigenFlux configuration ----
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument(
        "--eigenflux_r", type=int, default=8,
        help="LoRA rank used inside the EigenFlux adapter",
    )
    parser.add_argument(
        "--num_eigenvector_components", type=int, default=32,
        help="Number of eigenvector principal components",
    )
    parser.add_argument(
        "--num_gram_schmidt_components", type=int, default=0,
        help="Extra random orthogonal components added via Gram-Schmidt",
    )
    parser.add_argument(
        "--loading_source_index", type=int, default=0,
        help="Which checkpoint to use for computing initial loadings. "
             "Set to -1 for random loadings.",
    )
    parser.add_argument(
        "--adapter_name", type=str, default="default",
        help="PEFT adapter name",
    )

    # ---- Output ----
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where the EigenFlux PEFT adapter will be saved",
    )

    args = parser.parse_args()

    # ---- Resolve checkpoint list ----
    if args.lora_checkpoints is None:
        if args.lora_checkpoint is None:
            parser.error("Provide --lora_checkpoint or --lora_checkpoints")
        args.lora_checkpoints = [args.lora_checkpoint]

    if args.lora_names is None:
        args.lora_names = [f"subset_{i+1}" for i in range(len(args.lora_checkpoints))]

    if len(args.lora_checkpoints) != len(args.lora_names):
        parser.error("--lora_checkpoints and --lora_names must have equal length")

    args.total_components = args.num_eigenvector_components + args.num_gram_schmidt_components
    print(f"Total components: {args.total_components} "
          f"({args.num_eigenvector_components} eigen + "
          f"{args.num_gram_schmidt_components} Gram-Schmidt)")

    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Build aggregated LoRA dict ----
    print("\n[1/4] Loading LoRA checkpoints …")
    lora_dict = build_lora_dict(args.lora_checkpoints, args.lora_names)

    # Source LoRA for computing loadings
    source_idx = min(max(0, args.loading_source_index), len(args.lora_checkpoints) - 1)
    source_lora_sd = load_lora_weights_from_checkpoint(args.lora_checkpoints[source_idx])
    compute_loadings = args.loading_source_index >= 0
    if not compute_loadings:
        print("  Keeping loadings random (loading_source_index=-1)")

    # ---- Create EigenFlux PEFT model ----
    print("\n[2/4] Creating EigenFlux PEFT model …")
    base_vit = ViTModel.from_pretrained(args.model_name)
    eigenflux_config = EigenFluxConfig(
        r=args.eigenflux_r,
        num_components=args.total_components,
        use_rank_updates=False,
        num_rank_updates=0,
        target_modules=["query", "value"],
    )
    eigenflux_model = get_peft_model(base_vit, eigenflux_config, args.adapter_name)
    eigenflux_model.print_trainable_parameters()

    # ---- Compute and apply EigenFlux parameters ----
    print("\n[3/4] Computing EigenFlux …")
    compute_and_apply_eigenflux(
        eigenflux_model,
        lora_dict,
        source_lora_sd,
        args.num_eigenvector_components,
        args.adapter_name,
        compute_loadings=compute_loadings,
    )

    # Optionally extend with Gram-Schmidt orthogonal vectors
    if args.num_gram_schmidt_components > 0:
        print(f"  Adding {args.num_gram_schmidt_components} Gram-Schmidt vectors …")
        # Build a temporary state dict with only component tensors
        comp_sd = {}
        for name, module in eigenflux_model.named_modules():
            if hasattr(module, "EigenFlux_A") and args.adapter_name in module.EigenFlux_A:
                ef_a = module.EigenFlux_A[args.adapter_name]
                ef_b = module.EigenFlux_B[args.adapter_name]
                comp_sd[f"{name}.eigenflux_A.components"] = ef_a.components.data.cpu()
                comp_sd[f"{name}.eigenflux_B.components"] = ef_b.components.data.cpu()

        extended = add_gram_schmidt_vectors(comp_sd, args.num_gram_schmidt_components)

        # Re-apply extended components
        for name, module in eigenflux_model.named_modules():
            if hasattr(module, "EigenFlux_A") and args.adapter_name in module.EigenFlux_A:
                ef_a = module.EigenFlux_A[args.adapter_name]
                ef_b = module.EigenFlux_B[args.adapter_name]
                key_ac = f"{name}.eigenflux_A.components"
                key_bc = f"{name}.eigenflux_B.components"
                if key_ac in extended:
                    ef_a.components = torch.nn.Parameter(extended[key_ac].float())
                if key_bc in extended:
                    ef_b.components = torch.nn.Parameter(extended[key_bc].float())

    # ---- Save ----
    print(f"\n[4/4] Saving EigenFlux adapter to {args.output_dir} …")
    eigenflux_model.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
