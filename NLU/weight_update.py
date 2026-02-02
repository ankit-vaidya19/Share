import os
import sys
import argparse
from typing import Dict, List

import torch
from safetensors.torch import save_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peft import load_peft_weights
from utils import (
    combine_loras,
    get_eigenvectors,
    calculate_eigenflux,
    get_reconstruction,
    replace_key,
)


def combine_reconstructions(
    recons_dict: Dict[str, Dict[str, torch.Tensor]],
    state_dict: Dict[str, torch.Tensor],
    key_name: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Combine multiple reconstruction state dictionaries into a single dictionary.

    Similar to combine_loras but for reconstruction tensors (recons_A/recons_B).

    Args:
        recons_dict: Existing dictionary of combined reconstructions, organized by layer.
        state_dict: State dictionary of the reconstruction to add.
        key_name: Name identifier for this reconstruction.

    Returns:
        Updated recons_dict with the new reconstruction's weights added.
    """
    for key, value in state_dict.items():
        if "classifier" in key:
            continue
        try:
            recons_dict[key].update({key_name: value})
        except KeyError:
            recons_dict[key] = {key_name: value}
    return recons_dict


def update_eigenflux_weights(
    previous_weights: List[Dict[str, torch.Tensor]],
    previous_names: List[str],
    current_weights: Dict[str, torch.Tensor],
    current_name: str,
    num_components: int,
    unwind_tensor: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Update eigenflux weights by recomputing eigenvectors from all reconstructions.

    This function:
    1. Takes previous eigenflux weights and current eigenflux weights
    2. Reconstructs all of them using get_reconstruction
    3. Concatenates all reconstructions and performs eigendecomposition
    4. For previous weights only, calculates new eigenfluxes using new eigenvectors

    Args:
        previous_weights: List of previous eigenflux weight state dictionaries.
        previous_names: List of names for each previous weight set.
        current_weights: Current eigenflux weight state dictionary.
        current_name: Name for the current weight set.
        num_components: Number of principal components to retain.
        unwind_tensor: If True, flatten weight matrices to vectors before PCA.

    Returns:
        Dictionary mapping previous names to their updated eigenflux state dicts.
    """
    print("=" * 60)
    print("Updating EigenFlux Weights")
    print("=" * 60)

    # Step 1: Reconstruct all eigenflux weights
    print("Step 1: Reconstructing all eigenflux weights...")
    all_reconstructions = {}

    # Reconstruct previous weights
    for prev_weights, prev_name in zip(previous_weights, previous_names):
        print(f"  Reconstructing: {prev_name}")
        recons = get_reconstruction(prev_weights)
        all_reconstructions[prev_name] = recons

    # Reconstruct current weights
    print(f"  Reconstructing: {current_name}")
    current_recons = get_reconstruction(current_weights)
    all_reconstructions[current_name] = current_recons

    # Step 2: Combine all reconstructions for eigendecomposition
    print("Step 2: Combining reconstructions for eigendecomposition...")
    combined_recons = {}
    for name, recons_sd in all_reconstructions.items():
        combined_recons = combine_reconstructions(combined_recons, recons_sd, name)

    # Step 3: Compute new eigenvectors from combined reconstructions
    print("Step 3: Computing new eigenvectors from combined reconstructions...")
    new_eigenvectors = get_eigenvectors(combined_recons, unwind_tensor)

    # Step 4: For previous weights only, calculate new eigenfluxes
    print("Step 4: Calculating new eigenfluxes for previous weights...")
    updated_weights = {}

    for prev_name in previous_names:
        print(f"  Updating eigenflux for: {prev_name}")
        # Get the reconstruction for this previous weight (to use for loadings)
        prev_recons = all_reconstructions[prev_name]

        # Convert recons keys back to lora format for calculate_eigenflux
        # recons_A -> lora_A, recons_B -> lora_B
        lora_format_sd = {}
        for k, v in prev_recons.items():
            if "recons_A" in k:
                new_key = replace_key(k, "recons_A", "lora_A.weight")
                lora_format_sd[new_key] = v
            elif "recons_B" in k:
                new_key = replace_key(k, "recons_B", "lora_B.weight")
                lora_format_sd[new_key] = v

        # Calculate new eigenflux with new eigenvectors
        # Need to also convert eigenvector keys to match lora format
        eig_dict_lora_format = {}
        for k, v in new_eigenvectors.items():
            if "recons_A" in k:
                new_key = replace_key(k, "recons_A", "lora_A.weight")
                eig_dict_lora_format[new_key] = v
            elif "recons_B" in k:
                new_key = replace_key(k, "recons_B", "lora_B.weight")
                eig_dict_lora_format[new_key] = v

        new_eigenflux_sd = calculate_eigenflux(
            eig_dict_lora_format,
            lora_format_sd,
            num_components,
            loadings=True,
        )

        updated_weights[prev_name] = new_eigenflux_sd

    print("=" * 60)
    print("Weight update complete!")
    print("=" * 60)

    return updated_weights


def save_updated_weights(
    updated_weights: Dict[str, Dict[str, torch.Tensor]],
    output_dir: str,
) -> None:
    """
    Save updated eigenflux weights to disk.

    Args:
        updated_weights: Dictionary mapping names to updated eigenflux state dicts.
        output_dir: Base output directory for saving weights.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, weights in updated_weights.items():
        save_path = os.path.join(output_dir, f"{name}_updated.safetensors")
        print(f"Saving updated weights for {name} to {save_path}")
        save_file(weights, save_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update EigenFlux weights by recomputing eigenvectors from all tasks"
    )

    # Previous eigenflux weights
    parser.add_argument(
        "--previous_weight_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to previous EigenFlux adapter weights",
    )
    parser.add_argument(
        "--previous_weight_names",
        type=str,
        nargs="+",
        required=True,
        help="Names for each previous weight set (must match number of paths)",
    )

    # Current eigenflux weights
    parser.add_argument(
        "--current_weight_path",
        type=str,
        required=True,
        help="Path to current EigenFlux adapter weights",
    )
    parser.add_argument(
        "--current_weight_name",
        type=str,
        required=True,
        help="Name for the current weight set",
    )

    # EigenFlux configuration
    parser.add_argument(
        "--num_components",
        type=int,
        default=32,
        help="Number of eigenvector components to use (default: 32)",
    )

    # Processing options
    parser.add_argument(
        "--unwind_tensor",
        action="store_true",
        help="Unwind tensors when computing eigenvectors",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for updated EigenFlux weights",
    )

    args = parser.parse_args()

    # Validate arguments
    if len(args.previous_weight_paths) != len(args.previous_weight_names):
        parser.error(
            "Number of previous_weight_paths must match number of previous_weight_names"
        )

    return args


def main():
    args = parse_args()

    print("=" * 60)
    print("EigenFlux Weight Update")
    print("=" * 60)
    print(f"Previous weights: {args.previous_weight_names}")
    print(f"Current weight: {args.current_weight_name}")
    print(f"Number of components: {args.num_components}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Load previous eigenflux weights
    previous_weights = []
    for path in args.previous_weight_paths:
        print(f"Loading previous weights from: {path}")
        weights = load_peft_weights(path)
        previous_weights.append(weights)

    # Load current eigenflux weights
    print(f"Loading current weights from: {args.current_weight_path}")
    current_weights = load_peft_weights(args.current_weight_path)

    # Update eigenflux weights
    updated_weights = update_eigenflux_weights(
        previous_weights=previous_weights,
        previous_names=args.previous_weight_names,
        current_weights=current_weights,
        current_name=args.current_weight_name,
        num_components=args.num_components,
        unwind_tensor=args.unwind_tensor,
    )

    # Save updated weights
    save_updated_weights(updated_weights, args.output_dir)

    print("=" * 60)
    print("All weights updated and saved!")
    print("=" * 60)


if __name__ == "__main__":
    main()
