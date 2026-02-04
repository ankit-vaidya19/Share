import os
import re
import argparse
from typing import Dict, List

import torch
import torch.nn as nn
from peft import load_peft_weights
from diffusers import FluxPipeline
from safetensors.torch import save_file


# ============================================================================
# Utility Functions
# ============================================================================


def replace_key(text: str, substring: str, replacement: str) -> str:
    """Replace a substring and everything after it with a replacement string."""
    pattern = re.compile(re.escape(substring) + r".*", re.DOTALL)
    return re.sub(pattern, replacement, text)


def get_reconstruction(
    eigenflux_weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Reconstruct LoRA weights from EigenFlux components and loadings."""
    recons_sd = {}
    for k in eigenflux_weights.keys():
        if "eigenflux_A.components" in k:
            components = eigenflux_weights[k]
            loadings_key = k.replace("components", "loadings")
            loadings = eigenflux_weights[loadings_key]
            recons = torch.sum(
                components.unsqueeze(0) * loadings.t().unsqueeze(1),
                dim=-1,
            ).t()
            new_key = replace_key(k, "eigenflux_A.components", "recons_A")
            recons_sd.update({new_key: recons})
        elif "eigenflux_B.components" in k:
            components = eigenflux_weights[k]
            loadings_key = k.replace("components", "loadings")
            loadings = eigenflux_weights[loadings_key]
            recons = torch.sum(
                components.unsqueeze(0) * loadings.t().unsqueeze(1),
                dim=-1,
            ).t()
            new_key = replace_key(k, "eigenflux_B.components", "recons_B")
            recons_sd.update({new_key: recons})
    return recons_sd


def combine_reconstructions(
    recons_dict: Dict[str, Dict[str, torch.Tensor]],
    state_dict: Dict[str, torch.Tensor],
    key_name: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Combine multiple reconstruction state dictionaries into a single dictionary."""
    for key, value in state_dict.items():
        if "classifier" in key:
            continue
        try:
            recons_dict[key].update({key_name: value})
        except KeyError:
            recons_dict[key] = {key_name: value}
    return recons_dict


def eigendecomposition(matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Perform eigendecomposition on a centered covariance matrix."""
    mean = matrix.mean(axis=1, keepdim=True)
    matrix = matrix - mean
    cov = torch.mm(matrix, matrix.t())
    eigenvals, eigenvecs = torch.linalg.eig(cov)
    eigenvals = eigenvals.to(torch.float32)
    eigenvecs = eigenvecs.to(torch.float32)
    eigenvals, indices = eigenvals.sort(descending=True)
    eigenvecs = eigenvecs[:, indices]
    return {"eigenvalues": eigenvals, "eigenvectors": eigenvecs}


def get_eigenvectors(
    recons_dict: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute eigenvectors from combined reconstruction weight matrices."""
    eigen_dict = {}
    for layer_key in recons_dict.keys():
        tensor_list = []
        for lora_key in recons_dict[layer_key].keys():
            tensor = recons_dict[layer_key][lora_key]
            if tensor.shape[0] < tensor.shape[1]:
                tensor = tensor.t()
            tensor_list.append(tensor)
        concat_tensors = torch.cat(tensor_list, dim=1).to(torch.float32)
        eig = eigendecomposition(concat_tensors)
        eigen_dict.update({layer_key: eig})
    return eigen_dict


def calculate_eigenflux(
    eigenvectors: Dict[str, Dict[str, torch.Tensor]],
    lora_sd: Dict[str, torch.Tensor],
    num_components: int,
) -> Dict[str, torch.Tensor]:
    """Calculate EigenFlux components and loadings from eigenvectors."""
    eigenflux_sd = {}
    for k in lora_sd.keys():
        if "lora_A" in k:
            components = nn.Parameter(
                eigenvectors[k]["eigenvectors"][:, :num_components]
            ).contiguous()
            new_key_c = replace_key(k, "lora_A", "eigenflux_A.components")
            eigenflux_sd.update({new_key_c: components})
            loading_vals = nn.Parameter(
                torch.mm(components.t(), lora_sd[k].t()).squeeze(dim=1)
            )
            new_key_l = replace_key(k, "lora_A", "eigenflux_A.loadings")
            eigenflux_sd.update({new_key_l: loading_vals})
        elif "lora_B" in k:
            components = nn.Parameter(
                eigenvectors[k]["eigenvectors"][:, :num_components]
            ).contiguous()
            new_key_c = replace_key(k, "lora_B", "eigenflux_B.components")
            eigenflux_sd.update({new_key_c: components})
            loading_vals = nn.Parameter(
                torch.mm(components.t(), lora_sd[k]).squeeze(dim=1)
            )
            new_key_l = replace_key(k, "lora_B", "eigenflux_B.loadings")
            eigenflux_sd.update({new_key_l: loading_vals})
    return eigenflux_sd


def update_eigenflux_weights(
    previous_weights: List[Dict[str, torch.Tensor]],
    previous_names: List[str],
    current_weights: Dict[str, torch.Tensor],
    current_name: str,
    num_components: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Update eigenflux weights by recomputing eigenvectors from all reconstructions."""
    print("Updating EigenFlux weights...")

    # Step 1: Reconstruct all eigenflux weights
    all_reconstructions = {}
    for prev_weights, prev_name in zip(previous_weights, previous_names):
        recons = get_reconstruction(prev_weights)
        all_reconstructions[prev_name] = recons

    current_recons = get_reconstruction(current_weights)
    all_reconstructions[current_name] = current_recons

    # Step 2: Combine all reconstructions for eigendecomposition
    combined_recons = {}
    for name, recons_sd in all_reconstructions.items():
        combined_recons = combine_reconstructions(combined_recons, recons_sd, name)

    # Step 3: Compute new eigenvectors from combined reconstructions
    new_eigenvectors = get_eigenvectors(combined_recons)

    # Step 4: For previous weights only, calculate new eigenfluxes
    updated_weights = {}
    for prev_name in previous_names:
        prev_recons = all_reconstructions[prev_name]

        # Convert recons keys back to lora format
        lora_format_sd = {}
        for k, v in prev_recons.items():
            if "recons_A" in k:
                new_key = replace_key(k, "recons_A", "lora_A.weight")
                lora_format_sd[new_key] = v
            elif "recons_B" in k:
                new_key = replace_key(k, "recons_B", "lora_B.weight")
                lora_format_sd[new_key] = v

        # Convert eigenvector keys to lora format
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
        )
        updated_weights[prev_name] = new_eigenflux_sd

    return updated_weights


# ============================================================================
# Image Generation
# ============================================================================


def generate_image(pipe, weights, prompt, output_path):
    """Load weights into pipeline and generate image."""
    # Load LoRA weights into pipeline
    pipe.load_lora_weights(weights)

    # Generate image
    image = pipe(prompt, num_inference_steps=28, guidance_scale=3.5).images[0]

    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Saved image to {output_path}")

    # Unload weights
    pipe.unload_lora_weights()


# ============================================================================
# Argument Parsing
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continual Learning Pipeline for Flux Diffusion Models"
    )

    parser.add_argument(
        "--lora_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to eigenflux weights for each timestep",
    )
    parser.add_argument(
        "--lora_names",
        type=str,
        nargs="+",
        required=True,
        help="Names for each LoRA (must match number of paths)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="Prompts for image generation (one per task)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Base Flux model",
    )
    parser.add_argument(
        "--num_components",
        type=int,
        default=32,
        help="Number of components (default: 32)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory",
    )

    args = parser.parse_args()

    if len(args.lora_paths) != len(args.lora_names):
        parser.error("Number of lora_paths must match number of lora_names")

    return args


# ============================================================================
# Main Pipeline
# ============================================================================


def main():
    args = parse_args()

    print("=" * 60)
    print("Continual Learning Pipeline for Flux")
    print("=" * 60)
    print(f"Tasks: {args.lora_names}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Load pipeline
    print("Loading Flux pipeline...")
    pipe = FluxPipeline.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    # Load all eigenflux weights
    all_weights = []
    for path in args.lora_paths:
        print(f"Loading weights from: {path}")
        weights = load_peft_weights(path)
        all_weights.append(weights)

    # Process each timestep
    for t_idx, (weights, name, prompt) in enumerate(
        zip(all_weights, args.lora_names, args.prompts)
    ):
        print(f"\n{'='*60}")
        print(f"Timestep T-{t_idx}: {name}")
        print(f"{'='*60}")

        # Step 1: Reconstruct current LoRA and generate image
        print(f"Step 1: Reconstructing {name} and generating image...")
        recons = get_reconstruction(weights)
        output_path = os.path.join(args.output_dir, f"T{t_idx}", f"{name}.png")
        generate_image(pipe, recons, prompt, output_path)

        # Step 2: If previous timesteps exist, recalculate all and generate images
        if t_idx > 0:
            print(f"Step 2: Recalculating previous LoRAs with current knowledge...")

            previous_weights = all_weights[:t_idx]
            previous_names = args.lora_names[:t_idx]

            updated_weights = update_eigenflux_weights(
                previous_weights=previous_weights,
                previous_names=previous_names,
                current_weights=weights,
                current_name=name,
                num_components=args.num_components,
            )

            # Generate images for all updated previous LoRAs
            for prev_name, prev_updated in updated_weights.items():
                prev_idx = args.lora_names.index(prev_name)
                output_path = os.path.join(
                    args.output_dir, f"T{t_idx}", f"{prev_name}_updated.png"
                )
                generate_image(pipe, prev_updated, args.prompts[prev_idx], output_path)

                # Save updated weights
                weights_path = os.path.join(
                    args.output_dir, f"T{t_idx}", f"{prev_name}_updated.safetensors"
                )
                save_file(prev_updated, weights_path)
                print(f"Saved updated weights to {weights_path}")

    print("\n" + "=" * 60)
    print("Continual Learning Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
