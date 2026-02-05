"""
NLG Continual Learning Pipeline with EigenFlux

Processes 490 train LoRAs in batches of 50, incrementally computes
eigenfluxes with rank updates disabled. At each timestep:
1. Load batch T (new LoRAs) + reconstructions from T-1
2. Compute eigenvectors
3. Save eigenfluxes for ALL LoRAs seen so far
4. Save eigenvectors
5. Evaluate on train_subset and eval LoRAs
"""

import os
import json
import argparse
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from peft import load_peft_weights, LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file, load_file


# ============================================================================
# Utility Functions
# ============================================================================


def replace_key(text: str, substring: str, replacement: str) -> str:
    """Replace a substring and everything after it with a replacement string."""
    pattern = re.compile(re.escape(substring) + r".*", re.DOTALL)
    return re.sub(pattern, replacement, text)


def load_json(path: str) -> Dict:
    """Load JSON file and return dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def get_task_name(lora_path: str, prefix: str) -> str:
    """Extract task name from LoRA path."""
    match = re.search(r"task\d+", lora_path.replace(prefix, ""))
    return match.group(0) if match else lora_path.split("/")[-1]


# ============================================================================
# Batch Processing Functions
# ============================================================================


def get_lora_batches(
    train_json: str,
    batch_size: int = 50,
) -> Tuple[List[List[str]], List[str]]:
    """
    Load train.json and split into batches.

    Args:
        train_json: Path to train.json
        batch_size: Number of LoRAs per batch (default: 50)

    Returns:
        Tuple of (list of batches, full list of LoRA names)
    """
    data = load_json(train_json)
    all_loras = list(data.values())

    batches = []
    for i in range(0, len(all_loras), batch_size):
        batches.append(all_loras[i : i + batch_size])

    print(f"Loaded {len(all_loras)} LoRAs, split into {len(batches)} batches")
    return batches, all_loras


def consolidate_loras(
    lora_batch: List[str],
    base_path: str,
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """
    Concatenate all LoRAs in a batch by layer.

    Args:
        lora_batch: List of LoRA paths
        base_path: Base path where LoRAs are stored
        prefix: Prefix to remove from LoRA names

    Returns:
        Dictionary mapping layer keys to concatenated tensors
    """
    lora_dict = {}
    condensed_dict = {}

    for lora_path in tqdm(lora_batch, desc="Loading LoRAs"):
        name = lora_path.replace(prefix, "")
        path = os.path.join(base_path, f"{name}.safetensors")

        if os.path.exists(path):
            state_dict = load_file(path)
        else:
            # Try loading from HuggingFace
            state_dict = load_peft_weights(lora_path)

        for key, value in state_dict.items():
            if "classifier" in key:
                continue
            try:
                lora_dict[key].update({name: value})
            except KeyError:
                lora_dict[key] = {name: value}

    for layer_key in lora_dict.keys():
        tensor_list = []
        for lora_key in lora_dict[layer_key].keys():
            tensor = lora_dict[layer_key][lora_key].cpu()
            if tensor.shape[0] < tensor.shape[1]:
                tensor = tensor.t()
            tensor_list.append(tensor)
        concat_tensors = torch.cat(tensor_list, dim=1)
        condensed_dict.update({layer_key: concat_tensors})

    return condensed_dict


def consolidate_reconstructions(
    eigenflux_dir: str,
    task_names: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Load eigenfluxes from previous timestep and reconstruct, then consolidate.

    Args:
        eigenflux_dir: Directory containing saved eigenfluxes
        task_names: List of task names to load

    Returns:
        Dictionary mapping layer keys to concatenated reconstruction tensors
    """
    lora_dict = {}
    condensed_dict = {}

    for task_name in tqdm(task_names, desc="Loading reconstructions"):
        path = os.path.join(eigenflux_dir, task_name, "adapter_model.safetensors")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue

        eigenflux_sd = load_file(path)
        recons = get_reconstruction(eigenflux_sd)

        for key, value in recons.items():
            # Convert recons_A/B back to lora_A/B format
            if "recons_A" in key:
                new_key = replace_key(key, "recons_A", "lora_A.weight")
            elif "recons_B" in key:
                new_key = replace_key(key, "recons_B", "lora_B.weight")
            else:
                continue

            try:
                lora_dict[new_key].update({task_name: value})
            except KeyError:
                lora_dict[new_key] = {task_name: value}

    for layer_key in lora_dict.keys():
        tensor_list = []
        for lora_key in lora_dict[layer_key].keys():
            tensor = lora_dict[layer_key][lora_key].cpu()
            if tensor.shape[0] < tensor.shape[1]:
                tensor = tensor.t()
            tensor_list.append(tensor)
        concat_tensors = torch.cat(tensor_list, dim=1)
        condensed_dict.update({layer_key: concat_tensors})

    return condensed_dict


# ============================================================================
# Eigenflux Computation Functions
# ============================================================================


def get_components(
    state_dict: Dict[str, torch.Tensor],
    num_components: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute eigenvector components via SVD.

    Args:
        state_dict: Concatenated LoRA tensors by layer
        num_components: Number of components to retain

    Returns:
        Dictionary mapping layer keys to component matrices
    """
    to_return = {}
    for k in tqdm(state_dict.keys(), desc="Computing eigenvectors"):
        matrix = state_dict[k].to(torch.float32)
        mean = matrix.mean(axis=1, keepdim=True)
        matrix = matrix - mean
        U, S, V = torch.svd_lowrank(matrix, q=num_components, niter=10)
        to_return.update({k: U.contiguous()})
    return to_return


def get_loadings(
    component_dict: Dict[str, torch.Tensor],
    lora_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute eigenflux loadings for a single LoRA.

    Args:
        component_dict: Eigenvector components
        lora_sd: LoRA state dictionary

    Returns:
        EigenFlux state dictionary with components and loadings
    """
    to_return = {}
    for k in lora_sd.keys():
        if "classifier" in k:
            continue

        if "lora_A" in k:
            if k not in component_dict:
                continue
            components = component_dict[k].to(torch.float32)
            lora = lora_sd[k].t().to(torch.float32)
            loadings = torch.mm(components.t(), lora).squeeze(dim=1)

            new_key_c = replace_key(k, "lora_A", "eigenflux_A.components")
            new_key_l = replace_key(k, "lora_A", "eigenflux_A.loadings")
            to_return.update({new_key_c: components.contiguous()})
            to_return.update({new_key_l: loadings.contiguous()})

        elif "lora_B" in k:
            if k not in component_dict:
                continue
            components = component_dict[k].to(torch.float32)
            lora = lora_sd[k].to(torch.float32)
            loadings = torch.mm(components.t(), lora).squeeze(dim=1)

            new_key_c = replace_key(k, "lora_B", "eigenflux_B.components")
            new_key_l = replace_key(k, "lora_B", "eigenflux_B.loadings")
            to_return.update({new_key_c: components.contiguous()})
            to_return.update({new_key_l: loadings.contiguous()})

    return to_return


def get_reconstruction(
    eigenflux_weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct LoRA weights from EigenFlux components and loadings.

    Args:
        eigenflux_weights: EigenFlux state dictionary

    Returns:
        Reconstructed weights with recons_A/recons_B keys
    """
    recons_sd = {}
    for k in eigenflux_weights.keys():
        if "eigenflux_A.components" in k:
            components = eigenflux_weights[k].to(torch.float32)
            loadings_key = k.replace("components", "loadings")
            loadings = eigenflux_weights[loadings_key].to(torch.float32)
            recons = torch.sum(
                components.unsqueeze(0) * loadings.t().unsqueeze(1),
                dim=-1,
            ).t()
            new_key = replace_key(k, "eigenflux_A.components", "recons_A")
            recons_sd.update({new_key: recons.contiguous()})
        elif "eigenflux_B.components" in k:
            components = eigenflux_weights[k].to(torch.float32)
            loadings_key = k.replace("components", "loadings")
            loadings = eigenflux_weights[loadings_key].to(torch.float32)
            recons = torch.sum(
                components.unsqueeze(0) * loadings.t().unsqueeze(1),
                dim=-1,
            ).t()
            new_key = replace_key(k, "eigenflux_B.components", "recons_B")
            recons_sd.update({new_key: recons.contiguous()})
    return recons_sd


# ============================================================================
# Saving Functions
# ============================================================================


def save_eigenflux(
    eigenflux_sd: Dict[str, torch.Tensor],
    output_dir: str,
    task_name: str,
):
    """Save eigenflux weights to disk."""
    task_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    save_file(eigenflux_sd, os.path.join(task_dir, "adapter_model.safetensors"))


def save_eigenvectors(
    component_dict: Dict[str, torch.Tensor],
    output_dir: str,
):
    """Save eigenvector components to disk."""
    os.makedirs(output_dir, exist_ok=True)
    save_file(component_dict, os.path.join(output_dir, "eigenvectors.safetensors"))


# ============================================================================
# Main Pipeline Functions
# ============================================================================


def process_timestep(
    timestep: int,
    current_batch: List[str],
    prev_tasks: List[str],
    train_sub_loras: List[str],
    eval_loras: List[str],
    base_path: str,
    prefix: str,
    output_dir: str,
    num_components: int,
):
    """
    Process a single timestep of the continual learning pipeline.

    Args:
        timestep: Current timestep (1-indexed)
        current_batch: List of LoRA paths for this batch
        prev_tasks: List of task names from previous timesteps
        train_sub_loras: List of train_subset LoRA paths
        eval_loras: List of eval LoRA paths
        base_path: Base path for LoRA files
        prefix: Prefix to strip from LoRA paths
        output_dir: Output directory
        num_components: Number of eigenvector components
    """
    print(f"\n{'='*60}")
    print(f"TIMESTEP T{timestep}")
    print(f"{'='*60}")

    timestep_dir = os.path.join(output_dir, f"T{timestep}")
    os.makedirs(timestep_dir, exist_ok=True)

    # Step 1: Consolidate current batch
    print("\nStep 1: Loading current batch...")
    current_condensed = consolidate_loras(current_batch, base_path, prefix)

    # Step 2: If T > 1, load reconstructions from T-1 and combine
    if timestep > 1 and prev_tasks:
        print("\nStep 2: Loading reconstructions from T-1...")
        prev_dir = os.path.join(output_dir, f"T{timestep - 1}")
        prev_condensed = consolidate_reconstructions(prev_dir, prev_tasks)

        # Combine current + previous
        for key in current_condensed.keys():
            if key in prev_condensed:
                current_condensed[key] = torch.cat(
                    [prev_condensed[key], current_condensed[key]], dim=1
                )

    # Step 3: Compute eigenvectors
    print("\nStep 3: Computing eigenvectors...")
    components = get_components(current_condensed, num_components)

    # Step 4: Save eigenvectors
    print("\nStep 4: Saving eigenvectors...")
    save_eigenvectors(components, timestep_dir)

    # Step 5: Compute and save eigenfluxes for ALL LoRAs seen so far
    print("\nStep 5: Computing eigenfluxes for all LoRAs...")

    # Current batch
    current_tasks = []
    for lora_path in tqdm(current_batch, desc="Current batch"):
        task_name = get_task_name(lora_path, prefix)
        current_tasks.append(task_name)
        name = lora_path.replace(prefix, "")
        path = os.path.join(base_path, f"{name}.safetensors")

        if os.path.exists(path):
            lora_sd = load_file(path)
        else:
            lora_sd = load_peft_weights(lora_path)

        eigenflux_sd = get_loadings(components, lora_sd)
        save_eigenflux(eigenflux_sd, timestep_dir, task_name)

    # Previous tasks (re-compute with new eigenvectors)
    if prev_tasks:
        print("\nStep 6: Re-computing eigenfluxes for previous tasks...")
        prev_dir = os.path.join(output_dir, f"T{timestep - 1}")
        for task_name in tqdm(prev_tasks, desc="Previous tasks"):
            prev_path = os.path.join(prev_dir, task_name, "adapter_model.safetensors")
            if os.path.exists(prev_path):
                prev_eigenflux = load_file(prev_path)
                recons = get_reconstruction(prev_eigenflux)

                # Convert back to lora format
                lora_sd = {}
                for k, v in recons.items():
                    if "recons_A" in k:
                        new_key = replace_key(k, "recons_A", "lora_A.weight")
                    elif "recons_B" in k:
                        new_key = replace_key(k, "recons_B", "lora_B.weight")
                    else:
                        continue
                    lora_sd[new_key] = v

                eigenflux_sd = get_loadings(components, lora_sd)
                save_eigenflux(eigenflux_sd, timestep_dir, task_name)

    # Step 7: Evaluate on train_subset and eval
    print("\nStep 7: Computing eigenfluxes for train_subset and eval...")

    eval_tasks = []
    for lora_path in tqdm(train_sub_loras + eval_loras, desc="Eval LoRAs"):
        task_name = get_task_name(lora_path, prefix)
        eval_tasks.append(task_name)

        # Skip if already computed (in current batch)
        if task_name in current_tasks or task_name in prev_tasks:
            continue

        name = lora_path.replace(prefix, "")
        path = os.path.join(base_path, f"{name}.safetensors")

        if os.path.exists(path):
            lora_sd = load_file(path)
        else:
            lora_sd = load_peft_weights(lora_path)

        eigenflux_sd = get_loadings(components, lora_sd)
        save_eigenflux(eigenflux_sd, timestep_dir, task_name)

    # Return all tasks processed at this timestep
    all_tasks = list(set(current_tasks + prev_tasks))
    return all_tasks


# ============================================================================
# Argument Parsing
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="NLG Continual Learning Pipeline with EigenFlux"
    )

    parser.add_argument(
        "--train_json",
        type=str,
        default="train.json",
        help="Path to train.json with all training LoRAs",
    )
    parser.add_argument(
        "--train_subset_json",
        type=str,
        default="train_subset.json",
        help="Path to train_subset.json for evaluation",
    )
    parser.add_argument(
        "--eval_json",
        type=str,
        default="eval.json",
        help="Path to eval.json for evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of LoRAs per batch (default: 50)",
    )
    parser.add_argument(
        "--num_components",
        type=int,
        default=256,
        help="Number of eigenvector components (default: 256)",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/mnt/data1/ankit/Lots_of_LoRAs/iid/",
        help="Base path where LoRA safetensors are stored",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-",
        help="Prefix to strip from LoRA names",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cl_outputs",
        help="Output directory for eigenfluxes and eigenvectors",
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()

    print("=" * 60)
    print("NLG Continual Learning Pipeline")
    print("=" * 60)
    print(f"Train JSON: {args.train_json}")
    print(f"Train Subset JSON: {args.train_subset_json}")
    print(f"Eval JSON: {args.eval_json}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num components: {args.num_components}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # Load LoRA lists
    batches, all_train_loras = get_lora_batches(args.train_json, args.batch_size)
    train_sub_data = load_json(args.train_subset_json)
    eval_data = load_json(args.eval_json)

    train_sub_loras = list(train_sub_data.values())
    eval_loras = list(eval_data.values())

    print(f"Train subset: {len(train_sub_loras)} LoRAs")
    print(f"Eval: {len(eval_loras)} LoRAs")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each timestep
    prev_tasks = []
    for t_idx, batch in enumerate(batches):
        timestep = t_idx + 1
        prev_tasks = process_timestep(
            timestep=timestep,
            current_batch=batch,
            prev_tasks=prev_tasks,
            train_sub_loras=train_sub_loras,
            eval_loras=eval_loras,
            base_path=args.base_path,
            prefix=args.prefix,
            output_dir=args.output_dir,
            num_components=args.num_components,
        )

    print("\n" + "=" * 60)
    print("Continual Learning Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
