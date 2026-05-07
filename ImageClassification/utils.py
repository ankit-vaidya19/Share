import os
import json
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

# ============================================================================
# String / Key Utilities
# ============================================================================


def replace_key(text: str, substring: str, replacement: str) -> str:
    """Replace a substring and everything after it with a replacement string."""
    pattern = re.compile(re.escape(substring) + r".*", re.DOTALL)
    return re.sub(pattern, replacement, text)


# ============================================================================
# LoRA Aggregation
# ============================================================================


def combine_loras(
    lora_dict: Dict[str, Dict[str, torch.Tensor]],
    state_dict: Dict[str, torch.Tensor],
    key_name: str,
    noise: float = 1.0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Combine LoRA state dicts into a nested {layer_key: {adapter_name: tensor}} dict.

    Args:
        lora_dict: Existing combined dict (modified in-place).
        state_dict: A single LoRA adapter's state dict.
        key_name: Identifier for this adapter.
        noise: Optional scaling factor.

    Returns:
        Updated lora_dict.
    """
    for key, value in state_dict.items():
        if "classifier" in key:
            continue
        try:
            lora_dict[key].update({key_name: noise * value})
        except KeyError:
            lora_dict[key] = {key_name: noise * value}
    return lora_dict


# ============================================================================
# Eigendecomposition
# ============================================================================


def eigendecomposition(matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Perform eigendecomposition on a centered covariance matrix.

    Args:
        matrix: Input matrix of shape (features, samples).

    Returns:
        Dict with 'eigenvalues' and 'eigenvectors' sorted descending.
    """
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
    lora_dict: Dict[str, Dict[str, torch.Tensor]],
    unwind_tensor: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute eigenvectors from combined LoRA weight matrices.

    For each layer, concatenates the weight matrices from all adapters
    (transposing so that the longer dimension is treated as features)
    and performs eigendecomposition.

    Args:
        lora_dict: {layer_key: {adapter_name: weight_tensor}}.
        unwind_tensor: If True, flatten each weight matrix to a column vector
                       before concatenation (for very high-dimensional cases).

    Returns:
        {layer_key: {"eigenvalues": ..., "eigenvectors": ...}}.
    """
    eigen_dict = {}
    for layer_key in lora_dict.keys():
        tensor_list = []
        for lora_key in lora_dict[layer_key].keys():
            tensor = lora_dict[layer_key][lora_key]
            if unwind_tensor:
                tensor = tensor.reshape((tensor.shape[0] * tensor.shape[1], 1))
            # Ensure the feature dimension (larger) is axis 0.
            if tensor.shape[0] < tensor.shape[1]:
                tensor = tensor.t()
            tensor_list.append(tensor)
        concat_tensors = torch.cat(tensor_list, dim=1).to(torch.float32)
        eig = eigendecomposition(concat_tensors)
        eigen_dict[layer_key] = eig
    return eigen_dict


# ============================================================================
# EigenFlux Computation
# ============================================================================


def calculate_eigenflux(
    eigenvectors: Dict[str, Dict[str, torch.Tensor]],
    lora_sd: Dict[str, torch.Tensor],
    num_components: int,
    compute_loadings: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Project LoRA weights onto the principal eigenvector subspace to create
    EigenFlux components (and optionally loadings).

    Key naming convention mirrors the Share NLU approach.  For ViT LoRA
    checkpoints saved by EigenLoRA's ViTClassifier, weight keys look like::

        base_model.model.encoder.layer.0.attention.attention.query.lora_A.default.weight

    Components are written out using::

        base_model.model.encoder.layer.0.attention.attention.query.eigenflux_A.components

    Args:
        eigenvectors: {layer_key: {eigenvalues, eigenvectors}} from get_eigenvectors().
        lora_sd: The single LoRA adapter's state dict (layer_key → tensor).
        num_components: How many principal components to keep.
        compute_loadings: Whether to project the source LoRA onto the components
            to initialise the loadings (True = informed init; False = random).

    Returns:
        Flat state dict suitable for loading into an EigenFlux PEFT model.
    """
    eigenflux_sd = {}
    for k in lora_sd.keys():
        if "lora_A" in k:
            evecs = eigenvectors[k]["eigenvectors"]  # (in_features, in_features)
            # LoRA-A weight: (rank, in_features) → transposed to (in_features, rank)
            lora_A_weight = lora_sd[k]
            if lora_A_weight.shape[0] < lora_A_weight.shape[1]:
                lora_A_weight = lora_A_weight.t()  # (in_features, rank)
            components = evecs[
                :, :num_components
            ].contiguous()  # (in_features, num_components)
            new_key_c = replace_key(k, "lora_A", "eigenflux_A.components")
            eigenflux_sd[new_key_c] = components
            if compute_loadings:
                # loadings: (num_components, rank)
                loadings = torch.mm(components.t(), lora_A_weight)
                new_key_l = replace_key(k, "lora_A", "eigenflux_A.loadings")
                eigenflux_sd[new_key_l] = loadings

        elif "lora_B" in k:
            evecs = eigenvectors[k]["eigenvectors"]  # (out_features, out_features)
            # LoRA-B weight: (out_features, rank) — feature dim already first.
            lora_B_weight = lora_sd[k]
            if lora_B_weight.shape[0] < lora_B_weight.shape[1]:
                lora_B_weight = lora_B_weight.t()
            components = evecs[
                :, :num_components
            ].contiguous()  # (out_features, num_components)
            new_key_c = replace_key(k, "lora_B", "eigenflux_B.components")
            eigenflux_sd[new_key_c] = components
            if compute_loadings:
                # loadings: (num_components, rank)
                loadings = torch.mm(components.t(), lora_B_weight)
                new_key_l = replace_key(k, "lora_B", "eigenflux_B.loadings")
                eigenflux_sd[new_key_l] = loadings

    return eigenflux_sd


# ============================================================================
# Gram-Schmidt Extension
# ============================================================================


def gram_schmidt_normalization(
    matrix: torch.Tensor,
    vec: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Orthogonalise *vec* against the columns of *matrix* (Gram-Schmidt step)."""
    vec = vec.reshape(-1)
    for col in matrix.t():
        proj = (vec @ col) * col
        vec = vec - proj
    norm = torch.norm(vec)
    if norm < eps:
        raise ValueError("Vector is linearly dependent with existing basis")
    return vec / norm


def add_gram_schmidt_vectors(
    state_dict: Dict[str, torch.Tensor],
    num_random_vectors: int,
) -> Dict[str, torch.Tensor]:
    """
    Extend EigenFlux component matrices with additional random orthogonal vectors.

    Only processes keys that contain 'components'.
    """
    updated = {}
    mat_shape = None
    for k in state_dict.keys():
        if "components" in k:
            mat = state_dict[k].cpu()  # (features, num_components)
            for _ in range(num_random_vectors):
                rand_vec = torch.rand(mat.shape[0])
                new_vec = gram_schmidt_normalization(mat, rand_vec).unsqueeze(1)
                mat = torch.cat([mat, new_vec], dim=1)
            updated[k] = mat
            mat_shape = mat.shape
        else:
            updated[k] = state_dict[k]
    if mat_shape is not None:
        print(f"Components extended to {min(mat_shape)} columns")
    return updated


# ============================================================================
# Reconstruction (EigenFlux → effective LoRA weights)
# ============================================================================


def get_reconstruction_from_module(
    ef_a_module,
    ef_b_module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct the effective LoRA A and B weight matrices from EigenFlux
    module parameters.

    Returns:
        (recons_A, recons_B) where:
          - recons_A has shape (in_features, rank)   [= lora_A.weight.T]
          - recons_B has shape (out_features, rank)  [= lora_B.weight]
    """
    components_A = ef_a_module.components  # (in_features, num_components)
    loadings_A = ef_a_module.loadings  # (num_components, rank)
    recons_A = torch.sum(
        components_A.unsqueeze(0) * loadings_A.t().unsqueeze(1),
        dim=-1,
    ).t()  # (in_features, rank)

    components_B = ef_b_module.components  # (out_features, num_components)
    loadings_B = ef_b_module.loadings  # (num_components, rank)
    recons_B = torch.sum(
        components_B.unsqueeze(0) * loadings_B.t().unsqueeze(1),
        dim=-1,
    ).t()  # (out_features, rank)

    return recons_A, recons_B


def get_all_reconstructions(
    eigenflux_model,
    adapter_name: str = "default",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Walk all EigenFlux layers in *eigenflux_model* and collect reconstructed
    A and B weight tensors.

    Returns:
        {module_name: {"A": recons_A, "B": recons_B}}
    """
    reconstructions = {}
    for name, module in eigenflux_model.named_modules():
        if hasattr(module, "EigenFlux_A") and adapter_name in module.EigenFlux_A:
            ef_a = module.EigenFlux_A[adapter_name]
            ef_b = module.EigenFlux_B[adapter_name]
            with torch.no_grad():
                recons_A, recons_B = get_reconstruction_from_module(ef_a, ef_b)
            reconstructions[name] = {
                "A": recons_A.detach().cpu(),
                "B": recons_B.detach().cpu(),
            }
    return reconstructions


def build_combined_lora_dict(
    reconstructions_per_adapter: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Convert per-adapter reconstruction dicts into the lora_dict format expected
    by get_eigenvectors().

    Args:
        reconstructions_per_adapter: {adapter_label: {module_name: {"A": t, "B": t}}}

    Returns:
        {module_name + ".lora_A": {adapter_label: tensor},
         module_name + ".lora_B": {adapter_label: tensor}}
    """
    combined: Dict[str, Dict[str, torch.Tensor]] = {}
    for adapter_label, recons in reconstructions_per_adapter.items():
        for module_name, ab in recons.items():
            key_a = f"{module_name}.lora_A"
            key_b = f"{module_name}.lora_B"
            if key_a not in combined:
                combined[key_a] = {}
            if key_b not in combined:
                combined[key_b] = {}
            combined[key_a][adapter_label] = ab["A"]  # (in_features, rank)
            combined[key_b][adapter_label] = ab["B"]  # (out_features, rank)
    return combined


# ============================================================================
# Applying EigenFlux weights to PEFT model
# ============================================================================


def set_eigenflux_parameters(
    eigenflux_model,
    eigenflux_sd: Dict[str, torch.Tensor],
    adapter_name: str = "default",
) -> None:
    """
    Set EigenFlux component and loading tensors on a PEFT model directly,
    bypassing state_dict key-format issues.

    *eigenflux_sd* uses the key format produced by calculate_eigenflux():
        ``<module_name>.eigenflux_A.components``  → EigenFlux_A[adapter_name].components
        ``<module_name>.eigenflux_A.loadings``    → EigenFlux_A[adapter_name].loadings
        ``<module_name>.eigenflux_B.components``  → EigenFlux_B[adapter_name].components
        ``<module_name>.eigenflux_B.loadings``    → EigenFlux_B[adapter_name].loadings

    The function matches each PEFT module to its entry by constructing the
    corresponding key from the module's full name.
    """
    for module_name, module in eigenflux_model.named_modules():
        if not (hasattr(module, "EigenFlux_A") and adapter_name in module.EigenFlux_A):
            continue

        ef_a = module.EigenFlux_A[adapter_name]
        ef_b = module.EigenFlux_B[adapter_name]

        key_ac = f"{module_name}.eigenflux_A.components"
        key_al = f"{module_name}.eigenflux_A.loadings"
        key_bc = f"{module_name}.eigenflux_B.components"
        key_bl = f"{module_name}.eigenflux_B.loadings"

        if key_ac in eigenflux_sd:
            ef_a.components = nn.Parameter(eigenflux_sd[key_ac].clone().float())
        if key_al in eigenflux_sd:
            ef_a.loadings = nn.Parameter(eigenflux_sd[key_al].clone().float())
        if key_bc in eigenflux_sd:
            ef_b.components = nn.Parameter(eigenflux_sd[key_bc].clone().float())
        if key_bl in eigenflux_sd:
            ef_b.loadings = nn.Parameter(eigenflux_sd[key_bl].clone().float())


# ============================================================================
# Dataset Utilities  (adapted from EigenLoRA ImageClassification)
# ============================================================================


def dataloader_from_subset(
    train_set,
    test_set,
    class_indices: List[int],
    experiment_folder: str,
    subset_index: int,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/test DataLoaders restricted to *class_indices*, remapping
    labels to [0, len(class_indices)-1].

    Saves a JSON label-mapping file inside *experiment_folder/label_mappings/*.
    """
    label_mapping = {orig: new for new, orig in enumerate(class_indices)}

    mapping_path = os.path.join(
        experiment_folder, "label_mappings", f"subset_{subset_index}_mapping.json"
    )
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=2)

    def _get_label(dataset, idx):
        try:
            _, label = dataset[idx]
        except Exception:
            sample = dataset[idx]
            label = sample[1] if isinstance(sample, (tuple, list)) else sample["label"]
        return label

    train_indices = [
        i for i in range(len(train_set)) if _get_label(train_set, i) in class_indices
    ]
    test_indices = [
        i for i in range(len(test_set)) if _get_label(test_set, i) in class_indices
    ]

    class RemappedSubset(Subset):
        def __init__(self, dataset, indices, label_map):
            super().__init__(dataset, indices)
            self.label_map = label_map

        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            img, label = sample[0], sample[1]
            return img, self.label_map[label]

    train_subset = RemappedSubset(train_set, train_indices, label_mapping)
    test_subset = RemappedSubset(test_set, test_indices, label_mapping)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Subset {subset_index}: {len(train_subset)} train / {len(test_subset)} test")
    print(f"  Classes {class_indices} → [0, {len(class_indices) - 1}]")
    return train_loader, test_loader


def argument_check(args) -> None:
    """Validate CLI arguments."""
    if args.method == "eigenflux" and args.eigenflux_load_path is None:
        raise ValueError("--eigenflux_load_path must be set when --method eigenflux")
    if args.subset_size <= 0:
        raise ValueError("--subset_size must be a positive integer")
    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
