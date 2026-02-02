import re
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def replace_key(text: str, substring: str, replacement: str) -> str:
    """
    Replace a substring and everything after it with a replacement string.

    Args:
        text: The original string.
        substring: The substring to find and replace from.
        replacement: The string to replace with.

    Returns:
        The modified string with the replacement applied.
    """
    pattern = re.compile(re.escape(substring) + r".*", re.DOTALL)
    return re.sub(pattern, replacement, text)


def combine_loras(
    lora_dict: Dict[str, Dict[str, torch.Tensor]],
    state_dict: Dict[str, torch.Tensor],
    key_name: str,
    noise: float = 1.00,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Combine multiple LoRA state dictionaries into a single dictionary.

    This function aggregates LoRA weights from different adapters, grouped by
    layer key. Each layer contains a dictionary mapping adapter names to their
    corresponding weight tensors.

    Args:
        lora_dict: Existing dictionary of combined LoRAs, organized by layer.
        state_dict: State dictionary of the LoRA adapter to add.
        key_name: Name identifier for this LoRA adapter.
        noise: Optional scaling factor for the weights (default: 1.0).

    Returns:
        Updated lora_dict with the new adapter's weights added.

    Example:
        >>> lora_dict = {}
        >>> lora_dict = combine_loras(lora_dict, cola_weights, "cola")
        >>> lora_dict = combine_loras(lora_dict, qnli_weights, "qnli")
    """
    for key, value in state_dict.items():
        if "classifier" in key:
            continue
        try:
            lora_dict[key].update({key_name: noise * value})
        except KeyError:
            lora_dict[key] = {key_name: noise * value}
    return lora_dict


def calculate_eigenflux(
    eigenvectors: Dict[str, Dict[str, torch.Tensor]],
    lora_sd: Dict[str, torch.Tensor],
    num_components: int,
    loadings: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Calculate EigenFlux components and optionally loadings from eigenvectors.

    This function projects LoRA weights onto the principal eigenvector subspace
    to create EigenFlux components. If loadings=True, it also computes the
    projection coefficients (loadings) for initializing the adapter.

    Args:
        eigenvectors: Dictionary containing eigenvectors for each layer,
            with keys 'eigenvectors' and 'eigenvalues'.
        lora_sd: State dictionary of the source LoRA to compute loadings from.
        num_components: Number of principal components to retain.
        loadings: Whether to compute initial loadings (default: True).
            If False, only components are saved and loadings will be random.

    Returns:
        State dictionary containing EigenFlux components and optionally loadings.

    Note:
        If loadings=False, the returned state dict only contains components,
        and loadings will be randomly initialized during training.
    """
    eigenflux_sd = {}
    for k in lora_sd.keys():
        if "lora_A" in k:
            components = nn.Parameter(
                eigenvectors[k]["eigenvectors"][:, :num_components]
            ).contiguous()
            new_key_c = replace_key(k, "lora_A", "eigenflux_A.components")
            eigenflux_sd.update({new_key_c: components})
            if loadings:
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
            if loadings:
                loading_vals = nn.Parameter(
                    torch.mm(components.t(), lora_sd[k]).squeeze(dim=1)
                )
                new_key_l = replace_key(k, "lora_B", "eigenflux_B.loadings")
                eigenflux_sd.update({new_key_l: loading_vals})
    return eigenflux_sd


def add_classifier(
    eigenflux_dict: Dict[str, torch.Tensor],
    lora_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Add classifier weights from a LoRA state dict to an EigenFlux state dict.

    Args:
        eigenflux_dict: EigenFlux state dictionary to update.
        lora_dict: LoRA state dictionary containing classifier weights.

    Returns:
        Updated EigenFlux state dictionary with classifier weights added.
    """
    for key, value in lora_dict.items():
        if "classifier" in key:
            eigenflux_dict.update({key: value})
    return eigenflux_dict


def get_eigenvectors(
    lora_dict: Dict[str, Dict[str, torch.Tensor]],
    unwind_tensor: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute eigenvectors from combined LoRA weight matrices.

    This function performs PCA on the combined LoRA weights across multiple
    adapters to find the principal directions of variation.

    Args:
        lora_dict: Dictionary of combined LoRA weights organized by layer,
            where each layer maps adapter names to weight tensors.
        unwind_tensor: If True, flatten weight matrices to vectors before PCA.
            If False, compute eigenvectors along the larger dimension.

    Returns:
        Dictionary mapping layer keys to eigenvector dictionaries containing:
        - 'eigenvectors': Principal components sorted by eigenvalue (descending)
        - 'eigenvalues': Corresponding eigenvalues

    Note:
        unwind_tensor=True produces eigenvectors of size (m*n, 1) for m×n matrices.
        unwind_tensor=False produces eigenvectors of size (max(m,n), 1).
    """
    eigen_dict = {}
    for layer_key in lora_dict.keys():
        tensor_list = []
        for lora_key in lora_dict[layer_key].keys():
            tensor = lora_dict[layer_key][lora_key]
            if unwind_tensor:
                tensor = tensor.reshape((tensor.shape[0] * tensor.shape[1], 1))
            if tensor.shape[0] < tensor.shape[1]:
                tensor = tensor.t()
            tensor_list.append(tensor)
        concat_tensors = torch.cat(tensor_list, dim=1).to(torch.float32)
        eig = eigendecomposition(concat_tensors)
        eigen_dict.update({layer_key: eig})
    return eigen_dict


def eigendecomposition(
    matrix: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Perform eigendecomposition on a centered covariance matrix.

    This function centers the input matrix, computes its covariance,
    and returns eigenvectors sorted by eigenvalue in descending order.

    Args:
        matrix: Input matrix of shape (features, samples).

    Returns:
        Dictionary containing:
        - 'eigenvalues': Sorted eigenvalues (descending order)
        - 'eigenvectors': Corresponding eigenvectors as columns
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


def gram_schmidt_normalization(
    matrix: torch.Tensor,
    vec: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Orthogonalize a vector against existing basis vectors using Gram-Schmidt.

    This function projects out all components of `vec` that lie in the subspace
    spanned by the columns of `matrix`, then normalizes the result.

    Args:
        matrix: Matrix whose columns form the existing orthonormal basis.
        vec: Vector to orthogonalize against the basis.
        eps: Tolerance for detecting linear dependence (default: 1e-8).

    Returns:
        Normalized vector orthogonal to all columns of matrix.

    Raises:
        ValueError: If the vector is linearly dependent with the existing basis.
    """
    vec = vec.reshape(-1)
    for col in matrix.t():
        proj = (vec @ col) * col
        vec = vec - proj
    norm = torch.norm(vec)
    if norm < eps:
        raise ValueError("Vector is linearly dependent with existing basis")
    vec = vec / norm
    return vec


def add_gram_schmidt_vectors(
    state_dict: Dict[str, torch.Tensor],
    num_random_vectors: int,
) -> Dict[str, torch.Tensor]:
    """
    Extend EigenFlux components with random orthogonal vectors.

    This function adds additional orthogonal directions to the component
    matrices using Gram-Schmidt orthogonalization of random vectors.
    This increases the expressiveness of the EigenFlux subspace.

    Args:
        state_dict: EigenFlux state dictionary containing component matrices.
        num_random_vectors: Number of random orthogonal vectors to add.

    Returns:
        Updated state dictionary with extended component matrices.

    Example:
        >>> eigenflux_sd = add_gram_schmidt_vectors(eigenflux_sd, 32)
        >>> # Components now have 32 additional orthogonal directions
    """
    to_return = {}
    for k in state_dict.keys():
        if "components" in k:
            mat = state_dict[k].cpu()
            for i in range(num_random_vectors):
                rand_vec = torch.rand(max(mat.shape))
                new_vector = gram_schmidt_normalization(mat, rand_vec).unsqueeze(1)
                mat = torch.cat([mat, new_vector], dim=1)
            to_return[k] = mat
    print(f"Number of components --> {min(mat.shape)}")
    return to_return


def get_reconstruction(
    eigenflux_weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct LoRA weights from EigenFlux components and loadings.

    This function takes an EigenFlux state dictionary containing components
    and loadings, and reconstructs the original weight matrices.

    Args:
        eigenflux_weights: State dictionary containing EigenFlux components
            and loadings for each layer (keys with 'components' and 'loadings').

    Returns:
        State dictionary containing reconstructed weights with keys renamed
        from 'eigenflux_A' to 'recons_A' and 'eigenflux_B' to 'recons_B'.
    """
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
