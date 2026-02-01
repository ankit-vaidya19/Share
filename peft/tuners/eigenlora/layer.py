import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


# class EigenLoRA_A_incremental(nn.Module):
#     """
#     Class that implements a basic EigenLoRA A layer.
#     X,768 --> X,34
#     """

#     def __init__(
#         self,
#         num_components,
#         in_features,
#         out_features,
#         rank_update=2,
#     ):
#         super().__init__()
#         self.in_dim = in_features
#         self.out_dim = out_features
#         self.num_components = num_components
#         self.components = nn.Parameter(torch.rand((in_features, self.num_components)))
#         self.rank_update = rank_update
#         self.rank_update_vectors = nn.Parameter(
#             torch.rand((in_features, self.rank_update))
#         )  # trainable
#         self.loadings = nn.Parameter(torch.rand((self.num_components, self.out_dim)))
#         self.mode = "train"

#     def recalculate_eigenlora(self):
#         """
#         Recalculate the components using prev_components and the updated
#         rank update vectors
#         """
#         recons = torch.sum(
#             self.components.unsqueeze(0) * self.loadings.t().unsqueeze(1),
#             dim=-1,
#         ).t()
#         reconstructed = torch.cat((recons, self.rank_update_vectors), dim=1)
#         mean = reconstructed.mean(axis=1, keepdim=True)
#         reconstructed = reconstructed - mean
#         cov = torch.mm(reconstructed.t(), reconstructed)
#         eigenvals, eigenvecs = torch.linalg.eig(cov)
#         eigenvals = eigenvals.to(torch.float32)
#         eigenvecs = eigenvecs.to(torch.float32)
#         eigenvecs = torch.mm(reconstructed, eigenvecs)
#         eigenvecs = torch.nn.functional.normalize(eigenvecs, p=2, dim=0)
#         eigenvals, indices = eigenvals.sort(descending=True)
#         eigenvecs = eigenvecs[:, indices]
#         self.components = nn.Parameter(eigenvecs[:, : self.num_components])
#         self.loadings = nn.Parameter(
#             torch.mm(self.components.t(), recons).squeeze(dim=1)
#         )

#     def forward(self, x):
#         recons = torch.sum(
#             self.components.unsqueeze(0) * self.loadings.t().unsqueeze(1),
#             dim=-1,
#         ).t()
#         if self.mode == "train":
#             output = torch.cat(
#                 (torch.matmul(x, recons), torch.matmul(x, self.rank_update_vectors)),
#                 dim=2,
#             )
#         elif self.mode == "inference":
#             output = torch.matmul(x, recons)
#         return output


# class EigenLoRA_B_incremental(nn.Module):
#     """
#     Class that implements a basic EigenLoRA B layer.
#     X,34 --> X,768
#     """

#     def __init__(
#         self,
#         num_components,
#         in_features,
#         out_features,
#         rank_update=2,
#     ):
#         super().__init__()
#         self.in_dim = in_features
#         self.out_dim = out_features
#         self.num_components = num_components
#         self.components = nn.Parameter(torch.rand((out_features, self.num_components)))
#         self.rank_update = rank_update
#         self.rank_update_vectors = nn.Parameter(
#             torch.rand((out_features, self.rank_update))
#         )  # trainable
#         self.loadings = nn.Parameter(torch.rand((self.num_components, self.in_dim)))
#         self.mode = "train"

#     def recalculate_eigenlora(self):
#         """
#         Recalculate the components using prev_components and the updated
#         rank update vectors
#         """
#         recons = torch.sum(
#             self.components.unsqueeze(0) * self.loadings.t().unsqueeze(1),
#             dim=-1,
#         ).t()
#         reconstructed = torch.cat((recons, self.rank_update_vectors), dim=1)
#         mean = reconstructed.mean(axis=1, keepdim=True)
#         reconstructed = reconstructed - mean
#         cov = torch.mm(reconstructed.t(), reconstructed)
#         eigenvals, eigenvecs = torch.linalg.eig(cov)
#         eigenvals = eigenvals.to(torch.float32)
#         eigenvecs = eigenvecs.to(torch.float32)
#         eigenvecs = torch.mm(reconstructed, eigenvecs)
#         eigenvecs = torch.nn.functional.normalize(eigenvecs, p=2, dim=0)
#         eigenvals, indices = eigenvals.sort(descending=True)
#         eigenvecs = eigenvecs[:, indices]
#         self.components = nn.Parameter(eigenvecs[:, : self.num_components])
#         self.loadings = nn.Parameter(
#             torch.mm(self.components.t(), recons).squeeze(dim=1)
#         )

#     def forward(self, x):
#         recons = torch.sum(
#             self.components.unsqueeze(0) * self.loadings.t().unsqueeze(1),
#             dim=-1,
#         ).t()
#         if self.mode == "train":
#             output = torch.matmul(
#                 x, torch.cat((recons, self.rank_update_vectors), dim=1).t()
#             )
#         elif self.mode == "inference":
#             output = torch.matmul(x, recons.t())
#         return output


class EigenLoRA_A_incremental(nn.Module):
    """
    Class that implements a basic EigenLoRA A layer.
    X,768 --> X,34
    """

    def __init__(
        self,
        num_components,
        in_features,
        out_features,
        rank_update=2,
        use_rank_updates=False,
    ):
        super().__init__()
        self.in_dim = in_features  # feature size
        self.out_dim = out_features  # rank of EigenLoRA
        self.num_components = num_components
        self.components = nn.Parameter(torch.rand((in_features, self.num_components)))
        self.rank_update = rank_update
        self.rv_1 = nn.Parameter(
            torch.empty(in_features, self.rank_update).uniform_(-0.5, to=0.5)
        )
        self.rv_2 = nn.Parameter(
            torch.empty(self.rank_update, out_features).uniform_(-0.5, to=0.5)
        )
        if use_rank_updates == True:
            # self.rank_update_vectors = nn.Parameter(
            #     torch.rand((in_features, self.rank_update))
            # )  # trainable
            self.mode = "new_method"
        else:
            self.mode = "inference"
        self.loadings = nn.Parameter(
            torch.empty(self.num_components, self.out_dim).uniform_(-0.5, to=0.5)
        )

    def recalculate_eigenlora(self):
        """
        Recalculate the components using prev_components and the updated
        rank update vectors
        """
        recons = torch.sum(
            self.components.unsqueeze(0) * self.loadings.t().unsqueeze(1),
            dim=-1,
        ).t()
        rank_updates = torch.matmul(self.rv_1, self.rv_2)
        reconstructed = torch.cat((recons, rank_updates), dim=1)
        mean = reconstructed.mean(axis=1, keepdim=True)
        reconstructed = reconstructed - mean
        cov = torch.mm(reconstructed, reconstructed.t())
        eigenvals, eigenvecs = torch.linalg.eig(cov)
        eigenvals = eigenvals.to(torch.float32)
        eigenvecs = eigenvecs.to(torch.float32)
        eigenvals, indices = eigenvals.sort(descending=True)
        eigenvecs = eigenvecs[:, indices]
        components = eigenvecs[:, : self.num_components].contiguous()
        self.components = nn.Parameter(components)
        self.loadings = nn.Parameter(
            torch.mm(self.components.t(), recons).squeeze(dim=1)
        )
        self.mode = "inference"

    def forward(self, x):
        recons = torch.sum(
            self.components.unsqueeze(0) * self.loadings.t().unsqueeze(1),
            dim=-1,
        ).t()
        if self.mode == "train":
            output = torch.cat(
                (torch.matmul(x, recons), torch.matmul(x, self.rank_update_vectors)),
                dim=-1,
            )
        elif self.mode == "inference":
            output = torch.matmul(x, recons)
        elif self.mode == "new_method":
            op1 = torch.matmul(x, recons)
            op2 = torch.matmul(torch.matmul(x, self.rv_1), self.rv_2)
            output = op1 + op2
        return output


class EigenLoRA_B_incremental(nn.Module):
    """
    Class that implements a basic EigenLoRA B layer.
    X,34 --> X,768
    """

    def __init__(
        self,
        num_components,
        in_features,
        out_features,
        rank_update=2,
        use_rank_updates=False,
    ):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.num_components = num_components
        self.components = nn.Parameter(torch.rand((out_features, self.num_components)))
        self.rank_update = rank_update
        self.rv_1 = nn.Parameter(
            torch.empty(in_features, self.rank_update).uniform_(-0.5, to=0.5)
        )
        self.rv_2 = nn.Parameter(
            torch.empty(self.rank_update, out_features).uniform_(-0.5, to=0.5)
        )
        if use_rank_updates == True:
            self.mode = "new_method"
        else:
            self.mode = "inference"
        self.loadings = self.loadings = nn.Parameter(
            torch.empty(self.num_components, self.in_dim).uniform_(-0.5, to=0.5)
        )

    def recalculate_eigenlora(self):
        """
        Recalculate the components using prev_components and the updated
        rank update vectors
        """
        recons = torch.sum(
            self.components.unsqueeze(0) * self.loadings.t().unsqueeze(1),
            dim=-1,
        ).t()
        rank_updates = torch.matmul(self.rv_1, self.rv_2).t()
        reconstructed = torch.cat((recons, rank_updates), dim=1)
        mean = reconstructed.mean(axis=1, keepdim=True)
        reconstructed = reconstructed - mean
        cov = torch.mm(reconstructed, reconstructed.t())
        eigenvals, eigenvecs = torch.linalg.eig(cov)
        eigenvals = eigenvals.to(torch.float32)
        eigenvecs = eigenvecs.to(torch.float32)
        eigenvals, indices = eigenvals.sort(descending=True)
        eigenvecs = eigenvecs[:, indices]
        components = eigenvecs[:, : self.num_components].contiguous()
        self.components = nn.Parameter(components)
        self.loadings = nn.Parameter(
            torch.mm(self.components.t(), recons).squeeze(dim=1)
        )
        self.mode = "inference"

    def forward(self, x):
        recons = torch.sum(
            self.components.unsqueeze(0) * self.loadings.t().unsqueeze(1),
            dim=-1,
        ).t()
        if self.mode == "train":
            output = torch.matmul(
                x, torch.cat((recons, self.rank_update_vectors), dim=1).t()
            )
        elif self.mode == "inference":
            output = torch.matmul(x, recons.t())
        elif self.mode == "new_method":
            op1 = torch.matmul(x, recons.t())
            op2 = torch.matmul(torch.matmul(x, self.rv_1), self.rv_2)
            output = op1 + op2
        return output


class EigenLoRALayer(BaseTunerLayer):
    """
    Modified the LoraLayer class to include EigenLoRA. Changed the update_layer function to add EigenLoRA layer instead of a simple linear layer.
    """

    adapter_layer_names = ("eigenlora_A", "eigenlora_B")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.num_components = {}

        self.eigenlora_A = nn.ModuleDict({})
        self.eigenlora_B = nn.ModuleDict({})

        self._disable_adapters = False
        self.merged_adapters = []
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        """Not used yet"""
        return bool(self.merged_adapters)

    def recalculate(self, adapter_name):
        """For EigenFLux"""
        self.eigenlora_A[adapter_name].recalculate_eigenlora()
        self.eigenlora_B[adapter_name].recalculate_eigenlora()

    def set_mode(self, adapter_name, mode):
        """For 'training' or 'inference' EigenFlux"""
        self.eigenlora_A[adapter_name].mode = mode
        self.eigenlora_B[adapter_name].mode = mode

    def update_layer(
        self, adapter_name, r, num_components, num_rank_updates, use_rank_updates
    ):
        """Adds the eigenlora to base layer"""
        self.eigenlora_A[adapter_name] = EigenLoRA_A_incremental(
            num_components, self.in_features, r, num_rank_updates, use_rank_updates
        )
        self.eigenlora_B[adapter_name] = EigenLoRA_B_incremental(
            num_components, r, self.out_features, num_rank_updates, use_rank_updates
        )
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def set_adapter(self, adapter_names: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `List[str]`): Name of the adapter(s) to be activated.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names:
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.components.requires_grad = False
                    layer.loadings.requires_grad = True
                    if layer.rv_1 is not None:
                        layer.rv_1.requires_grad = True
                    if layer.rv_2 is not None:
                        layer.rv_2.requires_grad = True
                else:
                    layer.requires_grad_(False)

        self._active_adapter = adapter_names


class Linear(EigenLoRALayer, nn.Linear):
    """
    This is the class used to instantiate in the model.py file. Modified the forward method. Need to modify merge()
    """

    def __init__(
        self,
        base_layer,
        adapter_name,
        r,
        num_components,
        num_rank_updates,
        use_rank_updates,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        EigenLoRALayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, r, num_components, num_rank_updates, use_rank_updates
        )

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None
    ) -> None:
        """
        Not being used - needs edit.
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.eigenlora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    # def unmerge(self) -> None:
    #     if not self.merged:
    #         warnings.warn("Already unmerged. Nothing to do.")
    #         return

    #     while len(self.merged_adapters) > 0:
    #         active_adapter = self.merged_adapters.pop()
    #         if active_adapter in self.vera_lambda_d.keys():
    #             self.get_base_layer().weight.data -= self.get_delta_weight(
    #                 active_adapter
    #             )

    # def get_delta_weight(self, adapter) -> torch.Tensor:
    #     """
    #     Compute the delta weight for the given adapter.

    #     Args:
    #         adapter (str):
    #             The name of the adapter for which the delta weight should be computed.
    #     """
    #     vera_A = self.vera_A[adapter]
    #     vera_B = self.vera_B[adapter]

    #     device = vera_B.device
    #     dtype = vera_B.dtype

    #     # In case users wants to merge the adapter weights that are in
    #     # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    #     # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
    #     cast_to_fp32 = device.type == "cpu" and (
    #         dtype == torch.float16 or dtype == torch.bfloat16
    #     )

    #     lambda_d = self.vera_lambda_d[adapter]
    #     lambda_b = self.vera_lambda_b[adapter]

    #     if cast_to_fp32:
    #         vera_A = vera_A.float()
    #         vera_B = vera_B.float()
    #         lambda_d = lambda_d.float()
    #         lambda_b = lambda_b.float()

    #     sliced_A = vera_A[:, : self.in_features]
    #     sliced_B = vera_B[: self.out_features, :]
    #     lambda_b = lambda_b.unsqueeze(-1)
    #     lambda_d = lambda_d.unsqueeze(-1)
    #     output_tensor = transpose(
    #         (lambda_b * sliced_B) @ (lambda_d * sliced_A), self.fan_in_fan_out
    #     )

    #     if cast_to_fp32:
    #         output_tensor = output_tensor.to(dtype=dtype)

    #         # cast back the weights
    #         # TODO: why?
    #         self.vera_lambda_d[adapter].data = lambda_d.to(dtype)
    #         self.vera_lambda_b[adapter].data = lambda_b.to(dtype)

    #     return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            """Not used"""
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            """Not used"""
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.eigenlora_A.keys():
                    continue

                eigenlora_A = self.eigenlora_A[active_adapter]
                eigenlora_B = self.eigenlora_B[active_adapter]

                # As adapted layers may have different shapes and VeRA contains a single shared pair of A and B matrices,
                # we initialize these matrices with the largest required size for each dimension.
                # During the forward pass, required submatrices are sliced out from the shared vera_A and vera_B.
                x = x.to(eigenlora_A.components.dtype)  # datatype equivalency
                result = result + eigenlora_B(eigenlora_A(x))
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "eigenlora." + rep
