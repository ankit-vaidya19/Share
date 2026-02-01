import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class EigenLoRAConfig(PeftConfig):
    r: int = field(default=32, metadata={"help": "Rank of the Components"})
    num_components: int = field(
        default=15, metadata={"help": "Number of components to use"}
    )
    use_rank_updates: bool = field(
        default=False, metadata={"help": "Use rank update vectors"}
    )
    num_rank_updates: int = field(
        default=2, metadata={"help": "Number of rank update vectors to use"}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Vera. Can be 'none', 'all' or 'vera_only'"},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with Vera."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from Vera layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer"
                " pattern is not in the common layers pattern."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.EIGENLORA
        self.target_modules = (
            set(self.target_modules)
            if isinstance(self.target_modules, list)
            else self.target_modules
        )

    #     if not self.save_projection:
    #         warnings.warn(
    #             "Specified to not save vera_A and vera_B within the state dictionary, instead they will be restored "
    #             "using the PRNG key store in `config.projection_prng_key`. Consider setting `config.save_projection` "
    #             "to `True` to guarantee restoring the checkpoint correctly on all system configurations."
    #         )
