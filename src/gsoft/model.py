import re
from itertools import chain

import torch
from torch import nn
from tqdm import tqdm

from transformers import PreTrainedModel

from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_peft_model_state_dict,
)

from peft import PeftConfig

from gsoft.gsoft import GSOFTLinear


class GSOFTModel(BaseTuner):
    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)
    
    def disable_adapter_layers(self):
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """

        self._set_adapter_layers(enabled=False)

    def enable_adapter_layers(self):
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """

        self._set_adapter_layers(enabled=True)

    def merge_and_unload(
            self,
            progressbar: bool = False, 
            safe_merge: bool = False
        ):
        """
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model as a standalone model

        Args:
            progressbar (`bool`): 
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`): 
                whether to activate the safe merging check to check if there is any potential Nan in the adapter weights
        """

        pass

    def unload(self):
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base model.
        """
        pass

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)