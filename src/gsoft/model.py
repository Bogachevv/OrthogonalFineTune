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
    
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        raise NotImplemented("Specify target_modules in GSOFTConfig. Auto target_modules deducting is not implemented yet")
    
    @staticmethod
    def _check_target_module_exists(gsoft_config, key):
        return check_target_module_exists(gsoft_config, key)

    @staticmethod
    def _create_new_module(gsoft_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = GSOFTLinear(
                target,
                in_features=target.weight.shape[1],
                out_features=target.weight.shape[0], 
                **kwargs
                )
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                "Currently, only `torch.nn.Linear` is supported."
            )

        return new_module

    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
) -> None:
        kwargs = {
            'nblocks': peft_config.nblocks,
            'orthogonal': peft_config.orthogonal,
            'method': peft_config.method,
            'block_size': peft_config.block_size,
            'scale': peft_config.scale,
        }

        if isinstance(target, GSOFTLinear):
            raise NotImplemented("Smth strange, target is instance of GSOFTLinear")
        
        new_module = self._create_new_module(peft_config, adapter_name, target, **kwargs)
        if adapter_name not in self.active_adapters:
            # adding an additional adapter: it is not automatically trainable
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)
    
    @staticmethod
    def __get_layer(model, name):
        layer = model
        for attr in name.split("."):
            layer = getattr(layer, attr)
        return layer
    
    @staticmethod
    def __set_layer(model, name, layer):
        try:
            attrs, name = name.rsplit(".", 1)
            model = GSOFTModel.get_layer(model, attrs)
        except ValueError:
            pass
        setattr(model, name, layer)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        ## child layer wraps the original module, unpack it
        # if hasattr(child, "base_layer"):
        #     child = child.base_layer

        # if not hasattr(new_module, "base_layer"):
        #     if hasattr(new_module, "W_q"):  # HQQ
        #         new_module.W_q = child.W_q
        #     else:
        #         new_module.weight = child.weight
        #     if hasattr(child, "bias"):
        #         new_module.bias = child.bias

        # if getattr(child, "state", None) is not None:
        #     if hasattr(new_module, "base_layer"):
        #         new_module.base_layer.state = child.state
        #     else:
        #         new_module.state = child.state
        #     new_module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        raise NotImplemented()
        # for n, p in model.named_parameters():
        #     if self.prefix not in n:
        #         p.requires_grad = False

        # for active_adapter in self.active_adapters:
        #     bias = self.peft_config[active_adapter].bias
        #     if bias == "none":
        #         continue

        #     if bias == "all":
        #         for n, p in model.named_parameters():
        #             if "bias" in n:
        #                 p.requires_grad = True
        #     elif bias == "lora_only":
        #         for m in model.modules():
        #             if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
        #                 m.bias.requires_grad = True
        #     else:
        #         raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")