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

from gsoft import GSOFTLinear


class GSOFTConfig(PeftConfig):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class GSOFTModel(BaseTuner):
    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

        self._adapter_name_prefix = 'gsoft'
    
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

        return self._unload_and_optionally_merge(merge=True, progressbar=progressbar, safe_merge=safe_merge)

    def unload(self):
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base model.
        """
        
        return self._unload_and_optionally_merge(merge=False)

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)
    
    def _prepare_adapter_config(self, peft_config: GSOFTConfig, model_config: dict) -> GSOFTConfig:
        r"""
        A private method to eventually prepare the adapter config. For transformers based models, if
        `peft_config.target_modules` is None, we can automatically infer the target modules from the
        `TRANSFORMERS_MODELS_TO_XXX_TARGET_MODULES_MAPPING`. This method can be further refactored in the future to
        automatically infer it for all tuner models.

        Check out `peft.tuner.lora.LoraModel._prepare_adapter_config` for an example.

        Args:
            peft_config (`GSOFTConfig`):
                The adapter config.
            model_config (`dict`):
                The transformers model config, that config should contain the `model_type` key.
        """
        
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )

        return peft_config

    def _prepare_model(self, peft_config: GSOFTConfig, model: nn.Module):
        r"""
        A private method to modify the model structure before adapter is applied.

        See `peft.tuner.lora.LoraModel._prepare_model` for an example.

        Args:
            peft_config (`GSOFTConfig`):
                The prepared adapter config.
            model (`nn.Module`):
                The model that is going to be adapted.
        """
        pass

    def _check_target_module_exists(peft_config: GSOFTConfig, key: str) -> bool:
        r"""
        A helper private method to check if the passed module's key name matches any of the target modules in the
        `peft_config.target_modules` list. If it does, return `True`, else return `False`.

        Args:
            peft_config (`GSOFTConfig`):
                The adapter config.
            key (`str`):
                The module's key name.
        """
        return check_target_module_exists(peft_config, key)

    def _create_and_replace(
        self,
        gsoft_config: GSOFTConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        r"""
        Inplace replacement of the target module with the adapter layer. This method needs to be overridden by all the
        tuner classes.

        Check `peft.tuners.lora.LoraModel._create_and_replace` for an example.

        Args:
            peft_config (`GSOFTConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
            target (`nn.Module`):
                The target module.
            target_name (`str`):
                The target module's name.
            parent (`nn.Module`):
                The parent module.
            current_key (`str`):
                The key of the current target being adapted.
        """
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # TODO: implement kwargs generation by regex
        kwargs = {
            'nblocks': gsoft_config.get('nblocks', None),
            'orthogonal': gsoft_config.orthogonal,
            'method': gsoft_config.method,
            'block_size': gsoft_config.get('block_size', None),
            'scale': gsoft_config.scale,
        }
        
        if isinstance(target, GSOFTLinear):
            # update existing layer
            raise NotImplementedError()
        else:
            new_module = self._create_new_module(gsoft_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
    ):
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        named_modules = list(filter(
            lambda p: isinstance(p[1], GSOFTLinear),
            self.named_modules()
        ))

        for name, gs_linear in tqdm(named_modules, disable=not progressbar, desc=desc):
            parent, target, target_name = _get_submodules(self.model, name)
            
            new_module = gs_linear.merge() if merge else gs_linear.pre_layer
            self._replace_module(parent, target_name, new_module, target)
        
        return self.model

    def _replace_module(self, parent, child_name, new_module: GSOFTLinear, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

    @staticmethod
    def _create_new_module(gsoft_config, adapter_name, target, **kwargs):
        out_f, in_f = target.weight.shape

        # Do i need to set target.requires_grad = False?
        gs_linear = GSOFTLinear(
            pre_layer=target,
            in_features=in_f, out_features=out_f,
            **kwargs
        )

        return gs_linear

    def _mark_only_adapters_as_trainable(self, model: nn.Module):
        r"""
        A helper method to mark only the adapter layers as trainable (i.e. module.requires_grad = False) This needs to
        be overridden for all tuner classes to match the correct key names.

        Check `peft.tuners.lora.LoraModel._mark_only_adapters_as_trainable` for an example.
        """

        for name, param in model.named_parameters():
            if self._adapter_name_prefix not in name:
                param.requires_grad = False
