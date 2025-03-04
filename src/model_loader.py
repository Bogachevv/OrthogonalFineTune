import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from transformers.modeling_utils import load_sharded_checkpoint
from peft import get_peft_model, PeftConfig, LoraConfig, BOFTConfig, VeraConfig, PeftModel, TaskType

from gsoft_tmp_injector import inject_gsoft
from HydraLoRA import HydraLoraConfig, HydraLoraModel

from omegaconf import OmegaConf
import re


def warmup_boft():
    model = nn.Sequential()
    model.add_module('layer', nn.Linear(16, 32))
    
    adapter_config = BOFTConfig(
        inference_mode=False,
        boft_block_size=4,
        boft_n_butterfly_factor=2,
        bias='none',
        target_modules=['layer']
    )
    
    model_adapter = get_peft_model(model, adapter_config)
    assert model_adapter is not None, "model_adapter is None"


def get_dtype(config):
    torch_dtype = torch.float32
    if config.fp16:
        torch_dtype = torch.float16
    if config.bf16:
        torch_dtype = torch.bfloat16 # The flag bf16 overrides the flag fp16
    
    return torch_dtype


def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, 
        **OmegaConf.to_object(config.tokenizer_config)
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(config):
    torch_dtype = get_dtype(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map='auto',
        torch_dtype=torch_dtype,
    )

    return model


def get_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_num_trainable(model):
    total_params = get_total_parameters(model)
    trainable_params = get_trainable_parameters(model)
    frac = (float(trainable_params) / total_params) * 100
    
    print(f"trainable: {trainable_params}  |  total: {total_params}  |  trainable(%): {frac:.6f}")


@torch.no_grad()
def _hash_tensor(tensor):
    return hash(tuple(tensor.reshape(-1).tolist()))


@torch.no_grad()
def _freeze_lora_A(freeze_args, model) -> int:
    '''
        :return (int): Hashsum for lora_A tensors
    '''
    A_pattern = re.compile(r'.lora_A$')
    B_pattern = re.compile(r'.lora_B$')
    seed = freeze_args.get('init_seed', 42)
    reinit = freeze_args.get('reinit', False)

    adapter_hash = 0
    torch.manual_seed(seed)
    for layer_name, layer in model.named_modules():
        if A_pattern.findall(layer_name):
            if reinit:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(layer.default.weight, a=5 ** 0.5)
            layer.default.weight.requires_grad_(False)

            layer_hash = _hash_tensor(layer.default.weight)
            adapter_hash += layer_hash

    return adapter_hash


def _drop_freeze_args(adapter_config) -> tuple[dict, dict]:
    '''
        :return tuple[dict, dict]: freeze_args, adapter_config
    '''
    args_to_drop = ('freeze_A', 'init_seed', 'reinit')
    freeze_args = {}

    for arg in args_to_drop:
        if arg not in adapter_config: continue
        
        freeze_args[arg] = adapter_config[arg]
        del adapter_config[arg]

    return freeze_args, adapter_config


def _get_peft_part(config, model, ft_strategy):
    freeze_args = {}

    if ft_strategy == 'GSOFT':
        # TODO: Implement PEFT model instead of custom injection
        # adapter_config = OmegaConf.to_object(config.adapter_config.GSOFT_config)
        adapter_config = config.adapter_config.GSOFT_config
        model_adapter = inject_gsoft(adapter_config, model)

        print("WARNING: spaggety implementation")

        return model_adapter

    if ft_strategy == 'HydraLoRA':
        adapter_config = HydraLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=not config.adapter_config.peft_is_trainable, 
            **OmegaConf.to_object(config.adapter_config.HydraLoRA_config),
        )
        
        model_adapter = HydraLoraModel(adapter_config, model)
        print_num_trainable(model_adapter)

        return model_adapter
        
    if ft_strategy == 'LoRA':
        adapter_config = OmegaConf.to_object(config.adapter_config.LoRA_config)
        freeze_args, adapter_config = _drop_freeze_args(adapter_config)

        adapter_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=not config.adapter_config.peft_is_trainable, 
            **adapter_config,
        )
    elif ft_strategy == 'BOFT':
        warmup_boft()
        adapter_config = BOFTConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=not config.adapter_config.peft_is_trainable,
            **OmegaConf.to_object(config.adapter_config.BOFT_config)
        )
    elif ft_strategy == 'VeRA':
        adapter_config = VeraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=not config.adapter_config.peft_is_trainable,
            **OmegaConf.to_object(config.adapter_config.VeRA_config)
        )
    else:
        raise ValueError('Incorrect FT type')

    model_adapter = get_peft_model(model, adapter_config)
    if freeze_args.get('freeze_A', False):
        print(f"Freeze args: {freeze_args}")
        A_hash = _freeze_lora_A(freeze_args, model)
        print(f"Hashsum for lora_A tensors: {A_hash}")

    model_adapter.print_trainable_parameters()

    return model_adapter


def _unfreeze_layernorm(config, model):
    if not config.adapter_config.get('unfreeze_layernorm', False):
        return

    for name, param in model.named_parameters():
        if 'norm' in name:
            param.requires_grad = True


def _get_peft_new(config, model):
    ft_strategy_ls = config.adapter_config.ft_strategy
    ft_strategy_ls = [ft_strategy_ls] if isinstance(ft_strategy_ls, str) else ft_strategy_ls
    
    model_adapter = model
    for ft_strategy in ft_strategy_ls:
        model_adapter = _get_peft_part(config, model_adapter, ft_strategy=ft_strategy)

    _unfreeze_layernorm(config, model)
    
    print_num_trainable(model_adapter)
    return model_adapter    


def _get_peft_pretrained(config, model):
    adapter_pth = config.adapter_config.peft_pretrained_path

    if config.adapter_config.get('peft_as_model', False):
        print(f"Loading finetuned model from shards...")

        model = _get_peft_new(config, model)

        load_info = load_sharded_checkpoint(
            model=model,
            folder=adapter_pth,
            strict=False
        ) 

        print(load_info)

        if len(load_info.missing_keys) - ('lm_head.weight' in load_info.missing_keys) + len(load_info.unexpected_keys) > 0:
            raise RuntimeError("Can't load model")

        return model

    model_adapter = PeftModel.from_pretrained(
        model=model,
        model_id=adapter_pth,
        is_trainable=config.adapter_config.peft_is_trainable,
    )

    return model_adapter


def get_peft(config, model):
    if config.adapter_config.ft_strategy == 'none':
        return model

    if config.adapter_config.peft_pretrained:
        return _get_peft_pretrained(config, model)
    else:
        return _get_peft_new(config, model)


def get_pipeline(config, model, tokenizer):
    torch_dtype = get_dtype(config)

    pl = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype
    )

    return pl
