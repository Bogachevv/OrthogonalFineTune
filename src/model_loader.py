import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from peft import get_peft_model, PeftConfig, LoraConfig, BOFTConfig, PeftModel, TaskType

from omegaconf import OmegaConf


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


def _get_peft_new(config, model):
    if config.adapter_config.ft_strategy == 'LoRA':
        adapter_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=not config.adapter_config.peft_is_trainable, 
            **OmegaConf.to_object(config.adapter_config.LoRA_config),
        )
    elif config.adapter_config.ft_strategy == 'BOFT':
        warmup_boft()
        adapter_config = BOFTConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=not config.adapter_config.peft_is_trainable,
            **OmegaConf.to_object(config.adapter_config.BOFT_config)
        )
    else:
        raise ValueError('Incorrect FT type')

    model_adapter = get_peft_model(model, adapter_config)    
    model_adapter.print_trainable_parameters()

    return model_adapter


def _get_peft_pretrained(config, model):
    adapter_pth = config.adapter_config.peft_pretrained_path

    model_adapter = PeftModel.from_pretrained(
        model=model,
        model_id=adapter_pth,
        is_trainable=config.adapter_config.peft_is_trainable,
    )

    return model_adapter


def get_peft(config, model):
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
