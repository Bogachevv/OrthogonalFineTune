import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from peft import get_peft_model, LoraConfig, BOFTConfig, TaskType

from omegaconf import OmegaConf


def get_dtype(config):
    torch_dtype = torch.float32
    if config.fp16:
        torch_dtype = torch.float16
    if config.bf16:
        torch_dtype = torch.bloat16 # The flag bf16 overrides the flag fp16
    
    return torch_dtype


def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, 
        padding_side=config.padding_side,
    #     model_max_length=512,
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

def get_peft(config, model):
    # TODO: It is necessary to implement for other PEFT strategies

    if config.ft_strategy == 'LoRA':
        adapter_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            **OmegaConf.to_object(config.LoRA_config),
        )
    elif config.ft_strategy == 'BOFT':
        adapter_config = BOFTConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **OmegaConf.to_object(config.BOFT_config)
        )
    else:
        raise ValueError('Incorrect FT type')

    model_adapter = get_peft_model(model, adapter_config)    
    model_adapter.print_trainable_parameters()

    return model_adapter

def get_pipeline(config, model, tokenizer):
    torch_dtype = get_dtype(config)

    pl = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype
    )

    return pl
