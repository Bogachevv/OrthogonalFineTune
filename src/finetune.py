import torch
from torch import nn
import torch.nn.functional as F

import transformers
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

import wandb
from omegaconf import OmegaConf


def _get_optimizer_by_name(name: str):
    mapping = {
        'adam': torch.optim.Adamw,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }

    name = name.lower()
    if name in mapping:
        return mapping[name]
    
    raise ValueError(f'Incorrect optimizer name {name}, supported: {mapping.keys()}')


def get_trainer(config, model, tokenizer, train_dataset, val_dataset):
    training_args = SFTConfig(
        **OmegaConf.to_object(config.trainer_config),
    )

    optimizer_cls_and_kwargs = None
    optimizer_config = config.get('optimizer_config', None)
    if optimizer_config is not None:
        optim_name = optimizer_config['optim']
        optim_cls = _get_optimizer_by_name(optim_name)

        optim_kwargs = dict(optimizer_config)
        del optim_kwargs['optim'] # remove optimizer name from kwargs

        optimizer_cls_and_kwargs = optim_cls, optim_kwargs

        print(
            f"Running with custom optimizer: {optim_cls}\nWith kwargs:\n\t", 
            '\n\t'.join(
                lambda itm: f"{itm[0]}: {itm[1]}",
                optim_kwargs.items()
            )
        )

    if config.get('val_ds_size', None):
        if config.get('val_ds_seed', None) is not None:
            val_dataset = val_dataset.shuffle(config.val_ds_seed)
        
        val_dataset = val_dataset.select(
            range(config.val_ds_seed)
        )
        
    response_template = '<|start_header_id|>assistant<|end_header_id|>'
    print(f"WARNING: Collator response_template is fixed to {response_template}")
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        data_collator=collator,
    )

    return trainer