import torch
from torch import nn
import torch.nn.functional as F

import transformers
from trl import SFTConfig, SFTTrainer

import wandb
from omegaconf import OmegaConf


def get_trainer(config, model, tokenizer, train_dataset, val_dataset):
    training_args = SFTConfig(
        **OmegaConf.to_object(config.trainer_config),
    )

    if config.val_ds_size:
        if config.val_ds_seed:
            val_dataset = val_dataset.shuffle(config.val_ds_seed)
        
        val_dataset = val_dataset.select(
            range(config.val_ds_seed)
        )
        

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    return trainer