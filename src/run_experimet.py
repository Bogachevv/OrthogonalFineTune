import model_loader
import data_preparation
import evaluate
import finetune

from omegaconf import OmegaConf
from enum import Enum

import pickle

class Task(Enum):
    INFERENCE = 1,
    FINETUNE = 2,

def run_finetune(config, model, tokenizer, train_dataset, val_dataset):
    trainer = finetune.get_trainer(config, model, tokenizer, train_dataset, val_dataset)
    trainer.train()

    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

def run_inference(config, pl, test_dataset, task_idx=None):
    preds_df = evaluate.make_preds(
        config=config,
        pl=pl,
        test_dataset=test_dataset,
    )

    path = config.evaluation_config.dump_path
    task_idx = 0 if task_idx is None else task_idx
    path = path.format(task_idx)

    with open(path, 'wb') as f:
        pickle.dump(
            preds_df,
            file=f,
        )

def run_tasks(config):
    tasks = config.tasks

    tokenizer = model_loader.load_tokenizer(config)
    model = model_loader.load_model(config)
    model = model_loader.get_peft(config, model)
    pl = model_loader.get_pipeline(config, model, tokenizer)

    dataset = data_preparation.load_MMLU(config, tokenizer)
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    train_dataset  = dataset['auxiliary_train']

    for task in tasks:
        if task is Task.INFERENCE:
            run_inference(config, pl, test_dataset)
        elif task is Task.FINETUNE:
            run_finetune()
        else:            
            raise ValueError(f'Incorrect value of {task=}\ntask must be instance of Task enum')


def main():
    # TODO: move config to file
    config = OmegaConf.create({
        'model_name':   'meta-llama/Meta-Llama-3-8B-Instruct',
        'tasks': [Task.INFERENCE, Task.FINETUNE, Task.INFERENCE],
        'padding_side': 'left',
        'task_name':    'all',
        'max_length':   256,
        'n_shots': 2,
        'fp16': True,
        'bf16': False,
        'loader_config': {
            'num_proc': 2,
        },
        'LoRA_config': {
            'r': 16, 
            'lora_alpha': 32, 
            'lora_dropout': 0.05,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        },
        'evaluation_config':{
            'num_splits': 20,
            'max_new_tokens': 4,
            'batch_size': 1,
            'empty_cache': True,
            'dump_path': './preds_{0}.bin',
        },
        'trainer_config': {
            'output_dir': "bogachevv/Llama-3-8b-MMLU",
            'max_seq_length': 512,
            'dataset_text_field': 'text',
            'fp16': True,
            'full_determinism': False,
            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size':  1,
            'gradient_accumulation_steps': 8,
            'lr_scheduler_type': 'cosine_with_restarts',
            'lr_scheduler_kwargs':{
                'num_cycles': 6,
            },
            'warmup_steps': 100,
    #         'num_train_epochs': 2,
            'learning_rate': 1e-4,
            'max_steps': 2048,
            'weight_decay': 0.01,
    #         'warmup_ratio': 1e-2,
            'dataloader_num_workers': 2,
            'eval_strategy': "steps",
    #         'torch_empty_cache_steps': 16,
            'eval_steps': 16,
            'logging_steps': 16,
            'load_best_model_at_end': True,
            'seed': 42,
            'data_seed': 42,
            'report_to': 'wandb',
    #         'push_to_hub': True,
    #         'hub_model_id': 'LLama-LoRA-test',
    #         'hub_strategy': 'checkpoint',
            'save_strategy': "steps",
            'save_steps': 128,
        },
    })

    run_tasks(config)
