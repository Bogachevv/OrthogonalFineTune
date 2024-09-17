import model_loader
import data_preparation
import eval
import finetune

from omegaconf import OmegaConf
from enum import Enum

import pickle

class Task(Enum):
    INFERENCE = 1,
    FINETUNE = 2,
    VALIDATE = 3,

def run_finetune(config, model, tokenizer, train_dataset, val_dataset):
    trainer = finetune.get_trainer(config, model, tokenizer, train_dataset, val_dataset)
    trainer.train()

    model.save_pretrained(config.adapter_config.peft_pretrained_path)
    tokenizer.save_pretrained(config.adapter_config.peft_pretrained_path)

def run_inference(config, pl, test_dataset, task_idx=None):
    preds_df = eval.make_preds(
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

    dataset = data_preparation.load_MMLU(config, tokenizer)
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    train_dataset  = dataset['auxiliary_train']

    for i, task in enumerate(tasks):
        if isinstance(task, str):
            task = Task[task]

        if task is Task.INFERENCE:
            pl = model_loader.get_pipeline(config, model, tokenizer)
            run_inference(config, pl, test_dataset, task_idx=i)
        elif task is Task.VALIDATE:
            pl = model_loader.get_pipeline(config, model, tokenizer)
            run_inference(config, pl, validation_dataset, task_idx=i)
        elif task is Task.FINETUNE:
            run_finetune(config, model, tokenizer, train_dataset, validation_dataset)
        else:            
            raise ValueError(f'Incorrect value of {task=}\ntask must be instance of Task enum')


def run(cfg_path: str):
    config = OmegaConf.load(cfg_path)

    run_tasks(config)
