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
    MULTILANG = 4,


def run_finetune(config, model, tokenizer, train_dataset, val_dataset):
    trainer = finetune.get_trainer(config, model, tokenizer, train_dataset, val_dataset)
    trainer.train()

    model.save_pretrained(config.adapter_config.peft_pretrained_path)
    tokenizer.save_pretrained(config.adapter_config.peft_pretrained_path)

def run_inference(config, pl, test_dataset, task_idx=None, path: str = None):
    preds_df = eval.make_preds(
        config=config,
        pl=pl,
        test_dataset=test_dataset,
    )

    path = config.evaluation_config.get('dump_path', None) if path is None else path
    task_idx = 0 if task_idx is None else task_idx
    path = path.format(task_idx)

    with open(path, 'wb') as f:
        pickle.dump(
            preds_df,
            file=f,
        )


def run_multitask(config, pl, test_datasets, task_idx=None):
    base_path: str = config.evaluation_config.get('dump_path', None)
    
    for lang, dataset in test_datasets.items():
        path = base_path.format('{0}', lang)
        print(f"Running predictions for {lang}, {path=}\n")

        run_inference(config, pl, dataset, task_idx=task_idx, path=path)


def run_tasks(config):
    tasks = config.tasks

    tokenizer = model_loader.load_tokenizer(config)
    model = model_loader.load_model(config)
    model = model_loader.get_peft(config, model)

    dataset = data_preparation.load_dataset(config)

    for i, task in enumerate(tasks):
        print(f"Running task {i}: {task}")

        if isinstance(task, str):
            task = Task[task]

        if task is Task.INFERENCE:
            pl = model_loader.get_pipeline(config, model, tokenizer)
            run_inference(config, pl, dataset['test'], task_idx=i)
        elif task is Task.MULTILANG:
            pl = model_loader.get_pipeline(config, model, tokenizer)
            run_multitask(config, pl, dataset['test'], task_idx=i)
        elif task is Task.VALIDATE:
            pl = model_loader.get_pipeline(config, model, tokenizer)
            run_inference(config, pl, dataset['validation'], task_idx=i)
        elif task is Task.FINETUNE:
            run_finetune(config, model, tokenizer, dataset['train'], dataset['validation'])
        else:            
            raise ValueError(f'Incorrect value of {task=}\ntask must be instance of Task enum')


def run(cfg_path: str):
    config = OmegaConf.load(cfg_path)

    run_tasks(config)
