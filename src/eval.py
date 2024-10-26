import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from datasets import DatasetDict, Dataset

from omegaconf import OmegaConf

import categories


def _inference_model(eval_cfg, pl, test_dataset):
    model_preds = []

    with torch.inference_mode(), torch.amp.autocast('cuda'):
        for i, split in enumerate(np.array_split(np.arange(len(test_dataset)), eval_cfg.num_splits)):
            print(f"Run {i} with split [{split[0]}, {split[-1]}]", flush=True)

            model_pred = pl(
                test_dataset.select(split)['text_wa_answer'],
                return_full_text=False,
                max_new_tokens=eval_cfg.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                batch_size=eval_cfg.batch_size,
            )
            model_preds += model_pred

            if eval_cfg.empty_cache:
                torch.cuda.empty_cache()

    model_preds_merged = []
    for ls in model_preds:
        model_preds_merged += ls

    return model_preds_merged


def _process_prediction(pred):
    pred = pred['generated_text']
    
    pred = pred.strip().upper()
    
    pred = pred[0] if pred else 'I'
    pred = pred if pred in {'A', 'B', 'C', 'D'} else 'I'
    
    return pred


def _preds_to_df(model_preds, test_dataset: Dataset):
    preds_df: pd.DataFrame = test_dataset.to_pandas()
    preds_df.insert(
        loc=len(preds_df.columns),
        column='model_pred',
        value=model_preds
    )

    return preds_df


def make_preds(config, pl, test_dataset: Dataset):
    eval_cfg = config.evaluation_config
    
    model_preds = _inference_model(eval_cfg, pl, test_dataset)

    preds_df = _preds_to_df(model_preds, test_dataset)

    return preds_df
