import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

from omegaconf import OmegaConf

import categories


def _inference_model(eval_cfg, pl, test_dataset):
    model_preds = []

    with torch.inference_mode(), torch.cuda.amp.autocast():
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


def _preds_to_df(model_preds, test_dataset):
    preds_df = pd.DataFrame(model_preds)

    if 'subject' in test_dataset:
        preds_df['subject'] = test_dataset['subject']
        preds_df['category'] = preds_df['subject'].apply(lambda subcat: categories.subcat_to_cat.get(subcat, None))
    
    preds_df['pred'] = preds_df.apply(_process_prediction, axis=1)
    preds_df['true'] = list(map(lambda v: chr(v + ord('A')), test_dataset['answer']))
    preds_df['corr'] = (preds_df['pred'] == preds_df['true']).astype(np.int32)

    return preds_df


def make_preds(config, pl, test_dataset):
    eval_cfg = config.evaluation_config
    
    model_preds = _inference_model(eval_cfg, pl, test_dataset)

    preds_df = _preds_to_df(model_preds, test_dataset)

    return preds_df
