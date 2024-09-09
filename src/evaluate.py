import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BertTokenizer, BertModel 
from transformers import TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig, DataCollatorWithPadding
from transformers import pipeline
from peft import BOFTConfig, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import evaluate
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

import wandb
from omegaconf import OmegaConf

import pickle
import tqdm.notebook as tqdm


from metrics import compute_accuracy
import model


def make_preds(config, pl, test_dataset):
    model_preds = []
    eval_cfg = config.evaluation_config
    num_splits = eval_cfg.num_splits


    with torch.no_grad():
        for i, split in tqdm.tqdm(
            enumerate(np.array_split(np.arange(len(test_dataset)), num_splits)),
            total=num_splits,
        ):            
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

    model_preds = model_preds_merged

    for i in range(len(model_preds)):
        model_preds[i]['subject'] = test_dataset[i]['subject']

    return model_preds