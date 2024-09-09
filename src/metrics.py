import torch
from torch import nn
import torch.nn.functional as F

import evaluate

from omegaconf import OmegaConf


accuracy_metric = evaluate.load("accuracy")


def process_prediction(pred):
    pred = pred['generated_text']
    
    pred = pred.strip().upper()
    
    pred = pred[0] if pred else 'I'
    pred = pred if pred in {'A', 'B', 'C', 'D'} else 'I'
    
    return pred


def compute_accuracy(model_preds, labels):   
    model_preds = list(map(process_prediction, model_preds))
    
    model_preds  = torch.LongTensor(list(map(ord, model_preds)))
    actual_labels = ord('A') + labels
    incorrect_labels = actual_labels.new_full(actual_labels.shape, ord('I'))
    
    acc_res = accuracy_metric.compute(predictions=model_preds, references=actual_labels)['accuracy']
    corr_res = 1.0 - accuracy_metric.compute(predictions=model_preds, references=incorrect_labels)['accuracy']
    
    return {'accuracy': acc_res, 'correctness': corr_res}
