import json
from metrics import caculate_accuracy
import re

def extract_answer(pred):
    expr_float = re.compile(r'[+-]?([0-9]*[.])?[0-9]+')
    expr_word = re.compile(r'\w+')

    pred = pred.strip()

    float_match = expr_float.match(pred)
    if float_match is not None:
        return float_match.group()

    word_match = expr_word.match(pred)
    if word_match is not None:
        return word_match.group()

    return pred


def eval(predicted_sequences, ground_truths):
    predicted_sequences = list(map(extract_answer, predicted_sequences))

    accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result
