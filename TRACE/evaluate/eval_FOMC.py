import json
from metrics import caculate_accuracy


def extract_answer(pred):
    pred = pred.strip()[0]

    return pred


def eval(predicted_sequences, ground_truths):
    predicted_sequences = list(map(extract_answer, predicted_sequences))

    accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result
