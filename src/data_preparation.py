from datasets import load_dataset, DatasetDict

from omegaconf import OmegaConf

import functools

__all__ = ['load_MMLU']

def _prepare_question(examples):
    prompt = f"{examples['question']}\n"
    for letter, choice in zip(('A', 'B', 'C', 'D'), examples['choices']):
        prompt += f"{letter}. {choice}\n"

    answer = chr(65 + examples['answer'])
    
    return prompt, answer


def _prepare_prompt(examples, dev_dataset = None):
    if dev_dataset:
        yield from map(_prepare_question, dev_dataset)
    
    yield _prepare_question(examples)


def _prepare_instruction_text(example, *, tokenizer, config, few_shot_datasets):
    instructions = [
        {
            "role": "system", 
            "content": f"The following are multiple choice questions (with answers) about {example['subject']}. Output 'A', 'B', 'C', or 'D'. Full answer not needed."
        },
    ]

    if config['n_shots'] and example['subject']:
        few_shot_dataset = few_shot_datasets[example['subject']]
        few_shot_dataset = few_shot_dataset.select(range(config['n_shots']))
    else:
        few_shot_dataset = None
    
    for prompt, ans in _prepare_prompt(example, dev_dataset=few_shot_dataset):
        instructions.append({"role": "user", "content": prompt})
        instructions.append({"role": "assistant", "content": ans})
    
    text = tokenizer.apply_chat_template(
        instructions,
        tokenize=False
    )
    
    return {'text': text}


def _remove_answer(example):
    text_wa_answer = example['text']
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0][:-1]
    
    return {'text_wa_answer': text_wa_answer}


def load_MMLU(config, tokenizer) -> DatasetDict:
    mmlu_dataset =  load_dataset("cais/mmlu", config.task_name)

    few_shot_datasets = {
        subject: mmlu_dataset['dev'].filter(lambda row: row['subject'] == subject)
        for subject in set(mmlu_dataset['dev']['subject'])
    }

    instructions_datasets = mmlu_dataset.map(
        function=functools.partial(
            func=_prepare_instruction_text,
            tokenizer=tokenizer,
            config=config,
            few_shot_datasets=few_shot_datasets,
        ),
        batched=False, 
        num_proc=2
    )

    instructions_datasets['validation'] = instructions_datasets['validation'].map(
        function=_remove_answer, 
        batched=False
    )
    instructions_datasets['test'] = instructions_datasets['test'].map(
        function=_remove_answer, 
        batched=False
    )

    instructions_datasets.set_format("torch")

    return instructions_datasets