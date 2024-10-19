from datasets import load_dataset, DatasetDict

from omegaconf import OmegaConf

import functools

__all__ = ['load_MMLU', 'load_multilang_MMLU']

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


def _multilang_get_choices(example):
    choices = [
        example[ltr]
        for ltr in ('A', 'B', 'C', 'D')
    ]
    
    answer = ord(example['Answer']) - ord('A')
    
    return {'choices': choices, 'answer': answer}


def load_MMLU(config, tokenizer) -> DatasetDict:
    mmlu_dataset =  load_dataset("cais/mmlu", config.task_name)
    loader_config = config.loader_config

    few_shot_datasets = {
        subject: mmlu_dataset['dev'].filter(lambda row: row['subject'] == subject)
        for subject in set(mmlu_dataset['dev']['subject'])
    }

    instructions_datasets = mmlu_dataset.map(
        function=functools.partial(
            _prepare_instruction_text,
            tokenizer=tokenizer,
            config=config,
            few_shot_datasets=few_shot_datasets,
        ),
        batched=False, 
        num_proc=loader_config.num_proc,
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


def load_multilang_MMLU(config, tokenizer) -> DatasetDict:
    loader_config = config.loader_config
    langs = config.get('MMMLU_langs', list())

    if not langs:
        return DatasetDict()

    if config.get('n_shots', 0) != 0:
        raise ValueError('Incorrect value of n_shots')

    multilang_mmlu_dataset = DatasetDict()

    for lang in langs:
        multilang_mmlu_dataset[lang] = load_dataset('openai/MMMLU', lang)['test']

    multilang_mmlu_dataset = multilang_mmlu_dataset.map(_multilang_get_choices)
    multilang_mmlu_dataset = multilang_mmlu_dataset.remove_columns(['A', 'B', 'C', 'D', 'Answer', 'Unnamed: 0'])
    multilang_mmlu_dataset = multilang_mmlu_dataset.rename_column('Question', 'question')
    multilang_mmlu_dataset = multilang_mmlu_dataset.rename_column('Subject', 'subject')

    multilang_mmlu_dataset = multilang_mmlu_dataset.map(
        function=functools.partial(
            _prepare_instruction_text,
            tokenizer=tokenizer,
            config=config,
            few_shot_datasets=None,
        ),
        batched=False, 
        num_proc=loader_config.num_proc,
    )

    multilang_mmlu_dataset = multilang_mmlu_dataset.map(
        function=_remove_answer, 
        batched=False
    )

    multilang_mmlu_dataset.set_format("torch")

    return multilang_mmlu_dataset