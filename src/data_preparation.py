from datasets import load_dataset, DatasetDict

from omegaconf import OmegaConf

import functools

__all__ = ['load_MMLU', 'load_multilang_MMLU', 'load_ARC']


def _letter_to_int(choise):
    if isinstance(choise, str):
        return ord(choise) - ord('A')
    
    if isinstance(choise, int):
        return choise
    
    raise ValueError(f'Incorrect type of choise: {type(choise)=}')


def _int_to_letter(choise):
    return chr(choise + ord('A'))


def _prepare_question(examples):
    prompt = f"{examples['question']}\n"
    for letter, choice in zip(('A', 'B', 'C', 'D'), examples['choices']):
        prompt += f"{letter}. {choice}\n"

    answer = _int_to_letter(examples['answer'])
    
    return prompt, answer


def _prepare_prompt(examples, dev_dataset = None):
    if dev_dataset:
        yield from map(_prepare_question, dev_dataset)
    
    yield _prepare_question(examples)


def _prepare_instruction_text(example, *, tokenizer, config, few_shot_datasets):
    subject_info =  f" about {example['subject']}" if 'subject' in example else ''

    instructions = [
        {
            "role": "system", 
            "content": f"The following are multiple choice questions (with answers){subject_info}. Output 'A', 'B', 'C', or 'D'. Full answer not needed."
        },
    ]

    if config.dataset_loader_config.get('n_shots', None) and example.get('subject', None):
        few_shot_dataset = few_shot_datasets[example['subject']]
        few_shot_dataset = few_shot_dataset.select(range(config.dataset_loader_config['n_shots']))
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
    
    answer = _letter_to_int(example['Answer'])
    
    return {'choices': choices, 'answer': answer}


def load_MMLU(config, tokenizer) -> DatasetDict:
    dataset_loader_config = config.dataset_loader_config

    mmlu_dataset =  load_dataset("cais/mmlu", dataset_loader_config.task_name)

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
        num_proc=dataset_loader_config.num_proc,
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
    dataset_loader_config = config.dataset_loader_config
    langs = config.get('MMMLU_langs', list())

    if not langs:
        raise ValueError('Languages not specified')

    if dataset_loader_config.get('n_shots', 0) != 0:
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
        num_proc=dataset_loader_config.num_proc,
    )

    multilang_mmlu_dataset = multilang_mmlu_dataset.map(
        function=_remove_answer, 
        batched=False
    )

    multilang_mmlu_dataset.set_format("torch")

    mmlu_dataset = load_MMLU(config=config, tokenizer=tokenizer)
    multilang_mmlu_dataset['EN_US'] = mmlu_dataset['test']

    return multilang_mmlu_dataset


def load_ARC(config, tokenizer) -> DatasetDict:
    dataset_loader_config = config.dataset_loader_config

    arc_dataset =  load_dataset("allenai/ai2_arc", dataset_loader_config.task_name)

    arc_dataset = arc_dataset.map(lambda q: q['choices'], batched=False)
    arc_dataset = arc_dataset.map(lambda q: {'answer': _letter_to_int(q['answerKey'])}, batched=False)
    arc_dataset = arc_dataset.filter(lambda q: len(q['label']) == 4)

    arc_dataset = arc_dataset.remove_columns(['choices', 'answerKey', 'id', 'label'])
    arc_dataset = arc_dataset.rename_column('text', 'choices')

    instructions_datasets = arc_dataset.map(
        function=functools.partial(
            _prepare_instruction_text,
            tokenizer=tokenizer,
            config=config,
            few_shot_datasets=None,
        ),
        batched=False, 
        num_proc=dataset_loader_config.num_proc,
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