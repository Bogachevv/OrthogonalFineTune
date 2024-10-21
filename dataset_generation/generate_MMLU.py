import datasets
from datasets import load_dataset, DatasetDict

from transformers import AutoTokenizer

from omegaconf import OmegaConf

import functools

import sys


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

    instructions_datasets['train'] = instructions_datasets.pop('auxiliary_train')

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

def _load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, 
        **OmegaConf.to_object(config.tokenizer_config)
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def generate(config_pth, out_dir, ds_name: str, n_shots: int):
    config = OmegaConf.load(config_pth)

    tokenizer = _load_tokenizer(config)
    config['n_shots'] = n_shots

    if ds_name == 'MMLU':
        dataset = load_MMLU(config, tokenizer)
    elif ds_name == 'MMMLU':
        dataset = load_multilang_MMLU(config, tokenizer)
    else:
        raise ValueError(f"Incorrect {ds_name=}")

    dataset.save_to_disk(
        dataset_dict_path=out_dir,
        max_shard_size=config.get('max_shard_size', None),
        num_proc=config.get('num_proc', None),
    )


def main():
    if len(sys.argv) < 4:
        print('Usage:')
        print('python3 generate_common_MMLU.py confg_path.yaml output_dir ds_name [n_shots]')
        exit(1)

    cfg_path = sys.argv[1]
    out_dir = sys.argv[2]
    ds_name = sys.argv[3].strip()
    n_shots = int(sys.argv[4]) if (len(sys.argv) >= 5) else 0

    generate(
        config_pth=cfg_path, 
        out_dir=out_dir,
        ds_name=ds_name,
        n_shots=n_shots
    )


if __name__ == '__main__':
    main()
