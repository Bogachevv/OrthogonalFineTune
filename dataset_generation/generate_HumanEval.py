import datasets
from datasets import load_dataset, DatasetDict, Dataset

from transformers import AutoTokenizer

from omegaconf import OmegaConf

import functools

import sys


def prepare_prompt(example, tokenizer):
    system_instruction = 'You are a highly skilled programming assistant trained to provide helpful, clear, and accurate responses for software developers. You should compleete the following code. The following code already has the function prototype. You don`t have to rewrite it, just complete it.'
    user_instruction = example['prompt']

    instructions = [
        {'role': 'system', 'content': system_instruction},
        {'role': 'user', 'content': user_instruction},
    ]

    text = tokenizer.apply_chat_template(
        instructions,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {'text_wa_answer': text}


def _generate_dataset(config, tokenizer) -> DatasetDict:
    dataset =  load_dataset("openai/openai_humaneval")

    processor = functools.partial(
        prepare_prompt,
        tokenizer=tokenizer,
    )
    dataset = dataset.map(processor, batched=False, num_proc=config.num_proc)

    return dataset


def _load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, 
        **OmegaConf.to_object(config.tokenizer_config)
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def generate(config_pth, out_dir):
    config = OmegaConf.load(config_pth)

    tokenizer = _load_tokenizer(config)
    dataset: DatasetDict = _generate_dataset(config, tokenizer=tokenizer)

    dataset.save_to_disk(
        dataset_dict_path=out_dir,
        max_shard_size=config.get('max_shard_size', None),
        num_proc=config.get('num_proc', None),
    )


def main():
    if len(sys.argv) < 3:
        print('Usage:')
        print('python3 generate_code_ift.py confg_path.yaml output_dir')
        exit(1)

    cfg_path = sys.argv[1]
    out_dir = sys.argv[2]

    generate(config_pth=cfg_path, out_dir=out_dir)


if __name__ == '__main__':
    main()
