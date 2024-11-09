import datasets
from datasets import load_dataset, DatasetDict, Dataset

from transformers import AutoTokenizer

from omegaconf import OmegaConf

import functools

import sys

def prepare_prompt(example, tokenizer):
    system_instruction = 'You are a highly skilled programming assistant trained to provide helpful, clear, and accurate responses for software developers. Your primary goal is to assist users with writing, understanding, debugging, and optimizing code across various programming languages and frameworks. Provide direct, actionable answers. If code snippets are requested or would be beneficial, prioritize clear and well-commented code examples. Ensure that all code examples are syntactically correct and functional, especially for common languages (e.g., Python, JavaScript, Java, etc.). Use a supportive and professional tone. When generating code, include meaningful comments that clarify the purpose of each function, class, or critical section of code. Avoid overly obvious comments that repeat what the code states but focus on adding context.'
    user_instruction = example['instruction']
    assistant_instruction = example['response']

    instructions = [
        {'role': 'system', 'content': system_instruction},
        {'role': 'user', 'content': user_instruction},
        {'role': 'assistant', 'content': assistant_instruction}
    ]

    text = tokenizer.apply_chat_template(
        instructions,
        tokenize=False
    )

    return {'text': text}

def _generate_dataset(config, tokenizer) -> DatasetDict:
    dataset =  load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")

    processor = functools.partial(
        prepare_prompt,
        tokenizer=tokenizer,
    )
    dataset = dataset.map(processor, batched=False, num_proc=config.num_proc)
    
    dataset = dataset['train'].train_test_split(
        test_size=config.get('val_size', 2048),
        seed=config.get('split_seed', 42),
    )

    dataset = DatasetDict({
        'train': dataset['train'],
        'validation': dataset['test']
    })

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
