from datasets import load_dataset, DatasetDict, Dataset

from transformers import AutoTokenizer

from omegaconf import OmegaConf

import functools

import sys
import argparse


DEFAULT_SYSTEM_PROMPT = """"""


def _generate_instructions(example, tokenizer):
    system_prompt = DEFAULT_SYSTEM_PROMPT
    user_prompt = example['prompt']
    correct_answer = example['answer']

    instructions = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is {correct_answer}"}
    ]
    instructions_wa = [
        {"role": "assistant", "content": f"The correct answer is "}
    ]

    text = tokenizer.apply_chat_template(
        instructions + instructions_ans,
        tokenize=False
    )

    text_wa_answer = tokenizer.apply_chat_template(
        instructions + instructions_wa,
        tokenize=False,
        add_generation_prompt=False,
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]

    return {'text': text, 'text_wa_answer': text_wa_answer}


def _load_datasets(config, dataset_base_pth: str) -> dict[str, DatasetDict]:
    dataset_names = ['20Minuten', 'FOMC', 'MeetingBank', 'NumGLUE-ds', 'ScienceQA', 'C-STANCE', 'Lima', 'NumGLUE-cm', 'Py150']

    datasets = {
        dataset_name: load_dataset(
            f'{dataset_base_pth}/{dataset_name}',
            data_files = {"train": "train.json", "test": "test.json", "validation": "eval.json"}
        )
        for dataset_name in dataset_names
    }
    
    return datasets


def _process_datasets(config, datasets: dict[str, DatasetDict], tokenizer) -> dict[str, DatasetDict]:
    processor = functools.partial(
        _generate_instructions,
        tokenizer=tokenizer,
    )

    datasets = {
        dataset_name: dataset.map(processor, batched=False)
        for dataset_name, dataset in datasets.items()
    }

    return datasets


def _generate_datasets(config, tokenizer, dataset_base_pth: str) -> dict[str, DatasetDict]:
    datasets = _load_datasets(config=config, dataset_base_pth=dataset_base_pth)
    datasets = _process_datasets(config=config, datasets=datasets, tokenizer=tokenizer)

    return datasets


def _load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, 
        **OmegaConf.to_object(config.tokenizer_config)
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def generate(config_pth: str, out_dir: str, dataset_base_pth: str):
    config = OmegaConf.load(config_pth)

    tokenizer = _load_tokenizer(config)
    datasets: dict[str, DatasetDict] = _generate_datasets(config, tokenizer=tokenizer, dataset_base_pth=dataset_base_pth)

    for dataset_name in datasets.keys():
        datasets[dataset_name].save_to_disk(
            dataset_dict_path=f"{out_dir}/{dataset_name}",
            max_shard_size=config.get('max_shard_size', None),
            num_proc=config.get('num_proc', None),
        )


def main():
    parser = argparse.ArgumentParser(
        description='Loading from the HF hub and processing common reasoning dataset'
    )

    parser.add_argument('config', help='Path to config file')
    parser.add_argument('output', help='Path to output dir')
    parser.add_argument('source', help='Path to TRACE source')

    args = parser.parse_args()
    cfg_path = args.config
    out_dir = args.output
    dataset_base_pth = args.source

    print(f"Loading TRACE")
    generate(config_pth=cfg_path, out_dir=out_dir, dataset_base_pth=dataset_base_pth)


if __name__ == '__main__':
    main()
