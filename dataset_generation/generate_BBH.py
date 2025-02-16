import datasets
from datasets import load_dataset, DatasetDict

from transformers import AutoTokenizer

from omegaconf import OmegaConf

import functools
import requests

import sys
import argparse


def _get_few_shot_prompts(task_names: list[str]) -> dict[str, str]:
    base_url = 'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/refs/heads/main/cot-prompts/{0}.txt'

    few_shot_prompts = {}
    for task_name in task_names:
        prompt_resp = requests.get(base_url.format(task_name))
        if prompt_resp.status_code != 200:
            raise RuntimeError(f"Can't load {task_name} task. Response status code: {prompt_resp.status_code}")

        prompt = prompt_resp.text
        prompt = '\n'.join(prompt.split('\n')[2:])

        few_shot_prompts[task_name] = prompt
    
    return few_shot_prompts
        

def _get_inference_instructions(example, tokenizer, few_shot_prompt):
    instructions = [
        {"role": "system", "content": f""},
        {"role": "user", "content": f"{few_shot_prompt}\n{example['input']}"},
    ]

    instructions_wa = [
        {"role": "assistant", "content": f""}
    ]

    text_wa_answer = tokenizer.apply_chat_template(
        instructions + instructions_wa,
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text_wa_answer': text_wa_answer}


def _load_task(config, tokenizer, task_name: str, few_shot_prompt: str) -> datasets.Dataset:
    dataset =  dataset  = load_dataset(
        'maveriq/bigbenchhard',
        task_name,
        trust_remote_code=True,
    )

    dataset = dataset.map(
        function=functools.partial(
            _get_inference_instructions,
            tokenizer=tokenizer,
            few_shot_prompt=few_shot_prompt,
        ),
        batched=False, 
        num_proc=config.num_proc,
    )

    return dataset['train']


def load_BBH(config, tokenizer):
    task_names = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting']

    few_shot_prompts = _get_few_shot_prompts(task_names)
    
    dataset_ls = []
    for task_name in task_names:
        dataset = _load_task(config, tokenizer, task_name, few_shot_prompts[task_name])
        dataset = dataset.add_column(name=task_name, column=[task_name] * len(dataset))
        
        dataset_ls.append(dataset)
    
    dataset = datasets.concatenate_datasets(dataset_ls)

    return DatasetDict({
        'test': dataset
    })


def _load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, 
        **OmegaConf.to_object(config.tokenizer_config)
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def generate(config_pth, out_dir, n_shots: int):
    config = OmegaConf.load(config_pth)
    
    tokenizer = _load_tokenizer(config)
    dataset = load_BBH(config, tokenizer)

    dataset.save_to_disk(
        dataset_dict_path=out_dir,
        max_shard_size=config.get('max_shard_size', None),
        num_proc=config.get('num_proc', None),
    )


def main():
    parser = argparse.ArgumentParser(
        description='Loading from the HF hub and processing Big-Bench-Hard dataset'
    )

    parser.add_argument('config', help='Path to config file')
    parser.add_argument('output', help='Path to output dir')

    args = parser.parse_args()
    cfg_path = args.config
    out_dir = args.output

    generate(
        config_pth=cfg_path, 
        out_dir=out_dir,
    )


if __name__ == '__main__':
    main()
