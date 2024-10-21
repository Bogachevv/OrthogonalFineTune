import datasets
from datasets import load_dataset, DatasetDict, Dataset

from transformers import AutoTokenizer

from omegaconf import OmegaConf

import functools

import sys

def _get_BoolQ_instructions(example, tokenizer):
    instructions = [
        {"role": "system", "content": f"Please answer the following question with True or False. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"Question: {example['question']}\nAnswer format: True/False"},
    ]

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is {example['answer']}"}
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
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer}


def _get_PIQA_instructions(example, tokenizer):
    instructions = [
        {"role": "system", "content": f"Please choose the correct solution to the question. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"Question: {example['goal']}\nSolution1: {example['sol1']}\nSolution2: {example['sol2']}\nAnswer format: Solution1/Solution2"},
    ]

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is Solution{example['label']+1}"}
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
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer}


def _get_SIQA_instructions(example, tokenizer):
    instructions = [
        {"role": "system", "content": f"Please choose the correct answer to the question based on the context provided. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"Context: {example['context']}\nQuestion: {example['question']}\nA: {example['answerA']}\nB: {example['answerB']}\nC: {example['answerC']}\nAnswer format: A/B/C"},
    ]

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is {chr(int(example['label']) + ord('A') - 1)}"}
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
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer}


def _get_hellaswag_instructions(example, tokenizer):
    endings = '\n'.join(
        f'Ending{i}: {ending}'
        for i, ending in enumerate(example['endings'])
    )
    
    instructions = [
        {"role": "system", "content": f"Please choose the correct ending to complete the given sentence. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"{example['activity_label']}. {example['ctx']}\n{endings}\nAnswer format: Ending0/Ending1/Ending2/Ending3"},
    ]

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is Ending{example['label']}"}
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
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer}


def _get_winogrande_instructions(example, tokenizer):
    instructions = [
        {"role": "system", "content": f"Please choose the correct answer to fill in the blank to complete the given sentence. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"Sentence: {example['sentence']}\nOption1: {example['option1']}\nOption2: {example['option2']}\nAnswer format: Option1/Option2"},
    ]

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is Option{example['answer']}"}
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
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer}


def _get_ARC_instructions(example, tokenizer):
    answers = '\n'.join(
        f"{answer_label}: {answer_text}"
        for answer_label, answer_text in zip(example['choices']['label'], example['choices']['text'])
    )
    ans_format = '/'.join(
        f"{answer_label}"
        for answer_label in example['choices']['label']
    )
    
    instructions = [
        {"role": "system", "content": f"Please choose the correct answer to the question. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"Question: {example['question']}\n{answers}\nAnswer format: {ans_format}"},
    ]

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is {example['answerKey']}"}
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
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer}


def _get_OBQA_instructions(example, tokenizer):
    answers = '\n'.join(
        f"{answer_label}: {answer_text}"
        for answer_label, answer_text in zip(example['choices']['label'], example['choices']['text'])
    )
    ans_format = '/'.join(
        f"{answer_label}"
        for answer_label in example['choices']['label']
    )
    
    instructions = [
        {"role": "system", "content": f"Please choose the correct answer to the question. Follow the answer format, full answer not needed."},
        {"role": "user", "content": f"Question: {example['question_stem']}\n{answers}\nAnswer format: {ans_format}"},
    ]

    instructions_ans = [
        {"role": "assistant", "content": f"The correct answer is {example['answerKey']}"}
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


def _load_datasets(config) -> list[DatasetDict]:
    BoolQ_dataset = load_dataset('google/boolq')
    PIQA_dataset = load_dataset('ybisk/piqa', trust_remote_code=True)
    SIQA_dataset = load_dataset('allenai/social_i_qa', trust_remote_code=True)
    hellaswag_dataset = load_dataset("Rowan/hellaswag")
    winogrande_dataset = load_dataset('allenai/winogrande', 'winogrande_debiased', trust_remote_code=True)
    ARC_e_dataset = load_dataset("allenai/ai2_arc", 'ARC-Easy')
    ARC_c_dataset = load_dataset("allenai/ai2_arc", 'ARC-Challenge')
    OBQA_dataset = load_dataset("allenai/openbookqa", "main")

    return [
        BoolQ_dataset, 
        PIQA_dataset, 
        SIQA_dataset, 
        hellaswag_dataset, 
        winogrande_dataset, 
        ARC_e_dataset, 
        ARC_c_dataset, 
        OBQA_dataset
    ]


def _process_datasets(config, dataset_ls: list[DatasetDict], tokenizer) -> list[DatasetDict]:
    dataset_processors = (
        _get_BoolQ_instructions,
        _get_PIQA_instructions,
        _get_SIQA_instructions,
        _get_hellaswag_instructions,
        _get_winogrande_instructions,
        _get_ARC_instructions,
        _get_ARC_instructions,
        _get_OBQA_instructions
    )

    dataset_names = (
        'BoolQ', 'PIQA', 'SIQA', 'hellaswag', 'winogrande', 'ARC-E', 'ARC-C', 'OBQA'
    )

    new_dataset_ls = []
    for dataset, dataset_name, processor in zip(dataset_ls, dataset_names, dataset_processors):   
        processor = functools.partial(
            processor,
            tokenizer=tokenizer,
        )

        dataset = dataset.map(
            processor, 
            batched=False, 
            num_proc=config.num_proc,
        )

        for split_name in dataset:
            dataset[split_name] = dataset[split_name].add_column(
                name='task',
                column=[dataset_name] * len(dataset[split_name])
            )
        
        new_dataset_ls.append(dataset)

    return new_dataset_ls


def _generate_dataset(config, tokenizer) -> DatasetDict:
    dataset_ls = _load_datasets(config=config)
    dataset_ls = _process_datasets(config=config, dataset_ls=dataset_ls, tokenizer=tokenizer)

    train_dataset = datasets.concatenate_datasets([
        dataset['train'].select_columns(['task', 'text', 'text_wa_answer'])
        for dataset in dataset_ls
    ])

    validation_dataset = datasets.concatenate_datasets([
        dataset['validation'].select_columns(['task', 'text', 'text_wa_answer'])
        for dataset in dataset_ls
    ])

    return DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })


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
        print('python3 generate_common_reasoning.py confg_path.yaml output_dir')
        exit(1)

    cfg_path = sys.argv[1]
    out_dir = sys.argv[2]

    generate(config_pth=cfg_path, out_dir=out_dir)


if __name__ == '__main__':
    main()
