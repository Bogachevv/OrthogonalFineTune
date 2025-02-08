import sys
import argparse
import pathlib
import json
from omegaconf import OmegaConf


def get_model_name(task_idx, seq, base_ft_path):
    if task_idx == 0:
        return "meta-llama/Llama-3.2-1B-Instruct"
    
    prev_task_name = seq[task_idx - 1].strip()
    return f"{base_ft_path}/finetuned_model_{prev_task_name}"


def generate_config(base_config_pth, model_name: str, task_name: str, seq_no: int, max_new_tokens: int, batch_size: int, grad_acc_steps: int, num_epochs: int):
    print(f"Reading config from {base_config_pth}")
    config = OmegaConf.load(base_config_pth)
    base_config_pth = pathlib.Path(base_config_pth)

    config['model_name'] = model_name
    config['task_name'] = task_name
    config['seq_n'] = seq_no
    config['max_length'] = max_new_tokens

    config['trainer_config']['per_device_train_batch_size'] = batch_size
    config['trainer_config']['per_device_eval_batch_size'] = batch_size
    config['trainer_config']['gradient_accumulation_steps'] = grad_acc_steps
    config['trainer_config']['num_train_epochs'] = num_epochs

    new_config_pth = base_config_pth.parent / f"{task_name}.yaml"
    print(f"Writing config to {new_config_pth}")
    with open(new_config_pth, 'w') as f:
        print(OmegaConf.to_yaml(config, resolve=False, sort_keys=False), file=f)



def generate_configs(base_config_pth: str, base_ft_path: str, epochs_json_pth: str, seq_files_pth:str, seq_no: int, max_new_tokens: int, batch_size: int, grad_acc_steps: int):
    seq_path = f"{seq_files_pth}/seq-{seq_no}.txt"
    with open(seq_path, 'r') as f:
        seq = f.readlines()

    with open(epochs_json_pth, 'r') as f:
        epochs_dict = json.load(f)

    for i, task_name in enumerate(seq):
        task_name = task_name.strip()
        model_name = get_model_name(i, seq, base_ft_path)
        num_epochs = epochs_dict[task_name]

        generate_config(base_config_pth, model_name, task_name, seq_no, max_new_tokens, batch_size, grad_acc_steps, num_epochs)


def main():
    parser = argparse.ArgumentParser(
        description=''
    )

    parser.add_argument('--base_config',  required=True, type=str, help='Path to base config file')
    parser.add_argument('--base_ft_path', required=True, type=str, help='Path to base fine-tuned model dir')
    parser.add_argument('--epochs',       required=True, type=str, help='Path to epochs dictionary')
    parser.add_argument('--seq_files',    required=True, type=str, help='Path to sequences storage')
    parser.add_argument('--seq',          required=True, type=int, help='Datasets chain ID')
    parser.add_argument('--tokens',       required=True, type=int, help='Max new tokens')
    parser.add_argument('--bs',           required=True, type=int, help='Batch size')
    parser.add_argument('--acc_steps',    required=True, type=int, help='Gradient accumulation steps')

    args = parser.parse_args()
    base_config_pth = args.base_config
    base_ft_path = args.base_ft_path
    epochs_json_pth = args.epochs
    seq_files_pth = args.seq_files
    seq_no = args.seq
    max_new_tokens = args.tokens
    batch_size = args.bs
    grad_acc_steps = args.acc_steps

    generate_configs(base_config_pth, base_ft_path, epochs_json_pth, seq_files_pth, seq_no, max_new_tokens, batch_size, grad_acc_steps)


if __name__ == '__main__':
    main()
