import sys
import argparse
import pathlib

from omegaconf import OmegaConf


def generate_config(base_config_pth: str, task_name: str, max_new_tokens: int, batch_size: int):
    print(f"Reading config from {base_config_pth}")
    config = OmegaConf.load(base_config_pth)
    base_config_pth = pathlib.Path(base_config_pth)

    config['task_name'] = task_name
    config['evaluation_config']['max_new_tokens'] = max_new_tokens
    config['evaluation_config']['batch_size'] = batch_size

    new_config_pth = base_config_pth.parent / f"{task_name}.yaml"
    print(f"Writing config to {new_config_pth}")
    with open(new_config_pth, 'w') as f:
        print(OmegaConf.to_yaml(config, resolve=False, sort_keys=False), file=f)


def main():
    parser = argparse.ArgumentParser(
        description='Insert task_name and max_new_tokens into config'
    )

    parser.add_argument('base_config', help='Path to base config file')
    parser.add_argument('--task',   required=True, type=str, help='Task name')
    parser.add_argument('--tokens', required=True, type=int, help='Max new tokens')
    parser.add_argument('--bs',     required=True, type=int, help='Batch size')

    args = parser.parse_args()
    base_config_pth = args.base_config
    task_name = args.task
    max_new_tokens = args.tokens
    batch_size = args.bs

    generate_config(base_config_pth, task_name, max_new_tokens, batch_size)


if __name__ == '__main__':
    main()
