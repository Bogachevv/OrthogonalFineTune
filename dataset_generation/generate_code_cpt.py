import datasets
from datasets import load_dataset, DatasetDict

from omegaconf import OmegaConf
import sys


def _generate_dataset(config) -> DatasetDict:
    dataset = load_dataset("DKYoon/starcoder-python-200k")
    dataset = dataset.rename_column('input', 'text')
    dataset = dataset.remove_columns(['output'])
    dataset = dataset['train'].train_test_split(
        test_size=config.get('val_size', 2048),
        seed=config.get('split_seed', 42),

    )

    dataset = DatasetDict({
        'train': dataset['train'],
        'validation': dataset['test']
    })

    return dataset


def generate(config_pth, out_dir):
    config = OmegaConf.load(config_pth)
    dataset: DatasetDict = _generate_dataset(config)

    dataset.save_to_disk(
        dataset_dict_path=out_dir,
        max_shard_size=config.get('max_shard_size', None),
        num_proc=config.get('num_proc', None),
    )


def main():
    if len(sys.argv) < 3:
        print('Usage:')
        print('python3 generate_code_cpt.py confg_path.yaml output_dir')
        exit(1)

    cfg_path = sys.argv[1]
    out_dir = sys.argv[2]

    generate(config_pth=cfg_path, out_dir=out_dir)


if __name__ == '__main__':
    main()
