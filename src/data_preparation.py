from datasets import load_dataset, DatasetDict

from omegaconf import OmegaConf

__all__ = ['load_dataset']


def load_dataset(config) -> DatasetDict:
    dataset_path = config.dataset_path

    instruction_dataset = DatasetDict.load_from_disk(dataset_path)

    return instruction_dataset
