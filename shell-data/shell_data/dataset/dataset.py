import torch
import torchvision
import logging
from typing import (
    Union,
    Optional,
    List,
    Dict,
)
from shell_data.utils.config import DatasetConfig
import os


DATASET_NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "fashion_mnist": 10,
    "cifar100": 100,
}


class LabelToTensor:
    def __call__(self, label: int) -> torch.Tensor:
        if isinstance(label, int):
            label = torch.tensor(label)
        return label


def get_vision_dataset_subsets(dataset_name: Optional[str] = "mnist", train: Optional[bool] = True) -> List[torch.utils.data.Subset]:
    data_type = "train" if train else "test"
    if dataset_name == "mnist":
        dataset = torchvision.datasets.MNIST(
            root='./data/cv', train=train, download=True, transform=torchvision.transforms.ToTensor(),
            target_transform=LabelToTensor(),
        )
    elif dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root='./data/cv', train=train, download=True, transform=torchvision.transforms.ToTensor(),
            target_transform=LabelToTensor(),
        )
    elif dataset_name == "fashion_mnist":
        dataset = torchvision.datasets.FashionMNIST(
            root='./data/cv', train=train, download=True, transform=torchvision.transforms.ToTensor(),
            target_transform=LabelToTensor(),
        )
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root='./data/cv', train=train, download=True, transform=torchvision.transforms.ToTensor(),
            target_transform=LabelToTensor(),
        )
    else:
        raise ValueError(f"dataset {dataset_name} not supported")
    # dim of dataset (channel, height, width)
    logging.debug(f"dataset dim: {dataset[0][0].shape}")
    subsets = []
    if not os.path.exists(f"./data/cv/{dataset_name}"):
        os.makedirs(f"./data/cv/{dataset_name}")
    for i in range(DATASET_NUM_CLASSES[dataset_name]):
        # if ./data/mnist_{i}.pt exists, load it
        # otherwise, create it
        try:
            subset = torch.load(
                f'./data/cv/{dataset_name}/{dataset_name}_{data_type}_cls={i}.pt')
        except:
            logging.debug(
                f"creating subset for {dataset_name}_{data_type}_cls={i}.pt")
            if not isinstance(dataset.targets, torch.Tensor):
                dataset.targets = torch.tensor(dataset.targets)
            subset = torch.utils.data.Subset(
                dataset, torch.where(dataset.targets == i)[0])
            torch.save(
                subset, f'./data/cv/{dataset_name}/{dataset_name}_{data_type}_cls={i}.pt')

        subsets.append(subset)

    logging.debug(f"subsets length: {[len(subset) for subset in subsets]}")
    return subsets


def get_train_val_test_subsets(dataset_name):
    test_subsets = get_vision_dataset_subsets(
        dataset_name=dataset_name,
        train=False,
    )
    train_val_subsets = get_vision_dataset_subsets(
        dataset_name=dataset_name,
        train=True,
    )
    train_subsets, val_subsets = [], []
    for train_val_subset in train_val_subsets:
        train_subset, val_subset = torch.utils.data.random_split(
            train_val_subset, [0.6, 0.4])
        # train_subset, val_subset = torch.utils.data.random_split(
        #     train_val_subset, [0.5, 0.5])
        train_subsets.append(train_subset)
        val_subsets.append(val_subset)

    return train_subsets, val_subsets, test_subsets


def get_vision_dataset_subsets_save_memory(subsets: List[torch.utils.data.Subset], size: Optional[Union[int, float, Dict[int, int], Dict[int, float]]] = 1.0) -> List[torch.utils.data.Subset]:
    """
    Load a computer vision classification dataset and partition it into subsets of size `size` for each class.
    """
    # dim of dataset (channel, height, width)
    sized_subsets = []
    for i, subset in enumerate(subsets):
        # size is a float or int or dict
        if isinstance(size, dict):
            subset_size = size.get(i, 1.0)
        else:
            subset_size = size

        # convert float (fraction) to int
        if isinstance(subset_size, int):
            assert subset_size <= len(
                subset), f"size {subset_size} > len(subset) {len(subset)}"
        elif isinstance(subset_size, float):
            assert 0 <= subset_size <= 1.0, f"size {subset_size} not in [0, 1.0]"
            subset_size = int(subset_size * len(subset))
        else:
            raise ValueError(f"size {size} should be int or float")

        subset = torch.utils.data.Subset(
            subset, torch.randperm(len(subset))[:subset_size])
        sized_subsets.append(subset)

    logging.debug(
        f"subsets length: {[len(subset) for subset in sized_subsets]} with provided size {size}")
    return sized_subsets


class LifelongDataset:
    def __init__(self, train_subsets, val_subsets, test_subsets, cfg: Optional[DatasetConfig] = {}) -> None:
        if isinstance(cfg, DatasetConfig):
            cfg = cfg.__dict__
        self.train_size = cfg.get('train_size', 0.8)
        self.val_size = cfg.get('val_size', 0.2)
        self.test_size = cfg.get('test_size', 1.0)
        self.dataset_name = cfg.get('name', 'mnist')

        logging.debug(
            f"DATASET: {self.dataset_name}, train_size: {self.train_size}, val_size: {self.val_size}, test_size: {self.test_size}")

        self.train_datasets = get_vision_dataset_subsets_save_memory(train_subsets,
                                                                     self.train_size)
        self.val_datasets = get_vision_dataset_subsets_save_memory(val_subsets,
                                                                   self.val_size)
        self.test_datasets = get_vision_dataset_subsets_save_memory(test_subsets,
                                                                    self.test_size)

        self.num_cls_per_task = cfg.get('num_cls_per_task', 2)
        self.num_task_per_life = cfg.get('num_task_per_life', 5)
        self.num_tasks = self.num_cls_per_task * self.num_task_per_life
        assert len(
            self.train_datasets) >= self.num_tasks, f'len(subsets) >= {len(self.train_datasets)}, self.num_tasks = {self.num_tasks}'
        assert len(
            self.val_datasets) >= self.num_tasks, f'len(subsets) >= {len(self.val_datasets)}, self.num_tasks = {self.num_tasks}'
        assert len(
            self.test_datasets) >= self.num_tasks, f'len(subsets) >= {len(self.test_datasets)}, self.num_tasks = {self.num_tasks}'

        self.test_size = cfg.get('test_size', 0.2)  # float or int

        self.num_tot_cls = len(self.train_datasets)
        self.reset()

    def reset(self) -> None:
        self.perm = torch.randperm(self.num_tot_cls)[:self.num_tasks]

    def get_task_indices(self, time: int) -> List[int]:
        assert time < self.num_task_per_life, f'time < self.num_task_per_life, {time} < {self.num_task_per_life}'
        start = time * self.num_cls_per_task
        end = start + self.num_cls_per_task
        return self.perm[start:end].tolist()

    def get_data_tasks(self, time: int, split: str) -> List[torch.utils.data.Subset]:
        task_indices = self.get_task_indices(time)
        # logging.critical(f"Getting data {task_indices} for split {split}")
        if split == 'train':
            task_datasets = [self.train_datasets[i] for i in task_indices]
        elif split == 'val':
            task_datasets = [self.val_datasets[i] for i in task_indices]
        elif split == 'test':
            task_datasets = [self.test_datasets[i] for i in task_indices]
        else:
            raise ValueError(f"split {split} should be train, val or test")
        return task_datasets

    def get_train_dataset(self, time: int, kind="one") -> torch.utils.data.Dataset:
        def flatten_list(l): return [item for sublist in l for item in sublist]
        if kind == "one":
            return torch.utils.data.ConcatDataset(self.get_data_tasks(time, 'train'))
        elif kind == "all":
            return torch.utils.data.ConcatDataset(flatten_list([self.get_data_tasks(t, 'train') for t in range(time + 1)]))
        else:
            raise ValueError(f"kind {kind} should be one or all")

    def get_val_dataset(self, time: int, kind="one") -> torch.utils.data.Dataset:
        def flatten_list(l): return [item for sublist in l for item in sublist]
        if kind == "one":
            return torch.utils.data.ConcatDataset(self.get_data_tasks(time, 'val'))
        elif kind == "all":
            return torch.utils.data.ConcatDataset(flatten_list([self.get_data_tasks(t, 'val') for t in range(time + 1)]))
        else:
            raise ValueError(f"kind {kind} should be one or all")

    def get_test_dataset(self, time: int, kind="one") -> torch.utils.data.Dataset:
        def flatten_list(l): return [item for sublist in l for item in sublist]
        if kind == "one":
            return torch.utils.data.ConcatDataset(self.get_data_tasks(time, 'test'))
        elif kind == "all":
            return torch.utils.data.ConcatDataset(flatten_list([self.get_data_tasks(t, 'test') for t in range(time + 1)]))
        else:
            raise ValueError(f"kind {kind} should be one or all")
