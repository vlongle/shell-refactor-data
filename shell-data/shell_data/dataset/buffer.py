import torch
from abc import ABC, abstractmethod
from typing import (
    Tuple,
    Union,
    Dict,
    List,
)


class Buffer(ABC):
    @abstractmethod
    def add_data(self, data):
        pass

    @abstractmethod
    def get_data(self, batch_size):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def is_empty(self):
        return len(self) == 0


class SupervisedLearningBuffer(Buffer):
    """
    Infinite-size buffer for supervised learning tasks.
    Consisting of a tensor of features and a tensor of labels.
    """

    def __init__(self, dim, task):
        super().__init__()
        self.dim = dim
        self.X = torch.empty(0, *dim)
        label_type = torch.long if task == 'classification' else torch.float
        self.y = torch.empty(0, dtype=label_type)

    def add_data(self, data):
        x, y = data
        self.X = torch.cat((self.X, x))
        self.y = torch.cat((self.y, y))

    def get_data(self, batch_size):
        """
        Sample (without replacement) a batch of data from the buffer.
        """
        batch_size = min(batch_size, len(self))
        idx = torch.randperm(len(self.X))[:batch_size]
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


class ClassifcationBuffer(SupervisedLearningBuffer):
    def __init__(self, dim):
        super().__init__(dim, 'classification')


class RegressionBuffer(SupervisedLearningBuffer):
    def __init__(self, dim):
        super().__init__(dim, 'regression')


class BalancedClassificationBuffer(Buffer):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.buffers = [ClassifcationBuffer(dim) for _ in range(num_classes)]
        self.past_tasks = []

    def update_tasks(self, task_idx: List[int]):
        self.past_tasks += task_idx

    def add_data(self, data):
        x, y = data
        for i in range(self.num_classes):
            idx = y == i
            self.buffers[i].add_data((x[idx], y[idx]))

    def get_data(self, batch_size: Union[int, Dict[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample (without replacement) a batch of data from the buffer,
        making sure that each class is represented equally.

        If batch_size is an integer, then the batch size is the same for
        all non-empty classes. If batch_size is a dictionary, then the
        batch size for each class is specified by the dictionary.
        """
        assert isinstance(batch_size, (int, dict))
        if isinstance(batch_size, int):
            nonzero_num_classes = sum([len(b) > 0 for b in self.buffers])
            min_num_samples = min([len(b) for b_id, b in enumerate(
                self.buffers) if len(b) > 0 and b_id in self.past_tasks])
        X = torch.empty(0, *self.dim)
        y = torch.empty(0, dtype=torch.long)
        for b_id, b in enumerate(self.buffers):
            if isinstance(batch_size, int):
                cls_batch_size = min(
                    batch_size // nonzero_num_classes, min_num_samples)
            elif isinstance(batch_size, dict) and b_id in batch_size:
                cls_batch_size = batch_size[b_id]
            else:
                continue
            if len(b) < cls_batch_size:
                continue
            cls_data = b.get_data(cls_batch_size)
            X = torch.cat((X, cls_data[0]))
            y = torch.cat((y, cls_data[1]))
        return X, y

    def __len__(self):
        return sum([len(b) for b in self.buffers])
