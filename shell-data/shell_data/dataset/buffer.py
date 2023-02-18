import torch
from abc import ABC, abstractmethod
from typing import (
    Tuple,
    Union,
    Dict,
    List,
)
from shell_data.utils.utils import knn_dist
import logging


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

    def update_tasks(self, task_idx):
        pass


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

    def add_data(self, data, dedup=True):
        if dedup and len(self) > 0:
            data = self.dedup(data)
        x, y = data
        self.X = torch.cat((self.X, x))
        self.y = torch.cat((self.y, y))

    def dedup(self, data, ret_mask=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process data and remove the data that is already in the buffer.

        Return a mask: true if the data is NOT in the buffer. False otherwise.
        """
        if len(self) == 0:
            if ret_mask:
                return torch.ones(len(data[0]), dtype=torch.bool)
            return data

        x, y = data
        distances = knn_dist(x, self.X, k=1)
        # if distances = 0, then the data is already in the buffer and should be removed
        eps = 0.1  # HACK: because of floating point error
        mask = distances > eps
        logging.debug(f"No. of duplicates: {len(mask) - mask.sum()}")
        if ret_mask:
            return mask
        return x[mask], y[mask]

    def get_data(self, batch_size):
        """
        Sample (without replacement) a batch of data from the buffer.
        """
        batch_size = min(batch_size, len(self))
        idx = torch.randperm(len(self.X))[:batch_size]
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

    def save_buffer(self, path_name):
        torch.save(self.X, f"{path_name}_X.pt")
        torch.save(self.y, f"{path_name}_y.pt")

    def load(self, path_name):
        self.X = torch.load(f"{path_name}_X.pt")
        self.y = torch.load(f"{path_name}_y.pt")


class ClassifcationBuffer(SupervisedLearningBuffer):
    def __init__(self, dim, num_classes):
        super().__init__(dim, 'classification')
        self.num_classes = num_classes

    def get_cls_counts(self):
        # HACK: assume that the num_cls = 10
        return {f"cls_{i}": (self.y == i).sum().item() for i in range(self.num_classes)}


class ReservoirSamplingClassificationBuffer(ClassifcationBuffer):
    def __init__(self, dim, buffer_size, num_classes):
        super().__init__(dim, num_classes)
        self.buffer_size = buffer_size
        self._buffer_weights = torch.zeros(0)

    # https://avalanche-api.continualai.org/en/v0.1.0/_modules/avalanche/training/storage_policy.html#ReservoirSamplingBuffer
    def add_data(self, data, dedup=True):
        if len(data[0]) == 0:
            return
        if dedup and len(self) > 0:
            data = self.dedup(data)
        x, y = data
        # self.X = torch.cat((self.X, x))
        # self.y = torch.cat((self.y, y))
        new_weights = torch.rand(len(x))
        cat_weights = torch.cat([new_weights, self._buffer_weights])
        cat_x = torch.cat([x, self.X])
        cat_y = torch.cat([y, self.y])
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        buffer_idxs = sorted_idxs[:self.buffer_size]
        self.X = cat_x[buffer_idxs]
        self.y = cat_y[buffer_idxs]
        self._buffer_weights = sorted_weights[:self.buffer_size]


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

    def save_buffer(self, path):
        for buffer_id, buffer in enumerate(self.buffers):
            buffer.save_buffer(path_name=f'{path}_buffer_{buffer_id}')

    def load(self, path):
        for buffer_id, buffer in enumerate(self.buffers):
            buffer.load(path_name=f'{path}_buffer_{buffer_id}')

    def update_tasks(self, task_idx: List[int]):
        self.past_tasks += task_idx

    def get_cls_counts(self) -> dict:
        return {f"cls_{i}": len(b) for i, b in enumerate(self.buffers)}

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


def get_dataset_from_buffer(buffer: Buffer, data_size: int):
    buf_x, buf_y = buffer.get_data(
        batch_size=data_size
    )
    return torch.utils.data.TensorDataset(buf_x, buf_y)
