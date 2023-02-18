from abc import ABC, abstractmethod
from typing import (
    Optional,
    Callable,
)
import torch
from shell_data.utils.config import ShELLDataSharingConfig
from shell_data.task_model.task_model import TaskModel
from shell_data.dataset.dataset import LifelongDataset
from shell_data.utils.utils import train


class ShELLAgent(ABC):
    model: TaskModel
    ll_dataset: LifelongDataset

    def __init__(self, cfg: Optional[ShELLDataSharingConfig] = {}) -> None:
        pass

    @abstractmethod
    def learn_task(self, ll_time: int):
        pass

    @abstractmethod
    def test(self, ll_time: int, type_: str = "acc"):
        pass

    def train(self, train_dataloader, val_dataloader, n_epochs: int,
              val_every_n_epoch: int, patience: int, delta: float,
              val_func: Optional[Callable] = None, val_before=True, load_best_model=True):
        return train(self.model, train_dataloader, val_dataloader, n_epochs, val_every_n_epoch, patience, delta,
                     val_func=val_func, val_before=val_before, load_best_model=load_best_model)
