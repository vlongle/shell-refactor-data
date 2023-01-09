from abc import ABC, abstractmethod
from typing import (
    Optional,
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

    @abstractmethod
    def data_valuation(self, X: torch.tensor, y: torch.tensor, ll_time: int) -> torch.tensor:
        pass
    
    def train(self, train_dataloader, val_dataloader, n_epochs: int,
          val_every_n_epoch: int, patience: int, delta: float):
        return train(self.model, train_dataloader, val_dataloader, n_epochs, val_every_n_epoch, patience, delta)
    