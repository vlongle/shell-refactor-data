from abc import ABC, abstractmethod
from typing import Tuple
import torch.nn as nn
import torch
from shell_data.utils.config import TaskModelConfig
from typing import (
    Optional,
    Union,
)
from shell_data.task_model.task_nets import CIFAR10Net, MNISTNet
import logging


class TaskModel(ABC):
    net: nn.Module
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    device: Union[torch.device, str]
    """
    TaskModel contains a network, loss function, and optimizer.
    It also contains the logic for training, validation, and testing.
    """
    @abstractmethod
    def train_step(self, batch) -> float:
        pass

    @abstractmethod
    def val_step(self, batch) -> float:
        pass

    @abstractmethod
    def test_step(self, batch) -> float:
        pass

# NOTE: head_id is a pretty ugly hack


class SupervisedLearningTaskModel(TaskModel):
    """
    SupervisedLearningTaskModel must contain a attribute `net` which is a nn.Module.
    A criterion (loss function) and optimizer must be provided.
    """

    def to_device(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Move data batch to the same device as the model.
        """
        x, y = batch
        if x.device != self.device:
            x = x.to(self.device)
        if y.device != self.device:
            y = y.to(self.device)
        return x, y

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor], head_id: Optional[int] = None) -> float:
        x, y = self.to_device(batch)
        self.net.train()
        self.optimizer.zero_grad()
        y_hat = self.net(x)
        if head_id is not None:
            y_hat = y_hat[:, head_id]
        loss = self.criterion(y_hat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], head_id: Optional[int] = None) -> float:
        x, y = self.to_device(batch)
        self.net.eval()
        with torch.no_grad():
            y_hat = self.net(x)
            if head_id is not None:
                y_hat = y_hat[:, head_id]
            loss = self.criterion(y_hat, y)
            return loss.item()

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor], head_id: Optional[int] = None) -> float:
        return self.test_step(batch, head_id)


class ClassifcationTaskModel(SupervisedLearningTaskModel):
    def __init__(self, n_classes: int, cfg: Optional[TaskModelConfig] = {}) -> None:
        if isinstance(cfg, TaskModelConfig):
            cfg = cfg.__dict__
        SupervisedLearningTaskModel.__init__(self)
        self.task_name = cfg.get("name", "cifar10")
        self.device = cfg.get("device", "cuda")
        if self.task_name == "cifar10":
            self.net = CIFAR10Net(n_out=n_classes)
        elif self.task_name == "mnist":
            self.net = MNISTNet(n_out=n_classes)
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")
        self.cfg = cfg
        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def test_acc(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        x, y = self.to_device(batch)
        self.net.eval()
        with torch.no_grad():
            y_hat = self.net(x)
            # logging.debug(f"y: {torch.unique(y, return_counts=True)}, y_hat: {torch.unique(y_hat.argmax(dim=1), return_counts=True)}")
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            return acc.item()


class RegressionTaskModel(SupervisedLearningTaskModel):
    def __init__(self, n_out: int, cfg: Optional[TaskModelConfig] = {}) -> None:
        if isinstance(cfg, TaskModelConfig):
            cfg = cfg.__dict__
        SupervisedLearningTaskModel.__init__(self)
        self.task_name = cfg.get("name", "cifar10")
        self.device = cfg.get("device", "cuda")
        if self.task_name == "cifar10":
            self.net = CIFAR10Net(n_out=n_out)
        elif self.task_name == "mnist":
            self.net = MNISTNet(n_out=n_out)
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")
        self.cfg = cfg
        self.net.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def reset(self):
        self.net.reset_parameters()
