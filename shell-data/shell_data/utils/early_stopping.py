import torch.nn as nn
from copy import deepcopy
from typing import Optional


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Save the model with the best validation loss.

    Patience is the number of steps to wait before stopping.
    Delta (positive int): is the minimum change in the monitored quantity to qualify as an improvement.
    """

    def __init__(self, net: nn.Module, patience: Optional[int] = 3, delta: Optional[float] = 0.0) -> None:
        self.model = net
        self.val_loss_min = None
        self.patience = patience
        self.waiting_steps = 0
        self.best_model = None
        self.delta = delta
        assert self.delta >= 0.0, "Delta must be non-negative"

    def step(self, val_loss: float) -> bool:
        """
        Returns True if the model should stop training.
        """
        if self.val_loss_min is None:
            # init
            self.val_loss_min = val_loss
            self.best_model = deepcopy(self.model.state_dict())
            return False
        elif val_loss < self.val_loss_min - self.delta:
            self.val_loss_min = val_loss
            self.waiting_steps = 0
            self.best_model = deepcopy(self.model.state_dict())
            return False
        else:
            self.waiting_steps += 1
            return self.waiting_steps >= self.patience
