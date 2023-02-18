import torch.utils.data
from shell_data.utils.early_stopping import EarlyStopping
import logging
from typing import (
    Optional,
    Callable,
)
import numpy as np


def train(model, train_dataloader, val_dataloader, n_epochs: int,
          val_every_n_epoch: int, patience: int, delta: float,
          head_id: Optional[int] = None, val_func: Optional[Callable] = None, val_before=True,
          load_best_model=True):
    early_stopping = EarlyStopping(
        net=model.net,
        patience=patience, delta=delta)

    train_losses, val_losses = [], []
    global_step = 0
    train_loss = 0.0
    for epoch in range(n_epochs):
        epoch_train_losses = []
        # print("epoch:", epoch, "global_step:", global_step)
        if epoch % val_every_n_epoch == 0 and val_before:
            if val_func is not None and val_func(early_stopping=early_stopping, global_step=global_step,
                                                 epoch=epoch,
                                                 #  train_loss=np.mean(epoch_train_losses)):
                                                 train_loss=train_loss):
                logging.info(
                    f"Early stopping at epoch {epoch+1} with best val loss {early_stopping.val_loss_min:.3f}")
                break

        for batch in train_dataloader:
            train_loss = model.train_step(batch, head_id=head_id)
            train_losses.append(train_loss)
            epoch_train_losses.append(train_loss)
            global_step += 1

        if epoch % val_every_n_epoch == 0 and not val_before:
            if val_func is not None and val_func(early_stopping=early_stopping, global_step=global_step,
                                                 epoch=epoch,
                                                 train_loss=np.mean(epoch_train_losses)):
                logging.info(
                    f"Early stopping at epoch {epoch+1} with best val loss {early_stopping.val_loss_min:.3f}")
                break

    if load_best_model:
        model.net.load_state_dict(early_stopping.best_model)
    return train_losses, val_losses


def flatten_image(X: torch.tensor) -> torch.tensor:
    """
    Transform (batch_size, C, H, W) to (batch_size, C*H*W)
    """
    if len(X.shape) == 2:
        return X
    return X.view(X.size(0), -1)


def image_dist(images1: torch.tensor, images2: torch.tensor, p=2) -> torch.tensor:
    """
    Compute the Lp distance between pairs of images from the collection of images1 and images2
    images1 shape is (b1, C, H, W) and images2 shape is (b2, C, H, W)
    then the output is (b1, b2)
    """
    return torch.cdist(flatten_image(images1), flatten_image(images2), p=p)


def knn_dist(query_images: torch.tensor, anchor_images: torch.tensor, k=5, p=2) -> torch.tensor:
    """
    Compute the mean of distance from K-nearest neighbors of each image in query_images with the anchor_images.
    Distance is in Lp norm.
    """
    dist = image_dist(query_images, anchor_images, p=p)
    closest_dist, closest_idx = torch.topk(dist, k=k, dim=1, largest=False)
    return closest_dist.mean(dim=1)
