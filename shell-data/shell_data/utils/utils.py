from shell_data.dataset.buffer import Buffer
import torch.utils.data
from shell_data.utils.early_stopping import EarlyStopping
import logging
from typing import Optional


def get_dataset_from_buffer(buffer: Buffer, data_size: int):
    buf_x, buf_y = buffer.get_data(
        batch_size=data_size
    )
    return torch.utils.data.TensorDataset(buf_x, buf_y)


def train(model, train_dataloader, val_dataloader, n_epochs: int,
          val_every_n_epoch: int, patience: int, delta: float,
          head_id: Optional[int] = None):
    X_val, y_val = next(iter(val_dataloader))
    early_stopping = EarlyStopping(
        net=model.net,
        patience=patience, delta=delta)

    train_losses, val_losses = [], []
    for epoch in range(n_epochs):
        for batch in train_dataloader:
            loss = model.train_step(batch, head_id=head_id)
            train_losses.append(loss)
        if epoch % val_every_n_epoch == 0:
            val_loss = model.val_step((X_val, y_val), head_id=head_id)
            logging.info(
                f'epoch: {epoch+1} / {n_epochs}, loss: {loss:.3f} | val_loss {val_loss:.3f}')

            val_losses.append(val_loss)
            if early_stopping.step(val_loss):
                logging.info(
                    f"Early stopping at epoch {epoch+1} with best val loss {early_stopping.val_loss_min:.3f}")
                model.net.load_state_dict(early_stopping.best_model)
                break

    return train_losses, val_losses
