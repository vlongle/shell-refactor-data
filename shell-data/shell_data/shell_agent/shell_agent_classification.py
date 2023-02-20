from shell_data.utils.config import ShELLDataSharingConfig, validate_config
from shell_data.dataset.dataset import LifelongDataset
from shell_data.task_model.task_model import ClassifcationTaskModel
import torch
from typing import Optional, Tuple
import logging
from shell_data.dataset.buffer import BalancedClassificationBuffer, ClassifcationBuffer, get_dataset_from_buffer, ReservoirSamplingClassificationBuffer
from shell_data.router.router import (
    RandomShellRouter,
    NeuralShellRouter,
    OracleShellRouter,
)
from copy import deepcopy
import numpy as np
from shell_data.shell_agent.shell_agent import ShELLAgent
from shell_data.utils.record import Record, snapshot_conf_mat
from functools import partial

from torchsampler import ImbalancedDatasetSampler

from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


"""
NOTE: we're using the simple class-wise rebalancing dataloader technique
which duplicates the samples of the minority classes to balance the dataset.

There are more advanced techniques such as:
SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic samples
using k-nearest neighbors
Link: https://imbalanced-learn.org/stable/index.html
"""


class ShELLClassificationAgent(ShELLAgent):
    def __init__(self, train_subsets, val_subsets, test_subsets, cfg: Optional[ShELLDataSharingConfig] = {},
                 enable_validate_config: Optional[bool] = True, name: Optional[str] = "") -> None:
        if enable_validate_config:
            validate_config(cfg)
        ShELLAgent.__init__(self, cfg)
        self.name = name
        self.n_epochs = cfg.training.n_epochs
        self.buffer_n_epochs = cfg.experience_replay.n_epochs
        self.ll_dataset = LifelongDataset(
            train_subsets, val_subsets, test_subsets, cfg.dataset)
        self.train_batch_size = cfg.training.batch_size
        self.cfg = cfg

        if self.cfg.dataset.name == "cifar10":
            self.dim = (3, 32, 32)
        elif self.cfg.dataset.name == "mnist":
            self.dim = (1, 28, 28)
        elif self.cfg.dataset.name == "fashion_mnist":
            self.dim = (1, 28, 28)
        elif self.cfg.dataset.name == "cifar100":
            self.dim = (3, 32, 32)
        else:
            raise ValueError(f"Unknown dataset name: {self.cfg.dataset.name}")
        self.buffer = ReservoirSamplingClassificationBuffer(
            dim=self.dim, buffer_size=cfg.experience_replay.buffer_size,
            num_classes=self.cfg.num_classes,)

        self.sharing_buffer = ReservoirSamplingClassificationBuffer(
            dim=self.dim,
            buffer_size=self.cfg.sharing_buffer_size,
            num_classes=self.cfg.num_classes,
        )

        self.past_tasks = []

    def share_with(self, other: ShELLAgent, other_id: int, ll_time: int, record_name=None):
        if self.router is not None:
            self.router.share_with(
                other, other_id, ll_time, record_name=record_name)

    def update_tasks(self, ll_time):
        task_idx = self.ll_dataset.get_task_indices(ll_time)
        self.past_tasks += task_idx
        self.buffer.update_tasks(task_idx)

    def val_func(self, early_stopping, global_step, epoch, train_loss, ll_time, head_id, val_dataloader, record, metric="past_task_test_acc"):
        val_loss = self.shared_test_val(
            ll_time, metric_type="loss", split="val", kind="one")
        train_acc = self.shared_test_val(
            ll_time, metric_type="acc", split="train", kind="one")
        val_acc = self.shared_test_val(
            ll_time, metric_type="acc", split="val", kind="one")
        test_acc = self.shared_test_val(
            ll_time, metric_type="acc", split="test", kind="one")
        past_task_test_acc = self.shared_test_val(
            ll_time, metric_type="acc", split="test", kind="all")
        past_task_val_acc = self.shared_test_val(
            ll_time, metric_type="acc", split="val", kind="all")

        logging.info(
            f'epoch: {epoch+1}, step {global_step} loss: {train_loss:.3f} | val_loss {val_loss:.3f} | train_acc {train_acc:.3f} | val_acc {val_acc:.3f} | test_acc {test_acc:.3f} | past_task_test_acc {past_task_test_acc:.3f}')

        record.write({
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "past_task_test_acc": past_task_test_acc,
        })
        # val_losses.append(val_loss)
        # calculate the accuracy on the train/val/test set and log it to record
        # return early_stopping.step(val_loss)
        # return early_stopping.step(1 - val_acc)
        if metric == "past_task_test_acc":
            return early_stopping.step(1 - past_task_test_acc)
        elif metric == "val_acc":
            return early_stopping.step(1 - val_acc)
        elif metric == "test_acc":
            return early_stopping.step(1 - test_acc)
        elif metric == "past_task_val_acc":
            return early_stopping.step(1 - past_task_val_acc)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_task_buffer_dataset(self):
        buf_x, buf_y = self.buffer.get_data(
            batch_size=self.cfg.experience_replay.buffer_size)
        return torch.utils.data.TensorDataset(buf_x, buf_y)

    """
    NOTE: might can even use self.recognizer encoder to get the feature
    and decoder to get back images instead of flattening like this...
    """

    def rebalance(self, X, y):
        """
        Basically, https://arxiv.org/pdf/2105.02340.pdf (deepSMOTE)

        TODO: 
        1. throw away extreme class
        2. include SMOTE or other
        3. include VAE.
        """
        # return X, y

        # X.shape = (n, c, h, w) needs to reshape to(n, c*h*w)

        # with torch.no_grad():
        #     X = self.recognizer.embedding(
        #         X.to(self.cfg.task_model.device)).cpu()

        X = X.reshape(X.shape[0], -1)  # (n, embedding_dim)
        y = y.reshape(-1)
        # NOTE: SMOTE() expects that each class contains n_samples >= n_neighbors
        # that is used to find the k nearest neighbors for each sample.
        # so, we'll discard any class that is too small
        # use torch.unique to get the unique classes, and get their proportion
        # discard X and y if proportional is smaller than 5%
        # NOTE: this is a hacky way to do this, but it works for now
        y_unique, y_counts = torch.unique(y, return_counts=True)
        y_props = y_counts / y_counts.sum()
        good_cls = y_unique[y_props > 0.2 * y_props.max()]
        idx = torch.isin(y, good_cls)
        X = X[idx]
        y = y[idx]

        # X = X.reshape(X.shape[0], *self.dim)  # (n, embedding_dim)
        # return X, y

        # X_resampled, y_resampled = KMeansSMOTE().fit_resample(X, y)
        X_resampled, y_resampled = SVMSMOTE().fit_resample(X, y)
        # X_resampled, y_resampled = SMOTEENN().fit_resample(X, y)
        X_resampled = torch.from_numpy(X_resampled)  # (new_n, embedding_dim)
        y_resampled = torch.from_numpy(y_resampled)

        # with torch.no_grad():
        #     X_resampled = X_resampled.to(self.cfg.task_model.device)
        #     # reshape X_resampled to put it through the decoder
        #     X_resampled = X_resampled.reshape(
        #         X_resampled.shape[0], *self.recognizer.embedding_dim)

        #     X_resampled = self.recognizer.decoder(
        #         X_resampled).cpu()  # (new_n, c, h, w)

        X_resampled = X_resampled.reshape(
            X_resampled.shape[0], *self.dim)
        return X_resampled, y_resampled

    def get_buffer_dataset(self):
        buf_x, buf_y = self.buffer.get_data(
            batch_size=self.cfg.experience_replay.buffer_size)
        buf_x2, buf_y2 = self.sharing_buffer.get_data(
            batch_size=self.cfg.experience_replay.buffer_size)
        buf_x = torch.cat([buf_x, buf_x2], dim=0)
        buf_y = torch.cat([buf_y, buf_y2], dim=0)

        buf_x, buf_y = self.rebalance(buf_x, buf_y)

        return torch.utils.data.TensorDataset(buf_x, buf_y)

    def get_buffer_train_dataset(self, ll_time):
        task_train_dataset = self.ll_dataset.get_train_dataset(ll_time)
        # avoid catastrophic forgetting by experience replay:
        # augment the train_dataset with data from the buffer
        if not self.buffer.is_empty():
            buf_dataset = self.get_buffer_dataset()
            er_task_train_dataset = torch.utils.data.ConcatDataset(
                [task_train_dataset, buf_dataset])
        else:
            er_task_train_dataset = task_train_dataset

        return er_task_train_dataset, task_train_dataset

    def learn_task(self, ll_time: int, record_name=None, metric="val_acc", load_best_model=True):
        """
        ER learning: augment the train_dataset with data from the buffer
        to avoid catastrophic forgetting.
        """
        er_task_train_dataset, task_train_dataset = self.get_buffer_train_dataset(
            ll_time)
        val_dataset = self.ll_dataset.get_val_dataset(ll_time)

        def get_labels(dataset):
            return [int(dataset[i][1]) for i in range(len(dataset))]

        # task + er dataset
        # er_task_train_dataloader = torch.utils.data.DataLoader(
        #     er_task_train_dataset, batch_size=self.train_batch_size, shuffle=True,
        #     )
        er_task_train_dataloader = torch.utils.data.DataLoader(
            er_task_train_dataset, batch_size=self.train_batch_size,
            sampler=ImbalancedDatasetSampler(er_task_train_dataset, callback_get_label=get_labels))

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=min(128, len(val_dataset)), shuffle=False)

        # update past tasks
        self.update_tasks(ll_time)

        if record_name is None:
            record_name = f"record_{ll_time}.csv"
        record = Record(record_name)
        # local training
        self.train(er_task_train_dataloader, val_dataloader, n_epochs=self.n_epochs,
                   val_every_n_epoch=self.cfg.training.val_every_n_epoch,
                   patience=self.cfg.training.patience, delta=self.cfg.training.delta,
                   val_func=partial(self.val_func, val_dataloader=val_dataloader,
                                    head_id=None, ll_time=ll_time, record=record, metric=metric),
                   load_best_model=load_best_model)

        record.save()

    def add_buffer_task(self, ll_time: int):
        """
        Add data from task ll_time to the task buffer `self.buffer`. (i.e., not the sharing buffer)
        """
        _, task_train_dataset = self.get_buffer_train_dataset(
            ll_time)
        task_train_loader = torch.utils.data.DataLoader(
            task_train_dataset, batch_size=len(task_train_dataset), shuffle=False)
        # add task data to buffer
        for batch in task_train_loader:
            self.buffer.add_data(batch)

    def shared_test_val(self, ll_time: int, metric_type: str = "acc", split: str = None, kind="one",
                        ret_confusion_matrix=False, dataset=None):
        # either split or dataset must be specified
        assert split is not None or dataset is not None
        if dataset is None:
            if split == "test":
                dataset = self.ll_dataset.get_test_dataset(ll_time, kind)
            elif split == "val":
                dataset = self.ll_dataset.get_val_dataset(ll_time, kind)
            elif split == "train":
                dataset = self.ll_dataset.get_train_dataset(ll_time, kind)
            else:
                raise ValueError(f"Unknown split: {split}")
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=min(128, len(dataset)), shuffle=False)
        if metric_type == "acc":
            func = self.model.test_acc
        elif metric_type == "loss":
            func = self.model.test_step
        else:
            raise ValueError(f"Unknown type: {metric_type}")

        if ret_confusion_matrix:
            assert metric_type == "acc"

        metric = 0.0
        confusion_matrix = np.zeros(
            (self.cfg.num_classes, self.cfg.num_classes))
        for batch in dataloader:
            if ret_confusion_matrix:
                metric_step, confusion_step = func(
                    batch, ret_confusion_matrix=True)
                metric += metric_step
                confusion_matrix += confusion_step
            else:
                metric += func(batch)
        metric /= len(dataset)

        if ret_confusion_matrix:
            return metric, confusion_matrix
        return metric

    def test(self, ll_time: int, metric_type: str = "acc"):
        return self.shared_test_val(ll_time, metric_type=metric_type, split="test")

    def val(self, ll_time: int, metric_type: str = "acc"):
        return self.shared_test_val(ll_time, metric_type=metric_type, split="val")

    def save_model(self, path: str):
        torch.save(self.model.net.state_dict(), path)

    def save_buffer(self, path: str):
        self.buffer.save_buffer(path)
        # save past tasks
        torch.save(self.past_tasks, path + ".tasks")

    def load_model(self, path: str):
        self.model.net.load_state_dict(torch.load(path))

    def load_buffer(self, path: str):
        self.buffer.load(path)
        self.past_tasks = torch.load(path + ".tasks")
        self.buffer.update_tasks(self.past_tasks)

    def test_on_past_tasks(self, ll_time: int, split: Optional[str] = "test"):
        test_losses, test_accs = [], []
        for t in range(ll_time+1):
            test_loss = self.shared_test_val(
                t, metric_type="loss", split=split)
            test_acc = self.shared_test_val(t, metric_type="acc", split=split)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
        return test_losses, test_accs

    def learn_from_buffer(self, ll_time: int, kind="all", metric="past_task_val_acc", record_name=None, val_before=False, load_best_model=True):
        buf_train_dataset = self.get_buffer_dataset()

        def get_labels(dataset):
            return [int(dataset[i][1]) for i in range(len(dataset))]

        train_dataloader = torch.utils.data.DataLoader(
            buf_train_dataset, batch_size=self.train_batch_size,
            sampler=ImbalancedDatasetSampler(buf_train_dataset,  callback_get_label=get_labels))

        if record_name is None:
            record_name = f"record_integration_{ll_time}.csv"
        record = Record(record_name)

        self.train(train_dataloader, val_dataloader=None,
                   n_epochs=self.buffer_n_epochs,
                   val_every_n_epoch=self.cfg.training.val_every_n_epoch,
                   patience=self.cfg.training.patience, delta=self.cfg.training.delta,
                   val_func=partial(self.val_func, val_dataloader=None,
                                    head_id=None, ll_time=ll_time, record=record, metric=metric),
                   val_before=val_before, load_best_model=load_best_model)
        record.save()
