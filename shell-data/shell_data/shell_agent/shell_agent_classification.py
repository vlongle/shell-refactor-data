from shell_data.utils.config import ShELLDataSharingConfig, validate_config
from shell_data.dataset.dataset import LifelongDataset
from shell_data.task_model.task_model import ClassifcationTaskModel
import torch
from typing import Optional, Tuple
import logging
from shell_data.dataset.buffer import BalancedClassificationBuffer, ClassifcationBuffer, get_dataset_from_buffer
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

NUM_CLASSES = 10

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
                 enable_validate_config: Optional[bool] = True,) -> None:
        if enable_validate_config:
            validate_config(cfg)
        ShELLAgent.__init__(self, cfg)
        self.n_epochs = cfg.training.n_epochs
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
        else:
            raise ValueError(f"Unknown dataset name: {self.cfg.dataset.name}")
        self.buffer = BalancedClassificationBuffer(
            dim=self.dim, num_classes=NUM_CLASSES)
        # self.buffer = ClassifcationBuffer(
        #     dim=self.dim)

        self.past_tasks = []

    def init_model_router(self):
        """
        NOTE: in multi-agent setting, to ensure no routing and routing have the same task distribution,
        (random number generator is not used to generate the weights of the task model and the router).
        We initialize the task model and the router only after we have initialized the dataset
        for every agent (in the __init__ function)
        """
        self.model = ClassifcationTaskModel(
            n_classes=10, cfg=self.cfg.task_model)

        self.router_cfg = self.cfg.router
        if self.router_cfg.strategy == "random":
            self.router = RandomShellRouter(self, self.router_cfg)
        elif self.router_cfg.strategy == "neural":
            self.router = NeuralShellRouter(self, self.router_cfg)
        elif self.router_cfg.strategy == "oracle":
            self.router = OracleShellRouter(self, self.router_cfg)
        elif self.router_cfg.strategy == "no_routing":
            self.router = None
        else:
            raise ValueError(
                f"Unknown router name: {self.router_cfg.strategy}")

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

    def learn_task(self, ll_time: int, record_name=None, metric="test_acc"):
        """
        ER learning: augment the train_dataset with data from the buffer
        to avoid catastrophic forgetting.
        """
        task_train_dataset = self.ll_dataset.get_train_dataset(ll_time)
        # avoid catastrophic forgetting by experience replay:
        # augment the train_dataset with data from the buffer
        if not self.buffer.is_empty():
            buf_x, buf_y = self.buffer.get_data(
                # batch_size=self.cfg.experience_replay.train_size * (ll_time + 1) * self.cfg.experience_replay.factor)
                batch_size=self.cfg.experience_replay.train_size * len(self.past_tasks))
            buf_dataset = torch.utils.data.TensorDataset(buf_x, buf_y)
            er_task_train_dataset = torch.utils.data.ConcatDataset(
                [task_train_dataset, buf_dataset])
        else:
            er_task_train_dataset = task_train_dataset

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
                                    head_id=None, ll_time=ll_time, record=record, metric=metric))

        record.save()

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
            (NUM_CLASSES, NUM_CLASSES))
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

    def learn_from_buffer(self, ll_time: int, kind="all", metric="past_task_test_acc", record_name=None, val_before=True):
        buf_x, buf_y = self.buffer.get_data(
            batch_size=self.cfg.experience_replay.buffer_size)
        logging.critical(
            f"Buffer size: {len(buf_x)} cls distribution {torch.unique(buf_y, return_counts=True)}")
        buf_dataset = torch.utils.data.TensorDataset(buf_x, buf_y)
        # split buf_dataset into train and val
        # NOTE: breaking change. buf_val_dataset doesn't matter since
        # we're evaluating on the test data...
        # buf_train_dataset, buf_val_dataset = torch.utils.data.random_split(
        #     buf_dataset, [0.9, 0.1])

        buf_train_dataset = buf_dataset
        # train_dataloader = torch.utils.data.DataLoader(
        #     buf_train_dataset, batch_size=self.train_batch_size, shuffle=True)

        def get_labels(dataset):
            return [int(dataset[i][1]) for i in range(len(dataset))]

        train_dataloader = torch.utils.data.DataLoader(
            buf_train_dataset, batch_size=self.train_batch_size,
            sampler=ImbalancedDatasetSampler(buf_train_dataset,  callback_get_label=get_labels))

        # NOTE: no class balancing sampler...
        # train_dataloader = torch.utils.data.DataLoader(
        #     buf_train_dataset, batch_size=self.train_batch_size,
        #     shuffle=True)

        # val_dataset = self.ll_dataset.get_val_dataset(ll_time, kind)
        # val_dataset = torch.utils.data.ConcatDataset(
        #     [val_dataset, buf_val_dataset])

        # val_dataloader = torch.utils.data.DataLoader(
        #     val_dataset, batch_size=min(128, len(val_dataset)), shuffle=False)

        if record_name is None:
            record_name = f"record_integration_{ll_time}.csv"
        record = Record(record_name)

        self.train(train_dataloader, val_dataloader=None,
                   n_epochs=self.n_epochs,
                   val_every_n_epoch=self.cfg.training.val_every_n_epoch,
                   patience=self.cfg.training.patience, delta=self.cfg.training.delta,
                   val_func=partial(self.val_func, val_dataloader=None,
                                    head_id=None, ll_time=ll_time, record=record, metric=metric),
                   val_before=val_before)
        record.save()

    def data_valuation_oracle(self, X, y, ll_time):
        rewards = torch.zeros(X.shape[0])
        for cl in range(NUM_CLASSES):
            X_cl = X[y == cl]
            y_cl = y[y == cl]
            if len(X_cl) == 0:
                continue
            # check if y is in past tasks
            if cl in self.past_tasks:
                rewards[y == cl] = 1.0
            else:
                rewards[y == cl] = -1.0
        return rewards

    def data_valuation_clustering(self, X, y, ll_time):
        pass

    def data_valuation_class_perf(self, X, y, ll_time, metric="past_task_val_acc", record_name=None):
        """
        Contribution of each class to the improvement on `past_task_test_acc`
        """
        rewards = torch.zeros(X.shape[0])
        if metric == "past_task_test_acc":
            before_past_task_test_acc = self.shared_test_val(
                ll_time, metric_type="acc", split="test", kind="all")
        elif metric == "past_task_val_acc":
            before_past_task_test_acc = self.shared_test_val(
                ll_time, metric_type="acc", split="val", kind="all")
        else:
            raise NotImplementedError

        snapshot_conf_mat(self, ll_time, "before")

        for cl in range(NUM_CLASSES):
            X_cl = X[y == cl]
            y_cl = y[y == cl]
            if len(X_cl) == 0:
                continue
            agent_copy = deepcopy(self)
            agent_copy.buffer.add_data((X_cl, y_cl))

            buf_x, buf_y = agent_copy.buffer.get_data(
                batch_size=self.cfg.experience_replay.buffer_size)
            train_cls_dataset = torch.utils.data.TensorDataset(
                buf_x, buf_y)
            snapshot_conf_mat(
                agent_copy, ll_time, f"before_class_{cl}", train_dataset=train_cls_dataset)

            if record_name is not None:
                cls_record_name = f"{record_name}_record_cls_eval_{ll_time}_class_{cl}.csv"
            else:
                cls_record_name = f"record_cls_eval_{ll_time}_class_{cl}.csv"

            if metric == "past_task_test_acc":
                agent_copy.learn_from_buffer(
                    ll_time, kind="all", metric="past_task_test_acc", record_name=cls_record_name,
                    val_before=False,)
                after_past_task_test_acc = agent_copy.shared_test_val(
                    ll_time, metric_type="acc", split="test", kind="all")

            elif metric == "past_task_val_acc":
                agent_copy.learn_from_buffer(
                    ll_time, kind="all", metric="past_task_val_acc", record_name=cls_record_name,
                    val_before=False,)
                after_past_task_test_acc = agent_copy.shared_test_val(
                    ll_time, metric_type="acc", split="val", kind="all")

            # TODO: have to add agent name here...
            agent_copy.save_model(f"{ll_time}_class_{cl}")
            agent_copy.save_buffer(f"{ll_time}_class_{cl}")

            snapshot_conf_mat(
                agent_copy, ll_time, f"after_class_{cl}", train_dataset=train_cls_dataset)

            reward = (after_past_task_test_acc -
                      before_past_task_test_acc) / before_past_task_test_acc

            logging.critical(
                f"Class {cl} before {before_past_task_test_acc} after {after_past_task_test_acc} len {len(X_cl)} contribution: {reward}")
            rewards[y == cl] = reward

        return rewards

    def data_valuation(self, X: torch.tensor, y: torch.tensor, ll_time: int, record_name=None) -> Tuple[torch.tensor, torch.tensor]:
        if self.cfg.data_valuation.method == "oracle":
            func = self.data_valuation_oracle
        elif self.cfg.data_valuation.method == "clustering":
            func = self.data_valuation_oracle
        elif self.cfg.data_valuation.method == "performance":
            func = self.data_valuation_class_perf
        else:
            raise NotImplementedError
        scores = func(X, y, ll_time, record_name=record_name,
                      metric=self.cfg.data_valuation.metric)

        # keep the data (add to buffer) if score > threshold
        to_keeps = scores > self.cfg.data_valuation.threshold
        data_to_keep = (X[to_keeps], y[to_keeps])
        if len(X[to_keeps]) > 0:
            mask = self.buffer.dedup(data_to_keep, ret_mask=True)
            # if mask is False means that the data is already in the buffer, we
            # will change the scores to -1 so that the preference model will
            # not send it in the future
            # change the scores to -1 so that the preference model will not
            # send it in the future

            # scores[to_keeps][mask] = -1
            indices = torch.where(to_keeps)[0]
            scores.scatter_(-1, indices[~mask], -1)
            self.buffer.add_data(data_to_keep)

        return scores, to_keeps

    # def train_buffer_past_task(ll_time: int, buffer_size: int):
    #     pass

    # def learn_from_buffer(self, X_cl: torch.tensor, y_cl: torch.tensor, ll_time: int, buffer_size: int,
    #                       val_strategy="all"):

    #     # buf_dataset = get_dataset_from_buffer(self.buffer,
    #     #                                       buffer_size)
    #     # # concat X_cl and y_cl to buf_dataset
    #     # buf_dataset = torch.utils.data.ConcatDataset(
    #     #     [buf_dataset, torch.utils.data.TensorDataset(X_cl, y_cl)])
    #     # buf_x, buf_y = self.buffer.get_data(
    #     #     batch_size=len(X_cl) * len(self.past_tasks))

    #     # NOTE: make sure to sample such that each cls is equally represented.
    #     buf_x, buf_y = self.buffer.get_data(
    #         batch_size=buffer_size)

    #     # HACK
    #     if X_cl is not None:
    #         # if y_cl[0] is in buf_y then we need to randomly switch out
    #         # one of the samples in buf_y with X_cl
    #         num_buf_y = (buf_y == y_cl[0]).sum()
    #         if num_buf_y >= len(y_cl):
    #             # switch out len(y_cl) samples in buf_y with X_cl
    #             # idxs = torch.randperm(num_buf_y)[:len(y_cl)]
    #             # sanity check, calculate the avg distance between X_cl and buf_x
    #             # switch out samples
    #             # do it in place
    #             # buf_x[buf_y == y_cl[0]][idxs] = X_cl
    #             indices = torch.nonzero(buf_y == y_cl[0]).flatten()

    #             # Generate a random permutation of the indices
    #             perm = torch.randperm(indices.size(0))

    #             # Use the permutation to shuffle the indices
    #             shuffled_indices = indices[perm]

    #             # Pick the first 10 indices
    #             selected_indices = shuffled_indices[:len(y_cl)]

    #             # logging.warning(
    #             #     f"before avg distance: {torch.mean(torch.norm(X_cl - buf_x[selected_indices], dim=1))} | selected_indices {selected_indices} ")
    #             # Replace the elements at these indices in buf_x with elements from X_cl
    #             buf_x[selected_indices] = X_cl

    #             # pick random samples from buf_x with class y_cl[0] and replace them with X_cl
    #             # idxs = torch.randperm(num_buf_y)[:len(y_cl)]
    #             # buf_x[buf_y == y_cl[0]][idxs] = X_cl

    #             # logging.warning(
    #             #     f"before avg distance: {torch.mean(torch.norm(X_cl - buf_x[selected_indices], dim=1))} | selected_indices {selected_indices} ")
    #         else:
    #             buf_x = torch.cat([buf_x, X_cl], dim=0)
    #             buf_y = torch.cat([buf_y, y_cl], dim=0)

    #     # if y_cl[0] in self.past_tasks:
    #     #     buf_x = buf_x[buf_y != y_cl[0]]
    #     #     buf_y = buf_y[buf_y != y_cl[0]]

    #     # buf_x = torch.cat([buf_x, X_cl], dim=0)
    #     # buf_y = torch.cat([buf_y, y_cl], dim=0)
    #     buf_dataset = torch.utils.data.TensorDataset(buf_x, buf_y)

    #     # self.buffer.add_data((X_cl, y_cl))
    #     # buf_dataset = get_dataset_from_buffer(self.buffer,
    #     #                                       buffer_size)

    #     # TODO: problem with this!! if the class-wise data is small,
    #     # we will get a wrong estimate!
    #     logging.critical(
    #         f"LEARN_FROM_BUFFER: Unique buf_y: {torch.unique(buf_dataset.tensors[1], return_counts=True)}")
    #     if val_strategy == "current":
    #         val_dataset = self.ll_dataset.get_val_dataset(ll_time)
    #     elif val_strategy == "all":
    #         val_datasets = [self.ll_dataset.get_val_dataset(
    #             t) for t in range(ll_time+1)]
    #         val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    #     else:
    #         raise NotImplementedError
    #     train_dataloader = torch.utils.data.DataLoader(
    #         buf_dataset, batch_size=self.train_batch_size, shuffle=True)
    #     val_dataloader = torch.utils.data.DataLoader(
    #         val_dataset, batch_size=min(128, len(val_dataset)), shuffle=False)
    #     self.train(train_dataloader, val_dataloader,
    #                n_epochs=self.n_epochs,
    #                val_every_n_epoch=self.cfg.training.val_every_n_epoch,
    #                patience=self.cfg.training.patience, delta=self.cfg.training.delta)

    # def data_valuation(self, X: torch.tensor, y: torch.tensor, ll_time: int) -> torch.tensor:
    #     pass

    # # def data_valuation(self, X: torch.tensor, y: torch.tensor, ll_time: int) -> torch.tensor:
    # #     """
    # #     Class-wise performance-based valuation of data points.
    # #     1. Add the data points to the buffer
    # #     2. Sample data from the buffer, and train the model on it
    # #     3. Evaluate the model on the test or val set
    # #     """
    # #     rewards = torch.zeros(X.shape[0])
    # #     # TODO: investigate why split = "val" leads to negative rewards
    # #     # even for the desired class
    # #     # split = "test"
    # #     split = "val"
    # #     before_test_losses, before_test_accs = self.test_on_past_tasks(
    # #         ll_time, split=split)

    # #     for cl in range(NUM_CLASSES):
    # #         X_cl = X[y == cl]
    # #         y_cl = y[y == cl]
    # #         if len(X_cl) == 0:
    # #             continue
    # #         agent_copy = deepcopy(self)
    # #         # agent_copy.buffer.add_data((X_cl, y_cl))
    # #         # logging.debug(
    # #         #     f"\t buffer: {[len(b) for b in agent_copy.buffer.buffers]}")

    # #         # logging.warning(
    # #         #     f"\t buf_y: {torch.unique(buf_y, return_counts=True)}")

    # #         agent_copy.learn_from_buffer(X_cl, y_cl,
    # #                                      ll_time, buffer_size=self.cfg.data_valuation.train_size)
    # #         after_test_losses, after_test_accs = agent_copy.test_on_past_tasks(
    # #             ll_time, split=split)
    # #         test_loss_diffs, test_acc_diffs = [], []
    # #         for t in range(ll_time+1):
    # #             test_loss_diffs.append(
    # #                 (before_test_losses[t] - after_test_losses[t])/before_test_losses[t])
    # #             test_acc_diffs.append(after_test_accs[t] - before_test_accs[t])
    # #             logging.warning(
    # #                 f"\t task {t+1} diff test loss: {test_loss_diffs[t]:.3f}, diff test acc: {test_acc_diffs[t]:.3f}")

    # #         if self.cfg.data_valuation.strategy == "last_loss":
    # #             reward = test_loss_diffs[-1]
    # #         elif self.cfg.data_valuation.strategy == "last_acc":
    # #             reward = test_acc_diffs[-1] * 100
    # #         elif self.cfg.data_valuation.strategy == "mean_loss":
    # #             reward = np.mean(test_loss_diffs)
    # #         elif self.cfg.data_valuation.strategy == "mean_acc":
    # #             reward = np.mean(test_acc_diffs) * 100
    # #         elif self.cfg.data_valuation.strategy == "best_mean":
    # #             reward = max(np.mean(test_loss_diffs),
    # #                          np.mean(test_acc_diffs) * 100)
    # #         else:
    # #             raise ValueError(
    # #                 f"Unknown strategy: {self.cfg.data_valuation.strategy}")

    # #         logging.warning(
    # #             f"reward for class {cl}: {reward:.3f} No. of data: {len(X_cl)}")
    # #         rewards[y == cl] = reward

    # #     for cl in range(NUM_CLASSES):
    # #         X_cl = X[y == cl]
    # #         y_cl = y[y == cl]
    # #         if len(X_cl) == 0:
    # #             continue
    # #         reward = rewards[y == cl].mean()
    # #         if reward > self.cfg.data_valuation.threshold:
    # #             logging.warning(f"Class {cl} is valuable. Adding to buffer.")
    # #             self.learn_from_buffer(X_cl, y_cl,
    # #                                    ll_time, buffer_size=self.cfg.data_valuation.train_size)
    # #             self.buffer.add_data((X_cl, y_cl))
    # #         else:
    # #             logging.warning(f"Class {cl} is not valuable. Discarding.")

    # #     return rewards
