from shell_data.utils.config import ShELLDataSharingConfig, validate_config
from shell_data.dataset.dataset import LifelongDataset
from shell_data.task_model.task_model import ClassifcationTaskModel
import torch
from typing import Optional
import logging
from shell_data.dataset.buffer import BalancedClassificationBuffer
from shell_data.router.router import RandomShellRouter, NeuralShellRouter
from copy import deepcopy
import numpy as np
from shell_data.shell_agent.shell_agent import ShELLAgent
from shell_data.utils.utils import get_dataset_from_buffer

NUM_CLASSES = 10


class ShELLClassificationAgent(ShELLAgent):
    def __init__(self, train_subsets, val_subsets, test_subsets, cfg: Optional[ShELLDataSharingConfig] = {}) -> None:
        validate_config(cfg)
        ShELLAgent.__init__(self, cfg)
        self.n_epochs = cfg.training.n_epochs
        self.ll_dataset = LifelongDataset(
            train_subsets, val_subsets, test_subsets, cfg.dataset)
        self.train_batch_size = cfg.training.batch_size
        self.model = ClassifcationTaskModel(n_classes=10, cfg=cfg.task_model)
        self.cfg = cfg
        if cfg.dataset.name == "cifar10":
            self.dim = (3, 32, 32)
        elif cfg.dataset.name == "mnist":
            self.dim = (1, 28, 28)
        else:
            raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")
        self.buffer = BalancedClassificationBuffer(
            dim=self.dim, num_classes=NUM_CLASSES)

        self.past_tasks = []

        self.router_cfg = cfg.router
        if self.router_cfg.strategy == "random":
            self.router = RandomShellRouter(self, self.router_cfg)
        elif self.router_cfg.strategy == "neural":
            self.router = NeuralShellRouter(self, self.router_cfg)
        else:
            raise ValueError(
                f"Unknown router name: {self.router_cfg.strategy}")

    def update_tasks(self, ll_time):
        task_idx = self.ll_dataset.get_task_indices(ll_time)
        self.past_tasks += task_idx
        self.buffer.update_tasks(task_idx)

    def learn_task(self, ll_time: int):
        """
        ER learning: augment the train_dataset with data from the buffer
        to avoid catastrophic forgetting.
        """
        train_dataset = self.ll_dataset.get_train_dataset(ll_time)
        # avoid catastrophic forgetting by experience replay:
        # augment the train_dataset with data from the buffer
        if not self.buffer.is_empty():
            buf_x, buf_y = self.buffer.get_data(
                batch_size=self.cfg.experience_replay.train_size * (ll_time + 1) * self.cfg.experience_replay.factor)
            buf_dataset = torch.utils.data.TensorDataset(buf_x, buf_y)
            train_dataset = torch.utils.data.ConcatDataset(
                [train_dataset, buf_dataset])

        val_dataset = self.ll_dataset.get_val_dataset(ll_time)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.train_batch_size, shuffle=True)

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=len(val_dataset), shuffle=False)

        # update past tasks
        self.update_tasks(ll_time)

        # local training
        self.train(train_dataloader, val_dataloader, n_epochs=self.n_epochs,
                   val_every_n_epoch=self.cfg.training.val_every_n_epoch,
                   patience=self.cfg.training.patience, delta=self.cfg.training.delta)

        # add task data to buffer
        for batch in train_dataloader:
            self.buffer.add_data(batch)

    def shared_test_val(self, ll_time: int, type_: str = "acc", split: str = "test"):
        if split == "test":
            test_dataset = self.ll_dataset.get_test_dataset(ll_time)
        elif split == "val":
            test_dataset = self.ll_dataset.get_val_dataset(ll_time)
        else:
            raise ValueError(f"Unknown split: {split}")
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False)
        X_test, y_test = next(iter(test_dataloader))
        if type_ == "acc":
            return self.model.test_acc((X_test, y_test))
        elif type_ == "loss":
            return self.model.test_step((X_test, y_test))
        else:
            raise ValueError(f"Unknown type: {type_}")

    def test(self, ll_time: int, type_: str = "acc"):
        return self.shared_test_val(ll_time, type_=type_, split="test")

    def val(self, ll_time: int, type_: str = "acc"):
        return self.shared_test_val(ll_time, type_=type_, split="val")

    def save_model(self, path: str):
        torch.save(self.model.net.state_dict(), path)

    def load_model(self, path: str):
        self.model.net.load_state_dict(torch.load(path))

    def test_on_past_tasks(self, ll_time: int, split: Optional[str] = "test"):
        test_losses, test_accs = [], []
        for t in range(ll_time+1):
            test_loss = self.shared_test_val(t, type_="loss", split=split)
            test_acc = self.shared_test_val(t, type_="acc", split=split)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
        return test_losses, test_accs

    # def train_buffer_past_task(ll_time: int, buffer_size: int):
    #     pass

    def learn_from_buffer(self, X_cl: torch.tensor, y_cl: torch.tensor, ll_time: int, buffer_size: int,
                          val_strategy="all"):
        # buf_dataset = get_dataset_from_buffer(self.buffer,
        #                                       buffer_size)
        # # concat X_cl and y_cl to buf_dataset
        # buf_dataset = torch.utils.data.ConcatDataset(
        #     [buf_dataset, torch.utils.data.TensorDataset(X_cl, y_cl)])
        # buf_x, buf_y = self.buffer.get_data(
        #     batch_size=len(X_cl) * len(self.past_tasks))

        # NOTE: make sure to sample such that each cls is equally represented.
        buf_x, buf_y = self.buffer.get_data(
            batch_size=buffer_size)

        # if y_cl[0] is in buf_y then we need to randomly switch out
        # one of the samples in buf_y with X_cl
        num_buf_y = (buf_y == y_cl[0]).sum()
        if num_buf_y >= len(y_cl):
            # switch out len(y_cl) samples in buf_y with X_cl
            # idxs = torch.randperm(num_buf_y)[:len(y_cl)]
            # sanity check, calculate the avg distance between X_cl and buf_x
            # switch out samples
            # do it in place
            # buf_x[buf_y == y_cl[0]][idxs] = X_cl
            indices = torch.nonzero(buf_y == y_cl[0]).flatten()

            # Generate a random permutation of the indices
            perm = torch.randperm(indices.size(0))

            # Use the permutation to shuffle the indices
            shuffled_indices = indices[perm]

            # Pick the first 10 indices
            selected_indices = shuffled_indices[:len(y_cl)]

            # logging.warning(
            #     f"before avg distance: {torch.mean(torch.norm(X_cl - buf_x[selected_indices], dim=1))} | selected_indices {selected_indices} ")
            # Replace the elements at these indices in buf_x with elements from X_cl
            buf_x[selected_indices] = X_cl

            # pick random samples from buf_x with class y_cl[0] and replace them with X_cl
            # idxs = torch.randperm(num_buf_y)[:len(y_cl)]
            # buf_x[buf_y == y_cl[0]][idxs] = X_cl

            # logging.warning(
            #     f"before avg distance: {torch.mean(torch.norm(X_cl - buf_x[selected_indices], dim=1))} | selected_indices {selected_indices} ")
        else:
            buf_x = torch.cat([buf_x, X_cl], dim=0)
            buf_y = torch.cat([buf_y, y_cl], dim=0)

        # if y_cl[0] in self.past_tasks:
        #     buf_x = buf_x[buf_y != y_cl[0]]
        #     buf_y = buf_y[buf_y != y_cl[0]]

        # buf_x = torch.cat([buf_x, X_cl], dim=0)
        # buf_y = torch.cat([buf_y, y_cl], dim=0)
        buf_dataset = torch.utils.data.TensorDataset(buf_x, buf_y)

        # self.buffer.add_data((X_cl, y_cl))
        # buf_dataset = get_dataset_from_buffer(self.buffer,
        #                                       buffer_size)

        # TODO: problem with this!! if the class-wise data is small,
        # we will get a wrong estimate!
        logging.warning(
            f"unique buf_y: {torch.unique(buf_dataset.tensors[1], return_counts=True)}")
        if val_strategy == "current":
            val_dataset = self.ll_dataset.get_val_dataset(ll_time)
        elif val_strategy == "all":
            val_datasets = [self.ll_dataset.get_val_dataset(
                t) for t in range(ll_time+1)]
            val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        else:
            raise NotImplementedError
        train_dataloader = torch.utils.data.DataLoader(
            buf_dataset, batch_size=self.train_batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=len(val_dataset), shuffle=False)
        self.train(train_dataloader, val_dataloader,
                   n_epochs=self.n_epochs,
                   val_every_n_epoch=self.cfg.training.val_every_n_epoch,
                   patience=self.cfg.training.patience, delta=self.cfg.training.delta)

    def data_valuation(self, X: torch.tensor, y: torch.tensor, ll_time: int) -> torch.tensor:
        """
        Class-wise performance-based valuation of data points.
        1. Add the data points to the buffer 
        2. Sample data from the buffer, and train the model on it
        3. Evaluate the model on the test or val set 
        """
        rewards = torch.zeros(X.shape[0])
        # TODO: investigate why split = "val" leads to negative rewards
        # even for the desired class
        # split = "test"
        split = "val"
        before_test_losses, before_test_accs = self.test_on_past_tasks(
            ll_time, split=split)

        for cl in range(NUM_CLASSES):
            X_cl = X[y == cl]
            y_cl = y[y == cl]
            if len(X_cl) == 0:
                continue
            agent_copy = deepcopy(self)
            # agent_copy.buffer.add_data((X_cl, y_cl))
            # logging.debug(
            #     f"\t buffer: {[len(b) for b in agent_copy.buffer.buffers]}")

            # logging.warning(
            #     f"\t buf_y: {torch.unique(buf_y, return_counts=True)}")

            agent_copy.learn_from_buffer(X_cl, y_cl,
                                         ll_time, buffer_size=self.cfg.data_valuation.train_size)
            after_test_losses, after_test_accs = agent_copy.test_on_past_tasks(
                ll_time, split=split)
            test_loss_diffs, test_acc_diffs = [], []
            for t in range(ll_time+1):
                test_loss_diffs.append(
                    (before_test_losses[t] - after_test_losses[t])/before_test_losses[t])
                test_acc_diffs.append(after_test_accs[t] - before_test_accs[t])
                logging.warning(
                    f"\t task {t+1} diff test loss: {test_loss_diffs[t]:.3f}, diff test acc: {test_acc_diffs[t]:.3f}")

            if self.cfg.data_valuation.strategy == "last_loss":
                reward = test_loss_diffs[-1]
            elif self.cfg.data_valuation.strategy == "last_acc":
                reward = test_acc_diffs[-1] * 100
            elif self.cfg.data_valuation.strategy == "mean_loss":
                reward = np.mean(test_loss_diffs)
            elif self.cfg.data_valuation.strategy == "mean_acc":
                reward = np.mean(test_acc_diffs) * 100
            elif self.cfg.data_valuation.strategy == "best_mean":
                reward = max(np.mean(test_loss_diffs),
                             np.mean(test_acc_diffs) * 100)
            else:
                raise ValueError(
                    f"Unknown strategy: {self.cfg.data_valuation.strategy}")

            logging.warning(
                f"reward for class {cl}: {reward:.3f} No. of data: {len(X_cl)}")
            rewards[y == cl] = reward

        for cl in range(NUM_CLASSES):
            X_cl = X[y == cl]
            y_cl = y[y == cl]
            if len(X_cl) == 0:
                continue
            reward = rewards[y == cl].mean()
            if reward > self.cfg.data_valuation.threshold:
                logging.warning(f"Class {cl} is valuable. Adding to buffer.")
                self.learn_from_buffer(X_cl, y_cl,
                                       ll_time, buffer_size=self.cfg.data_valuation.train_size)
                self.buffer.add_data((X_cl, y_cl))
            else:
                logging.warning(f"Class {cl} is not valuable. Discarding.")

        return rewards
