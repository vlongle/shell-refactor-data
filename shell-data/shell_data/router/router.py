from shell_data.utils.config import RouterConfig
from abc import ABC, abstractmethod
from shell_data.shell_agent.shell_agent import ShELLAgent
import torch
from shell_data.router.explore_strategy import ShELLBotlzmanExploration
from shell_data.task_model.task_model import RegressionTaskModel
# from shell_data.task_model.task_nets import CIFAR10Net, MNISTNet
import torch.nn as nn
import numpy as np
from typing import Optional
from shell_data.dataset.buffer import RegressionBuffer, get_dataset_from_buffer
import logging
from shell_data.utils.utils import train
from functools import partial
from collections import defaultdict


class Router(ABC):
    """
    Router is the interface for all routers.
    """

    def __init__(self, agent: ShELLAgent, cfg: RouterConfig) -> None:
        super().__init__()
        self.agent = agent
        self.cfg = cfg
        # indices of the data that the receiver already kept
        self.to_keeps = defaultdict(list)

    @abstractmethod
    def share_with(self, other: ShELLAgent, other_id: int, ll_time: int):
        """
        Send `self.cfg.router.num_batches` batches of data to `other`.
        Each batch is of size `self.cfg.router.batch_size`.
        """
        pass

    @abstractmethod
    def get_candidate_data(self, other_id: int, ll_time: int):
        pass


class ConvenientRouter(Router):
    def __init__(self, agent: ShELLAgent, cfg: RouterConfig) -> None:
        super().__init__(agent, cfg)

    def get_candidate_data(self, other_id: int, ll_time: int, kind="all") -> torch.utils.data.Dataset:
        """
        Get all the data collected by this agent until now
        """
        # data = []
        # for t in range(ll_time+1):
        #     data.append(self.agent.ll_dataset.get_train_dataset(t))
        # return torch.utils.data.ConcatDataset(data)
        data = self.agent.ll_dataset.get_train_dataset(ll_time, kind=kind)
        # if dedup and other_id in self.to_keeps:
        #     # filter out the data that the receiver already kept
        #     data = data[~np.in1d(data.y, self.to_keeps[other_id])]
        return data

    def get_candidate_dataloader(self, other_id: int, ll_time: int) -> torch.utils.data.DataLoader:
        dataset = self.get_candidate_data(other_id, ll_time)
        logging.warning(f"Router available dataset size: {len(dataset)}")
        num_candidates = len(dataset) // self.cfg.num_batches
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=num_candidates, shuffle=True)
        return dataloader


class RandomShellRouter(ConvenientRouter):
    def __init__(self, agent: ShELLAgent, cfg: RouterConfig) -> None:
        super().__init__(agent, cfg)

    def share_with(self, other: ShELLAgent, other_id: int, ll_time: int, record_name=None):
        """
        Send `self.cfg.router.num_batches` batches of data to `other`.
        Each batch is of size `self.cfg.router.batch_size`.
        """
        logging.warning("Randomly routing...")
        dataloader = self.get_candidate_dataloader(other_id, ll_time)

        for X, y in dataloader:
            # pick randomly without replacement
            idx = torch.randperm(len(X))[:self.cfg.batch_size]
            scores, to_keeps = other.data_valuation(
                X[idx], y[idx], ll_time, record_name=record_name)
            self.to_keeps[other_id] += to_keeps.tolist()

    def reset(self):
        pass


class OracleShellRouter(ConvenientRouter):
    def __init__(self, agent: ShELLAgent, cfg: RouterConfig) -> None:
        super().__init__(agent, cfg)

    def share_with(self, other: ShELLAgent, other_id: int, ll_time: int):
        """
        Send `self.cfg.router.num_batches` batches of data to `other`.
        Each batch is of size `self.cfg.router.batch_size`.

        Select classes that are in other's `past_tasks`
        """
        logging.warning("oracle routing...")
        dataloader = self.get_candidate_dataloader(other_id, ll_time)

        for X, y in dataloader:
            # viable candidates are those that are in other's past_tasks
            viable_x = X[np.in1d(y, other.past_tasks)]
            viable_y = y[np.in1d(y, other.past_tasks)]
            logging.critical(f"viable {len(viable_x)} / {len(X)}")
            # pick randomly without replacement
            idx = torch.randperm(len(viable_x))[:self.cfg.batch_size]
            scores, to_keeps = other.data_valuation(
                viable_x[idx], viable_y[idx], ll_time)

            logging.critical(f"{scores} {len(scores)}")

            # this is wrong!!
            self.to_keeps[other_id] += to_keeps.tolist()

    def reset(self):
        pass


class NeuralShellRouter(ConvenientRouter):
    """
    Neural bandit with Boltzman Exploration
    Regression (pointwise) to estimate the value of each data point
    """

    def __init__(self, agent: ShELLAgent, cfg: RouterConfig) -> None:
        super().__init__(agent, cfg)
        self.n_heads = cfg.n_heads
        self.cfg = cfg
        self.exploration_strategy = ShELLBotlzmanExploration(
            n_heads=self.n_heads,
            cfg=self.cfg.explore)

        self.estimator = RegressionTaskModel(
            n_out=self.n_heads,
            cfg=self.cfg.estimator_task_model)

        if cfg.estimator_task_model.name == "cifar10":
            self.dim = (3, 32, 32)
        elif cfg.estimator_task_model.name == "mnist":
            self.dim = (1, 28, 28)
        else:
            raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")

        self.buffers = [RegressionBuffer(self.dim)
                        for _ in range(self.n_heads)]

    def reset(self):
        self.estimator.reset()
        self.exploration_strategy.reset()

    def get_Q(self, observations: np.ndarray, head_id: int, eval: Optional[bool] = True) -> np.ndarray:
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        observations = observations.to(self.estimator.device)
        if eval:
            self.estimator.net.eval()
            with torch.no_grad():
                Q = self.estimator.net(observations)[
                    :, head_id].squeeze().cpu().numpy()
        else:
            self.estimator.net.train()
            Q = self.estimator.net(observations)[:, head_id].squeeze()
        return Q

    def predict(self, obs: torch.tensor, other_agent_id: int) -> torch.tensor:
        Q_values = self.get_Q(obs, other_agent_id, eval=True)
        action = self.exploration_strategy.get_action(
            other_agent_id, Q_values)
        self.exploration_strategy.update(other_agent_id, action)
        return action

    def val_func(self, early_stopping, head_id, val_dataloader):
        val_loss = 0.0
        for val_batch in val_dataloader:
            val_batch_loss = self.model.val_step(val_batch, head_id=head_id)
            val_loss += val_batch_loss
        val_loss /= len(val_dataloader)
        # logging.info(
        #     f'epoch: {epoch+1} / {n_epochs}, loss: {train_loss:.3f} | val_loss {val_loss:.3f}')
        # val_losses.append(val_loss)
        # calculate the accuracy on the train/val/test set and log it to record
        return early_stopping.step(val_loss)

    # TODO: NOTE: regression loss is very problematic if the class-wise rewards
    # are very different (which is actually the desired behavior for RL and LTR algos),
    # but for supervised bandit regression, different scales of outputs are essentially
    # a "class-imbalance" problem. The network would just learn to predict the big scale
    # for some reasons...

    def update_from_buffer(self, other_id: int):
        # NOTE: we don't make sure that val and train sets are disjoint
        buffer = self.buffers[other_id]
        train_dataset = get_dataset_from_buffer(buffer, self.cfg.train_size)
        val_dataset = get_dataset_from_buffer(buffer, self.cfg.val_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=len(val_dataset), shuffle=True)
        return train(self.estimator, train_dataloader, val_dataloader, self.cfg.training.n_epochs, self.cfg.training.val_every_n_epoch, self.cfg.training.patience,
                     self.cfg.training.delta,
                     head_id=other_id,
                     val_func=partial(self.val_func, val_dataloader=val_dataloader,
                                      head_id=other_id,),
                     )

    # TODO: remember to reset in the outer loop!

    def share_with(self, other: ShELLAgent, other_id: int, ll_time: int):
        dataloader = self.get_candidate_dataloader(other_id, ll_time)
        logging.debug(f"Sharing with {other_id} at time {ll_time}")
        for step, (X, y) in enumerate(dataloader):
            if len(X) < self.cfg.batch_size:
                continue
            action = self.predict(X, other_id)
            rewards, to_keeps = other.data_valuation(
                X[action], y[action], ll_time)
            self.to_keeps[other_id] += to_keeps.tolist()
            # NOTE: online optimization without regression buffer
            # loss = self.estimator.train_step(
            #     (X[action], rewards), head_id=other_id)
            # logging.debug(f"\t Loss: {loss}")
            self.buffers[other_id].add_data((X[action], rewards))
            train_losses, val_losses = self.update_from_buffer(other_id)
            logging.warning(
                f"Routing step {step+1}, size {len(X)}, Train loss: {train_losses[-1]} - Val loss: {val_losses[-1]}")
