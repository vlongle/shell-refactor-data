from abc import ABC, abstractmethod
import numpy as np
from shell_data.utils.config import BoltzmanExplorationConfig
import logging


class ExplorationStrategy(ABC):
    @abstractmethod
    def get_action(self, Q_values: np.ndarray):
        """
        Choose actions given Q-values. 
        """
        pass

    @abstractmethod
    def update(self, actions: np.ndarray):
        """
        Update exploration params
        """
        pass


class BotlzmanExploration(ExplorationStrategy):
    def __init__(self, cfg: BoltzmanExplorationConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.init_eps = cfg.epsilon
        self.reset()

    def reset(self):
        self.eps = self.init_eps

    def get_action(self, Q_values: np.ndarray):
        logging.debug(f"eps = {self.eps}")
        assert len(
            Q_values) >= self.cfg.num_slates, f"len(Q_values) = {len(Q_values)} < {self.cfg.num_slates}"
        # numerically stable softmax
        Q_values = Q_values - np.max(Q_values)
        weights = np.exp(Q_values / self.eps)
        probs = weights / np.sum(weights)
        # sample without replacement if there's enough non-zero probs, otherwise
        # send all non-zero probs
        if np.sum(probs > 0) >= self.cfg.num_slates:
            actions = np.random.choice(
                np.arange(len(probs)), size=self.cfg.num_slates, replace=False, p=probs)
        else:
            actions = np.nonzero(probs)[0]
            logging.critical(
                f"sending all non-zero weights shape {actions.shape}\n")
        return actions

    def update(self, actions: np.ndarray):
        self.eps *= max(self.cfg.decay_rate *
                        self.eps, self.cfg.min_epsilon)


class ShELLBotlzmanExploration:
    def __init__(self, n_heads: int, cfg: BoltzmanExplorationConfig) -> None:
        self.n_heads = n_heads
        self.exploration_strategies = [
            BotlzmanExploration(cfg) for _ in range(n_heads)]

    def reset(self):
        for strategy in self.exploration_strategies:
            strategy.reset()

    def get_action(self, head_id: int, Q_values: np.ndarray):
        return self.exploration_strategies[head_id].get_action(Q_values)

    def update(self, head_id: int, actions: np.ndarray):
        self.exploration_strategies[head_id].update(actions)
