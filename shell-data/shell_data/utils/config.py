from dataclasses import dataclass
from typing import (
    Union,
    Dict,
)


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    # train_size: Union[float, int, Dict[int, int], Dict[int, float]] = 0.8
    train_size: Union[float, int] = 0.8
    val_size: Union[float, int] = 0.2
    test_size: Union[float, int] = 1.0
    num_cls_per_task: int = 2
    num_task_per_life: int = 5


@dataclass
class TaskModelConfig:
    name: str = "cifar10"
    device: str = "cuda"


@dataclass
class TrainingConfig:
    batch_size: int = 128
    n_epochs: int = 100
    patience: int = 5
    delta: float = 0.0
    val_every_n_epoch: int = 5


@dataclass
class ExperienceReplayConfig:
    train_size: int = 512
    factor: int = 1


@dataclass
class DataValuationConfig:
    strategy: str = "mean_acc"
    threshold: float = 0.0
    train_size: int = 512


@dataclass
class BoltzmanExplorationConfig:
    epsilon: float = 1.0
    min_epsilon: float = 0.01
    decay_rate: float = 0.9
    num_slates: int = 1  # how many actions to sample


@dataclass
class RouterConfig:
    strategy: str = "random"
    # params for communication constraint
    batch_size: int = 1
    num_batches: int = 1
    # params for training preference regressor
    train_size: int = 512  # how many samples from the regression buffer to train
    val_size: int = 128
    explore: BoltzmanExplorationConfig = BoltzmanExplorationConfig()
    n_heads: int = 5  # number of agents
    training: TrainingConfig = TrainingConfig()  # training the router estimator
    estimator_task_model: TaskModelConfig = TaskModelConfig()


@dataclass
class ShELLDataSharingConfig:
    n_agents: int = 2
    dataset: DatasetConfig = DatasetConfig()
    task_model: TaskModelConfig = TaskModelConfig()
    training: TrainingConfig = TrainingConfig()
    experience_replay: ExperienceReplayConfig = ExperienceReplayConfig()
    data_valuation: DataValuationConfig = DataValuationConfig()
    router: RouterConfig = RouterConfig()


def validate_config(cfg: ShELLDataSharingConfig):
    """
    Check that the config is valid
    """
    assert cfg.dataset.name == cfg.task_model.name == cfg.router.estimator_task_model.name
    # NOTE: specific to cifar10 and mnist
    assert cfg.dataset.num_cls_per_task * cfg.dataset.num_task_per_life <= 10
    assert cfg.router.explore.num_slates == cfg.router.batch_size
    assert cfg.router.n_heads == cfg.n_agents
