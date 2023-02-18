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
    buffer_size: int = 5000
    n_epochs: int = 20


@dataclass
class DataValuationConfig:
    # choices = [oracle, clustering, performance, random]
    method: str = "oracle"
    metric: str = "past_task_val_acc"  # choices = ["oracle"]
    threshold: float = 0.0
    train_size: int = 512


@dataclass
class BoltzmanExplorationConfig:
    epsilon: float = 1.0
    min_epsilon: float = 0.01
    decay_rate: float = 0.9
    num_slates: int = 1  # how many actions to sample


# Sender-first
@dataclass
class RouterConfig:
    # choices = ["random", "neural", "no_routing", "oracle"]
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
class ReceiverFirstConfig:
    num_queries: int = 1  # how many data points of my own to query
    num_neighbors: int = 1  # how many neighbors I expect to receive data from
    # choices = ["wrongly_predicted", "least_confidence", "entropy", "margin", "combine_wrong_least_confidence"]
    query_strategy: str = "wrongly_predicted"
    # ["ground_truth", "cac_contrastive", "no_outlier_detection"]
    outlier_method: str = "cac_contrastive"
    # how many std away from the mean to consider an outlier in the outlier score
    # outlier_std: float = 0.5
    outlier_std: float = 0.9  # NOTE: changed to percentile instead!
    query_score_threshold: float = 0.0
    # n_epochs: int = 200 # how long to train the recognizer
    n_epochs: int = 100  # how long to train the recognizer
    batch_size: int = 16  # batch size to train the recognizer
    cont_temp: float = 0.06  # the temperature of the contrastive loss
    cont_mode: str = "one"  # contrast mode

    # NOTE: we're experimenting with annealing the contamination level
    threshold_init_max_contam: float = 0.5
    threshold_decay_rate: float = 0.9
    funky_threshold: bool = False  # NOTE: for debugging


@dataclass
class ShELLDataSharingConfig:
    n_agents: int = 2
    dataset: DatasetConfig = DatasetConfig()
    task_model: TaskModelConfig = TaskModelConfig()
    training: TrainingConfig = TrainingConfig()
    experience_replay: ExperienceReplayConfig = ExperienceReplayConfig()
    data_valuation: DataValuationConfig = DataValuationConfig()
    router: RouterConfig = RouterConfig()
    receiver_first: ReceiverFirstConfig = ReceiverFirstConfig()
    # choices = ["sender_first", "receiver_first"]
    strategy: str = "sender_first"
    sharing_buffer_size: int = 1
    num_classes: int = 10


def validate_config(cfg: ShELLDataSharingConfig):
    """
    Check that the config is valid
    """
    assert cfg.dataset.name == cfg.task_model.name
    # NOTE: specific to cifar10 and mnist
    # assert cfg.dataset.num_cls_per_task * cfg.dataset.num_task_per_life <= 10
    if cfg.strategy == "sender_first":
        validate_router_config(cfg)


def validate_router_config(cfg: ShELLDataSharingConfig):
    if cfg.router.strategy != "no_routing":
        assert cfg.router.explore.num_slates == cfg.router.batch_size
        assert cfg.router.n_heads == cfg.n_agents
        assert cfg.dataset.name == cfg.task_model.name == cfg.router.estimator_task_model.name

    if isinstance(cfg.dataset.train_size, dict):
        train_size = min(cfg.dataset.train_size.values())
    else:
        train_size = cfg.dataset.train_size
    print(
        f"train_size: {train_size}, num_cls_per_task: {cfg.dataset.num_cls_per_task}")
    # NOTE: heuristic for the lower bound
    train_size *= cfg.dataset.num_cls_per_task
    assert cfg.router.num_batches * \
        cfg.router.batch_size <= train_size, f"train_size: {train_size}, num_batches: {cfg.router.num_batches}, batch_size: {cfg.router.batch_size}"
