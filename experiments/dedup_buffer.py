import time
import pandas as pd
from rich.logging import RichHandler
import logging
from lightning_lite.utilities.seed import seed_everything
from shell_data.dataset.dataset import get_train_val_test_subsets
import torch
import os
from shell_data.utils.config import (
    ShELLDataSharingConfig,
    DatasetConfig,
    TaskModelConfig,
    TrainingConfig,
    ExperienceReplayConfig,
    DataValuationConfig,
    RouterConfig,
    BoltzmanExplorationConfig,
)
from shell_data.utils.record import Record, snapshot_perf, snapshot_conf_mat
import numpy as np
from shell_data.shell_agent.shell_agent_classification import ShELLClassificationAgent
from itertools import combinations

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
seed_everything(0)

logging.basicConfig(level=logging.DEBUG,
                    format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, markup=False, show_time=False, show_path=False)],)


"""
NOTE: there might be intra-class positive transfer in this CIFAR-10 or something.
Somehow, adding cls=2 boost the test performance on cls=0,1,8.


TODO: see the training confusion matrix... Might need to increase the train size even further to avoid 
this weird mismatch between training and testing data...
"""


def main():
    start = time.time()
    # dataset_name = "fashion_mnist"
    dataset_name = "cifar10"
    # dataset_name = "mnist"
    train_subsets, val_subsets, test_subsets = get_train_val_test_subsets(
        dataset_name)

    size = 512

    if isinstance(size, int):
        print("size:", size)
    else:
        print("avg size: ", sum(size.values()) / len(size))

    num_cls_per_task = 3
    # num_cls_per_task = 2
    n_agents = 2
    num_task_per_life = 2
    buffer_integration_size = 50000  # sample all!
    batch_size = 32
    routing_method = "random"

    cfg = ShELLDataSharingConfig(
        n_agents=n_agents,
        dataset=DatasetConfig(
            name=dataset_name,
            train_size=size,
            # should val_size be the same as train_size?
            # val_size=min(size * 3, min([len(d) for d in val_subsets])),
            val_size=10,
            num_task_per_life=num_task_per_life,
            num_cls_per_task=num_cls_per_task,
        ),
        task_model=TaskModelConfig(
            name=dataset_name,
        ),
        training=TrainingConfig(
            n_epochs=200,
            batch_size=batch_size,
            patience=10,
        ),
        experience_replay=ExperienceReplayConfig(
            buffer_size=buffer_integration_size,
            # train_size=size // 2,
        ),
        data_valuation=DataValuationConfig(
            method="performance",  # control how the receiver perceives data and what to keep
        ),
        router=RouterConfig(
            strategy=routing_method,  # control how the sender decides which data point to send
            # num_batches=4,
            num_batches=1,
            # batch_size=32,  # sending half of the data
            batch_size=128 * 3,  # sending half of the data
            estimator_task_model=TaskModelConfig(
                name=dataset_name,
            ),
            explore=BoltzmanExplorationConfig(
                num_slates=128 * 3,
            ),
            n_heads=n_agents,
        ),
    )

    receiver = ShELLClassificationAgent(
        train_subsets, val_subsets, test_subsets, cfg)

    sender = ShELLClassificationAgent(
        train_subsets, val_subsets, test_subsets, cfg)

    receiver.ll_dataset.perm = torch.tensor([0, 1, 8, 9])

    # NOTE: this "oracle" doesn't model the receiver data need. If a
    # certain task already has enough data in the past, we should prioritize now
    sender.ll_dataset.perm = torch.tensor([0, 2, 1, 4])  # should send 1 at t=1
    # should probably prefer to send 3 instead of 1 at t=1
    # sender.ll_dataset.perm = torch.tensor([0, 2, 1, 3])
    sender.ll_dataset.perm = torch.tensor([0, 2, 3, 1])

    receiver.init_model_router()
    sender.init_model_router()

    record = Record("dedup.csv")

    receiver.learn_task(0)
    snapshot_perf(receiver, record, "receiver", ll_time=0, local=True)

    snapshot_conf_mat(receiver, ll_time=0, record_name="before")

    sender.share_with(receiver, other_id=0, ll_time=0)

    logging.critical("Receiver scouraging their buffer NOWWW...")

    # NOTE: learn_from_buffer might not be the same as the
    # actual model due to minibatch stuff...
    receiver.learn_from_buffer(
        ll_time=0, kind="all", metric="past_task_test_acc")
    # sender.share_with(receiver, other_id=0, ll_time=0)
    snapshot_perf(receiver, record, "receiver", ll_time=0, local=False)

    snapshot_conf_mat(receiver, ll_time=0, record_name="after")

    # receiver.learn_task(1)
    # sender.share_with(receiver, other_id=0, ll_time=1)
    # snapshot_perf(receiver, record, "receiver", ll_time=1, local=False)

    # receiver.learn_from_buffer(0)

    record.save()

    end = time.time()
    print(f"Time taken: {end - start}")


if __name__ == "__main__":
    main()
