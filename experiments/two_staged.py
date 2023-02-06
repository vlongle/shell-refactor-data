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
from copy import deepcopy

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
seed_everything(0)

logging.basicConfig(level=logging.DEBUG,
                    format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, markup=False, show_time=False, show_path=False),
                              # handlers=[
                              # log to file
                              logging.FileHandler("log_val.txt", mode="w")],
                    )


"""
NOTE: there might be intra-class positive transfer in this CIFAR-10 or something.
Somehow, adding cls=2 boost the test performance on cls=0,1,8.


TODO: see the training confusion matrix... Might need to increase the train size even further to avoid
this weird mismatch between training and testing data...
"""


def main():
    start = time.time()
    # dataset_name = "fashion_mnist"
    # dataset_name = "cifar10"
    dataset_name = "mnist"
    train_subsets, val_subsets, test_subsets = get_train_val_test_subsets(
        dataset_name)

    # size = {
    #     0: 512,
    #     1: 256,
    # }
    size = 64

    if isinstance(size, int):
        print("size:", size)
    else:
        print("avg size: ", sum(size.values()) / len(size))

    num_cls_per_task = 4
    n_agents = 2
    num_task_per_life = 1
    buffer_integration_size = 50000  # sample all!
    batch_size = 32
    # batch_size = 16
    routing_method = "random"

    cfg = ShELLDataSharingConfig(
        n_agents=n_agents,
        dataset=DatasetConfig(
            name=dataset_name,
            train_size=size,
            test_size=892,
            # test_size=1.0,
            # should val_size be the same as train_size?
            # val_size=min(size * 3, min([len(d) for d in val_subsets])),
            # val_size=10,
            # val_size=size * 8,  # NOTE: wtfff... this will probably not going
            # val_size=10,
            val_size=size,
            # val_size=500,
            # to fly with the reviewer!
            num_task_per_life=num_task_per_life,
            num_cls_per_task=num_cls_per_task,
        ),
        task_model=TaskModelConfig(
            name=dataset_name,
        ),
        training=TrainingConfig(
            n_epochs=200,
            # n_epochs=10,
            batch_size=batch_size,
            patience=10,
        ),
        experience_replay=ExperienceReplayConfig(
            buffer_size=buffer_integration_size,
            # train_size=size // 2,
        ),
        data_valuation=DataValuationConfig(
            method="performance",  # control how the receiver perceives data and what to keep
            metric="past_task_test_acc",
        ),
        router=RouterConfig(
            strategy=routing_method,  # control how the sender decides which data point to send
            # num_batches=4,
            num_batches=1,
            # batch_size=32,  # sending half of the data
            # batch_size=64 * 4,  # sending half of the data
            estimator_task_model=TaskModelConfig(
                name=dataset_name,
            ),
            # explore=BoltzmanExplorationConfig(
            #     num_slates=64 * 4,
            # ),
            n_heads=n_agents,
        ),
    )

    receiver = ShELLClassificationAgent(
        train_subsets, val_subsets, test_subsets, cfg)

    sender_cfg = deepcopy(cfg)
    sender_cfg.dataset.train_size = size
    sender_cfg.router.batch_size = size * 4
    sender_cfg.router.explore = BoltzmanExplorationConfig(
        num_slates=size * 4,
    )
    # HACK
    sender_cfg.dataset.val_size = 10
    sender = ShELLClassificationAgent(
        train_subsets, val_subsets, test_subsets, sender_cfg)

    # 0 and 1, 4 and 9 are hard to estabilish!
    receiver.ll_dataset.perm = torch.tensor([0, 3, 4, 9])
    sender.ll_dataset.perm = torch.tensor([0, 3, 4, 9])

    receiver.init_model_router()
    sender.init_model_router()

    # https://ai.stackexchange.com/questions/37577/why-does-mnist-provide-only-a-training-and-a-test-set-and-not-a-validation-set-a#:~:text=Training%20set%20%2D%20for%20learning%20purposes,metrics%20such%20as%20accuracy%20score

    record = Record("two_staged.csv")
    snapshot_perf(receiver, record, "two_staged_before", ll_time=0, local=True)
    snapshot_conf_mat(receiver, ll_time=0, record_name="two_staged_before")
    receiver.learn_task(0, metric="test_acc")
    snapshot_perf(receiver, record, "two_staged_after", ll_time=0, local=True)
    snapshot_conf_mat(receiver, ll_time=0, record_name="two_staged_after")

    # save the receiver model
    receiver.save_model("two_staged_receiver")

    sender.share_with(receiver, other_id=0, ll_time=0)
    snapshot_perf(receiver, record, "two_staged_after", ll_time=0, local=True)
    snapshot_conf_mat(receiver, ll_time=0, record_name="two_staged_after")

    record.save()
    end = time.time()
    print(f"Time taken: {end - start}")


if __name__ == "__main__":
    main()
