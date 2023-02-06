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
from shell_data.utils.utils import Record
import numpy as np
from shell_data.shell_agent.shell_agent_classification import ShELLClassificationAgent
from itertools import combinations

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
seed_everything(0)

logging.basicConfig(level=logging.DEBUG,
                    format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, markup=False, show_time=False, show_path=False)],)


def main():
    start = time.time()
    dataset_name = "fashion_mnist"
    # dataset_name = "mnist"
    train_subsets, val_subsets, test_subsets = get_train_val_test_subsets(
        dataset_name)

    # size = 128  # acc =

    # size = 258 # acc = 0.9576

    # size = 460  # acc = 0.9666

    # avg_size = 485.9, acc = 0.9617 # mildly imbalanced!
    # size = {
    #     0: 490,
    #     1: 248,
    #     2: 364,
    #     3: 760,
    #     4: 870,
    #     5: 625,
    #     6: 1000,
    #     7: 246,
    #     8: 128,
    #     9: 128,
    # }

    size = {
        # 0: 1012,
        # 1: 1012,
        # 2: 1012,
        # 0: 512,
        # 1: 512,
        # 2: 512,
        0: 128,
        1: 128,
        2: 128,
        3: 128,
        4: 128,
        5: 128,
        6: 128,
        7: 128,
        8: 128,
        9: 128,
        # 7: 1280,
        # 8: 1280,
        # 9: 1280,

        # 9: 0,
    }

    if isinstance(size, int):
        print("size:", size)
    else:
        print("avg size: ", sum(size.values()) / len(size))

    num_cls_per_task = 10
    n_agents = 1
    num_task_per_life = 1
    buffer_integration_size = 50000  # sample all!
    batch_size = 64
    routing_method = "oracle"

    cfg = ShELLDataSharingConfig(
        n_agents=n_agents,
        dataset=DatasetConfig(
            name=dataset_name,
            train_size=size,
            # should val_size be the same as train_size?
            # val_size=min(size * 3, min([len(d) for d in val_subsets])),
            val_size=512,
            num_task_per_life=num_task_per_life,
            num_cls_per_task=num_cls_per_task,
        ),
        task_model=TaskModelConfig(
            name=dataset_name,
        ),
        training=TrainingConfig(
            n_epochs=100,
            batch_size=batch_size,
        ),
        experience_replay=ExperienceReplayConfig(
            buffer_size=buffer_integration_size,
            # train_size=size // 2,
        ),
        data_valuation=DataValuationConfig(
            method="oracle",  # control how the receiver perceives data and what to keep
        ),
        router=RouterConfig(
            strategy=routing_method,  # control how the sender decides which data point to send
            num_batches=1,
            # batch_size=32,  # sending half of the data
            batch_size=0,  # sending half of the data
            estimator_task_model=TaskModelConfig(
                name=dataset_name,
            ),
            explore=BoltzmanExplorationConfig(
                # num_slates=32,
                num_slates=0,
            ),
            n_heads=n_agents,
        ),
    )

    agent = ShELLClassificationAgent(
        train_subsets, val_subsets, test_subsets, cfg)

    agent.init_model_router()

    if isinstance(size, dict):  # HACK
        size = str(sum(size.values()) / len(size))
        size = size.replace(".", "_")

    test, conf_mat = agent.shared_test_val(
        0, metric_type="acc", split="test", ret_confusion_matrix=True)
    print("Before test:", test)
    np.save(f"{dataset_name}_{size}_test_conf_before.npy", conf_mat)

    train, conf_mat = agent.shared_test_val(
        0, metric_type="acc", split="train", ret_confusion_matrix=True)
    print("Before train:", train)
    np.save(f"{dataset_name}_{size}_train_conf_before.npy", conf_mat)

    agent.learn_task(0, record_name=f"{dataset_name}_{size}.csv")

    # # add 128 samples from class 9 to the buffer
    # dataset = torch.utils.data.Subset(train_subsets[9], range(128))
    # X = torch.cat([x for x, y in dataset])
    # # add channel dimension
    # X = X.unsqueeze(1)
    # ys = torch.tensor([int(y) for x, y in dataset])
    # agent.buffer.add_data((X, ys))
    # agent.learn_from_buffer(0)

    test, conf_mat = agent.shared_test_val(
        0, metric_type="acc", split="test", ret_confusion_matrix=True)
    print("After test:", test)
    np.save(f"{dataset_name}_{size}_test_conf_after.npy", conf_mat)

    train, conf_mat = agent.shared_test_val(
        0, metric_type="acc", split="train", ret_confusion_matrix=True)
    print("After train:", train)
    np.save(f"{dataset_name}_{size}_train_conf_after.npy", conf_mat)

    end = time.time()
    print(f"Time taken: {end - start}")


if __name__ == "__main__":
    main()
