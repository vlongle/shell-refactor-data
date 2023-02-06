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
from shell_data.utils.utils import Record, snapshot_perf

from shell_data.shell_agent.shell_agent_classification import ShELLClassificationAgent
from itertools import combinations

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
seed_everything(0)

logging.basicConfig(level=logging.DEBUG,
                    format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, markup=False, show_time=False, show_path=False)],)

# record is a class that contains rows of a pandas dataframe
# write(dict) will add a row to the dataframe where the keys are the column names
# and the values are the values of the row

"""
TODO:
dedup the candidate by returning the keeps decision.
Debug why the final data distribution is not uniform?
"""


def main():
    start = time.time()
    dataset_name = "mnist"
    train_subsets, val_subsets, test_subsets = get_train_val_test_subsets(
        dataset_name)

    size = 128
    num_cls_per_task = 2
    n_agents = 10
    num_task_per_life = 5
    buffer_integration_size = 50000  # sample all!
    batch_size = 32
    routing_method = "oracle"

    cfg = ShELLDataSharingConfig(
        n_agents=n_agents,
        dataset=DatasetConfig(
            name=dataset_name,
            train_size=size,
            # should val_size be the same as train_size?
            val_size=min(size * 3, min([len(d) for d in val_subsets])),
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
            train_size=size // 2,
        ),
        data_valuation=DataValuationConfig(
            method="oracle",  # control how the receiver perceives data and what to keep
        ),
        router=RouterConfig(
            strategy=routing_method,  # control how the sender decides which data point to send
            num_batches=4,
            batch_size=32,  # sending half of the data
            estimator_task_model=TaskModelConfig(
                name=dataset_name,
            ),
            explore=BoltzmanExplorationConfig(
                num_slates=32,
            ),
            n_heads=n_agents,
        ),
    )

    agents = [ShELLClassificationAgent(
        train_subsets, val_subsets, test_subsets, cfg) for _ in range(n_agents)]

    for agent in agents:
        agent.init_model_router()

    record = Record(f"mnist_ll_{routing_method}_n{n_agents}.csv")

    for ll_time in range(num_task_per_life):

        # local training
        for agent_id, agent in enumerate(agents):
            agent.learn_task(ll_time)
            snapshot_perf(agent, record, agent_id, ll_time, local=True)

        # sharing
        for agent_i, agent_j in combinations(range(len(agents)), 2):
            logging.critical(f"agent {agent_i} share with agent {agent_j}")
            agents[agent_i].share_with(
                agents[agent_j], other_id=agent_j, ll_time=ll_time)
            logging.critical(f"agent {agent_j} share with agent {agent_i}")
            agents[agent_j].share_with(
                agents[agent_i], other_id=agent_i, ll_time=ll_time)

        # offline integration
        for agent_id, agent in enumerate(agents):
            agent.learn_from_buffer(ll_time)
            snapshot_perf(agent, record, agent_id, ll_time, local=False)

    record.save()
    end = time.time()
    print(f"Time taken: {end - start}")


if __name__ == "__main__":
    main()
