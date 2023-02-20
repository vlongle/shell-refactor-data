import time
from shell_data.shell_agent.shell_fleet import ShellFleet
from rich.logging import RichHandler
import logging
from lightning_lite.utilities.seed import seed_everything
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
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

dataset_name = "cifar100"


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
seed_everything(int(args.seed))

DIR = f"experiments/{dataset_name}/baseline_{args.seed}"

if not os.path.exists(DIR):
    os.makedirs(DIR)

logging.basicConfig(level=logging.DEBUG,
                    format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, markup=False, show_time=False, show_path=False),
                              logging.FileHandler(f"{DIR}/log.txt")],
                    )


def main():
    start = time.time()
    size = 256
    num_cls_per_task = 5
    num_task_per_life = 20
    n_epochs = 50
    buffer_size = size * 4
    n_agents = 1

    cfg = ShELLDataSharingConfig(
        n_agents=n_agents,
        dataset=DatasetConfig(
            name=dataset_name,
            train_size=size,
            val_size=size//2,
            num_task_per_life=num_task_per_life,
            num_cls_per_task=num_cls_per_task,
        ),
        task_model=TaskModelConfig(
            name=dataset_name,
        ),
        training=TrainingConfig(
            n_epochs=n_epochs,
            # basically not doing early stopping
            # batch_size=32,
            batch_size=64,
            patience=1000,
            val_every_n_epoch=1,
        ),
        experience_replay=ExperienceReplayConfig(
            buffer_size=buffer_size,
        ),
        router=RouterConfig(
            strategy="no_routing",
        ),
        num_classes=100,
    )

    fleet = ShellFleet(
        cfg,
        dir=DIR,
    )

    for t in range(num_task_per_life):
        logging.info(f"Task {t+1}/{num_task_per_life}")
        fleet.learn_task(ll_time=t)
    end = time.time()

    fleet.clean_models()
    logging.info(f"Total time: {end - start} seconds.")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
