import argparse
import time
from shell_data.utils.config import (
    ShELLDataSharingConfig,
    DatasetConfig,
    TaskModelConfig,
    TrainingConfig,
)
from shell_data.dataset.dataset import get_train_val_test_subsets
from shell_data.shell_agent.shell_agent_classification import ShELLClassificationAgent
import pandas as pd
import logging
from rich.logging import RichHandler
import os
import torch
from lightning_lite.utilities.seed import seed_everything
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)

logging.basicConfig(level=logging.DEBUG,
                    format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, markup=False, show_time=False, show_path=False)],)

parser = argparse.ArgumentParser(description='Data effect')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default="cifar10")
args = parser.parse_args()


def main():
    start = time.time()
    dataset_name = args.dataset
    seed = args.seed

    train_subsets, val_subsets, test_subsets = get_train_val_test_subsets(
        dataset_name)

    # sizes = [128, 256, 512, 1024]
    # sizes = [2048, 4096] # for cifar10
    sizes = [4096]
    # num_cls_per_task_list = [2, 4, 8, 10]
    num_cls_per_task_list = [10]

    df = pd.DataFrame(
        columns=['seed', 'size', 'num_cls_per_task', 'test_acc', 'val_acc'])

    logging.critical(f"ANALYSIS FOR {dataset_name}")
    logging.critical(
        f"Train size {min([len(d) for d in train_subsets])}, val size {min([len(d) for d in val_subsets])}, test size {min([len(d) for d in test_subsets])}")

    seed_everything(seed)
    for size in sizes:
        for num_cls_per_task in num_cls_per_task_list:
            cfg = ShELLDataSharingConfig(
                n_agents=1,
                dataset=DatasetConfig(
                    name=dataset_name,
                    train_size=size,
                    val_size=min(size, min([len(d) for d in val_subsets])),
                    num_task_per_life=1,
                    num_cls_per_task=num_cls_per_task,
                ),
                task_model=TaskModelConfig(
                    name=dataset_name,
                ),
                training=TrainingConfig(
                    n_epochs=100,
                )
            )
            agent = ShELLClassificationAgent(
                train_subsets, val_subsets, test_subsets, cfg,
                enable_validate_config=False,)
            agent.learn_task(0)
            test = agent.test(0)
            val = agent.val(0)
            logging.critical(
                f"size {size}, num_cls_per_task {num_cls_per_task}, test {test:.3f}, val {val:.3f}")
            df.loc[len(df)] = [seed, size, num_cls_per_task, test, val]

    # save df, blocking access to file
    # file_name = f"results/{dataset_name}_data_effect.csv"
    # with open(file_name, "a") as f:
    #     df.to_csv(f, index=False, mode="a",
    #               header=f.tell() == 0)
    end = time.time()
    logging.critical(f"Time taken {end-start:.3f} seconds")
    # save the model
    torch.save(agent.model.net.state_dict(),
               "vgg16_cifar10_data=4096_cls=10.pth")


if __name__ == '__main__':
    main()
