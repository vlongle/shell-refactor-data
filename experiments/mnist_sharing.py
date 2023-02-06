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
)
from shell_data.shell_agent.shell_agent_classification import ShELLClassificationAgent

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
seed_everything(0)

logging.basicConfig(level=logging.DEBUG,
                    format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, markup=False, show_time=False, show_path=False)],)


def main():
    # dataset_name = "mnist"
    dataset_name = "cifar10"
    train_subsets, val_subsets, test_subsets = get_train_val_test_subsets(
        dataset_name)
    size = 4096
    # num_cls_per_task = 2
    num_cls_per_task = 10

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

    receiver = ShELLClassificationAgent(
        train_subsets, val_subsets, test_subsets, cfg,
        enable_validate_config=False,)

    logging.critical(f"Data task cls: {receiver.ll_dataset.perm}")  # [5, 7]

    model_name = f"{dataset_name}_128_10.pt"
    buffer_name = f"{dataset_name}_10_buffer"

    # check if model exists
    if not os.path.exists(model_name) or os.path.exists(buffer_name):
        receiver.learn_task(0)
        receiver.save_model(model_name)
        receiver.save_buffer(buffer_name)
    else:
        receiver.load_model(model_name)
        receiver.load_buffer(buffer_name)

    test = receiver.test(0)
    val = receiver.val(0)

    logging.critical(f"Test: {test:.3f}, val: {val:.3f}")

    # sender1 = ShELLClassificationAgent(
    #     train_subsets, val_subsets, test_subsets, cfg,
    #     enable_validate_config=False,)

    # sender2 = ShELLClassificationAgent(
    #     train_subsets, val_subsets, test_subsets, cfg,
    #     enable_validate_config=False,)

    # sender1.ll_dataset.perm = [5, 1]  # should route 5
    # sender2.ll_dataset.perm = [7, 2]  # should route 7


if __name__ == "__main__":
    main()
