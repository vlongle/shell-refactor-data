import os
import torch
import hydra
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
from hydra.core.config_store import ConfigStore
from shell_data.dataset.dataset import get_vision_dataset_subsets
import logging
from rich.logging import RichHandler
from shell_data.shell_agent.shell_agent_classification import ShELLClassificationAgent
import time


from lightning_lite.utilities.seed import seed_everything
# seed_everything(69)
seed_everything(0)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)

"""
TODO: modify class balanced sampler to only care about balancing the classes from the current task!
"""
# logging.basicConfig(level=logging.DEBUG,
logging.basicConfig(level=logging.WARNING,
                    format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, markup=False, show_time=False, show_path=False)],)
# suppress hydra logging
logging.getLogger("hydra").setLevel(logging.WARNING)

cs = ConfigStore.instance()
cs.store(name="config_schema", node=ShELLDataSharingConfig)


def main():
    test_subsets = get_vision_dataset_subsets(
        dataset_name="cifar10",
        train=False,
    )
    train_val_subsets = get_vision_dataset_subsets(
        dataset_name="cifar10",
        train=True,
    )
    train_subsets, val_subsets = [], []
    for train_val_subset in train_val_subsets:
        train_subset, val_subset = torch.utils.data.random_split(
            train_val_subset, [0.9, 0.1])
        train_subsets.append(train_subset)
        val_subsets.append(val_subset)

    print("train_subsets", [len(train_subset)
          for train_subset in train_subsets])
    print("val_subsets", [len(val_subset) for val_subset in val_subsets])
    print("test_subsets", [len(test_subset) for test_subset in test_subsets])

    start = time.time()
    dataset_cfg = DatasetConfig(
        name="cifar10",
        train_size=128,
        # train_size=256,
        # train_size={
        #     0: 64,
        #     1: 256,
        # },
        # train_size=1024,
        # val_size=256,  # TODO: NOTE: this is a bit sus since val_size
        val_size=128,  # TODO: NOTE: this is a bit sus since val_size
        # dist should mirror train_size (due to train val split)
        test_size=1.0,
    )
    task_model_cfg = TaskModelConfig(
        name="cifar10",
        device="cuda",
    )
    training_cfg = TrainingConfig(
        batch_size=64,
        # batch_size=128,
        # batch_size=256,
        # n_epochs=200,
        n_epochs=300,
        val_every_n_epoch=10,
        patience=10,
    )
    er_config = ExperienceReplayConfig(
        # train_size=128 * 2,
        train_size=256,
        factor=2,
    )
    dv_config = DataValuationConfig(
        # strategy="mean_acc",
        # strategy="mean_loss",
        strategy="best_mean",
        # threshold=0.01,
        threshold=0.0,
        train_size=64,
        # train_size=512,
        # threshold=0.01,
    )

    cfg = ShELLDataSharingConfig(
        dataset=dataset_cfg,
        task_model=task_model_cfg,
        training=training_cfg,
        experience_replay=er_config,
        data_valuation=dv_config,
    )
    agent = ShELLClassificationAgent(
        train_subsets, val_subsets, test_subsets, cfg)
    # agent.ll_dataset.perm = torch.tensor([0, 1, 2, 3])
    agent.ll_dataset.perm = torch.tensor([0, 1, 2, 3, 4, 5])
    agent.learn_task(0)
    # agent.save_model("model.pt")
    # agent.model.net.reset_parameters()
    # agent.load_model("model.pt")

    test_before = agent.test(0)
    logging.critical(f"task 0: test {test_before:.2f}")
    val_before = agent.val(0)
    logging.critical(f" val: {val_before:.2f}")

    # print("acc task 1:",    agent.test(1))
    # print("acc task 2:",    agent.test(2))
    # agent.learn_task(1)
    # print("acc task 0:",    agent.test(0))
    # print("acc task 1:",    agent.test(1))
    # print("acc task 2:",    agent.test(2))
    # agent.learn_task(2)
    # print("acc task 0:",    agent.test(0))
    # print("acc task 1:",    agent.test(1))
    # print("acc task 2:",    agent.test(2))

    dataset2_cfg = DatasetConfig(
        name="cifar10",
        train_size=128,
        # train_size=256,
        # train_size=1024,
        val_size=128,
        test_size=1.0,
    )

    router_cfg = RouterConfig(
        # strategy="random",
        strategy="neural",
        num_batches=4,
        batch_size=64,
        explore=BoltzmanExplorationConfig(
            num_slates=64,
            decay_rate=0.8,
        ),
        train_size=64 * 2,
        val_size=64,
        training=TrainingConfig(
            n_epochs=200,
            patience=10,
        ),
    )

    cfg2 = ShELLDataSharingConfig(
        dataset=dataset2_cfg,
        router=router_cfg,
    )

    # sender = ShELLClassificationAgent(
    #     train_subsets, val_subsets, test_subsets, cfg2)
    # sender.ll_dataset.perm = torch.tensor([0, 9, 1, 2])

    # logging.critical(
    #     f"Before sharing, data dist {[len(b) for b in agent.buffer.buffers]}")
    # sender.router.share_with(agent, other_id=0, ll_time=0)

    # logging.critical(
    #     f"After sharing, data dist {[len(b) for b in agent.buffer.buffers]}")

    # # agent.learn_from_buffer(0, buffer_size=256)
    # test_after = agent.test(0)
    # logging.critical(
    #     f"test task 0: before {test_before:.2f}, after {test_after:.2f}, improvement {((test_after - test_before) / test_before) * 100:.2f}%")
    # val_after = agent.val(0)
    # logging.critical(
    #     f"val task 0: before {val_before:.2f}, after {val_after:.2f}, improvement {((val_after - val_before) / val_before) * 100:.2f}%")

    logging.critical("SHARE AGAIN")
    logging.critical(
        f"before second round of sharing, data dist {[len(b) for b in agent.buffer.buffers]}")
    sender = ShELLClassificationAgent(
        train_subsets, val_subsets, test_subsets, cfg2)
    sender.ll_dataset.perm = torch.tensor([1, 8, 0, 3])

    sender.router.share_with(agent, other_id=0, ll_time=0)

    logging.critical(
        f"After sharing, data dist {[len(b) for b in agent.buffer.buffers]}")

    # agent.learn_from_buffer(0, buffer_size=256)
    test_after = agent.test(0)
    logging.critical(
        f"acc task 0: before {test_before:.2f}, after {test_after:.2f}, improvement {((test_after- test_before) / test_before) * 100:.2f}%")

    val_after = agent.val(0)
    logging.critical(
        f"val task 0: before {val_before:.2f}, after {val_after:.2f}, improvement {((val_after - val_before) / val_before) * 100:.2f}%")

    logging.critical("AGAIN\n\n\n")

    sender = ShELLClassificationAgent(
        train_subsets, val_subsets, test_subsets, cfg2)
    sender.ll_dataset.perm = torch.tensor([0, 9, 1, 2])

    logging.critical(
        f"Before sharing, data dist {[len(b) for b in agent.buffer.buffers]}")
    sender.router.share_with(agent, other_id=0, ll_time=0)

    logging.critical(
        f"After sharing, data dist {[len(b) for b in agent.buffer.buffers]}")

    # agent.learn_from_buffer(0, buffer_size=256)
    test_after = agent.test(0)
    logging.critical(
        f"test task 0: before {test_before:.2f}, after {test_after:.2f}, improvement {((test_after - test_before) / test_before) * 100:.2f}%")

    val_after = agent.val(0)
    logging.critical(
        f"val task 0: before {val_before:.2f}, after {val_after:.2f}, improvement {((val_after - val_before) / val_before) * 100:.2f}%")

    # ds = dataset.get_train_dataset(0)
    # dl = torch.utils.data.DataLoader(ds, batch_size=len(ds))
    # X, y = next(iter(dl))
    # agent.data_valuation(X, y, 0)

    # # ANOTHER ONE
    # dataset = LifelongDataset(dataset2_cfg)
    # dataset.perm = torch.tensor([8, 1, 1, 2])

    # ds = dataset.get_train_dataset(0)
    # dl = torch.utils.data.DataLoader(ds, batch_size=len(ds))
    # X, y = next(iter(dl))
    # agent.data_valuation(X, y, 0)

    # # logging.critical("learn task 1")

    # # agent.learn_task(1)
    # # ds = dataset.get_train_dataset(1)
    # # dl = torch.utils.data.DataLoader(ds, batch_size=len(ds))
    # # X, y = next(iter(dl))
    # # agent.data_valuation(X, y, 1)

    # end = time.time()
    # print("takes", end - start, "seconds")
    # return


if __name__ == "__main__":
    main()
