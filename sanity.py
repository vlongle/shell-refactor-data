from rich.logging import RichHandler
from shell_data.dataset.dataset import get_train_val_test_subsets
from shell_data.utils.config import (
    ShELLDataSharingConfig,
    validate_config,
)
from hydra.core.config_store import ConfigStore
import hydra
import torch
import os
from itertools import combinations
from lightning_lite.utilities.seed import seed_everything
from shell_data.shell_agent.shell_agent_classification import ShELLClassificationAgent
import logging
import time

SEED = 0
seed_everything(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)

cs = ConfigStore.instance()
cs.store(name="config_schema", node=ShELLDataSharingConfig)

logging.basicConfig(level=logging.CRITICAL,
                    # logging.basicConfig(level=logging.DEBUG,
                    format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, markup=False, show_time=False, show_path=False)],)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: ShELLDataSharingConfig):
    # print(cfg)
    validate_config(cfg)

    start = time.time()

    train_subsets, val_subsets, test_subsets = get_train_val_test_subsets(
        cfg.dataset.name)

    agents = [ShELLClassificationAgent(
        train_subsets, val_subsets, test_subsets, cfg)
        for _ in range(cfg.n_agents)]

    # reset the ll_dataset again and reset the random seed to ensure
    # correct randomness across all routing methods
    seed_everything(SEED)
    for agent_id, agent in enumerate(agents):
        agent.ll_dataset.reset()
        logging.critical(f"agent {agent_id} data: {agent.ll_dataset.perm}")

    # before sharing
    before_test_res = torch.zeros(
        (cfg.n_agents, cfg.dataset.num_task_per_life, cfg.dataset.num_task_per_life))
    before_val_res = torch.zeros(
        (cfg.n_agents, cfg.dataset.num_task_per_life, cfg.dataset.num_task_per_life))

    before_data_dist = torch.zeros(
        (cfg.n_agents, cfg.dataset.num_task_per_life, 10))

    # after sharing
    after_test_res = torch.zeros(
        (cfg.n_agents, cfg.dataset.num_task_per_life, cfg.dataset.num_task_per_life))

    after_val_res = torch.zeros(
        (cfg.n_agents, cfg.dataset.num_task_per_life, cfg.dataset.num_task_per_life))

    after_data_dist = torch.zeros(
        (cfg.n_agents, cfg.dataset.num_task_per_life, 10))

    for ll_time in range(cfg.dataset.num_task_per_life):
        for agent_id, agent in enumerate(agents):
            agent.learn_task(ll_time)
            logging.critical(
                f"agent {agent_id} finished learning task {ll_time}")
            before_data_dist[agent_id, ll_time, :] = torch.tensor(
                [len(b) for b in agent.buffer.buffers])
            logging.critical(
                f"BEFORE sharing, data dist {before_data_dist[agent_id, ll_time, :]}")

            # test on all past task
            for t in range(ll_time+1):
                acc = agent.test(t)
                val = agent.val(t)
                logging.critical(
                    f"\t task {t} test acc: {acc:.3f}, val acc: {val:.3f}")

                before_test_res[agent_id, ll_time, t] = acc
                before_val_res[agent_id, ll_time, t] = val

        logging.critical("\n")

    root_dir = f"results/seed_{SEED}_strategy_{cfg.router.strategy}_n_agents_{cfg.n_agents}_n_tasks_{cfg.dataset.num_task_per_life}"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # save the results
    torch.save(before_test_res, os.path.join(
        root_dir, "before_test_res.pt"))
    torch.save(before_val_res, os.path.join(
        root_dir, "before_val_res.pt"))
    torch.save(after_test_res, os.path.join(
        root_dir, "after_test_res.pt"))
    torch.save(after_val_res, os.path.join(
        root_dir, "after_val_res.pt"))

    torch.save(before_data_dist,
               os.path.join(root_dir, "before_data_dist.pt"))
    torch.save(after_data_dist,
               os.path.join(root_dir, "after_data_dist.pt"))

    end = time.time()
    print(f"Time taken: {end - start:.3f} sec")


if __name__ == "__main__":
    main()
