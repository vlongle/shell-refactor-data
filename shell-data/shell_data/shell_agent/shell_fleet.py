import torch
from shell_data.utils.config import ShELLDataSharingConfig
from shell_data.shell_agent.shell_agent_receiver_first import ShellAgentReceiverFirst
from shell_data.shell_agent.shell_agent_sender_first import ShellAgentSenderFirst

from shell_data.utils.record import Record, snapshot_perf, snapshot_conf_mat
from shell_data.dataset.dataset import get_train_val_test_subsets
from multiprocessing import Pool, SimpleQueue
from multiprocessing.pool import ThreadPool
from itertools import combinations
import logging
import pickle
import os

NUM_CLASSES = 10

# TODO: clean up all the pickle, otherwise we'll run out of memory real quick!
# TODO: make sure that the sharing data is always trained (even before putting
# into the buffer!) Just do train on them!


class ShellFleet:
    def __init__(self, cfg: ShELLDataSharingConfig, dir="."):
        self.cfg = cfg
        self.n_agents = cfg.n_agents
        self.dataset_name = cfg.dataset.name
        train_subsets, val_subsets, test_subsets = get_train_val_test_subsets(
            self.dataset_name)

        if cfg.strategy == "receiver_first":
            agent_cls = ShellAgentReceiverFirst
        elif cfg.strategy == "sender_first":
            agent_cls = ShellAgentSenderFirst
        else:
            raise NotImplementedError

        self.agents = [
            agent_cls(
                train_subsets, val_subsets, test_subsets, cfg, name=f"agent_{i}")
            for i in range(self.n_agents)]

        self.dir = dir
        self.init()
        self.global_record = Record(f"{self.dir}/global.csv")

        self.save_natural_task_dist()

    def save_natural_task_dist(self):
        record = Record(f"{self.dir}/natural_cls_dist.csv")
        # (agent_name, task_id, cls_id)
        for agent in self.agents:
            cls_dist = agent.ll_dataset.perm
            for ll_time in range(self.cfg.dataset.num_task_per_life):
                for i in range(self.cfg.dataset.num_cls_per_task):
                    record.write(
                        {
                            "agent_name": agent.name,
                            "ll_time": ll_time,
                            "cls_id":  cls_dist[ll_time * self.cfg.dataset.num_cls_per_task + i].item(),
                        }
                    )
        record.save()

    def init(self):
        for agent in self.agents:
            agent.init()

    def clean_models(self):
        # delete all .pickle files in the directory
        for file in os.listdir(self.dir):
            if file.endswith(".pickle"):
                os.remove(os.path.join(self.dir, file))

    def save_model_buffer(self, ll_time, info=""):
        for agent in self.agents:
            agent.save_model(
                f"{self.dir}/{agent.name}_{ll_time}_{info}_model.pt")
            # NOTE: don't save buffer to avoid running out of memory!
            # agent.save_buffer(
            #     f"{self.dir}/{agent.name}_{ll_time}_{info}_buffer")

    def snapshot_perf_fleet(self, ll_time, info=""):
        for agent in self.agents:
            snapshot_perf(agent, self.global_record, ll_time, info=info)
            train_dataset, _ = agent.get_buffer_train_dataset(ll_time)
            snapshot_conf_mat(
                agent, ll_time, record_name=f"{self.dir}/{agent.name}_{ll_time}_{info}", train_dataset=train_dataset)

    def checkpoint(self, ll_time, info=""):
        self.snapshot_perf_fleet(ll_time, info=info)
        # self.save_model_buffer(ll_time, info=info)
        self.global_record.save()

    def pickle_agent(self, agent, info=""):
        with open(f"{self.dir}/{agent.name}_{info}.pickle", "wb") as f:
            pickle.dump(agent, f)

    def learn_task_agent(self, agent, ll_time, metric="val_acc", load_best_model=False):
        record_name = f"{self.dir}/{agent.name}_task_{ll_time}"
        agent.learn_task(ll_time, record_name=record_name,
                         metric=metric, load_best_model=load_best_model)
        self.pickle_agent(agent, info=f"task_{ll_time}")

    def load_agents(self, info=""):
        for i, agent in enumerate(self.agents):
            with open(f"{self.dir}/{agent.name}_{info}.pickle", "rb") as f:
                self.agents[i] = pickle.load(f)

    def learn_task(self, ll_time, metric="val_acc"):
        for agent in self.agents:
            self.learn_task_agent(agent, ll_time, metric=metric)
        # with Pool(processes=self.n_agents) as pool:
        #     pool.starmap(self.learn_task_agent, [
        #                  (agent, ll_time, metric) for agent in self.agents])

        self.load_agents(info=f"task_{ll_time}")

        self.checkpoint(ll_time, info="after_learn")

        for agent in self.agents:
            agent.add_buffer_task(ll_time)

    def learn_from_buffer_agent(self, agent, ll_time, val_before=False, load_best_model=False):
        record_name = f"{self.dir}/{agent.name}_learn_buffer_{ll_time}"
        agent.learn_from_buffer(ll_time, record_name=record_name,
                                val_before=val_before, load_best_model=load_best_model)

        self.pickle_agent(agent, info=f"learn_buffer_{ll_time}")

    def learn_from_buffer(self, ll_time):

        # for agent in self.agents:
        #     self.learn_from_buffer_agent(agent, ll_time)

        with Pool(processes=self.n_agents) as pool:
            pool.starmap(self.learn_from_buffer_agent, [
                         (agent, ll_time) for agent in self.agents])
        self.load_agents(info=f"learn_buffer_{ll_time}")


class ShellFleetSenderFirst(ShellFleet):
    """
    Sender-first
    """

    def __init__(self, cfg: ShELLDataSharingConfig, dir="."):
        super().__init__(cfg, dir=dir)
        self.sharing_record = Record(f"{self.dir}/sharing.csv")

    def checkpoint_sharing(self, ll_time):
        """
        Row, ll_time, receiver, sender, query_cls_0, ..., query_cls_n, data_cls_0, ..., data_cls_n
        """
        for agent_i, agent_j in combinations(range(len(self.agents)), 2):
            self.checkpoint_sharing_agent(ll_time, agent_i, agent_j)
            self.checkpoint_sharing_agent(ll_time, agent_j, agent_i)
        self.sharing_record.save()

    def checkpoint_sharing_agent(self, ll_time, sender: int, receiver: int):
        _, data = self.agents[sender].router.outgoing_data[(ll_time, receiver)]
        data_counts = {f"data_cls_{i}": (
            data == i).sum().item() for i in range(NUM_CLASSES)}

        self.sharing_record.write(
            {
                "ll_time": ll_time,
                "sender": sender,
                "receiver": receiver,
            } | data_counts
        )

    def share_with(self, ll_time):
        for agent_i, agent_j in combinations(range(len(self.agents)), 2):
            logging.critical(f"agent {agent_i} share with agent {agent_j}")
            self.agents[agent_i].router.share_with(
                self.agents[agent_j], other_id=agent_j, ll_time=ll_time,
                record_name=f"from_{agent_i}_to_{agent_j}")
            logging.critical(f"agent {agent_j} share with agent {agent_i}")
            self.agents[agent_j].router.share_with(
                self.agents[agent_i], other_id=agent_i, ll_time=ll_time,
                record_name=f"from_{agent_j}_to_{agent_i}")

        self.checkpoint_sharing(ll_time)

        self.learn_from_buffer(ll_time)
        self.checkpoint(ll_time, info="after_share_with")

    def reset_routers(self):
        for agent in self.agents:
            agent.router.reset()


class ShellFleetReceiverFirst(ShellFleet):
    def __init__(self, cfg: ShELLDataSharingConfig, dir="."):
        super().__init__(cfg, dir=dir)
        self.sharing_record = Record(f"{self.dir}/sharing.csv")

    def train_data_searcher_agent(self, agent, ll_time):
        record_name = f"{self.dir}/{agent.name}_searcher_task_{ll_time}"
        agent.train_data_searcher(ll_time, record_name=record_name)
        self.pickle_agent(agent, info=f"searcher_task_{ll_time}")

    def train_data_searcher(self, ll_time):
        """
        Train the outlier and the open set recognition model!
        """
        with Pool(processes=self.n_agents) as pool:
            pool.starmap(self.train_data_searcher_agent, [
                         (agent, ll_time) for agent in self.agents])
        self.load_agents(info=f"searcher_task_{ll_time}")

    def compute_query_agent(self, agent, ll_time):
        agent.compute_query(ll_time)
        self.pickle_agent(agent, info=f"query_task_{ll_time}")

    def compute_query(self, ll_time):
        with Pool(processes=self.n_agents) as pool:
            pool.starmap(self.compute_query_agent, [
                (agent, ll_time) for agent in self.agents])
        self.load_agents(info=f"query_task_{ll_time}")

    def seek_data(self, ll_time):
        """
        NOTE: we are not doing pickling agents here because
        seek_data is implemented sequentially!
        """
        for agent_i, agent_j in combinations(range(len(self.agents)), 2):
            self.seek_data_agent(agent_i, agent_j, ll_time)
            self.seek_data_agent(agent_j, agent_i, ll_time)

        for agent in self.agents:
            for (ll_time, requester), data in agent.outgoing_data.items():
                self.agents[requester].sharing_buffer.add_data(data)

    def seek_data_agent(self, agent_i: int, agent_j: int, ll_time: int):
        self.agents[agent_i].image_search(
            queries=self.agents[agent_j].queries[ll_time], ll_time=ll_time, requester=agent_j)

        # NOTE: TODO: FOR DEBUG pickle
        # self.pickle_agent(self.agents[agent_i], info=f"seek_data_{ll_time}")

    def checkpoint_sharing_agent(self, ll_time, requester: int, provider: int):
        _, query = self.agents[requester].queries[ll_time]
        _, data = self.agents[provider].outgoing_data[(ll_time, requester)]
        _, outlier = self.agents[provider].outliers[(ll_time, requester)]
        query_counts = {f"query_cls_{i}": (
            query == i).sum().item() for i in range(NUM_CLASSES)}
        data_counts = {f"data_cls_{i}": (
            data == i).sum().item() for i in range(NUM_CLASSES)}
        outlier_counts = {f"outlier_cls_{i}": (
            outlier == i).sum().item() for i in range(NUM_CLASSES)}

        self.sharing_record.write(
            {
                "ll_time": ll_time,
                "requester": requester,
                "provider": provider,
            } | query_counts | data_counts | outlier_counts
        )

    def checkpoint_sharing(self, ll_time):
        """
        Row, ll_time, receiver, sender, query_cls_0, ..., query_cls_n, data_cls_0, ..., data_cls_n
        """
        for agent_i, agent_j in combinations(range(len(self.agents)), 2):
            self.checkpoint_sharing_agent(ll_time, agent_i, agent_j)
            self.checkpoint_sharing_agent(ll_time, agent_j, agent_i)
        self.sharing_record.save()

    def share_with(self, ll_time):
        self.train_data_searcher(ll_time)
        logging.info(
            f"Train data search completed at {ll_time+1}/{self.cfg.dataset.num_task_per_life}")
        self.compute_query(ll_time)
        logging.info(
            f"Compute query completed at {ll_time+1}/{self.cfg.dataset.num_task_per_life}")
        self.seek_data(ll_time)
        logging.info(
            f"Seek data completed at {ll_time+1}/{self.cfg.dataset.num_task_per_life}")

        self.checkpoint_sharing(ll_time)
        self.learn_from_buffer(ll_time)
        self.checkpoint(ll_time, info="after_share_with")
