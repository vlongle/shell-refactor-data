from shell_data.shell_agent.shell_agent_classification import ShELLClassificationAgent
import torch
from shell_data.dataset.buffer import BalancedClassificationBuffer, ClassifcationBuffer, get_dataset_from_buffer, ReservoirSamplingClassificationBuffer
from shell_data.utils.record import Record, snapshot_perf, snapshot_conf_mat
from copy import deepcopy
import logging
from typing import List, Tuple, Optional
from shell_data.router.router import (
    RandomShellRouter,
    NeuralShellRouter,
    OracleShellRouter,
)
from shell_data.task_model.task_model import ClassifcationTaskModel

NUM_CLASSES = 10


class ShellAgentSenderFirst(ShELLClassificationAgent):
    def init(self):
        """
        NOTE: in multi-agent setting, to ensure no routing and routing have the same task distribution,
        (random number generator is not used to generate the weights of the task model and the router).
        We initialize the task model and the router only after we have initialized the dataset
        for every agent (in the __init__ function)
        """
        self.model = ClassifcationTaskModel(
            n_classes=self.cfg.num_classes, cfg=self.cfg.task_model)

        self.router_cfg = self.cfg.router
        if self.router_cfg.strategy == "random":
            self.router = RandomShellRouter(self, self.router_cfg)
        elif self.router_cfg.strategy == "neural":
            self.router = NeuralShellRouter(self, self.router_cfg)
        elif self.router_cfg.strategy == "oracle":
            self.router = OracleShellRouter(self, self.router_cfg)
        elif self.router_cfg.strategy == "no_routing":
            self.router = None
        else:
            raise ValueError(
                f"Unknown router name: {self.router_cfg.strategy}")

    def data_valuation_oracle(self, X, y, ll_time, metric, record_name):
        rewards = torch.zeros(X.shape[0])
        for cl in range(NUM_CLASSES):
            X_cl = X[y == cl]
            y_cl = y[y == cl]
            if len(X_cl) == 0:
                continue
            # check if y is in past tasks
            if cl in self.past_tasks:
                rewards[y == cl] = 1.0
            else:
                rewards[y == cl] = -1.0
        return rewards

    def data_valuation_clustering(self, X, y, ll_time, metric, record_name):
        pass

    def data_valuation_class_perf(self, X, y, ll_time, metric="past_task_val_acc", record_name=None):
        """
        Contribution of each class to the improvement on `past_task_test_acc`
        """
        rewards = torch.zeros(X.shape[0])
        if metric == "past_task_test_acc":
            before_past_task_test_acc = self.shared_test_val(
                ll_time, metric_type="acc", split="test", kind="all")
        elif metric == "past_task_val_acc":
            before_past_task_test_acc = self.shared_test_val(
                ll_time, metric_type="acc", split="val", kind="all")
        else:
            raise NotImplementedError

        snapshot_conf_mat(self, ll_time, "before")

        for cl in range(NUM_CLASSES):
            X_cl = X[y == cl]
            y_cl = y[y == cl]
            if len(X_cl) == 0:
                continue
            agent_copy = deepcopy(self)
            agent_copy.buffer.add_data((X_cl, y_cl))

            buf_x, buf_y = agent_copy.buffer.get_data(
                batch_size=self.cfg.experience_replay.buffer_size)
            train_cls_dataset = torch.utils.data.TensorDataset(
                buf_x, buf_y)
            snapshot_conf_mat(
                agent_copy, ll_time, f"before_class_{cl}", train_dataset=train_cls_dataset)

            if record_name is not None:
                cls_record_name = f"{record_name}_record_cls_eval_{ll_time}_class_{cl}.csv"
            else:
                cls_record_name = f"record_cls_eval_{ll_time}_class_{cl}.csv"

            if metric == "past_task_test_acc":
                agent_copy.learn_from_buffer(
                    ll_time, kind="all", metric="past_task_test_acc", record_name=cls_record_name,
                    val_before=False,)
                after_past_task_test_acc = agent_copy.shared_test_val(
                    ll_time, metric_type="acc", split="test", kind="all")

            elif metric == "past_task_val_acc":
                agent_copy.learn_from_buffer(
                    ll_time, kind="all", metric="past_task_val_acc", record_name=cls_record_name,
                    val_before=False,)
                after_past_task_test_acc = agent_copy.shared_test_val(
                    ll_time, metric_type="acc", split="val", kind="all")

            # TODO: have to add agent name here...
            # agent_copy.save_model(f"{ll_time}_class_{cl}")
            # agent_copy.save_buffer(f"{ll_time}_class_{cl}")

            snapshot_conf_mat(
                agent_copy, ll_time, f"after_class_{cl}", train_dataset=train_cls_dataset)

            reward = (after_past_task_test_acc -
                      before_past_task_test_acc) / (before_past_task_test_acc + 1e-8)

            logging.critical(
                f"Class {cl} before {before_past_task_test_acc} after {after_past_task_test_acc} len {len(X_cl)} contribution: {reward}")
            rewards[y == cl] = reward

        return rewards

    def data_valuation_random(self, X, y, ll_time, metric, record_name):
        # rewards is always 1
        rewards = torch.ones(X.shape[0])
        return rewards

    def data_valuation(self, X: torch.tensor, y: torch.tensor, ll_time: int, record_name=None) -> Tuple[torch.tensor, torch.tensor]:
        if self.cfg.data_valuation.method == "oracle":
            func = self.data_valuation_oracle
        elif self.cfg.data_valuation.method == "clustering":
            func = self.data_valuation_oracle
        elif self.cfg.data_valuation.method == "performance":
            func = self.data_valuation_class_perf
        elif self.cfg.data_valuation.method == "random":
            func = self.data_valuation_random
        else:
            raise NotImplementedError
        scores = func(X, y, ll_time, record_name=record_name,
                      metric=self.cfg.data_valuation.metric)

        # keep the data (add to buffer) if score > threshold
        to_keeps = scores > self.cfg.data_valuation.threshold
        data_to_keep = (X[to_keeps], y[to_keeps])
        if len(X[to_keeps]) > 0:
            mask = self.sharing_buffer.dedup(data_to_keep, ret_mask=True)
            # if mask is False means that the data is already in the buffer, we
            # will change the scores to -1 so that the preference model will
            # not send it in the future
            # change the scores to -1 so that the preference model will not
            # send it in the future

            # scores[to_keeps][mask] = -1
            indices = torch.where(to_keeps)[0]
            scores.scatter_(-1, indices[~mask], -1)
            self.sharing_buffer.add_data(data_to_keep)

        return scores, to_keeps
