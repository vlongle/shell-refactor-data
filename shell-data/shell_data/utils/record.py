import pandas as pd
import numpy as np


class Record:
    def __init__(self, path: str):
        self.path = path
        self.df = pd.DataFrame()

    def write(self, row: dict):
        # use pandas.concat to append a row to the dataframe
        self.df = pd.concat([self.df, pd.DataFrame(row, index=[0])])

    def save(self):
        self.df.to_csv(self.path, index=False)


def snapshot_conf_mat(agent, ll_time, record_name, train_dataset=None):
    if train_dataset is not None:
        _, train_conf = agent.shared_test_val(
            ll_time, metric_type="acc", split=None, ret_confusion_matrix=True, dataset=train_dataset)
    else:
        _, train_conf = agent.shared_test_val(
            ll_time, metric_type="acc", split="train", ret_confusion_matrix=True)
    _, val_conf = agent.shared_test_val(
        ll_time, metric_type="acc", split="val", ret_confusion_matrix=True)
    _, test_conf = agent.shared_test_val(
        ll_time, metric_type="acc", split="test", ret_confusion_matrix=True)
    _, past_task_test_conf = agent.shared_test_val(
        ll_time, metric_type="acc", split="test", kind="all", ret_confusion_matrix=True)

    np.save(f"{record_name}_train_conf_mat_{ll_time}.npy", train_conf)
    np.save(f"{record_name}_val_conf_mat_{ll_time}.npy", val_conf)
    np.save(f"{record_name}_test_conf_mat_{ll_time}.npy", test_conf)
    np.save(f"{record_name}_past_task_test_conf_mat_{ll_time}.npy",
            past_task_test_conf)


def snapshot_perf(agent, record, agent_id, ll_time, local=True):
    train_acc = agent.shared_test_val(
        ll_time, metric_type="acc", split="train")
    val_acc = agent.shared_test_val(
        ll_time, metric_type="acc", split="val")
    test_acc = agent.shared_test_val(
        ll_time, metric_type="acc", split="test")
    past_task_test_acc = agent.shared_test_val(
        ll_time, metric_type="acc", split="test", kind="all")

    record.write({
        "agent_id": agent_id,
        "ll_time": ll_time,
        "local": local,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "past_task_test_acc": past_task_test_acc,
    } | agent.buffer.get_cls_counts()
    )
