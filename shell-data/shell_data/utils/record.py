import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import ConfusionMatrix as CM
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
    _, past_task_val_conf = agent.shared_test_val(
        ll_time, metric_type="acc", split="val", kind="all", ret_confusion_matrix=True)

    np.save(f"{record_name}_train_conf_mat_{ll_time}.npy", train_conf)
    np.save(f"{record_name}_val_conf_mat_{ll_time}.npy", val_conf)
    np.save(f"{record_name}_test_conf_mat_{ll_time}.npy", test_conf)
    np.save(f"{record_name}_past_task_test_conf_mat_{ll_time}.npy",
            past_task_test_conf)
    np.save(f"{record_name}_past_task_val_conf_mat_{ll_time}.npy",
            past_task_val_conf)


def snapshot_perf(agent, record, ll_time, info=""):
    train_acc = agent.shared_test_val(
        ll_time, metric_type="acc", split="train")
    val_acc = agent.shared_test_val(
        ll_time, metric_type="acc", split="val")
    test_acc = agent.shared_test_val(
        ll_time, metric_type="acc", split="test")
    past_task_train_acc = agent.shared_test_val(
        ll_time, metric_type="acc", split="train", kind="all")
    past_task_test_acc = agent.shared_test_val(
        ll_time, metric_type="acc", split="test", kind="all")
    past_task_val_acc = agent.shared_test_val(
        ll_time, metric_type="acc", split="val", kind="all")

    task_cls_ct = agent.buffer.get_cls_counts()
    sharing_cls_ct = agent.sharing_buffer.get_cls_counts()
    # add them together
    cls_ct = {k: task_cls_ct[k] + sharing_cls_ct[k] for k in task_cls_ct}
    record.write({
        "agent_name": agent.name,
        "ll_time": ll_time,
        "info": info,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "past_task_test_acc": past_task_test_acc,
        "past_task_val_acc": past_task_val_acc,
        "past_task_train_acc": past_task_train_acc,
    } | cls_ct
    )


# calculate the per-class accuracy from the confusion matrix
def per_class_acc(conf_mat):
    return np.diag(conf_mat) / conf_mat.sum(axis=1)


def per_class_true_labels(conf_mat):
    return conf_mat.sum(axis=1)


def acc(conf_mat):
    # total accuracy
    return np.diag(conf_mat).sum() / conf_mat.sum()


def predicted_cls_freq(conf_mat):
    # frequency that a class is predicted
    return conf_mat.sum(axis=0) / conf_mat.sum()


def to_features(X):
    return X.view(X.size(0), -1)


def load_snapshot_conf_mats(record_name, ll_time):
    train_conf = np.load(f"{record_name}_train_conf_mat_{ll_time}.npy")
    test_conf = np.load(f"{record_name}_test_conf_mat_{ll_time}.npy")
    val_conf = np.load(f"{record_name}_val_conf_mat_{ll_time}.npy")
    past_test_conf = np.load(
        f"{record_name}_past_task_test_conf_mat_{ll_time}.npy")
    past_val_conf = np.load(
        f"{record_name}_past_task_val_conf_mat_{ll_time}.npy")
    return train_conf, test_conf, val_conf, past_test_conf, past_val_conf


def summarize_conf(conf_mat):
    return {
        "acc": acc(conf_mat),
        "per_class_acc": per_class_acc(conf_mat),
        "per_class_true_labels": per_class_true_labels(conf_mat),
        "predicted_cls_freq": predicted_cls_freq(conf_mat),
    }


def summarize_confs(confs, names):
    # return 3 dfs: first df has acc follows by per_class_acc
    # second df has per_class_true_lables
    # third has predicted_cls_freq
    df1 = pd.DataFrame(columns=["name", "acc"] +
                       [f"per_class_acc_{i}" for i in range(10)])
    df2 = pd.DataFrame(
        columns=["name"] + [f"per_class_true_labels_{i}" for i in range(10)])
    df3 = pd.DataFrame(columns=["name"] +
                       [f"predicted_cls_freq_{i}" for i in range(10)])
    # summarize the confs first
    sums = [summarize_conf(conf) for conf in confs]
    for i, name in enumerate(names):
        df1.loc[i] = [name] + [sums[i]["acc"]] + list(sums[i]["per_class_acc"])
        df2.loc[i] = [name] + list(sums[i]["per_class_true_labels"])
        df3.loc[i] = [name] + list(sums[i]["predicted_cls_freq"])
    return df1, df2, df3


def viz_conf(conf, name=""):
    fig, ax = plt.subplots(figsize=(10, 7))
    ConfusionMatrixDisplay(conf).plot(cmap="Blues", values_format="", ax=ax)
    plt.title(name)
    print("accuracy", per_class_acc(conf))
    print("overall accuracy", acc(conf))
    print("predicted freq:", predicted_cls_freq(conf))
    print("class freq:", per_class_true_labels(conf))
