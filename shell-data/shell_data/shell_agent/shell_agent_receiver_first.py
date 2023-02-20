import torch.nn.functional as F
from shell_data.shell_agent.shell_agent_classification import ShELLClassificationAgent
import torch
from shell_data.dataset.buffer import BalancedClassificationBuffer, ClassifcationBuffer, get_dataset_from_buffer, ReservoirSamplingClassificationBuffer
from shell_data.utils.record import Record, snapshot_perf, snapshot_conf_mat
from copy import deepcopy
import logging
from typing import List, Tuple, Optional
import torchvision.transforms as transforms
from shell_data.task_model.task_nets import MNISTRecognizer
from shell_data.shell_agent.contrastive import SupConLoss
from pytorch_ood.loss import CACLoss, IILoss, CenterLoss
import torch.nn as nn
from shell_data.shell_agent.receiver_first import least_confidence_scorer, margin_scorer, entropy_scorer, wrong_prediction_scorer, combine_wrong_least_confidence_scorer
from pytorch_ood.detector import (
    ODIN,
    EnergyBased,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
    MCD,
)
from shell_data.utils.config import ShELLDataSharingConfig
from shell_data.task_model.task_model import ClassifcationTaskModel
from functools import partial
import numpy as np
from pythresh.thresholds.all import ALL
from pythresh.thresholds.filter import FILTER

# TODO: move all ops to cuda to do it faster!

NUM_CLASSES = 10


def image_search(queries, database, reducer_callable, n_neighbors=10, p=2, metric="distance"):
    query_embed = reducer_callable(X=queries)
    database_embed = reducer_callable(X=database)
    if metric == "distance":
        dist = torch.cdist(query_embed, database_embed, p=p)
    elif metric == "cosine":
        dist = 1 - torch.stack([F.cosine_similarity(query_embed[i], database_embed)
                               for i in range(len(query_embed))])
    else:
        raise ValueError(f"metric {metric} is not supported")
    closest_dist, closest_idx = torch.topk(
        dist, k=n_neighbors, dim=1, largest=False)
    return closest_dist, closest_idx


def contrastive_callable(model, X):
    """
    This function is called by the ContrastiveModel
    """
    encoded_images = model.embedding(X).view(X.shape[0], -1)
    return encoded_images


class ShellAgentReceiverFirst(ShELLClassificationAgent):

    def init(self):
        self.queries = {}  # queries[ll_time]
        self.outgoing_data = {}  # outgoing_data[(ll_time, requester)]
        self.outliers = {}  # outliers[(ll_time, requester)]
        self.model = ClassifcationTaskModel(
            n_classes=self.cfg.num_classes, cfg=self.cfg.task_model)

        if self.cfg.receiver_first.query_strategy == "wrongly_predicted":
            self.scorer = wrong_prediction_scorer
        elif self.cfg.receiver_first.query_strategy == "least_confidence":
            self.scorer = least_confidence_scorer
        elif self.cfg.receiver_first.query_strategy == "margin":
            self.scorer = margin_scorer
        elif self.cfg.receiver_first.query_strategy == "entropy":
            self.scorer = entropy_scorer
        elif self.cfg.receiver_first.query_strategy == "combine_wrong_least_confidence":
            self.scorer = combine_wrong_least_confidence_scorer
        else:
            raise ValueError(
                f"Query strategy {self.cfg.receiver_first.query_strategy} not supported.")

    def remap_labels(self, ys):
        for i, y in enumerate(torch.unique(ys)):
            ys[ys == y] = i
        return ys

    def train_data_searcher(self, ll_time, record_name=None):
        """
        Train the data searcher using the task buffer...

        NOTE: adding decoder in and pray for a curve
        """

        if record_name is None:
            record_name = f"receiver_first_{self.name}_{ll_time}"
        record = Record(record_name)

        buf_x, buf_y = self.buffer.get_data(
            batch_size=self.cfg.experience_replay.buffer_size)
        buf_x = buf_x.to(self.cfg.task_model.device)
        buf_y = buf_y.to(self.cfg.task_model.device)
        buf_y_remapped = self.remap_labels(buf_y)
        dataset = torch.utils.data.TensorDataset(buf_x, buf_y_remapped)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cfg.receiver_first.batch_size, shuffle=True)

        # n_classes = self.cfg.dataset.num_cls_per_task * (ll_time + 1)
        n_classes = torch.unique(buf_y).shape[0]

        # TODO: initialize the conv weight of recognizer from the pretrained
        # classifier (try to see if this makes the image similarity searcher better, converge faster ect...)
        self.recognizer = MNISTRecognizer(n_out=n_classes)
        # NOTE: using embedding from the pretrained model on the task
        # self.recognizer.embedding.load_state_dict(
        #     self.model.net.embedding.state_dict()
        # )

        self.recognizer.to(self.cfg.task_model.device)
        losses = []
        cont_losses = []
        cac_losses = []
        rec_losses = []

        optimizer = torch.optim.Adam(self.recognizer.parameters(), lr=1e-3)

        scl = SupConLoss(device=self.cfg.task_model.device,
                         temperature=self.cfg.receiver_first.cont_temp,
                         contrast_mode=self.cfg.receiver_first.cont_mode,)
        self.cac = CACLoss(n_classes=n_classes)
        self.cac._centers = self.cac._centers.to(self.cfg.task_model.device)

        rl = nn.MSELoss()

        global_step = 0
        for epoch in range(self.cfg.receiver_first.n_epochs):
            if epoch % 25 == 1:
                logging.info(
                    f"Epoch {epoch+1}/{self.cfg.receiver_first.n_epochs} loss {np.nanmean(losses):.3f}, cont_loss {np.nanmean(cont_losses):.3f}, rec_loss {np.nanmean(rec_losses):.3f},"
                    f"cac loss {np.nanmean(cac_losses):.3f} (n={n_classes})")
            for batch in dataloader:
                x, y = batch

                # CONTRASIVE LOSS
                train_transform = transforms.Compose([
                    # transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                    transforms.RandomResizedCrop(
                        size=28, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                ])

                encoded_transformed_images = self.recognizer.embedding(
                    train_transform(x))
                encoded_images = self.recognizer.embedding(x)

                encoded_transformed_images = encoded_transformed_images.view(
                    encoded_transformed_images.shape[0], -1)
                encoded_images = encoded_images.view(
                    encoded_images.shape[0], -1)

                features = torch.cat(
                    [encoded_transformed_images.unsqueeze(1),
                        encoded_images.unsqueeze(1)], dim=1)

                cont_loss = scl(features, y)

                # RECONSTRUCTION LOSS (autoencoder)
                rec_loss = rl(x, self.recognizer.reconstruct(x))
                # rec_loss = torch.tensor([0.0])

                # CAC LOSS
                # check device of recognizer
                z = self.recognizer(x)
                distances = self.cac.calculate_distances(z)
                cac_loss = self.cac(distances, y)

                loss = cont_loss + cac_loss + rec_loss
                # loss = cont_loss + cac_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                cont_losses.append(cont_loss.item())
                cac_losses.append(cac_loss.item())
                rec_losses.append(rec_loss.item())

                record.write(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss.item(),
                        "cont_loss": cont_loss.item(),  # contrastive loss (image similarity search)
                        "cac_loss": cac_loss.item(),  # class anchoring clustering (open set recognition)
                        "rec_loss": rec_loss.item(),  # reconstruction loss (autoencoder)
                    }
                )

                global_step += 1

        # use the Mahalanobis detector
        # self.detector = Mahalanobis(self.recognizer.features)
        # self.detector.fit(dataloader, device=self.cfg.task_model.device)

        # NOTE: move this out, so that the outlier score
        # are calculated at once!
        # calculate the outlier score on the train set
        # train
        # train_cac_scores = self.outlier_score_cac(buf_x)
        # train_detector_scores = self.outlier_score_detector(buf_x)

        # self.cac_upper_bound = train_cac_scores.mean(
        # ) + self.cfg.receiver_first.outlier_std * train_cac_scores.std()
        # self.detector_upper_bound = train_detector_scores.mean(
        # ) + self.cfg.receiver_first.outlier_std * train_detector_scores.std()

        record.save()
        return losses, cont_losses, cac_losses, rec_losses

    def _compute_query(self, ll_time):
        val_dataset = self.ll_dataset.get_val_dataset(ll_time, kind="all")
        X_val = torch.stack([x for x, y in val_dataset]).to(
            self.cfg.task_model.device)
        y_val = torch.tensor([y for x, y in val_dataset]).to(
            self.cfg.task_model.device)
        scores = self.scorer(X=X_val, y=y_val, model=self.model.net)
        rank = torch.argsort(scores, descending=True)
        top_k = rank[:self.cfg.receiver_first.num_queries]
        top_k = top_k[scores[top_k] >
                      self.cfg.receiver_first.query_score_threshold]
        return X_val[top_k].cpu(), y_val[top_k].cpu()

    def compute_query(self, ll_time):
        self.queries[ll_time] = self._compute_query(ll_time)
        return self.queries[ll_time]

    def outlier_score_cac(self, x):
        x = x.to(self.cfg.task_model.device)
        with torch.no_grad():
            # TODO: if x is too large (> 128), we need to split it
            # to avoid CUDA out of memory
            if x.shape[0] > 128:
                dist = []
                for i in range(0, x.shape[0], 128):
                    dist.append(self.cac.calculate_distances(
                        self.recognizer(x[i:i + 128])))
                dist = torch.cat(dist)
            else:
                dist = self.cac.calculate_distances(self.recognizer(x))
            outlier_score = self.cac.score(dist)
        return outlier_score

    def outlier_score_detector(self, x):
        x = x.to(self.cfg.task_model.device)
        with torch.no_grad():
            # TODO: if x is too large (> 128), we need to split it
            # to avoid CUDA out of memory
            if x.shape[0] > 128:
                outlier_score = []
                for i in range(0, x.shape[0], 128):
                    outlier_score.append(self.detector(x[i:i + 128]))
                outlier_score = torch.cat(outlier_score)
            else:
                outlier_score = self.detector(x)
        return outlier_score

    def remove_outliers(self, queries, ll_time=None, requester=None):
        """
        outlier_mask returns 1 for outliers and 0 for inliers
        """
        X, y = queries
        if X.shape[0] == 0:
            if ll_time is not None and requester is not None:
                self.outliers[(ll_time, requester)] = X, y
            return X, y
        X = X.to(self.cfg.task_model.device)
        y = y.to(self.cfg.task_model.device)

        buf_x, _ = self.buffer.get_data(
            batch_size=self.cfg.experience_replay.buffer_size)

        buf_x = torch.concat([buf_x.cuda(), X.cuda()])
        comb_cac_scores = self.outlier_score_cac(buf_x)
        # comb_detector_scores = self.outlier_score_detector(buf_x)

        # cac_upper_bound = np.quantile(
        #     comb_cac_scores.cpu(), self.cfg.receiver_first.outlier_std)
        # detector_upper_bound = np.quantile(
        #     comb_detector_scores.cpu(), self.cfg.receiver_first.outlier_std)

        # max_contam = self.cfg.receiver_first.threshold_init_max_contam * \
        #     (self.cfg.receiver_first.threshold_decay_rate) ** ll_time

        # linearly decay from 0.5 to 0.1 instead!
        # https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler
        # max_contam = 0.5 - 0.4 * \
        #     (ll_time / (self.cfg.dataset.num_task_per_life - 1))

        # piecewise constant max_contam so that 0,1: 0.5, 2,3: 0.3; 4: 0.1

        max_contam = 0.5
        # if self.cfg.receiver_first.funky_threshold:
        #     if ll_time in [0, 1]:
        #         max_contam = 0.5
        #     elif ll_time in [2, 3]:
        #         max_contam = 0.3
        #     elif ll_time in [4]:
        #         max_contam = 0.1

        # if self.cfg.receiver_first.outlier_method == "cac_contrastive":
        #     cac_scores = self.outlier_score_cac(X)
        #     detector_scores = self.outlier_score_detector(X)
        #     # AND (we can also use OR), but AND balances the outlier detection rate, and accuracy
        #     outlier_mask = (cac_scores > cac_upper_bound) & (
        #         detector_scores > detector_upper_bound)
        # elif self.cfg.receiver_first.outlier_method == "ground_truth":
        if self.cfg.receiver_first.outlier_method == "ground_truth":
            # get the index of y where y is not in self.past_tasks
            outlier_mask = torch.tensor(
                [y_i not in self.past_tasks for y_i in y])
        elif self.cfg.receiver_first.outlier_method == "pythresh_cac":
            thres = ALL(method="median", max_contam=max_contam)
            # TODO: should eval on cac_scores or detector_scores?
            # eval returns 0 for inliers and 1 for outliers
            comb_outlier_mask = thres.eval(comb_cac_scores.cpu())
            outlier_mask = torch.tensor(comb_outlier_mask[-X.shape[0]:]).bool()
        # elif self.cfg.receiver_first.outlier_method == "pythresh_detector":
        #     thres = ALL(method="median", max_contam=max_contam)
        #     comb_outlier_mask = thres.eval(comb_detector_scores.cpu())
        #     outlier_mask = torch.tensor(comb_outlier_mask[-X.shape[0]:]).bool()
        elif self.cfg.receiver_first.outlier_method == "pythresh_filter":
            thres = FILTER()
            comb_outlier_mask = thres.eval(comb_cac_scores.cpu())
            outlier_mask = torch.tensor(comb_outlier_mask[-X.shape[0]:]).bool()
        elif self.cfg.receiver_first.outlier_method == "no_outlier_detection":
            outlier_mask = torch.tensor([False] * X.shape[0])
        else:
            raise NotImplementedError

        if ll_time is not None and requester is not None:
            self.outliers[(ll_time, requester)
                          ] = X[outlier_mask], y[outlier_mask]
        return X[~outlier_mask], y[~outlier_mask]

    def _image_search(self, queries, ll_time, requester):
        queries = self.remove_outliers(queries, ll_time, requester)
        query_X = queries[0]
        if len(query_X) == 0:
            return torch.empty((0, *self.dim)), torch.empty((0,))
        buf_x, buf_y = self.buffer.get_data(
            batch_size=self.cfg.experience_replay.buffer_size)
        buf_x = buf_x.to(self.cfg.task_model.device)
        buf_y = buf_y.to(self.cfg.task_model.device)
        closest_dist, closest_idx = image_search(query_X, buf_x,  reducer_callable=partial(
            contrastive_callable, model=self.recognizer), n_neighbors=self.cfg.receiver_first.num_neighbors)

        return buf_x[closest_idx.flatten()].cpu(), buf_y[closest_idx.flatten()].cpu()

    def image_search(self, queries, ll_time, requester):
        self.outgoing_data[(ll_time, requester)] = self._image_search(
            queries, ll_time, requester)
        return self.outgoing_data[(ll_time, requester)]
