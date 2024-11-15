import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
# import torch
from sklearn.cluster import KMeans
import numpy as np


def loss_calc_cosin(pred1, pred2):
    # n, c, h, w = pred1.size()
    pred1 = pred1.view(-1).cuda()
    pred2 = pred2.view(-1).cuda()
    # print(pred1)
    # print(pred2)
    output = torch.matmul(pred1, pred2) / (torch.norm(pred1) * torch.norm(pred2))
    return output


def loss_calc_cosin_out(pred1, pred2):
    # n, c, h, w = pred1.size()
    pred1 = pred1.view(-1).cuda()
    pred2 = pred2.view(-1).cuda()
    # print(pred1)
    # print(pred2)
    output = torch.abs(1 - (torch.matmul(pred1, pred2) / (torch.norm(pred1) * torch.norm(pred2))))
    return output


def loss_calc_dist(pred1, pred2):
    # print(pred1.shape)
    # print(pred1.shape)
    n, c, h, w = pred1.size()
    pred1 = pred1.view(-1).cuda()
    pred2 = pred2.view(-1).cuda()
    # print(pred1)
    # print(pred2)
    output = torch.sum(torch.abs(pred1 - pred2)) / (h * w * c)
    return output


def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                         - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)


@MODELS.register_module()
class PCLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 loss_weight=0.01,
                 loss_name='loss_pcl'):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name


    def forward(self, Proto, feat, labels):
        """
        Args:
            C: NUM_CLASSES A: feat_dim B: batch_size H: feat_high W: feat_width N: number of pixels except IGNORE_LABEL
            Proto: shape: (C, A) the mean representation of each class
            feat: shape (BHW, A) -> (N, A)
            labels: shape (BHW, ) -> (N, )
        Returns:
        """
        assert not Proto.requires_grad
        assert not labels.requires_grad
        assert feat.requires_grad
        # assert feat.dim() == 2
        # assert labels.dim() == 1
        # remove IGNORE_LABEL pixels
        # mask = (labels != 255)
        # labels = labels[mask]
        # feat = feat[mask]

        # feat = F.normalize(feat, p=2, dim=1)
        # Proto = F.normalize(Proto, p=2, dim=1)
        #
        # # print('feat shape : ', feat.shape)
        # # print('proto shape : ', Proto.shape)
        #
        # logits = feat.mm(Proto.permute(1, 0).contiguous())
        # logits = logits / 0.5
        #
        # ce_criterion = nn.CrossEntropyLoss()
        # loss = ce_criterion(logits, labels)
        feat_np = feat.detach().cpu().numpy()

        # 使用KMeans找到19个聚类中心
        # kmeans = KMeans(n_clusters=19, random_state=0).fit(feat_np)
        kmeans = KMeans(n_clusters=19, init=Proto.detach().cpu().numpy(), n_init=1).fit(feat_np)

        # 获取聚类中心
        centroids = kmeans.cluster_centers_

        # 将聚类中心转换回PyTorch张量（如果需要的话）
        centroids_tensor = torch.from_numpy(centroids).cuda()
        loss = loss_calc_cosin_out(Proto, centroids_tensor)

        return loss * self.loss_weight

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name