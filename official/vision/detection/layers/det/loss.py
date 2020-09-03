# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine.functional as F
from megengine.core import Tensor


def binary_cross_entropy_with_logits(logits: Tensor, targets: Tensor) -> Tensor:
    r"""Binary Cross Entropy

    Args:
        logits (Tensor):
            the predicted logits
        targets (Tensor):
            the assigned targets with the same shape as logits

    Returns:
        the calculated binary cross entropy.
    """
    return -(targets * F.logsigmoid(logits) + (1 - targets) * F.logsigmoid(-logits))


def sigmoid_focal_loss(
    logits: Tensor, targets: Tensor, alpha: float = -1, gamma: float = 0,
) -> Tensor:
    r"""Focal Loss for Dense Object Detection:
    <https://arxiv.org/pdf/1708.02002.pdf>

    .. math::

        FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)

    Args:
        logits (Tensor):
            the predicted logits
        targets (Tensor):
            the assigned targets with the same shape as logits
        alpha (float):
            parameter to mitigate class imbalance. Default: -1
        gamma (float):
            parameter to mitigate easy/hard loss imbalance. Default: 0

    Returns:
        the calculated focal loss.
    """
    scores = F.sigmoid(logits)
    loss = binary_cross_entropy_with_logits(logits, targets)
    if gamma != 0:
        loss *= (targets * (1 - scores) + (1 - targets) * scores) ** gamma
    if alpha >= 0:
        loss *= targets * alpha + (1 - targets) * (1 - alpha)
    return loss


def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    r"""Smooth L1 Loss

    Args:
        pred (Tensor):
            the predictions
        target (Tensor):
            the assigned targets with the same shape as pred
        beta (int):
            the parameter of smooth l1 loss.

    Returns:
        the calculated smooth l1 loss.
    """
    x = pred - target
    abs_x = F.abs(x)
    if beta < 1e-5:
        loss = abs_x
    else:
        in_loss = 0.5 * x ** 2 / beta
        out_loss = abs_x - 0.5 * beta
        loss = F.where(abs_x < beta, in_loss, out_loss)
    return loss


def iou_loss(
    pred: Tensor, target: Tensor, box_mode: str = "xyxy", loss_type: str = "iou", eps: float = 1e-8,
) -> Tensor:
    if box_mode == "ltrb":
        pred = F.concat([-pred[..., :2], pred[..., 2:]], axis=-1)
        target = F.concat([-target[..., :2], target[..., 2:]], axis=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    pred_area = F.clamp(pred[..., 2] - pred[..., 0], lower=0) * F.clamp(
        pred[..., 3] - pred[..., 1], lower=0
    )
    target_area = F.clamp(target[..., 2] - target[..., 0], lower=0) * F.clamp(
        target[..., 3] - target[..., 1], lower=0
    )

    w_intersect = F.clamp(
        F.minimum(pred[..., 2], target[..., 2])
        - F.maximum(pred[..., 0], target[..., 0]),
        lower=0,
    )
    h_intersect = F.clamp(
        F.minimum(pred[..., 3], target[..., 3])
        - F.maximum(pred[..., 1], target[..., 1]),
        lower=0,
    )

    area_intersect = w_intersect * h_intersect
    area_union = pred_area + target_area - area_intersect
    ious = area_intersect / F.clamp(area_union, lower=eps)

    if loss_type == "iou":
        loss = -F.log(F.clamp(ious, lower=eps))
    elif loss_type == "linear_iou":
        loss = 1 - ious
    elif loss_type == "giou":
        g_w_intersect = F.maximum(pred[..., 2], target[..., 2]) - F.minimum(
            pred[..., 0], target[..., 0]
        )
        g_h_intersect = F.maximum(pred[..., 3], target[..., 3]) - F.minimum(
            pred[..., 1], target[..., 1]
        )
        ac_union = g_w_intersect * g_h_intersect
        gious = ious - (ac_union - area_union) / F.clamp(ac_union, lower=eps)
        loss = 1 - gious
    return loss


def softmax_loss(logits: Tensor, targets: Tensor, ignore_label: int = -1) -> Tensor:
    log_prob = F.log_softmax(logits, axis=1)
    mask = targets != ignore_label
    vtargets = targets * mask
    loss = -(F.indexing_one_hot(log_prob, vtargets.astype("int32"), 1) * mask).sum()
    loss = loss / F.maximum(mask.sum(), 1)
    return loss
