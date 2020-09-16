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

from official.vision.detection import layers


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
    logits: Tensor,
    targets: Tensor,
    alpha: float = -1,
    gamma: float = 0,
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


def softmax_loss(logits: Tensor, targets: Tensor, ignore_label: int = -1) -> Tensor:
    log_prob = F.log_softmax(logits, axis=1)
    mask = targets != ignore_label
    vtargets = targets * mask
    loss = -(F.indexing_one_hot(log_prob, vtargets.astype("int32"), 1) * mask).sum()
    loss = loss / F.maximum(mask.astype(loss.dtype).sum(), 1)
    return loss
