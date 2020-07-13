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


def get_focal_loss(
    score: Tensor,
    label: Tensor,
    ignore_label: int = -1,
    background: int = 0,
    alpha: float = 0.5,
    gamma: float = 0,
    norm_type: str = "fg",
) -> Tensor:
    r"""Focal Loss for Dense Object Detection:
    <https://arxiv.org/pdf/1708.02002.pdf>

    .. math::

        FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)

    Args:
        score (Tensor):
            the predicted score with the shape of :math:`(B, A, C)`
        label (Tensor):
            the assigned label of boxes with shape of :math:`(B, A)`
        ignore_label (int):
            the value of ignore class. Default: -1
        background (int):
            the value of background class. Default: 0
        alpha (float):
            parameter to mitigate class imbalance. Default: 0.5
        gamma (float):
            parameter to mitigate easy/hard loss imbalance. Default: 0
        norm_type (str): current support 'fg', 'none':
            'fg': loss will be normalized by number of fore-ground samples
            'none": not norm

    Returns:
        the calculated focal loss.
    """
    class_range = F.arange(1, score.shape[2] + 1)

    label = F.add_axis(label, axis=2)
    pos_part = (1 - score) ** gamma * F.log(F.clamp(score, 1e-8))
    neg_part = score ** gamma * F.log(F.clamp(1 - score, 1e-8))

    pos_loss = -(label == class_range) * pos_part * alpha
    neg_loss = (
        -(label != class_range) * (label != ignore_label) * neg_part * (1 - alpha)
    )
    loss = pos_loss + neg_loss

    if norm_type == "fg":
        fg_mask = (label != background) * (label != ignore_label)
        return loss.sum() / F.maximum(fg_mask.sum(), 1)
    elif norm_type == "none":
        return loss.sum()
    else:
        raise NotImplementedError


def get_smooth_l1_loss(
    pred_bbox: Tensor,
    gt_bbox: Tensor,
    label: Tensor,
    beta: int = 1,
    background: int = 0,
    ignore_label: int = -1,
    norm_type: str = "fg",
) -> Tensor:
    r"""Smooth l1 loss used in RetinaNet.

    Args:
        pred_bbox (Tensor):
            the predicted bbox with the shape of :math:`(B, A, 4)`
        gt_bbox (Tensor):
            the ground-truth bbox with the shape of :math:`(B, A, 4)`
        label (Tensor):
            the assigned label of boxes with shape of :math:`(B, A)`
        beta (int):
            the parameter of smooth l1 loss. Default: 1
        background (int):
            the value of background class. Default: 0
        ignore_label (int):
            the value of ignore class. Default: -1
        norm_type (str): current support 'fg', 'all', 'none':
            'fg': loss will be normalized by number of fore-ground samples
            'all': loss will be normalized by number of all samples
            'none': not norm
    Returns:
        the calculated smooth l1 loss.
    """
    pred_bbox = pred_bbox.reshape(-1, 4)
    gt_bbox = gt_bbox.reshape(-1, 4)
    label = label.reshape(-1)

    fg_mask = (label != background) * (label != ignore_label)

    losses = get_smooth_l1_base(pred_bbox, gt_bbox, beta)
    if norm_type == "fg":
        loss = (losses.sum(axis=1) * fg_mask).sum() / F.maximum(fg_mask.sum(), 1)
    elif norm_type == "all":
        all_mask = label != ignore_label
        loss = (losses.sum(axis=1) * fg_mask).sum() / F.maximum(all_mask.sum(), 1)
    else:
        raise NotImplementedError

    return loss


def get_smooth_l1_base(
    pred_bbox: Tensor, gt_bbox: Tensor, beta: float,
):
    r"""

    Args:
        pred_bbox (Tensor):
            the predicted bbox with the shape of :math:`(N, 4)`
        gt_bbox (Tensor):
            the ground-truth bbox with the shape of :math:`(N, 4)`
        beta (int):
            the parameter of smooth l1 loss.

    Returns:
        the calculated smooth l1 loss.
    """
    x = pred_bbox - gt_bbox
    abs_x = F.abs(x)
    if beta < 1e-5:
        loss = abs_x
    else:
        in_loss = 0.5 * x ** 2 / beta
        out_loss = abs_x - 0.5 * beta

        # FIXME: F.where cannot handle 0-shape tensor yet
        # loss = F.where(abs_x < beta, in_loss, out_loss)
        in_mask = abs_x < beta
        loss = in_loss * in_mask + out_loss * (1 - in_mask)
    return loss


def softmax_loss(score, label, ignore_label=-1):
    max_score = F.zero_grad(score.max(axis=1, keepdims=True))
    score -= max_score
    log_prob = score - F.log(F.exp(score).sum(axis=1, keepdims=True))
    mask = label != ignore_label
    vlabel = label * mask
    loss = -(F.indexing_one_hot(log_prob, vlabel.astype("int32"), 1) * mask).sum()
    loss = loss / F.maximum(mask.sum(), 1)
    return loss
