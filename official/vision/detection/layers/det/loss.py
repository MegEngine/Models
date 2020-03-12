# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine as mge
import megengine.functional as F
import numpy as np

from megengine.core import tensor, Tensor

from official.vision.detection.layers import basic


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
    mask = 1 - (label == ignore_label)
    valid_label = label * mask

    score_shp = score.shape
    zero_mat = mge.zeros(
        F.concat([score_shp[0], score_shp[1], score_shp[2] + 1], axis=0),
        dtype=np.float32,
    )
    one_mat = mge.ones(
        F.concat([score_shp[0], score_shp[1], tensor(1)], axis=0), dtype=np.float32,
    )

    one_hot = basic.indexing_set_one_hot(
        zero_mat, 2, valid_label.astype(np.int32), one_mat
    )[:, :, 1:]
    pos_part = F.power(1 - score, gamma) * one_hot * F.log(score)
    neg_part = F.power(score, gamma) * (1 - one_hot) * F.log(1 - score)
    loss = -(alpha * pos_part + (1 - alpha) * neg_part).sum(axis=2) * mask

    if norm_type == "fg":
        positive_mask = label > background
        return loss.sum() / F.maximum(positive_mask.sum(), 1)
    elif norm_type == "none":
        return loss.sum()
    else:
        raise NotImplementedError


def get_smooth_l1_loss(
    pred_bbox: Tensor,
    gt_bbox: Tensor,
    label: Tensor,
    sigma: int = 3,
    background: int = 0,
    ignore_label: int = -1,
    fix_smooth_l1: bool = False,
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
        sigma (int):
            the parameter of smooth l1 loss. Default: 1
        background (int):
            the value of background class. Default: 0
        ignore_label (int):
            the value of ignore class. Default: -1
        fix_smooth_l1 (bool):
            is to use huber loss, default is False to use original smooth-l1
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

    valid_mask = 1 - (label == ignore_label)
    fg_mask = (1 - (label == background)) * valid_mask

    losses = get_smooth_l1_base(pred_bbox, gt_bbox, sigma, is_fix=fix_smooth_l1)
    if norm_type == "fg":
        loss = (losses.sum(axis=1) * fg_mask).sum() / F.maximum(fg_mask.sum(), 1)
    elif norm_type == "all":
        raise NotImplementedError
    else:
        raise NotImplementedError

    return loss


def get_smooth_l1_base(
    pred_bbox: Tensor, gt_bbox: Tensor, sigma: float, is_fix: bool = False,
):
    r"""

    Args:
        pred_bbox (Tensor):
            the predicted bbox with the shape of :math:`(N, 4)`
        gt_bbox (Tensor):
            the ground-truth bbox with the shape of :math:`(N, 4)`
        sigma (int):
            the parameter of smooth l1 loss.
        is_fix (bool):
            is to use huber loss, default is False to use original smooth-l1

    Returns:
        the calculated smooth l1 loss.
    """
    if is_fix:
        sigma = 1 / sigma
        cond_point = sigma
        x = pred_bbox - gt_bbox
        abs_x = F.abs(x)
        in_mask = abs_x < cond_point
        out_mask = 1 - in_mask
        in_loss = 0.5 * (x ** 2)
        out_loss = sigma * abs_x - 0.5 * (sigma ** 2)
        loss = in_loss * in_mask + out_loss * out_mask
    else:
        sigma2 = sigma ** 2
        cond_point = 1 / sigma2
        x = pred_bbox - gt_bbox
        abs_x = F.abs(x)
        in_mask = abs_x < cond_point
        out_mask = 1 - in_mask
        in_loss = 0.5 * (sigma * x) ** 2
        out_loss = abs_x - 0.5 / sigma2
        loss = in_loss * in_mask + out_loss * out_mask
    return loss
