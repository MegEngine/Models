#!/usr/bin/python3
# -*- coding:utf-8 -*-

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
    eps = 1e-5
    score = F.clamp(score, eps, 1-eps)
    pos_part = (1 - score) ** gamma * F.log(score)
    neg_part = score ** gamma * F.log(1 - score)

    pos_loss = -(label == class_range) * pos_part * alpha
    neg_loss = -(label != class_range) * (label != ignore_label) * neg_part * (1 - alpha)
    loss = pos_loss + neg_loss

    if norm_type == "fg":
        fg_mask = (label != background) * (label != ignore_label)
        return loss.sum() / F.maximum(fg_mask.sum(), 1)
    elif norm_type == "none":
        return loss.sum()
    else:
        raise NotImplementedError
