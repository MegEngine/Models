# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from abc import ABCMeta, abstractmethod

import numpy as np

import megengine.functional as F
from megengine import Tensor


class BoxCoderBase(metaclass=ABCMeta):
    """Boxcoder class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def encode(self) -> Tensor:
        pass

    @abstractmethod
    def decode(self) -> Tensor:
        pass


class BoxCoder(BoxCoderBase, metaclass=ABCMeta):
    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        reg_mean=[0.0, 0.0, 0.0, 0.0],
        reg_std=[1.0, 1.0, 1.0, 1.0],
    ):
        """
        Args:
            reg_mean(np.ndarray): [x0_mean, x1_mean, y0_mean, y1_mean] or None
            reg_std(np.ndarray):  [x0_std, x1_std, y0_std, y1_std] or None

        """
        self.reg_mean = np.array(reg_mean, dtype="float32")[None, :]
        self.reg_std = np.array(reg_std, dtype="float32")[None, :]
        super().__init__()

    @staticmethod
    def _box_ltrb_to_cs_opr(bbox, addaxis=None):
        """ transform the left-top right-bottom encoding bounding boxes
        to center and size encodings"""
        bbox_width = bbox[:, 2] - bbox[:, 0]
        bbox_height = bbox[:, 3] - bbox[:, 1]
        bbox_ctr_x = bbox[:, 0] + 0.5 * bbox_width
        bbox_ctr_y = bbox[:, 1] + 0.5 * bbox_height
        if addaxis is None:
            return bbox_width, bbox_height, bbox_ctr_x, bbox_ctr_y
        else:
            return (
                F.expand_dims(bbox_width, addaxis),
                F.expand_dims(bbox_height, addaxis),
                F.expand_dims(bbox_ctr_x, addaxis),
                F.expand_dims(bbox_ctr_y, addaxis),
            )

    def encode(self, bbox: Tensor, gt: Tensor) -> Tensor:
        bbox_width, bbox_height, bbox_ctr_x, bbox_ctr_y = self._box_ltrb_to_cs_opr(bbox)
        gt_width, gt_height, gt_ctr_x, gt_ctr_y = self._box_ltrb_to_cs_opr(gt)

        target_dx = (gt_ctr_x - bbox_ctr_x) / bbox_width
        target_dy = (gt_ctr_y - bbox_ctr_y) / bbox_height
        target_dw = F.log(gt_width / bbox_width)
        target_dh = F.log(gt_height / bbox_height)
        target = F.stack([target_dx, target_dy, target_dw, target_dh], axis=1)

        target -= self.reg_mean
        target /= self.reg_std
        return target

    def decode(self, anchors: Tensor, deltas: Tensor) -> Tensor:
        deltas *= self.reg_std
        deltas += self.reg_mean

        (
            anchor_width,
            anchor_height,
            anchor_ctr_x,
            anchor_ctr_y,
        ) = self._box_ltrb_to_cs_opr(anchors, 1)
        pred_ctr_x = anchor_ctr_x + deltas[:, 0::4] * anchor_width
        pred_ctr_y = anchor_ctr_y + deltas[:, 1::4] * anchor_height
        pred_width = anchor_width * F.exp(deltas[:, 2::4])
        pred_height = anchor_height * F.exp(deltas[:, 3::4])

        pred_x1 = pred_ctr_x - 0.5 * pred_width
        pred_y1 = pred_ctr_y - 0.5 * pred_height
        pred_x2 = pred_ctr_x + 0.5 * pred_width
        pred_y2 = pred_ctr_y + 0.5 * pred_height

        pred_box = F.stack([pred_x1, pred_y1, pred_x2, pred_y2], axis=2)
        pred_box = pred_box.reshape(pred_box.shape[0], -1)

        return pred_box


class PointCoder(BoxCoderBase, metaclass=ABCMeta):
    def encode(self, point: Tensor, gt: Tensor) -> Tensor:
        return F.concat([point - gt[..., :2], gt[..., 2:] - point], axis=-1)

    def decode(self, anchors: Tensor, deltas: Tensor) -> Tensor:
        return F.stack([
            F.expand_dims(anchors[:, 0], axis=1) - deltas[:, 0::4],
            F.expand_dims(anchors[:, 1], axis=1) - deltas[:, 1::4],
            F.expand_dims(anchors[:, 0], axis=1) + deltas[:, 2::4],
            F.expand_dims(anchors[:, 1], axis=1) + deltas[:, 3::4],
        ], axis=2).reshape(deltas.shape)


def get_iou(boxes1: Tensor, boxes2: Tensor, return_ioa=False) -> Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1 (Tensor): boxes tensor with shape (N, 4)
        boxes2 (Tensor): boxes tensor with shape (M, 4)
        return_ioa (Bool): wheather return Intersection over Boxes1 or not, default: False

    Returns:
        iou (Tensor): IoU matrix, shape (N,M).
    """
    b_box1 = F.expand_dims(boxes1, axis=1)
    b_box2 = F.expand_dims(boxes2, axis=0)

    iw = F.minimum(b_box1[:, :, 2], b_box2[:, :, 2]) - F.maximum(
        b_box1[:, :, 0], b_box2[:, :, 0]
    )
    ih = F.minimum(b_box1[:, :, 3], b_box2[:, :, 3]) - F.maximum(
        b_box1[:, :, 1], b_box2[:, :, 1]
    )
    inter = F.maximum(iw, 0) * F.maximum(ih, 0)

    area_box1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_box2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = F.expand_dims(area_box1, axis=1) + F.expand_dims(area_box2, axis=0) - inter
    overlaps = F.maximum(inter / union, 0)

    if return_ioa:
        ioa = F.maximum(inter / area_box1, 0)
        return overlaps, ioa

    return overlaps


def get_clipped_boxes(boxes, hw):
    """ Clip the boxes into the image region."""
    # x1 >=0
    box_x1 = F.clip(boxes[:, 0::4], lower=0, upper=hw[1])
    # y1 >=0
    box_y1 = F.clip(boxes[:, 1::4], lower=0, upper=hw[0])
    # x2 < im_info[1]
    box_x2 = F.clip(boxes[:, 2::4], lower=0, upper=hw[1])
    # y2 < im_info[0]
    box_y2 = F.clip(boxes[:, 3::4], lower=0, upper=hw[0])

    clip_box = F.concat([box_x1, box_y1, box_x2, box_y2], axis=1)

    return clip_box


def filter_boxes(boxes, size=0):
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    keep = (width > size) & (height > size)
    return keep
