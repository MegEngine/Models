# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
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
        self.reg_mean = np.array(reg_mean, dtype=np.float32)[None, :]
        self.reg_std = np.array(reg_std, dtype=np.float32)[None, :]
        super().__init__()

    @staticmethod
    def _concat_new_axis(t1, t2, t3, t4, axis=1):
        return F.concat(
            [
                F.add_axis(t1, -1),
                F.add_axis(t2, -1),
                F.add_axis(t3, -1),
                F.add_axis(t4, -1),
            ],
            axis=axis,
        )

    @staticmethod
    def _box_ltrb_to_cs_opr(bbox, addaxis=None):
        """ transform the left-top right-bottom encoding bounding boxes
        to center and size encodings"""
        # FIXME
        bbox_width = F.remove_axis(bbox[:, 2] - bbox[:, 0], 1)
        bbox_height = F.remove_axis(bbox[:, 3] - bbox[:, 1], 1)
        bbox_ctr_x = F.remove_axis(bbox[:, 0], 1) + 0.5 * bbox_width
        bbox_ctr_y = F.remove_axis(bbox[:, 1], 1) + 0.5 * bbox_height
        if addaxis is None:
            return bbox_width, bbox_height, bbox_ctr_x, bbox_ctr_y
        else:
            return (
                F.add_axis(bbox_width, addaxis),
                F.add_axis(bbox_height, addaxis),
                F.add_axis(bbox_ctr_x, addaxis),
                F.add_axis(bbox_ctr_y, addaxis),
            )

    def encode(self, bbox: Tensor, gt: Tensor) -> Tensor:
        (bbox_width, bbox_height, bbox_ctr_x, bbox_ctr_y,) = self._box_ltrb_to_cs_opr(
            bbox
        )
        gt_width, gt_height, gt_ctr_x, gt_ctr_y = self._box_ltrb_to_cs_opr(gt)

        target_dx = (gt_ctr_x - bbox_ctr_x) / bbox_width
        target_dy = (gt_ctr_y - bbox_ctr_y) / bbox_height
        target_dw = F.log(gt_width / bbox_width)
        target_dh = F.log(gt_height / bbox_height)
        target = self._concat_new_axis(target_dx, target_dy, target_dw, target_dh)

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

        pred_box = self._concat_new_axis(pred_x1, pred_y1, pred_x2, pred_y2, 2)
        pred_box = pred_box.reshape(pred_box.shape[0], -1)

        return pred_box


def get_iou(boxes1: Tensor, boxes2: Tensor, return_ignore=False) -> Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    box = boxes1
    gt = boxes2
    target_shape = (boxes1.shape[0], boxes2.shape[0], 4)

    b_box = F.add_axis(boxes1, 1).broadcast(*target_shape)
    b_gt = F.add_axis(boxes2[:, :4], 0).broadcast(*target_shape)

    iw = F.minimum(b_box[:, :, 2], b_gt[:, :, 2]) - F.maximum(
        b_box[:, :, 0], b_gt[:, :, 0]
    )
    ih = F.minimum(b_box[:, :, 3], b_gt[:, :, 3]) - F.maximum(
        b_box[:, :, 1], b_gt[:, :, 1]
    )
    inter = F.remove_axis(F.maximum(iw, 0) * F.maximum(ih, 0), 2)  # FIXME

    area_box = F.remove_axis((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]), 1)  # FIXME
    area_gt = F.remove_axis((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]), 1)  # FIXME

    area_target_shape = (box.shape[0], gt.shape[0])

    b_area_box = F.add_axis(area_box, 1).broadcast(*area_target_shape)
    b_area_gt = F.add_axis(area_gt, 0).broadcast(*area_target_shape)

    union = b_area_box + b_area_gt - inter
    overlaps = F.maximum(inter / union, 0)

    if return_ignore:
        overlaps_ignore = F.maximum(inter / b_area_box, 0)
        gt_ignore_mask = F.add_axis((gt[:, 4] == -1), 0).broadcast(*area_target_shape)
        overlaps *= 1 - gt_ignore_mask
        overlaps_ignore *= gt_ignore_mask
        return overlaps, overlaps_ignore

    return overlaps


def get_clipped_box(boxes, hw):
    """ Clip the boxes into the image region."""
    # x1 >=0
    box_x1 = F.clamp(boxes[:, 0::4], lower=0, upper=hw[1])
    # y1 >=0
    box_y1 = F.clamp(boxes[:, 1::4], lower=0, upper=hw[0])
    # x2 < im_info[1]
    box_x2 = F.clamp(boxes[:, 2::4], lower=0, upper=hw[1])
    # y2 < im_info[0]
    box_y2 = F.clamp(boxes[:, 3::4], lower=0, upper=hw[0])

    clip_box = F.concat([box_x1, box_y1, box_x2, box_y2], axis=1)

    return clip_box


def filter_boxes(boxes, size=0):
    # FIXME
    width = F.remove_axis(boxes[:, 2] - boxes[:, 0], 1)
    height = F.remove_axis(boxes[:, 3] - boxes[:, 1], 1)
    keep = (width > size) * (height > size)
    return keep
