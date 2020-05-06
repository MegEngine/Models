#!/usr/bin/python3
# -*- coding:utf-8 -*-

import megengine as mge
import megengine.functional as F
import numpy as np
from .box_utils import get_iou


def match_anchors_and_boxes(
    anchors, gt_boxes, box_coder,
    neg_thresh, pos_thresh, allow_low_quality=True
):
    total_anchors = anchors.shape[0]
    overlaps = get_iou(anchors, gt_boxes[:, :4])
    argmax_overlaps = F.argmax(overlaps, axis=1)

    max_overlaps = overlaps.ai[
        F.linspace(0, total_anchors - 1, total_anchors).astype(np.int32),
        argmax_overlaps,
    ]

    labels = mge.tensor([-1]).broadcast(total_anchors)
    labels = labels * (max_overlaps >= neg_thresh)
    labels = labels * (max_overlaps < pos_thresh) + (max_overlaps >= pos_thresh)

    bbox_targets = box_coder.encode(
        anchors, gt_boxes.ai[argmax_overlaps, :4]
    )

    if allow_low_quality:
        gt_argmax_overlaps = F.argmax(overlaps, axis=0)
        labels = labels.set_ai(gt_boxes[:, 4])[gt_argmax_overlaps]
        matched_low_bbox_targets = box_coder.encode(
            anchors.ai[gt_argmax_overlaps, :], gt_boxes[:, :4]
        )
        bbox_targets = bbox_targets.set_ai(matched_low_bbox_targets)[
            gt_argmax_overlaps, :
        ]

    return labels, bbox_targets


def match_proposals_and_boxes(proposals, gt_boxes, match_thresh):
    total_proposals = proposals.shape[0]
    overlaps = get_iou(proposals, gt_boxes[:, :4])
    argmax_overlaps = F.argmax(overlaps, axis=1)

    max_overlaps = overlaps.ai[
        F.linspace(0, total_proposals - 1, total_proposals).astype(np.int32),
        argmax_overlaps,
    ]

    labels = mge.zeros(total_proposals)
    labels = labels + (max_overlaps >= match_thresh)
    return argmax_overlaps, labels
