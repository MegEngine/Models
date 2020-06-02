# -*- coding:utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine as mge
import megengine.functional as F
import megengine.module as M

from official.vision.detection import layers


class RCNN(M.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.box_coder = layers.BoxCoder(
            reg_mean=cfg.bbox_normalize_means,
            reg_std=cfg.bbox_normalize_stds
        )

        # roi head
        self.in_features = cfg.rcnn_in_features
        self.stride = cfg.rcnn_stride
        self.pooling_method = cfg.pooling_method
        self.pooling_size = cfg.pooling_size

        self.fc1 = M.Linear(256 * self.pooling_size[0] * self.pooling_size[1], 1024)
        self.fc2 = M.Linear(1024, 1024)
        for l in [self.fc1, self.fc2]:
            M.init.normal_(l.weight, std=0.01)
            M.init.fill_(l.bias, 0)

        # box predictor
        self.pred_cls = M.Linear(1024, cfg.num_classes + 1)
        self.pred_delta = M.Linear(1024, (cfg.num_classes + 1) * 4)
        M.init.normal_(self.pred_cls.weight, std=0.01)
        M.init.normal_(self.pred_delta.weight, std=0.001)
        for l in [self.pred_cls, self.pred_delta]:
            M.init.fill_(l.bias, 0)

    def forward(self, fpn_fms, rcnn_rois, im_info=None, gt_boxes=None):
        rcnn_rois, labels, bbox_targets = self.get_ground_truth(rcnn_rois, im_info, gt_boxes)

        fpn_fms = [fpn_fms[x] for x in self.in_features]
        pool_features = layers.roi_pool(
            fpn_fms, rcnn_rois, self.stride,
            self.pooling_size, self.pooling_method,
        )
        flatten_feature = F.flatten(pool_features, start_axis=1)
        roi_feature = F.relu(self.fc1(flatten_feature))
        roi_feature = F.relu(self.fc2(roi_feature))
        pred_cls = self.pred_cls(roi_feature)
        pred_delta = self.pred_delta(roi_feature)

        if self.training:
            # loss for classification
            loss_rcnn_cls = layers.softmax_loss(pred_cls, labels)
            # loss for regression
            pred_delta = pred_delta.reshape(-1, self.cfg.num_classes + 1, 4)
            loss_rcnn_loc = layers.smooth_l1_loss_rcnn(
                pred_delta, bbox_targets, labels, self.cfg.rcnn_smooth_l1_beta
            )
            loss_dict = {
                'loss_rcnn_cls': loss_rcnn_cls,
                'loss_rcnn_loc': loss_rcnn_loc
            }
            return loss_dict
        else:
            # slice 1 for removing background
            pred_scores = F.softmax(pred_cls, axis=1)[:, 1:]
            pred_delta = pred_delta[:, 4:].reshape(-1, 4)
            target_shape = (rcnn_rois.shapeof(0), self.cfg.num_classes, 4)
            # rois (N, 4) -> (N, 1, 4) -> (N, 80, 4) -> (N * 80, 4)
            base_rois = F.add_axis(rcnn_rois[:, 1:5], 1).broadcast(target_shape).reshape(-1, 4)
            pred_bbox = self.box_coder.decode(base_rois, pred_delta)
            return pred_bbox, pred_scores

    def get_ground_truth(self, rpn_rois, im_info, gt_boxes):
        if not self.training:
            return rpn_rois, None, None

        return_rois = []
        return_labels = []
        return_bbox_targets = []

        # get per image proposals and gt_boxes
        for bid in range(self.cfg.batch_per_gpu):
            num_valid_boxes = im_info[bid, 4]
            gt_boxes_per_img = gt_boxes[bid, :num_valid_boxes, :]
            batch_inds = mge.ones((gt_boxes_per_img.shapeof(0), 1)) * bid
            # if config.proposal_append_gt:
            gt_rois = F.concat([batch_inds, gt_boxes_per_img[:, :4]], axis=1)
            batch_roi_mask = (rpn_rois[:, 0] == bid)
            _, batch_roi_inds = F.cond_take(batch_roi_mask == 1, batch_roi_mask)
            # all_rois : [batch_id, x1, y1, x2, y2]
            all_rois = F.concat([rpn_rois.ai[batch_roi_inds], gt_rois])

            overlaps_normal, overlaps_ignore = layers.get_iou_with_ignore(
                all_rois[:, 1:5], gt_boxes_per_img
            )

            max_overlaps_normal = overlaps_normal.max(axis=1)
            gt_assignment_normal = F.argmax(overlaps_normal, axis=1)

            max_overlaps_ignore = overlaps_ignore.max(axis=1)
            gt_assignment_ignore = F.argmax(overlaps_ignore, axis=1)

            ignore_assign_mask = (max_overlaps_normal < self.cfg.fg_threshold) * (
                max_overlaps_ignore > max_overlaps_normal)
            max_overlaps = (
                max_overlaps_normal * (1 - ignore_assign_mask) +
                max_overlaps_ignore * ignore_assign_mask
            )
            gt_assignment = (
                gt_assignment_normal * (1 - ignore_assign_mask) +
                gt_assignment_ignore * ignore_assign_mask
            )
            gt_assignment = gt_assignment.astype("int32")
            labels = gt_boxes_per_img.ai[gt_assignment, 4]

            # ---------------- get the fg/bg labels for each roi ---------------#
            fg_mask = (max_overlaps >= self.cfg.fg_threshold) * (labels != self.cfg.ignore_label)
            bg_mask = (max_overlaps < self.cfg.bg_threshold_high) * (
                max_overlaps >= self.cfg.bg_threshold_low)

            num_fg_rois = self.cfg.num_rois * self.cfg.fg_ratio

            fg_inds_mask = self._bernoulli_sample_masks(fg_mask, num_fg_rois, 1)
            num_bg_rois = self.cfg.num_rois - fg_inds_mask.sum()
            bg_inds_mask = self._bernoulli_sample_masks(bg_mask, num_bg_rois, 1)

            labels = labels * fg_inds_mask

            keep_mask = fg_inds_mask + bg_inds_mask
            _, keep_inds = F.cond_take(keep_mask == 1, keep_mask)
            # Add next line to avoid memory exceed
            keep_inds = keep_inds[:F.minimum(self.cfg.num_rois, keep_inds.shapeof(0))]
            # labels
            labels = labels.ai[keep_inds].astype("int32")
            rois = all_rois.ai[keep_inds]
            target_boxes = gt_boxes_per_img.ai[gt_assignment.ai[keep_inds], :4]
            bbox_targets = self.box_coder.encode(rois[:, 1:5], target_boxes)
            bbox_targets = bbox_targets.reshape(-1, 4)

            return_rois.append(rois)
            return_labels.append(labels)
            return_bbox_targets.append(bbox_targets)

        if self.cfg.batch_per_gpu == 1:
            return F.zero_grad(rois), F.zero_grad(labels), F.zero_grad(bbox_targets)
        else:
            return_rois = F.concat(return_rois, axis=0)
            return_labels = F.concat(return_labels, axis=0)
            return_bbox_targets = F.concat(return_bbox_targets, axis=0)
            return (
                F.zero_grad(return_rois),
                F.zero_grad(return_labels),
                F.zero_grad(return_bbox_targets)
            )

    def _bernoulli_sample_masks(self, masks, num_samples, sample_value):
        """ Using the bernoulli sampling method"""
        sample_mask = (masks == sample_value)
        num_mask = sample_mask.sum()
        num_final_samples = F.minimum(num_mask, num_samples)
        # here, we use the bernoulli probability to sample the anchors
        sample_prob = num_final_samples / num_mask
        uniform_rng = mge.random.uniform(sample_mask.shapeof(0))
        after_sampled_mask = (uniform_rng <= sample_prob) * sample_mask
        return after_sampled_mask
