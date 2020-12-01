# -*- coding:utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine.functional as F
import megengine.module as M

from official.vision.detection import layers


class RCNN(M.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.box_coder = layers.BoxCoder(cfg.rcnn_reg_mean, cfg.rcnn_reg_std)

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
        self.pred_delta = M.Linear(1024, cfg.num_classes * 4)
        M.init.normal_(self.pred_cls.weight, std=0.01)
        M.init.normal_(self.pred_delta.weight, std=0.001)
        for l in [self.pred_cls, self.pred_delta]:
            M.init.fill_(l.bias, 0)

    def forward(self, fpn_fms, rcnn_rois, im_info=None, gt_boxes=None):
        rcnn_rois, labels, bbox_targets = self.get_ground_truth(
            rcnn_rois, im_info, gt_boxes
        )

        fpn_fms = [fpn_fms[x] for x in self.in_features]
        pool_features = layers.roi_pool(
            fpn_fms, rcnn_rois, self.stride, self.pooling_size, self.pooling_method,
        )
        flatten_feature = F.flatten(pool_features, start_axis=1)
        roi_feature = F.relu(self.fc1(flatten_feature))
        roi_feature = F.relu(self.fc2(roi_feature))
        pred_logits = self.pred_cls(roi_feature)
        pred_offsets = self.pred_delta(roi_feature)

        if self.training:
            # loss for rcnn classification
            loss_rcnn_cls = F.loss.cross_entropy(pred_logits, labels, axis=1)
            # loss for rcnn regression
            pred_offsets = pred_offsets.reshape(-1, self.cfg.num_classes, 4)
            num_samples = labels.shape[0]
            fg_mask = labels > 0
            loss_rcnn_bbox = layers.smooth_l1_loss(
                pred_offsets[fg_mask, labels[fg_mask] - 1],
                bbox_targets[fg_mask],
                self.cfg.rcnn_smooth_l1_beta,
            ).sum() / F.maximum(num_samples, 1)

            loss_dict = {
                "loss_rcnn_cls": loss_rcnn_cls,
                "loss_rcnn_bbox": loss_rcnn_bbox,
            }
            return loss_dict
        else:
            # slice 1 for removing background
            pred_scores = F.softmax(pred_logits, axis=1)[:, 1:]
            pred_offsets = pred_offsets.reshape(-1, 4)
            target_shape = (rcnn_rois.shape[0], self.cfg.num_classes, 4)
            # rois (N, 4) -> (N, 1, 4) -> (N, 80, 4) -> (N * 80, 4)
            base_rois = F.broadcast_to(
                F.expand_dims(rcnn_rois[:, 1:5], axis=1), target_shape).reshape(-1, 4)
            pred_bbox = self.box_coder.decode(base_rois, pred_offsets)
            return pred_bbox, pred_scores

    def get_ground_truth(self, rpn_rois, im_info, gt_boxes):
        if not self.training:
            return rpn_rois, None, None

        return_rois = []
        return_labels = []
        return_bbox_targets = []

        # get per image proposals and gt_boxes
        for bid in range(gt_boxes.shape[0]):
            num_valid_boxes = im_info[bid, 4].astype("int32")
            gt_boxes_per_img = gt_boxes[bid, :num_valid_boxes, :]
            batch_inds = F.full((gt_boxes_per_img.shape[0], 1), bid)
            gt_rois = F.concat([batch_inds, gt_boxes_per_img[:, :4]], axis=1)
            batch_roi_mask = rpn_rois[:, 0] == bid
            # all_rois : [batch_id, x1, y1, x2, y2]
            all_rois = F.concat([rpn_rois[batch_roi_mask], gt_rois])

            overlaps = layers.get_iou(all_rois[:, 1:], gt_boxes_per_img)

            max_overlaps = overlaps.max(axis=1)
            gt_assignment = F.argmax(overlaps, axis=1).astype("int32")
            labels = gt_boxes_per_img[gt_assignment, 4]

            # ---------------- get the fg/bg labels for each roi ---------------#
            fg_mask = (max_overlaps >= self.cfg.fg_threshold) & (labels >= 0)
            bg_mask = (
                (max_overlaps >= self.cfg.bg_threshold_low)
                & (max_overlaps < self.cfg.bg_threshold_high)
            )

            num_fg_rois = int(self.cfg.num_rois * self.cfg.fg_ratio)
            fg_inds_mask = layers.sample_labels(fg_mask, num_fg_rois, True, False)
            num_bg_rois = int(self.cfg.num_rois - fg_inds_mask.sum())
            bg_inds_mask = layers.sample_labels(bg_mask, num_bg_rois, True, False)

            labels[bg_inds_mask] = 0

            keep_mask = fg_inds_mask | bg_inds_mask
            labels = labels[keep_mask].astype("int32")
            rois = all_rois[keep_mask]
            target_boxes = gt_boxes_per_img[gt_assignment[keep_mask], :4]
            bbox_targets = self.box_coder.encode(rois[:, 1:], target_boxes)
            bbox_targets = bbox_targets.reshape(-1, 4)

            return_rois.append(rois)
            return_labels.append(labels)
            return_bbox_targets.append(bbox_targets)

        return (
            F.concat(return_rois, axis=0).detach(),
            F.concat(return_labels, axis=0).detach(),
            F.concat(return_bbox_targets, axis=0).detach(),
        )
