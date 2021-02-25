# -*- coding:utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine.functional as F
import megengine.module as M

from official.vision.detection import layers


class RPN(M.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.box_coder = layers.BoxCoder(cfg.rpn_reg_mean, cfg.rpn_reg_std)

        # check anchor settings
        assert len(set(len(x) for x in cfg.anchor_scales)) == 1
        assert len(set(len(x) for x in cfg.anchor_ratios)) == 1
        self.num_cell_anchors = len(cfg.anchor_scales[0]) * len(cfg.anchor_ratios[0])

        self.stride_list = np.array(cfg.rpn_stride).astype(np.float32)
        rpn_channel = cfg.rpn_channel
        self.in_features = cfg.rpn_in_features

        self.anchor_generator = layers.AnchorBoxGenerator(
            anchor_scales=cfg.anchor_scales,
            anchor_ratios=cfg.anchor_ratios,
            strides=cfg.rpn_stride,
            offset=self.cfg.anchor_offset,
        )

        self.matcher = layers.Matcher(
            cfg.match_thresholds, cfg.match_labels, cfg.match_allow_low_quality
        )

        self.rpn_conv = M.Conv2d(256, rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = M.Conv2d(
            rpn_channel, self.num_cell_anchors, kernel_size=1, stride=1
        )
        self.rpn_bbox_offsets = M.Conv2d(
            rpn_channel, self.num_cell_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_offsets]:
            M.init.normal_(l.weight, std=0.01)
            M.init.fill_(l.bias, 0)

    def forward(self, features, im_info, boxes=None):
        # prediction
        features = [features[x] for x in self.in_features]

        # get anchors
        anchors_list = self.anchor_generator(features)

        pred_cls_logit_list = []
        pred_bbox_offset_list = []
        for x in features:
            t = F.relu(self.rpn_conv(x))
            scores = self.rpn_cls_score(t)
            pred_cls_logit_list.append(
                scores.reshape(
                    scores.shape[0],
                    self.num_cell_anchors,
                    scores.shape[2],
                    scores.shape[3],
                )
            )
            bbox_offsets = self.rpn_bbox_offsets(t)
            pred_bbox_offset_list.append(
                bbox_offsets.reshape(
                    bbox_offsets.shape[0],
                    self.num_cell_anchors,
                    4,
                    bbox_offsets.shape[2],
                    bbox_offsets.shape[3],
                )
            )
        # get rois from the predictions
        rpn_rois = self.find_top_rpn_proposals(
            pred_cls_logit_list, pred_bbox_offset_list, anchors_list, im_info
        )

        if self.training:
            rpn_labels, rpn_offsets = self.get_ground_truth(
                anchors_list, boxes, im_info[:, 4].astype(np.int32)
            )
            pred_cls_logits, pred_bbox_offsets = self.merge_rpn_score_box(
                pred_cls_logit_list, pred_bbox_offset_list
            )

            fg_mask = rpn_labels > 0
            valid_mask = rpn_labels >= 0
            num_valid = valid_mask.sum()

            # rpn classification loss
            loss_rpn_cls = F.loss.binary_cross_entropy(
                pred_cls_logits[valid_mask], rpn_labels[valid_mask]
            )

            # rpn regression loss
            loss_rpn_bbox = layers.smooth_l1_loss(
                pred_bbox_offsets[fg_mask],
                rpn_offsets[fg_mask],
                self.cfg.rpn_smooth_l1_beta,
            ).sum() / F.maximum(num_valid, 1)

            loss_dict = {"loss_rpn_cls": loss_rpn_cls, "loss_rpn_bbox": loss_rpn_bbox}
            return rpn_rois, loss_dict
        else:
            return rpn_rois

    def find_top_rpn_proposals(
        self, rpn_cls_score_list, rpn_bbox_offset_list, anchors_list, im_info
    ):
        prev_nms_top_n = (
            self.cfg.train_prev_nms_top_n
            if self.training
            else self.cfg.test_prev_nms_top_n
        )
        post_nms_top_n = (
            self.cfg.train_post_nms_top_n
            if self.training
            else self.cfg.test_post_nms_top_n
        )

        return_rois = []

        for bid in range(im_info.shape[0]):
            batch_proposal_list = []
            batch_score_list = []
            batch_level_list = []
            for l, (rpn_cls_score, rpn_bbox_offset, anchors) in enumerate(
                zip(rpn_cls_score_list, rpn_bbox_offset_list, anchors_list)
            ):
                # get proposals and scores
                offsets = rpn_bbox_offset[bid].transpose(2, 3, 0, 1).reshape(-1, 4)
                proposals = self.box_coder.decode(anchors, offsets)

                scores = rpn_cls_score[bid].transpose(1, 2, 0).flatten()
                scores.detach()
                # prev nms top n
                scores, order = F.topk(scores, descending=True, k=prev_nms_top_n)
                proposals = proposals[order]

                batch_proposal_list.append(proposals)
                batch_score_list.append(scores)
                batch_level_list.append(F.full_like(scores, l))

            # gather proposals, scores, level
            proposals = F.concat(batch_proposal_list, axis=0)
            scores = F.concat(batch_score_list, axis=0)
            levels = F.concat(batch_level_list, axis=0)

            proposals = layers.get_clipped_boxes(proposals, im_info[bid])
            # filter invalid proposals and apply total level nms
            keep_mask = layers.filter_boxes(proposals)
            proposals = proposals[keep_mask]
            scores = scores[keep_mask]
            levels = levels[keep_mask]
            nms_keep_inds = layers.batched_nms(
                proposals, scores, levels, self.cfg.rpn_nms_threshold, post_nms_top_n
            )

            # generate rois to rcnn head, rois shape (N, 5), info [batch_id, x1, y1, x2, y2]
            rois = F.concat([proposals, scores.reshape(-1, 1)], axis=1)
            rois = rois[nms_keep_inds]
            batch_inds = F.full((rois.shape[0], 1), bid)
            batch_rois = F.concat([batch_inds, rois[:, :4]], axis=1)
            return_rois.append(batch_rois)

        return_rois = F.concat(return_rois, axis=0)
        return return_rois.detach()

    def merge_rpn_score_box(self, rpn_cls_score_list, rpn_bbox_offset_list):
        final_rpn_cls_score_list = []
        final_rpn_bbox_offset_list = []

        for bid in range(rpn_cls_score_list[0].shape[0]):
            batch_rpn_cls_score_list = []
            batch_rpn_bbox_offset_list = []

            for i in range(len(self.in_features)):
                rpn_cls_scores = rpn_cls_score_list[i][bid].transpose(1, 2, 0).flatten()
                rpn_bbox_offsets = (
                    rpn_bbox_offset_list[i][bid].transpose(2, 3, 0, 1).reshape(-1, 4)
                )

                batch_rpn_cls_score_list.append(rpn_cls_scores)
                batch_rpn_bbox_offset_list.append(rpn_bbox_offsets)

            batch_rpn_cls_scores = F.concat(batch_rpn_cls_score_list, axis=0)
            batch_rpn_bbox_offsets = F.concat(batch_rpn_bbox_offset_list, axis=0)

            final_rpn_cls_score_list.append(batch_rpn_cls_scores)
            final_rpn_bbox_offset_list.append(batch_rpn_bbox_offsets)

        final_rpn_cls_scores = F.concat(final_rpn_cls_score_list, axis=0)
        final_rpn_bbox_offsets = F.concat(final_rpn_bbox_offset_list, axis=0)
        return final_rpn_cls_scores, final_rpn_bbox_offsets

    def get_ground_truth(self, anchors_list, batched_gt_boxes, batched_num_gts):
        anchors = F.concat(anchors_list, axis=0)
        labels_list = []
        offsets_list = []

        for bid in range(batched_gt_boxes.shape[0]):
            gt_boxes = batched_gt_boxes[bid, :batched_num_gts[bid]]

            overlaps = layers.get_iou(gt_boxes[:, :4], anchors)
            matched_indices, labels = self.matcher(overlaps)

            offsets = self.box_coder.encode(anchors, gt_boxes[matched_indices, :4])

            # sample positive labels
            num_positive = int(self.cfg.num_sample_anchors * self.cfg.positive_anchor_ratio)
            labels = layers.sample_labels(labels, num_positive, 1, -1)
            # sample negative labels
            num_positive = (labels == 1).sum().astype(np.int32)
            num_negative = self.cfg.num_sample_anchors - num_positive
            labels = layers.sample_labels(labels, num_negative, 0, -1)

            labels_list.append(labels)
            offsets_list.append(offsets)

        return (
            F.concat(labels_list, axis=0).detach(),
            F.concat(offsets_list, axis=0).detach(),
        )
