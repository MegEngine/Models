# -*- coding:utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
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

        self.anchors_generator = layers.AnchorBoxGenerator(
            anchor_scales=cfg.anchor_scales,
            anchor_ratios=cfg.anchor_ratios,
            strides=cfg.rpn_stride,
        )

        match_thresh = [cfg.rpn_negative_overlap, cfg.rpn_positive_overlap]
        match_label = [0, cfg.ignore_label, 1]
        self.matcher = layers.Matcher(match_thresh, match_label, cfg.allow_low_quality)

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
        all_anchors_list = self.anchors_generator(features)

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
            pred_bbox_offset_list, pred_cls_logit_list, all_anchors_list, im_info
        )

        if self.training:
            rpn_labels, rpn_bbox_targets = self.get_ground_truth(
                boxes, im_info, all_anchors_list
            )
            pred_cls_logits, pred_bbox_offsets = self.merge_rpn_score_box(
                pred_cls_logit_list, pred_bbox_offset_list
            )

            # rpn classification loss
            valid_mask = rpn_labels != -1
            num_samples = valid_mask.sum()

            # loss_rpn_cls = layers.softmax_loss(pred_cls_logits, rpn_labels)
            loss_rpn_cls = layers.binary_cross_entropy_with_logits(
                pred_cls_logits[valid_mask], rpn_labels[valid_mask].astype("float32")
            ).sum() / F.maximum(num_samples, 1.0)

            # rpn regression loss
            fg_mask = rpn_labels > 0
            loss_rpn_loc = layers.smooth_l1_loss(
                pred_bbox_offsets[fg_mask],
                rpn_bbox_targets[fg_mask],
                self.cfg.rpn_smooth_l1_beta,
            ).sum() / F.maximum(num_samples, 1.0)

            loss_dict = {"loss_rpn_cls": loss_rpn_cls, "loss_rpn_loc": loss_rpn_loc}
            return rpn_rois, loss_dict
        else:
            return rpn_rois

    def find_top_rpn_proposals(
        self, rpn_bbox_offset_list, rpn_cls_score_list, all_anchors_list, im_info
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

        batch_per_gpu = self.cfg.batch_per_gpu if self.training else 1
        nms_threshold = self.cfg.rpn_nms_threshold

        list_size = len(rpn_bbox_offset_list)

        return_rois = []

        for bid in range(batch_per_gpu):
            batch_proposal_list = []
            batch_score_list = []
            batch_level_list = []
            for l in range(list_size):
                # get proposals and scores
                offsets = rpn_bbox_offset_list[l][bid].transpose(2, 3, 0, 1).reshape(-1, 4)
                all_anchors = all_anchors_list[l]
                proposals = self.box_coder.decode(all_anchors, offsets)

                scores = rpn_cls_score_list[l][bid].transpose(1, 2, 0).reshape(-1)
                scores.detach()
                # prev nms top n
                scores, order = F.topk(scores, descending=True, k=prev_nms_top_n)
                proposals = proposals[order, :]

                batch_proposal_list.append(proposals)
                batch_score_list.append(scores)
                batch_level_list.append(F.full_like(scores, l))

            # gather proposals, scores, level
            proposals = F.concat(batch_proposal_list, axis=0)
            scores = F.concat(batch_score_list, axis=0)
            levels = F.concat(batch_level_list, axis=0)

            proposals = layers.get_clipped_box(proposals, im_info[bid, :])
            # filter invalid proposals and apply total level nms
            keep_mask = layers.filter_boxes(proposals)
            _, keep_inds = F.cond_take(keep_mask == 1, keep_mask)
            proposals = proposals[keep_inds, :]
            scores = scores[keep_inds]
            levels = levels[keep_inds]
            nms_keep_inds = F.batched_nms(proposals, scores, levels, nms_threshold)
            nms_keep_inds = nms_keep_inds[:min(post_nms_top_n, nms_keep_inds.shape[0])]

            # generate rois to rcnn head, rois shape (N, 5), info [batch_id, x1, y1, x2, y2]
            rois = F.concat([proposals, scores.reshape(-1, 1)], axis=1)
            rois = rois[nms_keep_inds, :]
            batch_inds = F.full((rois.shape[0], 1), bid)
            batch_rois = F.concat([batch_inds, rois[:, :4]], axis=1)
            return_rois.append(batch_rois)

        return_rois = F.concat(return_rois, axis=0)
        return return_rois.detach()

    def merge_rpn_score_box(self, rpn_cls_score_list, rpn_bbox_offset_list):
        final_rpn_cls_score_list = []
        final_rpn_bbox_offset_list = []

        for bid in range(self.cfg.batch_per_gpu):
            batch_rpn_cls_score_list = []
            batch_rpn_bbox_offset_list = []

            for i in range(len(self.in_features)):
                rpn_cls_scores = rpn_cls_score_list[i][bid].transpose(1, 2, 0).reshape(-1)
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

    def _per_level_ground_truth(self, anchors, gt_boxes, im_info):
        # get the gt boxes
        valid_gt_boxes = gt_boxes[:im_info[4], :]
        # compute the iou matrix, (num_gt, num_anchors) shape
        iou_matrix = layers.get_iou(valid_gt_boxes[:, :4], anchors)
        matched_indices, labels = self.matcher(iou_matrix)
        # compute the targets
        bbox_targets = self.box_coder.encode(anchors, valid_gt_boxes[matched_indices, :4])
        return labels, bbox_targets

    def get_ground_truth(self, gt_boxes, im_info, all_anchors_list):
        gt_labels_list = []
        gt_bbox_targets_list = []

        for bid in range(self.cfg.batch_per_gpu):
            batch_labels_list = []
            batch_bbox_targets_list = []
            for anchors in all_anchors_list:
                rpn_labels_perlvl, rpn_bbox_targets_perlvl = self._per_level_ground_truth(
                    anchors, gt_boxes[bid], im_info[bid],
                )
                batch_labels_list.append(rpn_labels_perlvl)
                batch_bbox_targets_list.append(rpn_bbox_targets_perlvl)

            concated_batch_labels = F.concat(batch_labels_list, axis=0)
            concated_batch_bbox_targets = F.concat(batch_bbox_targets_list, axis=0)

            # sample labels
            num_positive = int(self.cfg.num_sample_anchors * self.cfg.positive_anchor_ratio)
            # sample positive labels
            concated_batch_labels = concated_batch_labels.detach()
            concated_batch_labels = layers.sample_labels(
                concated_batch_labels, num_positive, 1, self.cfg.ignore_label
            )
            # sample negative labels
            num_positive = (concated_batch_labels == 1).sum()
            num_negative = self.cfg.num_sample_anchors - num_positive
            concated_batch_labels = layers.sample_labels(
                concated_batch_labels, num_negative, 0, self.cfg.ignore_label
            )

            gt_labels_list.append(concated_batch_labels)
            gt_bbox_targets_list.append(concated_batch_bbox_targets)
        gt_labels = F.concat(gt_labels_list, axis=0)
        gt_bbox_targets = F.concat(gt_bbox_targets_list, axis=0)
        return gt_labels.detach(), gt_bbox_targets.detach()
