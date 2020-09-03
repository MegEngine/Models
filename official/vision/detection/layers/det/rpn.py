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
import megengine.random as rand

from official.vision.detection import layers


class RPN(M.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.box_coder = layers.BoxCoder(cfg.rpn_reg_mean, cfg.rpn_reg_std)
        self.num_cell_anchors = len(cfg.anchor_scales) * len(cfg.anchor_ratios)

        self.stride_list = np.array(cfg.rpn_stride).astype(np.float32)
        rpn_channel = cfg.rpn_channel
        self.in_features = cfg.rpn_in_features

        anchor_scales = [[x] for x in [32, 64, 128, 256, 512]]
        anchor_ratios = [[0.5, 1, 2]]
        stride = [4, 8, 16, 32, 64]
        self.anchors_generator = layers.DefaultAnchorGenerator(
            anchor_scales=anchor_scales,
            anchor_ratios=anchor_ratios,
            strides=stride,
        )

        self.rpn_conv = M.Conv2d(256, rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = M.Conv2d(
            rpn_channel, self.num_cell_anchors * 2, kernel_size=1, stride=1
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
                    2,
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
        # sample from the predictions
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
            loss_rpn_cls = layers.softmax_loss(pred_cls_logits, rpn_labels)
            # rpn regression loss
            fg_mask = (rpn_labels > 0).astype("float32")
            num_samples = (rpn_labels != -1).astype("int32").sum()
            loss_rpn_loc = (layers.smooth_l1_loss(
                pred_bbox_offsets,
                rpn_bbox_targets,
                self.cfg.rpn_smooth_l1_beta,
            ).sum(axis=1) * fg_mask).sum() / F.maximum(num_samples, 1.0)

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

                scores = rpn_cls_score_list[l][bid, 1].transpose(1, 2, 0).reshape(-1)
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
            keep_inds = F.batched_nms(proposals, scores, levels, nms_threshold)
            keep_inds = keep_inds[:post_nms_top_n]

            # generate rois to rcnn head, rois shape (N, 5), info [batch_id, x1, y1, x2, y2]
            rois = F.concat([proposals, scores.reshape(-1, 1)], axis=1)
            rois = rois[keep_inds]
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
                rpn_cls_scores = rpn_cls_score_list[i][bid].transpose(2, 3, 1, 0).reshape(-1, 2)
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

    def per_level_gt(self, gt_boxes, im_info, anchors, allow_low_quality_matches=True):
        # TODO: use Matcher instead
        ignore_label = self.cfg.ignore_label
        # get the gt boxes
        valid_gt_boxes = gt_boxes[:im_info[4], :]
        # compute the iou matrix
        overlaps = layers.get_iou(anchors, valid_gt_boxes[:, :4])
        # match the dtboxes
        max_overlaps = F.max(overlaps, axis=1)
        argmax_overlaps = F.argmax(overlaps, axis=1)
        # all ignore
        labels = F.full((anchors.shape[0],), ignore_label).astype("int32")
        # set negative ones
        labels = labels * (max_overlaps >= self.cfg.rpn_negative_overlap).astype("int32")
        # set positive ones
        # FIXME: bool astype
        fg_mask = (max_overlaps >= self.cfg.rpn_positive_overlap).astype("int32")
        if allow_low_quality_matches:
            # make sure that max iou of gt matched
            gt_argmax_overlaps = F.argmax(overlaps, axis=0)
            num_valid_boxes = valid_gt_boxes.shape[0]
            gt_id = F.linspace(
                0, num_valid_boxes - 1, num_valid_boxes, device=argmax_overlaps.device
            ).astype("int32")
            argmax_overlaps[gt_argmax_overlaps] = gt_id
            max_overlaps[gt_argmax_overlaps] = F.ones((num_valid_boxes,))
            # FIXME：bool astype
            fg_mask = (max_overlaps >= self.cfg.rpn_positive_overlap).astype("int32")

        # set positive ones
        _, fg_mask_ind = F.cond_take(fg_mask == 1, fg_mask)
        labels[fg_mask_ind] = F.ones((fg_mask_ind.shape[0],), dtype=labels.dtype)
        # compute the targets
        bbox_targets = self.box_coder.encode(anchors, valid_gt_boxes[argmax_overlaps, :4])
        return labels, bbox_targets

    def get_ground_truth(self, gt_boxes, im_info, all_anchors_list):
        final_labels_list = []
        final_bbox_targets_list = []

        for bid in range(self.cfg.batch_per_gpu):
            batch_labels_list = []
            batch_bbox_targets_list = []
            for anchors in all_anchors_list:
                rpn_labels_perlvl, rpn_bbox_targets_perlvl = self.per_level_gt(
                    gt_boxes[bid], im_info[bid], anchors,
                )
                batch_labels_list.append(rpn_labels_perlvl)
                batch_bbox_targets_list.append(rpn_bbox_targets_perlvl)

            concated_batch_labels = F.concat(batch_labels_list, axis=0)
            concated_batch_bbox_targets = F.concat(batch_bbox_targets_list, axis=0)

            # sample labels
            num_positive = self.cfg.num_sample_anchors * self.cfg.positive_anchor_ratio
            # sample positive
            concated_batch_labels = concated_batch_labels.detach()
            concated_batch_labels = self._bernoulli_sample_labels(
                concated_batch_labels, num_positive, 1, self.cfg.ignore_label
            )
            # sample negative
            # FIXME bool astype
            num_positive = (concated_batch_labels == 1).astype("int32").sum()
            num_negative = self.cfg.num_sample_anchors - num_positive
            concated_batch_labels = self._bernoulli_sample_labels(
                concated_batch_labels, num_negative, 0, self.cfg.ignore_label
            )

            final_labels_list.append(concated_batch_labels)
            final_bbox_targets_list.append(concated_batch_bbox_targets)
        final_labels = F.concat(final_labels_list, axis=0)
        final_bbox_targets = F.concat(final_bbox_targets_list, axis=0)
        return final_labels.detach(), final_bbox_targets.detach()

    def _bernoulli_sample_labels(
        self, labels, num_samples, sample_value, ignore_label=-1
    ):
        """ Using the bernoulli sampling method"""
        # TODO rewrite this logic
        # FIXME bool astype
        sample_label_mask = (labels == sample_value).astype("int32")
        # NOTE sum to int
        num_mask = sample_label_mask.sum()
        num_final_samples = F.minimum(num_mask, num_samples)
        # here, we use the bernoulli scoreability to sample the anchors
        uniform_rng = rand.uniform(sample_label_mask.shape[0])
        sample_score = (
            (
                num_final_samples.astype("float32") / num_mask.astype("float32")
            ).broadcast(uniform_rng.shape)
        )
        to_ignore_mask = (uniform_rng >= sample_score).astype("int32") * sample_label_mask
        labels = labels * (1 - to_ignore_mask) + to_ignore_mask * ignore_label

        return labels
