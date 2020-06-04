# -*- coding:utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine as mge
import megengine.random as rand
import megengine.functional as F
import megengine.module as M
from official.vision.detection import layers
from official.vision.detection.tools.gpu_nms import batched_nms


class RPN(M.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.box_coder = layers.BoxCoder()

        self.stride_list = cfg.rpn_stride
        rpn_channel = cfg.rpn_channel
        self.in_features = cfg.rpn_in_features
        self.anchors_generator = layers.DefaultAnchorGenerator(
            cfg.anchor_base_size,
            cfg.anchor_scales,
            cfg.anchor_aspect_ratios,
            cfg.anchor_offset,
        )
        self.rpn_conv = M.Conv2d(256, rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = M.Conv2d(
            rpn_channel, cfg.num_cell_anchors,
            kernel_size=1, stride=1
        )
        self.rpn_bbox_offsets = M.Conv2d(
            rpn_channel, cfg.num_cell_anchors * 4,
            kernel_size=1, stride=1
        )

        for l in [self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_offsets]:
            M.init.normal_(l.weight, std=0.01)
            M.init.fill_(l.bias, 0)

    def forward(self, features, im_info, boxes=None):
        # prediction
        features = [features[x] for x in self.in_features]

        # get anchors
        all_anchors_list = [
            self.anchors_generator(fm, stride)
            for fm, stride in zip(features, self.stride_list)
        ]

        pred_cls_score_list = []
        pred_bbox_offsets_list = []
        for x in features:
            t = F.relu(self.rpn_conv(x))
            scores = self.rpn_cls_score(t)
            pred_cls_score_list.append(scores)
            bbox_offsets = self.rpn_bbox_offsets(t)
            pred_bbox_offsets_list.append(
                bbox_offsets.reshape(
                    bbox_offsets.shape[0], self.cfg.num_cell_anchors, 4,
                    bbox_offsets.shape[2], bbox_offsets.shape[3]
                )
            )
        # sample from the predictions
        rpn_rois = self.find_top_rpn_proposals(
            pred_bbox_offsets_list, pred_cls_score_list,
            all_anchors_list, im_info
        )

        if self.training:
            rpn_labels, rpn_bbox_targets = self.get_ground_truth(
                boxes, im_info, all_anchors_list)
            pred_cls_score, pred_bbox_offsets = self.merge_rpn_score_box(
                pred_cls_score_list, pred_bbox_offsets_list
            )

            # rpn loss
            valid_labels, valid_inds = F.cond_take(rpn_labels >= 0, rpn_labels)
            loss_rpn_cls = F.binary_cross_entropy(
                F.sigmoid(pred_cls_score.ai[valid_inds]),
                valid_labels.astype("int32")
            )
            # loss_rpn_cls = layers.softmax_loss(pred_cls_score, rpn_labels)
            loss_rpn_loc = layers.get_smooth_l1_loss(
                pred_bbox_offsets,
                rpn_bbox_targets,
                rpn_labels,
                self.cfg.rpn_smooth_l1_beta,
                norm_type="all"
            )
            loss_dict = {
                "loss_rpn_cls": loss_rpn_cls,
                "loss_rpn_loc": loss_rpn_loc
            }
            return rpn_rois, loss_dict
        else:
            return rpn_rois

    def find_top_rpn_proposals(
        self, rpn_bbox_offsets_list, rpn_cls_prob_list,
        all_anchors_list, im_info
    ):
        prev_nms_top_n = self.cfg.train_prev_nms_top_n \
            if self.training else self.cfg.test_prev_nms_top_n
        post_nms_top_n = self.cfg.train_post_nms_top_n \
            if self.training else self.cfg.test_post_nms_top_n

        batch_per_gpu = self.cfg.batch_per_gpu if self.training else 1
        nms_threshold = self.cfg.rpn_nms_threshold

        list_size = len(rpn_bbox_offsets_list)

        return_rois = []

        for bid in range(batch_per_gpu):
            batch_proposals_list = []
            batch_probs_list = []
            batch_level_list = []
            for l in range(list_size):
                # get proposals and probs
                offsets = rpn_bbox_offsets_list[l][bid].dimshuffle(2, 3, 0, 1).reshape(-1, 4)
                all_anchors = all_anchors_list[l]
                proposals = self.box_coder.decode(all_anchors, offsets)

                probs = rpn_cls_prob_list[l][bid].dimshuffle(1, 2, 0).reshape(1, -1)
                # prev nms top n
                probs, order = F.argsort(probs, descending=True)
                num_proposals = F.minimum(probs.shapeof(1), prev_nms_top_n)
                probs = probs.reshape(-1)[:num_proposals]
                order = order.reshape(-1)[:num_proposals]
                proposals = proposals.ai[order, :]

                batch_proposals_list.append(proposals)
                batch_probs_list.append(probs)
                batch_level_list.append(mge.ones(probs.shapeof(0)) * l)

            proposals = F.concat(batch_proposals_list, axis=0)
            scores = F.concat(batch_probs_list, axis=0)
            level = F.concat(batch_level_list, axis=0)

            proposals = layers.get_clipped_box(proposals, im_info[bid, :])
            # filter empty
            keep_mask = layers.filter_boxes(proposals)
            _, keep_inds = F.cond_take(keep_mask == 1, keep_mask)
            proposals = proposals.ai[keep_inds, :]
            scores = scores.ai[keep_inds]
            level = level.ai[keep_inds]

            # gather the proposals and probs
            # sort nms by scores
            scores, order = F.argsort(scores.reshape(1, -1), descending=True)
            order = order.reshape(-1)
            proposals = proposals.ai[order, :]
            level = level.ai[order]

            # apply total level nms
            rois = F.concat([proposals, scores.reshape(-1, 1)], axis=1)
            keep_inds = batched_nms(proposals, scores, level, nms_threshold, post_nms_top_n)
            rois = rois.ai[keep_inds]

            # rois shape (N, 5), info [batch_id, x1, y1, x2, y2]
            batch_inds = mge.ones((rois.shapeof(0), 1)) * bid
            batch_rois = F.concat([batch_inds, rois[:, :4]], axis=1)
            return_rois.append(batch_rois)

        return F.zero_grad(F.concat(return_rois, axis=0))

    def merge_rpn_score_box(self, rpn_cls_score_list, rpn_bbox_offsets_list):
        final_rpn_cls_score_list = []
        final_rpn_bbox_offsets_list = []

        for bid in range(self.cfg.batch_per_gpu):
            batch_rpn_cls_score_list = []
            batch_rpn_bbox_offsets_list = []

            for i in range(len(self.in_features)):
                rpn_cls_score = rpn_cls_score_list[i][bid] \
                    .dimshuffle(1, 2, 0).reshape(-1)
                rpn_bbox_offsets = rpn_bbox_offsets_list[i][bid] \
                    .dimshuffle(2, 3, 0, 1).reshape(-1, 4)

                batch_rpn_cls_score_list.append(rpn_cls_score)
                batch_rpn_bbox_offsets_list.append(rpn_bbox_offsets)

            batch_rpn_cls_score = F.concat(batch_rpn_cls_score_list, axis=0)
            batch_rpn_bbox_offsets = F.concat(batch_rpn_bbox_offsets_list, axis=0)

            final_rpn_cls_score_list.append(batch_rpn_cls_score)
            final_rpn_bbox_offsets_list.append(batch_rpn_bbox_offsets)

        final_rpn_cls_score = F.concat(final_rpn_cls_score_list, axis=0)
        final_rpn_bbox_offsets = F.concat(final_rpn_bbox_offsets_list, axis=0)
        return final_rpn_cls_score, final_rpn_bbox_offsets

    def per_level_gt(
        self, gt_boxes, im_info, anchors, allow_low_quality_matches=True
    ):
        ignore_label = self.cfg.ignore_label
        # get the gt boxes
        valid_gt_boxes = gt_boxes[:im_info[4], :]
        # compute the iou matrix
        overlaps = layers.get_iou(anchors, valid_gt_boxes[:, :4])
        # match the dtboxes
        a_shp0 = anchors.shape[0]
        max_overlaps = F.max(overlaps, axis=1)
        argmax_overlaps = F.argmax(overlaps, axis=1)
        # all ignore
        labels = mge.ones(a_shp0).astype("int32") * ignore_label
        # set negative ones
        labels = labels * (max_overlaps >= self.cfg.rpn_negative_overlap)
        # set positive ones
        fg_mask = (max_overlaps >= self.cfg.rpn_positive_overlap)
        const_one = mge.tensor(1.0)
        if allow_low_quality_matches:
            # make sure that max iou of gt matched
            gt_argmax_overlaps = F.argmax(overlaps, axis=0)
            num_valid_boxes = valid_gt_boxes.shapeof(0)
            gt_id = F.linspace(0, num_valid_boxes - 1, num_valid_boxes).astype("int32")
            argmax_overlaps = argmax_overlaps.set_ai(gt_id)[gt_argmax_overlaps]
            max_overlaps = max_overlaps.set_ai(
                const_one.broadcast(num_valid_boxes)
            )[gt_argmax_overlaps]
            fg_mask = (max_overlaps >= self.cfg.rpn_positive_overlap)
        # set positive ones
        _, fg_mask_ind = F.cond_take(fg_mask == 1, fg_mask)
        labels = labels.set_ai(const_one.broadcast(fg_mask_ind.shapeof(0)))[fg_mask_ind]
        # compute the targets
        bbox_targets = self.box_coder.encode(
            anchors, valid_gt_boxes.ai[argmax_overlaps, :4]
        )
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
            concated_batch_labels = self._bernoulli_sample_labels(
                concated_batch_labels,
                num_positive, 1, self.cfg.ignore_label
            )
            # sample negative
            num_positive = (concated_batch_labels == 1).sum()
            num_negative = self.cfg.num_sample_anchors - num_positive
            concated_batch_labels = self._bernoulli_sample_labels(
                concated_batch_labels,
                num_negative, 0, self.cfg.ignore_label
            )

            final_labels_list.append(concated_batch_labels)
            final_bbox_targets_list.append(concated_batch_bbox_targets)

        final_labels = F.concat(final_labels_list, axis=0)
        final_bbox_targets = F.concat(final_bbox_targets_list, axis=0)
        return F.zero_grad(final_labels), F.zero_grad(final_bbox_targets)

    def _bernoulli_sample_labels(
        self, labels, num_samples, sample_value, ignore_label=-1
    ):
        """ Using the bernoulli sampling method"""
        sample_label_mask = (labels == sample_value)
        num_mask = sample_label_mask.sum()
        num_final_samples = F.minimum(num_mask, num_samples)
        # here, we use the bernoulli probability to sample the anchors
        sample_prob = num_final_samples / num_mask
        uniform_rng = rand.uniform(sample_label_mask.shapeof(0))
        to_ignore_mask = (uniform_rng >= sample_prob) * sample_label_mask
        labels = labels * (1 - to_ignore_mask) + to_ignore_mask * ignore_label

        return labels
