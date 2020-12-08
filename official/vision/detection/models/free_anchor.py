# -*- coding: utf-8 -*-
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

import official.vision.classification.resnet.model as resnet
from official.vision.detection import layers


class FreeAnchor(M.Module):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.anchor_generator = layers.AnchorBoxGenerator(
            anchor_scales=self.cfg.anchor_scales,
            anchor_ratios=self.cfg.anchor_ratios,
            strides=self.cfg.stride,
            offset=self.cfg.anchor_offset,
        )
        self.box_coder = layers.BoxCoder(cfg.reg_mean, cfg.reg_std)

        self.stride_list = np.array(cfg.stride, dtype=np.float32)
        self.in_features = cfg.in_features

        # ----------------------- build backbone ------------------------ #
        bottom_up = getattr(resnet, cfg.backbone)(
            norm=layers.get_norm(cfg.resnet_norm), pretrained=cfg.backbone_pretrained
        )
        del bottom_up.fc

        # ----------------------- build FPN ----------------------------- #
        in_channels_p6p7 = 2048
        out_channels = 256
        self.backbone = layers.FPN(
            bottom_up=bottom_up,
            in_features=["res3", "res4", "res5"],
            out_channels=out_channels,
            norm=cfg.fpn_norm,
            top_block=layers.LastLevelP6P7(in_channels_p6p7, out_channels),
        )

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        # ----------------------- build head ------------------ #
        self.head = layers.BoxHead(cfg, feature_shapes)

    def preprocess_image(self, image):
        padded_image = layers.get_padded_tensor(image, 32, 0.0)
        normed_image = (
            padded_image
            - np.array(self.cfg.img_mean, dtype=np.float32)[None, :, None, None]
        ) / np.array(self.cfg.img_std, dtype=np.float32)[None, :, None, None]
        return normed_image

    def forward(self, image, im_info, gt_boxes=None):
        image = self.preprocess_image(image)
        features = self.backbone(image)
        features = [features[f] for f in self.in_features]

        box_logits, box_offsets = self.head(features)

        box_logits_list = [
            _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, self.cfg.num_classes)
            for _ in box_logits
        ]
        box_offsets_list = [
            _.transpose(0, 2, 3, 1).reshape(image.shape[0], -1, 4) for _ in box_offsets
        ]

        anchors_list = self.anchor_generator(features)

        all_level_box_logits = F.concat(box_logits_list, axis=1)
        all_level_box_offsets = F.concat(box_offsets_list, axis=1)
        all_level_anchors = F.concat(anchors_list, axis=0)

        if self.training:

            loss_dict = self.get_losses(
                all_level_anchors, all_level_box_logits,
                all_level_box_offsets, gt_boxes, im_info
            )
            self.cfg.losses_keys = list(loss_dict.keys())
            return loss_dict
        else:
            # currently not support multi-batch testing
            assert image.shape[0] == 1

            transformed_box = self.box_coder.decode(
                all_level_anchors, all_level_box_offsets[0]
            )
            transformed_box = transformed_box.reshape(-1, 4)

            scale_w = im_info[0, 1] / im_info[0, 3]
            scale_h = im_info[0, 0] / im_info[0, 2]
            transformed_box = transformed_box / F.concat(
                [scale_w, scale_h, scale_w, scale_h], axis=0
            )
            clipped_box = layers.get_clipped_boxes(
                transformed_box, im_info[0, 2:4]
            ).reshape(-1, 4)
            all_level_box_scores = F.sigmoid(all_level_box_logits)
            return all_level_box_scores[0], clipped_box

    def get_losses(self, anchors, pred_logits, pred_offsets, gt_boxes, im_info):
        # pylint: disable=too-many-statements
        def positive_bag_loss(logits, axis=1):
            weight = 1.0 / (1.0 - logits)
            weight /= weight.sum(axis=axis, keepdims=True)
            bag_prob = (weight * logits).sum(axis=1)
            return -layers.safelog(bag_prob)

        def negative_bag_loss(logits, gamma):
            return (logits ** gamma) * (-layers.safelog(1.0 - logits))

        pred_scores = F.sigmoid(pred_logits)
        box_prob_list = []
        positive_losses = []
        clamp_eps = 1e-7
        bucket_size = self.cfg.bucket_size

        for bid in range(im_info.shape[0]):
            boxes_info = gt_boxes[bid, : im_info[bid, 4].astype("int32")]
            # id 0 is used for background classes, so -1 first
            labels = boxes_info[:, 4].astype("int32") - 1

            pred_box = self.box_coder.decode(anchors, pred_offsets[bid]).detach()
            overlaps = layers.get_iou(boxes_info[:, :4], pred_box).detach()
            thresh1 = self.cfg.box_iou_threshold
            thresh2 = F.clip(
                overlaps.max(axis=1, keepdims=True),
                lower=thresh1 + clamp_eps, upper=1.0
            )
            gt_pred_prob = F.clip(
                (overlaps - thresh1) / (thresh2 - thresh1), lower=0, upper=1.0)

            image_boxes_prob = F.zeros(pred_logits.shape[1:]).detach()
            # guarantee that nonzero_idx is not empty
            if gt_pred_prob.max() > clamp_eps:
                _, nonzero_idx = F.cond_take(gt_pred_prob != 0, gt_pred_prob)
                # since nonzeros is only 1 dim, use num_anchor to get real indices
                num_anchors = gt_pred_prob.shape[1]
                anchors_idx = nonzero_idx % num_anchors
                gt_idx = nonzero_idx // num_anchors
                image_boxes_prob[anchors_idx, labels[gt_idx]] = gt_pred_prob[gt_idx, anchors_idx]

            box_prob_list.append(image_boxes_prob)

            # construct bags for objects
            match_quality_matrix = layers.get_iou(boxes_info[:, :4], anchors).detach()
            num_gt = match_quality_matrix.shape[0]
            _, matched_idx = F.topk(
                match_quality_matrix,
                k=bucket_size,
                descending=True,
                no_sort=True,
            )

            matched_idx = matched_idx.detach()
            matched_idx_flatten = matched_idx.reshape(-1)
            gather_idx = labels.reshape(-1, 1)
            gather_idx = F.broadcast_to(gather_idx, (num_gt, bucket_size))

            gather_src = pred_scores[bid, matched_idx_flatten]
            gather_src = gather_src.reshape(num_gt, bucket_size, -1)
            matched_score = F.indexing_one_hot(gather_src, gather_idx, axis=2)

            topk_anchors = anchors[matched_idx_flatten]
            boxes_broad_cast = F.broadcast_to(
                F.expand_dims(boxes_info[:, :4], axis=1), (num_gt, bucket_size, 4)
            ).reshape(-1, 4)

            matched_offsets = self.box_coder.encode(topk_anchors, boxes_broad_cast)

            reg_loss = layers.smooth_l1_loss(
                pred_offsets[bid, matched_idx_flatten],
                matched_offsets,
                beta=self.cfg.smooth_l1_beta
            ).sum(axis=-1) * self.cfg.reg_loss_weight
            matched_reg_scores = F.exp(-reg_loss)

            positive_losses.append(
                positive_bag_loss(
                    matched_score * matched_reg_scores.reshape(-1, bucket_size), axis=1
                )
            )

        num_foreground = im_info[:, 4].sum()
        pos_loss = F.concat(positive_losses).sum() / F.maximum(1.0, num_foreground)
        box_probs = F.stack(box_prob_list, axis=0)

        neg_loss = negative_bag_loss(
            pred_scores * (1 - box_probs), self.cfg.focal_loss_gamma
        ).sum() / F.maximum(1.0, num_foreground * bucket_size)

        alpha = self.cfg.focal_loss_alpha
        pos_loss = pos_loss * alpha
        neg_loss = neg_loss * (1 - alpha)
        loss_dict = {
            "total_loss": pos_loss + neg_loss,
            "pos_loss": pos_loss,
            "neg_loss": neg_loss,
        }
        return loss_dict


class FreeAnchorConfig:
    def __init__(self):
        self.backbone = "resnet50"
        self.backbone_pretrained = True
        self.resnet_norm = "FrozenBN"
        self.fpn_norm = None
        self.backbone_freeze_at = 2
        self.box_iou_threshold = 0.6
        self.bucket_size = 50

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="coco",
            root="train2017",
            ann_file="annotations/instances_train2017.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="coco",
            root="val2017",
            ann_file="annotations/instances_val2017.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 80
        self.img_mean = [103.530, 116.280, 123.675]  # BGR
        self.img_std = [57.375, 57.120, 58.395]
        self.stride = [8, 16, 32, 64, 128]
        self.reg_mean = [0.0, 0.0, 0.0, 0.0]
        self.reg_std = [0.1, 0.1, 0.2, 0.2]
        self.in_features = ["p3", "p4", "p5", "p6", "p7"]

        self.anchor_scales = [
            [x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)]
            for x in [32, 64, 128, 256, 512]
        ]
        self.anchor_ratios = [[0.5, 1, 2]]
        self.anchor_offset = 0.5

        self.class_aware_box = False
        self.cls_prior_prob = 0.02

        # ------------------------ loss cfg -------------------------- #
        self.focal_loss_alpha = 0.5
        self.focal_loss_gamma = 2
        self.smooth_l1_beta = 0  # use L1 loss
        self.reg_loss_weight = 0.75
        self.num_losses = 3

        # ------------------------ training cfg ---------------------- #
        self.train_image_short_size = (640, 672, 704, 736, 768, 800)
        self.train_image_max_size = 1333

        self.basic_lr = 0.01 / 16  # The basic learning rate for single-image
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.log_interval = 20
        self.nr_images_epoch = 80000
        self.max_epoch = 18
        self.warm_iters = 500
        self.lr_decay_rate = 0.1
        self.lr_decay_stages = [12, 16]

        # ------------------------ testing cfg ----------------------- #
        self.test_image_short_size = 800
        self.test_image_max_size = 1333
        self.test_max_boxes_per_image = 100
        self.test_vis_threshold = 0.3
        self.test_cls_threshold = 0.05
        self.test_nms = 0.5
