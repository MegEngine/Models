# -*- coding: utf-8 -*-
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

import official.vision.classification.resnet.model as resnet
from official.vision.detection import layers


class RetinaNet(M.Module):
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

        self.in_features = cfg.in_features

        # ----------------------- build backbone ------------------------ #
        bottom_up = getattr(resnet, cfg.backbone)(
            norm=layers.get_norm(cfg.backbone_norm), pretrained=cfg.backbone_pretrained
        )
        del bottom_up.fc

        # ----------------------- build FPN ----------------------------- #
        self.backbone = layers.FPN(
            bottom_up=bottom_up,
            in_features=cfg.fpn_in_features,
            out_channels=cfg.fpn_out_channels,
            norm=cfg.fpn_norm,
            top_block=layers.LastLevelP6P7(
                cfg.fpn_top_in_channel, cfg.fpn_out_channels, cfg.fpn_top_in_feature
            ),
            strides=cfg.fpn_in_strides,
            channels=cfg.fpn_in_channels,
        )

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        # ----------------------- build RetinaNet Head ------------------ #
        self.head = layers.BoxHead(cfg, feature_shapes)

        self.matcher = layers.Matcher(
            cfg.match_thresholds, cfg.match_labels, cfg.match_allow_low_quality
        )

    def preprocess_image(self, image):
        padded_image = layers.get_padded_tensor(image, 32, 0.0)
        normed_image = (
            padded_image
            - np.array(self.cfg.img_mean, dtype="float32")[None, :, None, None]
        ) / np.array(self.cfg.img_std, dtype="float32")[None, :, None, None]
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
            gt_labels, gt_offsets = self.get_ground_truth(
                all_level_anchors, gt_boxes, im_info[:, 4].astype("int32"),
            )

            all_level_box_logits = all_level_box_logits.reshape(-1, self.cfg.num_classes)
            all_level_box_offsets = all_level_box_offsets.reshape(-1, 4)

            gt_labels = gt_labels.flatten()
            gt_offsets = gt_offsets.reshape(-1, 4)

            valid_mask = gt_labels >= 0
            fg_mask = gt_labels > 0
            num_fg = fg_mask.sum()

            gt_targets = F.zeros_like(all_level_box_logits)
            gt_targets[fg_mask, gt_labels[fg_mask] - 1] = 1

            loss_cls = layers.sigmoid_focal_loss(
                all_level_box_logits[valid_mask],
                gt_targets[valid_mask],
                alpha=self.cfg.focal_loss_alpha,
                gamma=self.cfg.focal_loss_gamma,
            ).sum() / F.maximum(num_fg, 1)

            loss_bbox = layers.smooth_l1_loss(
                all_level_box_offsets[fg_mask],
                gt_offsets[fg_mask],
                beta=self.cfg.smooth_l1_beta,
            ).sum() / F.maximum(num_fg, 1) * self.cfg.loss_bbox_weight

            total = loss_cls + loss_bbox
            loss_dict = {
                "total_loss": total,
                "loss_cls": loss_cls,
                "loss_bbox": loss_bbox,
            }
            self.cfg.losses_keys = list(loss_dict.keys())
            return loss_dict
        else:
            # currently not support multi-batch testing
            assert image.shape[0] == 1

            pred_boxes = self.box_coder.decode(
                all_level_anchors, all_level_box_offsets[0]
            )
            pred_boxes = pred_boxes.reshape(-1, 4)

            scale_w = im_info[0, 1] / im_info[0, 3]
            scale_h = im_info[0, 0] / im_info[0, 2]
            pred_boxes = pred_boxes / F.concat(
                [scale_w, scale_h, scale_w, scale_h], axis=0
            )
            clipped_boxes = layers.get_clipped_boxes(
                pred_boxes, im_info[0, 2:4]
            ).reshape(-1, 4)
            pred_score = F.sigmoid(all_level_box_logits)[0]
            return pred_score, clipped_boxes

    def get_ground_truth(self, anchors, batched_gt_boxes, batched_num_gts):
        labels_list = []
        offsets_list = []

        for bid in range(batched_gt_boxes.shape[0]):
            gt_boxes = batched_gt_boxes[bid, :batched_num_gts[bid]]

            overlaps = layers.get_iou(gt_boxes[:, :4], anchors)
            match_indices, labels = self.matcher(overlaps)
            gt_boxes_matched = gt_boxes[match_indices]

            fg_mask = labels == 1
            labels[fg_mask] = gt_boxes_matched[fg_mask, 4].astype("int32")
            offsets = self.box_coder.encode(anchors, gt_boxes_matched[:, :4])

            labels_list.append(labels)
            offsets_list.append(offsets)

        return (
            F.stack(labels_list, axis=0).detach(),
            F.stack(offsets_list, axis=0).detach(),
        )


class RetinaNetConfig:
    # pylint: disable=too-many-statements
    def __init__(self):
        self.backbone = "resnet50"
        self.backbone_pretrained = True
        self.backbone_norm = "FrozenBN"
        self.backbone_freeze_at = 2
        self.fpn_norm = None
        self.fpn_in_features = ["res3", "res4", "res5"]
        self.fpn_in_strides = [8, 16, 32]
        self.fpn_in_channels = [512, 1024, 2048]
        self.fpn_out_channels = 256
        self.fpn_top_in_feature = "res5"
        self.fpn_top_in_channel = 2048

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

        # ----------------------- net cfg ------------------------- #
        self.stride = [8, 16, 32, 64, 128]
        self.in_features = ["p3", "p4", "p5", "p6", "p7"]
        self.reg_mean = [0.0, 0.0, 0.0, 0.0]
        self.reg_std = [1.0, 1.0, 1.0, 1.0]

        self.anchor_scales = [
            [x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)] for x in [32, 64, 128, 256, 512]
        ]
        self.anchor_ratios = [[0.5, 1, 2]]
        self.anchor_offset = 0.5

        self.match_thresholds = [0.4, 0.5]
        self.match_labels = [0, -1, 1]
        self.match_allow_low_quality = True
        self.class_aware_box = False
        self.cls_prior_prob = 0.01

        # ------------------------ loss cfg -------------------------- #
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2
        self.smooth_l1_beta = 0  # use L1 loss
        self.loss_bbox_weight = 1.0
        self.num_losses = 3

        # ------------------------ training cfg ---------------------- #
        self.train_image_short_size = (640, 672, 704, 736, 768, 800)
        self.train_image_max_size = 1333

        self.basic_lr = 0.01 / 16  # The basic learning rate for single-image
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.log_interval = 20
        self.nr_images_epoch = 80000
        self.max_epoch = 54
        self.warm_iters = 500
        self.lr_decay_rate = 0.1
        self.lr_decay_stages = [42, 50]

        # ------------------------ testing cfg ----------------------- #
        self.test_image_short_size = 800
        self.test_image_max_size = 1333
        self.test_max_boxes_per_image = 100
        self.test_vis_threshold = 0.3
        self.test_cls_threshold = 0.05
        self.test_nms = 0.5
