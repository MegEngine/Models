# -*- coding: utf-8 -*-
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
import numpy as np

from official.vision.classification.resnet.model import resnet50
from official.vision.detection import layers


class RetinaNet(M.Module):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg, batch_size):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size

        self.anchor_gen = layers.DefaultAnchorGenerator(
            base_size=4,
            anchor_scales=self.cfg.anchor_scales,
            anchor_ratios=self.cfg.anchor_ratios,
        )
        self.box_coder = layers.BoxCoder(reg_mean=cfg.reg_mean, reg_std=cfg.reg_std)

        self.stride_list = np.array([8, 16, 32, 64, 128]).astype(np.float32)
        self.in_features = ["p3", "p4", "p5", "p6", "p7"]

        # ----------------------- build the backbone ------------------------ #
        bottom_up = resnet50(norm=layers.get_norm(self.cfg.resnet_norm))

        # ------------ freeze the weights of resnet stage1 and stage 2 ------ #
        if self.cfg.backbone_freeze_at >= 1:
            for p in bottom_up.conv1.parameters():
                p.requires_grad = False
        if self.cfg.backbone_freeze_at >= 2:
            for p in bottom_up.layer1.parameters():
                p.requires_grad = False

        # ----------------------- build the FPN ----------------------------- #
        in_channels_p6p7 = 2048
        out_channels = 256
        self.backbone = layers.FPN(
            bottom_up=bottom_up,
            in_features=["res3", "res4", "res5"],
            out_channels=out_channels,
            norm="",
            top_block=layers.LastLevelP6P7(in_channels_p6p7, out_channels),
        )

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        # ----------------------- build the RetinaNet Head ------------------ #
        self.head = layers.RetinaNetHead(cfg, feature_shapes)

        self.inputs = {
            "image": mge.tensor(
                np.random.random([2, 3, 224, 224]).astype(np.float32), dtype="float32",
            ),
            "im_info": mge.tensor(
                np.random.random([2, 5]).astype(np.float32), dtype="float32",
            ),
            "gt_boxes": mge.tensor(
                np.random.random([2, 100, 5]).astype(np.float32), dtype="float32",
            ),
        }

    def preprocess_image(self, image):
        normed_image = (
            image - self.cfg.img_mean[None, :, None, None]
        ) / self.cfg.img_std[None, :, None, None]
        return layers.get_padded_tensor(normed_image, 32, 0.0)

    def forward(self, inputs):
        image = self.preprocess_image(inputs["image"])
        features = self.backbone(image)
        features = [features[f] for f in self.in_features]

        box_cls, box_delta = self.head(features)

        box_cls_list = [
            _.dimshuffle(0, 2, 3, 1).reshape(self.batch_size, -1, self.cfg.num_classes)
            for _ in box_cls
        ]
        box_delta_list = [
            _.dimshuffle(0, 2, 3, 1).reshape(self.batch_size, -1, 4) for _ in box_delta
        ]

        anchors_list = [
            self.anchor_gen(features[i], self.stride_list[i]) for i in range(5)
        ]

        all_level_box_cls = F.sigmoid(F.concat(box_cls_list, axis=1))
        all_level_box_delta = F.concat(box_delta_list, axis=1)
        all_level_anchors = F.concat(anchors_list, axis=0)

        if self.training:
            box_gt_cls, box_gt_delta = self.get_ground_truth(
                all_level_anchors,
                inputs["gt_boxes"],
                inputs["im_info"][:, 4].astype(np.int32),
            )
            rpn_cls_loss = layers.get_focal_loss(
                all_level_box_cls,
                box_gt_cls,
                alpha=self.cfg.focal_loss_alpha,
                gamma=self.cfg.focal_loss_gamma,
            )
            rpn_bbox_loss = (
                layers.get_smooth_l1_loss(all_level_box_delta, box_gt_delta, box_gt_cls)
                * self.cfg.reg_loss_weight
            )

            total = rpn_cls_loss + rpn_bbox_loss
            return total, rpn_cls_loss, rpn_bbox_loss
        else:
            # currently not support multi-batch testing
            assert self.batch_size == 1

            transformed_box = self.box_coder.decode(
                all_level_anchors, all_level_box_delta[0],
            )
            transformed_box = transformed_box.reshape(-1, 4)

            scale_w = inputs["im_info"][0, 1] / inputs["im_info"][0, 3]
            scale_h = inputs["im_info"][0, 0] / inputs["im_info"][0, 2]
            transformed_box = transformed_box / F.concat(
                [scale_w, scale_h, scale_w, scale_h], axis=0
            )
            clipped_box = layers.get_clipped_box(
                transformed_box, inputs["im_info"][0, 2:4]
            ).reshape(-1, 4)
            return all_level_box_cls[0], clipped_box

    def get_ground_truth(self, anchors, batched_gt_boxes, batched_valid_gt_box_number):
        total_anchors = anchors.shape[0]
        labels_cat_list = []
        bbox_targets_list = []

        for b_id in range(self.batch_size):
            gt_boxes = batched_gt_boxes[b_id, : batched_valid_gt_box_number[b_id]]

            overlaps = layers.get_iou(anchors, gt_boxes[:, :4])
            argmax_overlaps = F.argmax(overlaps, axis=1)

            max_overlaps = overlaps.ai[
                F.linspace(0, total_anchors - 1, total_anchors).astype(np.int32),
                argmax_overlaps,
            ]

            labels = mge.tensor([-1]).broadcast(total_anchors)
            labels = labels * (max_overlaps >= self.cfg.negative_thresh)
            labels = labels * (max_overlaps < self.cfg.positive_thresh) + (
                max_overlaps >= self.cfg.positive_thresh
            )

            bbox_targets = self.box_coder.encode(
                anchors, gt_boxes.ai[argmax_overlaps, :4]
            )

            labels_cat = gt_boxes.ai[argmax_overlaps, 4]
            labels_cat = labels_cat * (1.0 - F.less_equal(F.abs(labels), 1e-5))
            ignore_mask = F.less_equal(F.abs(labels + 1), 1e-5)
            labels_cat = labels_cat * (1 - ignore_mask) - ignore_mask

            # assign low_quality boxes
            if self.cfg.allow_low_quality:
                gt_argmax_overlaps = F.argmax(overlaps, axis=0)
                labels_cat = labels_cat.set_ai(gt_boxes[:, 4])[gt_argmax_overlaps]
                matched_low_bbox_targets = self.box_coder.encode(
                    anchors.ai[gt_argmax_overlaps, :], gt_boxes[:, :4]
                )
                bbox_targets = bbox_targets.set_ai(matched_low_bbox_targets)[
                    gt_argmax_overlaps, :
                ]

            labels_cat_list.append(F.add_axis(labels_cat, 0))
            bbox_targets_list.append(F.add_axis(bbox_targets, 0))

        return (
            F.zero_grad(F.concat(labels_cat_list, axis=0)),
            F.zero_grad(F.concat(bbox_targets_list, axis=0)),
        )


class RetinaNetConfig:
    def __init__(self):
        self.resnet_norm = "FrozenBN"
        self.backbone_freeze_at = 2

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="coco",
            root="train2017",
            ann_file="instances_train2017.json"
        )
        self.test_dataset = dict(
            name="coco",
            root="val2017",
            ann_file="instances_val2017.json"
        )
        self.train_image_short_size = 800
        self.train_image_max_size = 1333
        self.num_classes = 80
        self.img_mean = np.array([103.530, 116.280, 123.675])  # BGR
        self.img_std = np.array([57.375, 57.120, 58.395])
        self.reg_mean = None
        self.reg_std = np.array([0.1, 0.1, 0.2, 0.2])

        self.anchor_ratios = np.array([0.5, 1, 2])
        self.anchor_scales = np.array([2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)])
        self.negative_thresh = 0.4
        self.positive_thresh = 0.5
        self.allow_low_quality = True
        self.class_aware_box = False
        self.cls_prior_prob = 0.01

        # ------------------------ loss cfg -------------------------- #
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2
        self.reg_loss_weight = 1.0 / 4.0

        # ------------------------ training cfg ---------------------- #
        self.basic_lr = 0.01 / 16.0  # The basic learning rate for single-image
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.log_interval = 20
        self.nr_images_epoch = 80000
        self.max_epoch = 18
        self.warm_iters = 500
        self.lr_decay_rate = 0.1
        self.lr_decay_sates = [12, 16, 17]

        # ------------------------ testing cfg ----------------------- #
        self.test_image_short_size = 800
        self.test_image_max_size = 1333
        self.test_max_boxes_per_image = 100
        self.test_vis_threshold = 0.3
        self.test_cls_threshold = 0.05
        self.test_nms = 0.5
