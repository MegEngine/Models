# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M

import official.vision.classification.resnet.model as resnet
from official.vision.detection import layers


class RetinaNet(M.Module):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg, batch_size):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size

        self.anchor_generator = layers.DefaultAnchorGenerator(
            anchor_scales=self.cfg.anchor_scales,
            anchor_ratios=self.cfg.anchor_ratios,
            strides=self.cfg.stride,
            offset=self.cfg.anchor_offset,
        )
        self.box_coder = layers.BoxCoder(cfg.reg_mean, cfg.reg_std)

        self.stride_list = np.array(cfg.stride, dtype=np.float32)
        self.in_features = ["p3", "p4", "p5", "p6", "p7"]

        # ----------------------- build the backbone ------------------------ #
        bottom_up = getattr(resnet, cfg.backbone)(
            norm=layers.get_norm(cfg.resnet_norm), pretrained=cfg.backbone_pretrained
        )
        del bottom_up.fc

        # # ------------ freeze the weights of resnet stage1 and stage 2 ------ #
        # if self.cfg.backbone_freeze_at >= 1:
        #     for p in bottom_up.conv1.parameters():
        #         p.requires_grad = False
        # if self.cfg.backbone_freeze_at >= 2:
        #     for p in bottom_up.layer1.parameters():
        #         p.requires_grad = False

        # ----------------------- build the FPN ----------------------------- #
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

        # ----------------------- build the RetinaNet Head ------------------ #
        self.head = layers.RetinaNetHead(cfg, feature_shapes)

        self.matcher = layers.Matcher(
            cfg.match_thresholds, cfg.match_labels, cfg.match_allow_low_quality
        )

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
            _.transpose(0, 2, 3, 1).reshape(self.batch_size, -1, self.cfg.num_classes)
            for _ in box_logits
        ]
        box_offsets_list = [
            _.transpose(0, 2, 3, 1).reshape(self.batch_size, -1, 4) for _ in box_offsets
        ]

        anchors_list = self.anchor_generator(features)

        all_level_box_logits = F.concat(box_logits_list, axis=1)
        all_level_box_offsets = F.concat(box_offsets_list, axis=1)
        all_level_anchors = F.concat(anchors_list, axis=0)

        if self.training:
            gt_labels, gt_offsets = self.get_ground_truth(
                all_level_anchors, gt_boxes, im_info[:, 4].astype(np.int32),
            )

            all_level_box_logits = all_level_box_logits.reshape(-1, self.cfg.num_classes)
            all_level_box_offsets = all_level_box_offsets.reshape(-1, 4)

            gt_labels = gt_labels.flatten()
            gt_offsets = gt_offsets.reshape(-1, 4)

            valid_mask = gt_labels >= 0
            fg_mask = gt_labels > 0
            num_fg = fg_mask.astype(np.float32).sum()

            gt_targets = F.zeros_like(all_level_box_logits)
            gt_targets[fg_mask, gt_labels[fg_mask] - 1] = 1

            rpn_cls_loss = layers.sigmoid_focal_loss(
                all_level_box_logits[valid_mask],
                gt_targets[valid_mask],
                alpha=self.cfg.focal_loss_alpha,
                gamma=self.cfg.focal_loss_gamma,
            ).sum() / F.maximum(1, num_fg)

            rpn_bbox_loss = layers.smooth_l1_loss(
                all_level_box_offsets[fg_mask],
                gt_offsets[fg_mask],
                beta=self.cfg.smooth_l1_beta,
            ).sum() / F.maximum(1, num_fg)
            rpn_bbox_loss *= self.cfg.reg_loss_weight

            total = rpn_cls_loss + rpn_bbox_loss
            loss_dict = {
                "total_loss": total,
                "loss_cls": rpn_cls_loss,
                "loss_loc": rpn_bbox_loss,
            }
            self.cfg.losses_keys = list(loss_dict.keys())
            return loss_dict
        else:
            # currently not support multi-batch testing
            assert self.batch_size == 1

            transformed_box = self.box_coder.decode(
                all_level_anchors, all_level_box_offsets[0]
            )
            transformed_box = transformed_box.reshape(-1, 4)

            scale_w = im_info[0, 1] / im_info[0, 3]
            scale_h = im_info[0, 0] / im_info[0, 2]
            transformed_box = transformed_box / F.concat(
                [scale_w, scale_h, scale_w, scale_h], axis=0
            )
            clipped_box = layers.get_clipped_box(
                transformed_box, im_info[0, 2:4]
            ).reshape(-1, 4)
            all_level_box_scores = F.sigmoid(all_level_box_logits)
            return all_level_box_scores[0], clipped_box

    def get_ground_truth(self, anchors, batched_gt_boxes, batched_valid_gt_box_number):
        labels_list = []
        offsets_list = []

        for b_id in range(self.batch_size):
            gt_boxes = batched_gt_boxes[b_id, : batched_valid_gt_box_number[b_id]]

            overlaps = layers.get_iou(gt_boxes[:, :4], anchors)
            match_indices, labels = self.matcher(overlaps)
            gt_boxes_matched = gt_boxes[match_indices]

            fg_mask = labels == 1
            labels[fg_mask] = gt_boxes_matched[fg_mask, 4].astype(np.int32)
            offsets = self.box_coder.encode(anchors, gt_boxes_matched[:, :4])

            labels_list.append(labels)
            offsets_list.append(offsets)

        return (
            F.stack(labels_list, axis=0).detach(),
            F.stack(offsets_list, axis=0).detach(),
        )


class RetinaNetConfig:
    def __init__(self):
        self.backbone = "resnet50"
        self.backbone_pretrained = True
        self.resnet_norm = "FrozenBN"
        self.fpn_norm = None
        self.backbone_freeze_at = 2

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
        self.reg_std = [1.0, 1.0, 1.0, 1.0]

        self.anchor_scales = [
            [x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)]
            for x in [32, 64, 128, 256, 512]
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
        self.reg_loss_weight = 1.0
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
