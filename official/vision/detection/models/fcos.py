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
import megengine.distributed as dist
import megengine.functional as F
import megengine.module as M

import official.vision.classification.resnet.model as resnet
from official.vision.detection import layers


class FCOS(M.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1904.01355).
    """

    def __init__(self, cfg, batch_size):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size

        self.anchor_generator = layers.AnchorPointGenerator(
            cfg.num_anchors,
            strides=self.cfg.stride,
            offset=self.cfg.anchor_offset,
        )
        self.box_coder = layers.PointCoder()

        self.in_features = ["p3", "p4", "p5", "p6", "p7"]

        # ----------------------- build the backbone ------------------------ #
        bottom_up = getattr(resnet, cfg.backbone)(
            norm=layers.get_norm(cfg.resnet_norm), pretrained=cfg.backbone_pretrained
        )
        del bottom_up.fc

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

        # ----------------------- build the FCOS Head ----------------------- #
        self.head = layers.PointHead(cfg, feature_shapes)

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

        box_logits, box_offsets, box_ctrness = self.head(features)

        box_logits_list = [
            _.transpose(0, 2, 3, 1).reshape(self.batch_size, -1, self.cfg.num_classes)
            for _ in box_logits
        ]
        box_offsets_list = [
            _.transpose(0, 2, 3, 1).reshape(self.batch_size, -1, 4) for _ in box_offsets
        ]
        box_ctrness_list = [
            _.transpose(0, 2, 3, 1).reshape(self.batch_size, -1, 1) for _ in box_ctrness
        ]

        anchors_list = self.anchor_generator(features)

        all_level_box_logits = F.concat(box_logits_list, axis=1)
        all_level_box_offsets = F.concat(box_offsets_list, axis=1)
        all_level_box_ctrness = F.concat(box_ctrness_list, axis=1)

        if self.training:
            gt_labels, gt_offsets, gt_ctrness = self.get_ground_truth(
                anchors_list, gt_boxes, im_info[:, 4].astype(np.int32),
            )

            all_level_box_logits = all_level_box_logits.reshape(-1, self.cfg.num_classes)
            all_level_box_offsets = all_level_box_offsets.reshape(-1, 4)
            all_level_box_ctrness = all_level_box_ctrness.flatten()

            gt_labels = gt_labels.flatten()
            gt_offsets = gt_offsets.reshape(-1, 4)
            gt_ctrness = gt_ctrness.flatten()

            valid_mask = gt_labels >= 0
            fg_mask = gt_labels > 0
            num_fg = fg_mask.sum()
            sum_ctr = gt_ctrness[fg_mask].sum()
            # add detach() to avoid syncing across ranks in backward
            num_fg = layers.all_reduce_mean(num_fg).detach()
            sum_ctr = layers.all_reduce_mean(sum_ctr).detach()

            gt_targets = F.zeros_like(all_level_box_logits)
            gt_targets[fg_mask, gt_labels[fg_mask] - 1] = 1

            cls_loss = layers.sigmoid_focal_loss(
                all_level_box_logits[valid_mask],
                gt_targets[valid_mask],
                alpha=self.cfg.focal_loss_alpha,
                gamma=self.cfg.focal_loss_gamma,
            ).sum() / F.maximum(1, num_fg)

            bbox_loss = (
                layers.iou_loss(
                    all_level_box_offsets[fg_mask],
                    gt_offsets[fg_mask],
                    box_mode="ltrb",
                    loss_type=self.cfg.iou_loss_type,
                ) * gt_ctrness[fg_mask]
            ).sum() / F.maximum(1, sum_ctr) * self.cfg.bbox_loss_weight

            ctr_loss = layers.binary_cross_entropy_with_logits(
                all_level_box_ctrness[fg_mask],
                gt_ctrness[fg_mask],
            ).sum() / F.maximum(1, num_fg)

            total = cls_loss + bbox_loss + ctr_loss
            loss_dict = {
                "total_loss": total,
                "loss_cls": cls_loss,
                "loss_bbox": bbox_loss,
                "loss_ctr": ctr_loss,
            }
            self.cfg.losses_keys = list(loss_dict.keys())
            return loss_dict
        else:
            # currently not support multi-batch testing
            assert self.batch_size == 1

            all_level_anchors = F.concat(anchors_list, axis=0)
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
            all_level_box_scores = F.sqrt(
                F.sigmoid(all_level_box_logits) * F.sigmoid(all_level_box_ctrness)
            )
            return all_level_box_scores[0], clipped_box

    def get_ground_truth(self, anchors_list, batched_gt_boxes, batched_valid_gt_box_number):
        labels_list = []
        offsets_list = []
        ctrness_list = []

        all_level_anchors = F.concat(anchors_list, axis=0)
        for b_id in range(self.batch_size):
            gt_boxes = batched_gt_boxes[b_id, : batched_valid_gt_box_number[b_id]]

            offsets = self.box_coder.encode(
                all_level_anchors, F.add_axis(gt_boxes[:, :4], axis=1)
            )

            object_sizes_of_interest = F.concat([
                F.add_axis(
                    mge.tensor(size, dtype=np.float32), axis=0
                ).broadcast(anchors_i.shape[0], 2)
                for anchors_i, size in zip(anchors_list, self.cfg.object_sizes_of_interest)
            ], axis=0)
            max_offsets = F.max(offsets, axis=2)
            is_cared_in_the_level = \
                (max_offsets >= F.add_axis(object_sizes_of_interest[:, 0], axis=0)) & \
                (max_offsets <= F.add_axis(object_sizes_of_interest[:, 1], axis=0))

            if self.cfg.center_sampling_radius > 0:
                gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:4]) / 2
                is_in_boxes = []
                for stride, anchors_i in zip(self.cfg.stride, anchors_list):
                    radius = stride * self.cfg.center_sampling_radius
                    center_boxes = F.concat([
                        F.maximum(gt_centers - radius, gt_boxes[:, :2]),
                        F.minimum(gt_centers + radius, gt_boxes[:, 2:4]),
                    ], axis=1)
                    center_offsets = self.box_coder.encode(
                        anchors_i, F.add_axis(center_boxes, axis=1)
                    )
                    is_in_boxes.append(F.min(center_offsets, axis=2) > 0)
                is_in_boxes = F.concat(is_in_boxes, axis=1)
            else:
                is_in_boxes = F.min(offsets, axis=2) > 0

            gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            areas = F.add_axis(gt_area, axis=1).broadcast(offsets.shape[:2])  # FIXME: repeat
            areas[~is_cared_in_the_level] = float("inf")
            areas[~is_in_boxes] = float("inf")

            match_indices = F.argmin(areas, axis=0)
            gt_boxes_matched = gt_boxes[match_indices]
            anchor_min_area = F.remove_axis(
                F.gather(areas, 0, F.add_axis(match_indices, axis=0)), axis=0
            )

            labels = gt_boxes_matched[:, 4].astype(np.int32)
            labels[anchor_min_area == float("inf")] = 0
            offsets = self.box_coder.encode(all_level_anchors, gt_boxes_matched[:, :4])

            left_right = offsets[:, [0, 2]]
            top_bottom = offsets[:, [1, 3]]
            ctrness = F.sqrt(
                F.clamp(F.min(left_right, axis=1) / F.max(left_right, axis=1), lower=0)
                * F.clamp(F.min(top_bottom, axis=1) / F.max(top_bottom, axis=1), lower=0)
            )

            labels_list.append(labels)
            offsets_list.append(offsets)
            ctrness_list.append(ctrness)

        return (
            F.stack(labels_list, axis=0).detach(),
            F.stack(offsets_list, axis=0).detach(),
            F.stack(ctrness_list, axis=0).detach(),
        )


class FCOSConfig:
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

        self.num_anchors = 1
        self.anchor_offset = 0.5

        self.object_sizes_of_interest = [
            [-1, 64], [64, 128], [128, 256], [256, 512], [512, float("inf")]
        ]
        self.center_sampling_radius = 1.5
        self.class_aware_box = False
        self.cls_prior_prob = 0.01

        # ------------------------ loss cfg -------------------------- #
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2
        self.iou_loss_type = "giou"
        self.bbox_loss_weight = 1.0
        self.num_losses = 4

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
        self.test_nms = 0.6
