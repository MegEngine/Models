# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import math
from typing import List

import megengine.module as M
from megengine import Tensor

from official.vision.detection import layers


class RetinaNetHead(M.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    """

    def __init__(self, cfg, input_shape: List[layers.ShapeSpec]):
        super().__init__()

        in_channels = input_shape[0].channels
        num_classes = cfg.num_classes
        num_convs = 4
        prior_prob = cfg.cls_prior_prob
        num_anchors = [
            len(cfg.anchor_scales[i]) * len(cfg.anchor_ratios[i])
            for i in range(len(input_shape))
        ]

        assert (
            len(set(num_anchors)) == 1
        ), "not support different number of anchors between levels"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(M.ReLU())
            bbox_subnet.append(
                M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(M.ReLU())

        self.cls_subnet = M.Sequential(*cls_subnet)
        self.bbox_subnet = M.Sequential(*bbox_subnet)
        self.cls_score = M.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = M.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [
            self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred
        ]:
            for layer in modules.modules():
                if isinstance(layer, M.Conv2d):
                    M.init.normal_(layer.weight, mean=0, std=0.01)
                    M.init.fill_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        M.init.fill_(self.cls_score.bias, bias_value)

    def forward(self, features: List[Tensor]):
        logits, offsets = [], []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            offsets.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, offsets
