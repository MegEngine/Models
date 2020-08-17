# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import List
from abc import ABCMeta, abstractmethod
import math
import numpy as np

from megengine import Tensor, tensor
import megengine.functional as F


def meshgrid(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    mesh_shape = (y.shape[0], x.shape[0])
    mesh_x = x.broadcast(mesh_shape)
    mesh_y = y.reshape(-1, 1).broadcast(mesh_shape)
    return mesh_x, mesh_y


def create_anchor_grid(featmap_size, offsets, stride):
    step_x, step_y = featmap_size
    shift = offsets * stride

    grid_x = F.arange(shift, step_x * stride + shift, step=stride)
    grid_y = F.arange(shift, step_y * stride + shift, step=stride)
    grids_x, grids_y = meshgrid(grid_y, grid_x)
    return grids_x.reshape(-1), grids_y.reshape(-1)


class BaseAnchorGenerator(metaclass=ABCMeta):
    """base class for anchor generator.
    """

    def __init__(self):
        pass

    @abstractmethod
    def generate_anchors_by_features(self, sizes) -> List[Tensor]:
        pass

    def __call__(self, featmaps):
        feat_sizes = [fmap.shape[-2:] for fmap in featmaps]
        return self.generate_anchors_by_features(feat_sizes)

    @property
    def anchor_dim(self):
        return 4


class DefaultAnchorGenerator(BaseAnchorGenerator):
    """default retinanet anchor generator.
    This class generate anchors by feature map in level.
    Args:
        base_size (int): anchor base size.
        anchor_scales (np.ndarray): anchor scales based on stride.
            The practical anchor scale is anchor_scale * stride
        anchor_ratios(np.ndarray): anchor aspect ratios.
        offset (float): center point offset.default is 0.
    """

    def __init__(
        self,
        anchor_scales: list = [[32], [64], [128], [256], [512]],
        anchor_ratios: list = [[0.5, 1, 2]],
        strides: list = [4, 8, 16, 32, 64],
        offset: float = 0,
    ):
        super().__init__()
        self.anchor_scales = np.array(anchor_scales, dtype=np.float32)
        self.anchor_ratios = np.array(anchor_ratios, dtype=np.float32)
        self.strides = strides
        self.offset = offset
        self.num_features = len(strides)

        self.base_anchors = self._different_level_anchors(anchor_scales, anchor_ratios)

    def _different_level_anchors(self, scales, ratios):
        if len(scales) == 1:
            scales *= self.num_features
        assert len(scales) == self.num_features

        if len(ratios) == 1:
            ratios *= self.num_features
        assert len(ratios) == self.num_features
        return [
            tensor(self.generate_base_anchors(scale, ratio))
            # self.generate_base_anchors(scale, ratio)
            for scale, ratio in zip(scales, ratios)
        ]

    def generate_base_anchors(self, scales, ratios):
        base_anchors = []
        areas = [s ** 2.0 for s in scales]
        for area in areas:
            for ratio in ratios:
                w = math.sqrt(area / ratio)
                h = ratio * w
                # center-based anchor
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                base_anchors.append([x0, y0, x1, y1])
        return base_anchors

    def generate_anchors_by_features(self, sizes):
        all_anchors = []
        assert len(sizes) == self.num_features, (
            "input features expected {}, got {}".format(self.num_features, len(sizes))
        )
        for size, stride, base_anchor in zip(sizes, self.strides, self.base_anchors):
            grid_x, grid_y = create_anchor_grid(size, self.offset, stride)
            # FIXME: If F.stack works, change to stack
            grid_x, grid_y = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)
            grids = F.concat([grid_x, grid_y, grid_x, grid_y], axis=1)
            all_anchors.append(
                (grids.reshape(-1, 1, 4) + base_anchor.reshape(1, -1, 4)).reshape(-1, 4)
            )
        return all_anchors
