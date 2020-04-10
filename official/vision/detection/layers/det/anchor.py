# -*- coding: utf-8 -*-
# Copyright 2018-2019 Open-MMLab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
# ---------------------------------------------------------------------
from abc import ABCMeta, abstractmethod

import megengine.functional as F
import numpy as np

from megengine.core import tensor, Tensor


class BaseAnchorGenerator(metaclass=ABCMeta):
    """base class for anchor generator.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_anchors_by_feature(self) -> Tensor:
        pass


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
        base_size=8,
        anchor_scales: np.ndarray = np.array([2, 3, 4]),
        anchor_ratios: np.ndarray = np.array([0.5, 1, 2]),
        offset: float = 0,
    ):
        super().__init__()
        self.base_size = base_size
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.offset = offset

    def _whctrs(self, anchor):
        """convert anchor box into (w, h, ctr_x, ctr_y)
        """
        w = anchor[:, 2] - anchor[:, 0] + 1
        h = anchor[:, 3] - anchor[:, 1] + 1
        x_ctr = anchor[:, 0] + 0.5 * (w - 1)
        y_ctr = anchor[:, 1] + 0.5 * (h - 1)

        return w, h, x_ctr, y_ctr

    def get_plane_anchors(self, anchor_scales: np.ndarray):
        """get anchors per location on feature map.
        The anchor number is anchor_scales x anchor_ratios
        """
        base_anchor = tensor([0, 0, self.base_size - 1, self.base_size - 1])
        base_anchor = F.add_axis(base_anchor, 0)
        w, h, x_ctr, y_ctr = self._whctrs(base_anchor)
        # ratio enumerate
        size = w * h
        size_ratios = size / self.anchor_ratios

        ws = size_ratios.sqrt().round()
        hs = (ws * self.anchor_ratios).round()

        # scale enumerate
        anchor_scales = anchor_scales[None, ...]
        ws = F.add_axis(ws, 1)
        hs = F.add_axis(hs, 1)
        ws = (ws * anchor_scales).reshape(-1, 1)
        hs = (hs * anchor_scales).reshape(-1, 1)

        anchors = F.concat(
            [
                x_ctr - 0.5 * (ws - 1),
                y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1),
                y_ctr + 0.5 * (hs - 1),
            ],
            axis=1,
        )

        return anchors.astype(np.float32)

    def get_center_offsets(self, featmap, stride):
        f_shp = featmap.shape
        fm_height, fm_width = f_shp[-2], f_shp[-1]

        shift_x = F.linspace(0, fm_width - 1, fm_width) * stride
        shift_y = F.linspace(0, fm_height - 1, fm_height) * stride

        # make the mesh grid of shift_x and shift_y
        mesh_shape = (fm_height, fm_width)
        broad_shift_x = shift_x.reshape(-1, shift_x.shape[0]).broadcast(*mesh_shape)
        broad_shift_y = shift_y.reshape(shift_y.shape[0], -1).broadcast(*mesh_shape)

        flatten_shift_x = F.add_axis(broad_shift_x.reshape(-1), 1)
        flatten_shift_y = F.add_axis(broad_shift_y.reshape(-1), 1)

        centers = F.concat(
            [flatten_shift_x, flatten_shift_y, flatten_shift_x, flatten_shift_y, ],
            axis=1,
        )
        if self.offset > 0:
            centers = centers + self.offset * stride
        return centers

    def get_anchors_by_feature(self, featmap, stride):
        # shifts shape: [A, 4]
        shifts = self.get_center_offsets(featmap, stride)
        # plane_anchors shape: [B, 4], e.g. B=9
        plane_anchors = self.get_plane_anchors(self.anchor_scales * stride)

        all_anchors = F.add_axis(plane_anchors, 0) + F.add_axis(shifts, 1)
        all_anchors = all_anchors.reshape(-1, 4)

        return all_anchors

    def __call__(self, featmap, stride):
        return self.get_anchors_by_feature(featmap, stride)
