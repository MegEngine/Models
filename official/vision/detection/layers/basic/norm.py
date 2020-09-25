# -*- coding: utf-8 -*-
# Copyright 2019 - present, Facebook, Inc
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
# All Megvii Modifications are Copyright (C) 2014-2020 Megvii Inc. All rights reserved.
# ---------------------------------------------------------------------
import numpy as np

import megengine.functional as F
import megengine.module as M
from megengine import Parameter


class FrozenBatchNorm2d(M.Module):
    """
    BatchNorm2d, which the weight, bias, running_mean, running_var
    are immutable.
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.weight = Parameter(np.ones(num_features, dtype=np.float32))
        self.bias = Parameter(np.zeros(num_features, dtype=np.float32))

        self.running_mean = Parameter(np.zeros((1, num_features, 1, 1), dtype=np.float32))
        self.running_var = Parameter(np.ones((1, num_features, 1, 1), dtype=np.float32))

    def forward(self, x):
        scale = self.weight.reshape(1, -1, 1, 1) * (
            1.0 / F.sqrt(self.running_var + self.eps)
        )
        bias = self.bias.reshape(1, -1, 1, 1) - self.running_mean * scale
        return x * scale.detach() + bias.detach()


class GroupNorm(M.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(np.ones(num_channels, dtype=np.float32))
            self.bias = Parameter(np.zeros(num_channels, dtype=np.float32))
        else:
            self.weight = None
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            M.init.ones_(self.weight)
            M.init.zeros_(self.bias)

    def forward(self, x):
        output = x.reshape(x.shape[0], self.num_groups, -1)
        mean = F.mean(output, axis=2, keepdims=True)
        mean2 = F.mean(output ** 2, axis=2, keepdims=True)
        var = mean2 - mean * mean

        output = (output - mean) / F.sqrt(var + self.eps)
        output = output.reshape(x.shape)
        if self.affine:
            output = self.weight.reshape(1, -1, 1, 1) * output + \
                self.bias.reshape(1, -1, 1, 1)

        return output


def get_norm(norm):
    """
    Args:
        norm (str): currently support "BN", "SyncBN", "FrozenBN" and "GN"

    Returns:
        M.Module or None: the normalization layer
    """
    if norm is None:
        return None
    norm = {
        "BN": M.BatchNorm2d,
        "SyncBN": M.SyncBatchNorm,
        "FrozenBN": FrozenBatchNorm2d,
        "GN": GroupNorm,
    }[norm]
    return norm
