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
from functools import partial

import numpy as np

import megengine.functional as F
import megengine.module as M
from megengine import Parameter


class GroupNorm(M.Module):
    def __init__(self, num_groups, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(np.ones(num_features, dtype=np.float32))
            self.bias = Parameter(np.zeros(num_features, dtype=np.float32))
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
        var = F.mean(output * output, axis=2, keepdims=True) - mean * mean

        output = (output - mean) / F.sqrt(var + self.eps)
        output = output.reshape(x.shape)
        if self.affine:
            output = self.weight.reshape(1, -1, 1, 1) * output + \
                self.bias.reshape(1, -1, 1, 1)

        return output

    def _module_info_string(self) -> str:
        s = "{num_groups}, {num_features}, eps={eps}, affine={affine}"
        return s.format(**self.__dict__)


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
        "FrozenBN": partial(M.BatchNorm2d, freeze=True),
        "GN": GroupNorm,
    }[norm]
    return norm
