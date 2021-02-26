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
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2021 Megvii Inc. All rights reserved.
# ---------------------------------------------------------------------
from functools import partial

import megengine.module as M
from megengine.module.normalization import GroupNorm, InstanceNorm, LayerNorm


def get_norm(norm):
    """
    Args:
        norm (str): currently support "BN", "SyncBN", "FrozenBN", "GN", "LN" and "IN"

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
        "LN": LayerNorm,
        "IN": InstanceNorm,
    }[norm]
    return norm
