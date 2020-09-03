# -*- coding:utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import math
import numpy as np

import megengine.functional as F


def roi_pool(
    rpn_fms, rois, stride, pool_shape, roi_type="roi_align",
):
    rois = rois.detach()
    assert len(stride) == len(rpn_fms)
    canonical_level = 4
    canonical_box_size = 224
    min_level = int(math.log2(stride[0]))
    max_level = int(math.log2(stride[-1]))

    num_fms = len(rpn_fms)
    box_area = (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])
    level_assignments = F.floor(
        canonical_level + F.log(F.sqrt(box_area) / canonical_box_size) / np.log(2)
    ).astype("int32")
    level_assignments = F.minimum(level_assignments, max_level)
    level_assignments = F.maximum(level_assignments, min_level)
    level_assignments = level_assignments - min_level

    # avoid empty assignment
    level_assignments = F.concat(
        [level_assignments, F.arange(0, num_fms, dtype="int32", device=level_assignments.device)],
    )
    rois = F.concat([rois, F.zeros((num_fms, rois.shape[-1]))])

    pool_list, inds_list = [], []
    for i in range(num_fms):
        _, inds = F.cond_take(level_assignments == i, level_assignments)
        level_rois = rois[inds]

        if roi_type == "roi_pool":
            pool_fm = F.roi_pooling(
                rpn_fms[i], level_rois, pool_shape, mode="max", scale=1.0 / stride[i]
            )
        elif roi_type == "roi_align":
            pool_fm = F.roi_align(
                rpn_fms[i],
                level_rois,
                pool_shape,
                mode="average",
                spatial_scale=1.0 / stride[i],
                sample_points=2,
                aligned=True,
            )
        pool_list.append(pool_fm)
        inds_list.append(inds)

    fm_order = F.argsort(F.concat(inds_list, axis=0))
    pool_feature = F.concat(pool_list, axis=0)
    pool_feature = pool_feature[fm_order][:-num_fms]

    return pool_feature
