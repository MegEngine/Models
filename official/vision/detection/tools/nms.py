# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np


def py_cpu_nms(dets, thresh):
    x1 = np.ascontiguousarray(dets[:, 0])
    y1 = np.ascontiguousarray(dets[:, 1])
    x2 = np.ascontiguousarray(dets[:, 2])
    y2 = np.ascontiguousarray(dets[:, 3])

    areas = (x2 - x1) * (y2 - y1)
    order = dets[:, 4].argsort()[::-1]
    keep = list()

    while order.size > 0:
        pick_idx = order[0]
        keep.append(pick_idx)
        order = order[1:]

        xx1 = np.maximum(x1[pick_idx], x1[order])
        yy1 = np.maximum(y1[pick_idx], y1[order])
        xx2 = np.minimum(x2[pick_idx], x2[order])
        yy2 = np.minimum(y2[pick_idx], y2[order])

        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / np.maximum(areas[pick_idx] + areas[order] - inter, 1e-5)

        order = order[iou <= thresh]

    return keep
