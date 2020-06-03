# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
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
        pick_ind = order[0]
        keep.append(pick_ind)

        xx1 = np.maximum(x1[pick_ind], x1[order[1:]])
        yy1 = np.maximum(y1[pick_ind], y1[order[1:]])
        xx2 = np.minimum(x2[pick_ind], x2[order[1:]])
        yy2 = np.minimum(y2[pick_ind], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[pick_ind] + areas[order[1:]] - inter)

        order = order[np.where(iou <= thresh)[0] + 1]

    return keep
