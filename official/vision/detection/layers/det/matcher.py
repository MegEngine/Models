# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine.functional as F


class Matcher:

    def __init__(self, thresholds, labels, allow_low_quality_matches=False):
        assert len(thresholds) + 1 == len(labels), "thresholds and labels are not matched"
        assert all(low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:]))
        thresholds.append(float("inf"))
        thresholds.insert(0, -float("inf"))

        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, matrix):
        """
        matrix(tensor): A two dim tensor with shape of (N, M). N is number of GT-boxes,
            while M is the number of anchors in detection.
        """
        assert len(matrix.shape) == 2
        max_scores = matrix.max(axis=0)
        match_indices = F.argmax(matrix, axis=0)

        # default ignore label: -1
        labels = F.full_like(match_indices, -1)

        for label, low, high in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            mask = (max_scores >= low) & (max_scores < high)
            labels[mask] = label

        if self.allow_low_quality_matches:
            mask = (matrix == F.max(matrix, axis=1, keepdims=True)).sum(axis=0) > 0
            labels[mask] = 1

        return match_indices, labels
