# -*- coding:utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine.functional as F
from megengine.random import uniform


def sample_labels(labels, num_samples, label_value, ignore_label=-1):
    """sample N labels with label value = sample_labels

    Args:
        labels(Tensor): shape of label is (N,)
        num_samples(int):
        label_value(int):

    Returns:
        label(Tensor): label after sampling
    """
    assert labels.ndim == 1, "Only tensor of dim 1 is supported."
    mask = (labels == label_value)
    num_valid = mask.sum()
    if num_valid <= num_samples:
        return labels

    random_tensor = F.zeros_like(labels).astype("float32")
    random_tensor[mask] = uniform(size=num_valid)
    _, invalid_inds = F.topk(random_tensor, k=num_samples - num_valid)

    labels[invalid_inds] = ignore_label
    return labels
