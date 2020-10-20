# -*- coding:utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
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
    num_class = mask.sum()
    if num_class <= num_samples:
        return labels

    topk_tensor = F.zeros_like(labels).astype("float32")
    topk_tensor[mask] = uniform(size=num_class)
    _, select_inds = F.topk(topk_tensor, k=num_samples - num_class)

    labels[select_inds] = ignore_label
    return labels


def sample_mask_from_labels(labels, num_sample, sample_value):
    """generate mask for labels using sampling method.

    Args:
        labels (Tensor):
        num_sample (int):
        sample_value (int):

    Returns:
        sample_mask (Tensor)
    """
    assert labels.ndim == 1, "Only tensor of dim 1 is supported."
    # TODO: support bool mask
    sample_mask = (labels == sample_value).astype("float32")
    num_mask = sample_mask.sum().astype("int32")
    if num_mask <= num_sample:
        return sample_mask

    random_tensor = sample_mask * uniform(size=labels.shape)
    _, sampled_idx = F.topk(random_tensor, k=num_sample - num_mask)
    sample_mask[sampled_idx] = F.zeros(sampled_idx.shape)

    return sample_mask
