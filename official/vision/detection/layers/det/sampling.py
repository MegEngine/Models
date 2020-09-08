#!/usr/bin/python3
# -*- coding:utf-8 -*-

from megengine.random import uniform
import megengine.functional as F


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
    _, mask_inds = F.cond_take(labels == label_value, labels)
    num_class = mask_inds.shape[0]
    if num_class <= num_samples:
        return labels

    topk_tensor = F.zeros_like(labels).astype("float32")
    topk_tensor[mask_inds] = uniform(num_class).astype("float32")
    _, mask_inds = F.topk(topk_tensor, k=num_samples-num_class)

    labels[mask_inds] = ignore_label
    return labels
