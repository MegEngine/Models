# Copyright (c) 2020 Kwot Sin Lee
# This code is licensed under MIT license
# (https://github.com/kwotsin/mimicry/blob/master/LICENSE)
# ------------------------------------------------------------------------------
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
# ------------------------------------------------------------------------------
"""
Loads randomly sampled images from datasets for computing metrics.
"""
import os

import numpy as np
import megengine.data.transform as T

from . import data_utils


def get_random_images(dataset, num_samples):
    """
    Randomly sample without replacement num_samples images.

    Args:
        dataset (Dataset): Torch Dataset object for indexing elements.
        num_samples (int): The number of images to randomly sample.

    Returns:
        Tensor: Batch of num_samples images in np array form [N, H, W, C](0-255).
    """
    choices = np.random.choice(range(len(dataset)),
                               size=num_samples,
                               replace=False)

    images = []
    for choice in choices:
        img = np.array(dataset[choice][0])
        img = np.expand_dims(img, axis=0)
        images.append(img)
    images = np.concatenate(images, axis=0)

    return images


def get_cifar10_images(num_samples, root=None, **kwargs):
    """
    Loads randomly sampled CIFAR-10 training images.

    Args:
        num_samples (int): The number of images to randomly sample.
        root (str): The root directory where all datasets are stored.

    Returns:
        Tensor: Batch of num_samples images in np array form.
    """
    dataset = data_utils.load_cifar10_dataset(root=root, **kwargs)

    images = get_random_images(dataset, num_samples)

    return images


def get_dataset_images(dataset_name, num_samples=50000, **kwargs):
    """
    Randomly sample num_samples images based on input dataset name.

    Args:
        dataset_name (str): Dataset name to load images from.
        num_samples (int): The number of images to randomly sample.

    Returns:
        Tensor: Batch of num_samples images from the specific dataset in np array form.
    """
    if dataset_name == "cifar10":
        images = get_cifar10_images(num_samples, **kwargs)

    elif dataset_name == "cifar10_test":
        images = get_cifar10_images(num_samples, split='test', **kwargs)

    else:
        raise ValueError("Invalid dataset name {}.".format(dataset_name))

    # Check shape and permute if needed
    if images.shape[1] == 3:
        images = images.transpose((0, 2, 3, 1))

    # Ensure the values lie within the correct range, otherwise there might be some
    # preprocessing error from the library causing ill-valued scores.
    if np.min(images) < 0 or np.max(images) > 255:
        raise ValueError(
            'Image pixel values must lie between 0 to 255 inclusive.')

    return images
