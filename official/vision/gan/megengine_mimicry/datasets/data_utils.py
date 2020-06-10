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
Script for loading datasets.
"""
import os

import megengine.data as data
import megengine.data.transform as T


def load_dataset(root, name, **kwargs):
    """
    Loads different datasets specifically for GAN training.
    By default, all images are normalized to values in the range [-1, 1].

    Args:
        root (str): Path to where datasets are stored.
        name (str): Name of dataset to load.

    Returns:
        Dataset: Torch Dataset object for a specific dataset.
    """
    if name == "cifar10":
        return load_cifar10_dataset(root, **kwargs)

    else:
        raise ValueError("Invalid dataset name {} selected.".format(name))


def load_cifar10_dataset(root=None,
                         split='train',
                         download=True,
                         **kwargs):
    """
    Loads the CIFAR-10 dataset.

    Args:
        root (str): Path to where datasets are stored.
        split (str): The split of data to use.
        download (bool): If True, downloads the dataset.

    Returns:
        Dataset: Torch Dataset object.
    """
    dataset_dir = root
    if dataset_dir and not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Build datasets
    if split == "train":
        dataset = data.dataset.CIFAR10(root=dataset_dir,
                                       train=True,
                                       download=download,
                                       **kwargs)
    elif split == "test":
        dataset = data.dataset.CIFAR10(root=dataset_dir,
                                       train=False,
                                       download=download,
                                       **kwargs)
    else:
        raise ValueError("split argument must one of ['train', 'val']")

    return dataset
