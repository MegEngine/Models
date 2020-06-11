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
import numpy as np


def _normalize_images(images):
    """
    Given a tensor of (megengine BGR) images, uses the torchvision
    normalization method to convert floating point data to integers. See reference
    at: https://pytorch.org/docs/stable/_modules/torchvision/utils.html#save_image

    The function uses the normalization from make_grid and save_image functions.

    Args:
        images (Tensor): Batch of images of shape (N, 3, H, W).

    Returns:
        ndarray: Batch of normalized (0-255) RGB images of shape (N, H, W, 3).
    """
    # Shift the image from [-1, 1] range to [0, 1] range.
    min_val = float(images.min())
    max_val = float(images.max())

    images = (images - min_val) / (max_val - min_val + 1e-5)

    images = np.clip(images * 255 + 0.5, 0, 255).astype("uint8")

    images = np.transpose(images, [0, 2, 3, 1])

    # NOTE: megengine(opencv) uses BGR, while TF uses RGB. Needs conversion.
    images = images[:, :, :, ::-1]

    return images
