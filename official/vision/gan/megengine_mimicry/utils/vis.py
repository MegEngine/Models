# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import math

import cv2
import megengine


def normalize_image(tensor: megengine.Tensor, scale=255):
    """normalize image tensors of any range to [0, scale=255]"""
    mi = tensor.min()
    ma = tensor.max()
    tensor = scale * (tensor - mi) / (ma - mi + 1e-9)
    return tensor


def make_grid(
        tensor: megengine.Tensor,  # [N,C,H,W]
        nrow: int = 8,
        padding: int = 2,
        background: float = 0,
        normalize: bool = False,
) -> megengine.Tensor:
    """align [N, C, H, W] image tensor to [H, W, 3] image grids, for visualization"""
    if normalize:
        tensor = normalize_image(tensor, scale=255)  # normalize to 0-255 scale

    c = tensor.shape[1]
    assert c in (1, 3), "only support color/grayscale images, got channel = {}".format(c)
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = megengine.ones((num_channels, height * ymaps + padding, width * xmaps + padding), "float32") * background
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = grid.set_subtensor(tensor[k])[:,
                                                 y * height + padding: (y + 1) * height,
                                                 x * width + padding: (x + 1) * width]
            k = k + 1
    c, h, w = grid.shape
    grid = grid.dimshuffle(1, 2, 0)  # [C,H,W] -> [H,W,C]
    grid = grid.broadcast(h, w, 3)   # [H,W,C] -> [H,W,3]
    return grid


def save_image(image, path):
    if isinstance(image, megengine.Tensor):
        image = image.numpy()
    cv2.imwrite(path, image)
