# Copyright (c) 2020 Kwot Sin Lee
# This code is licensed under MIT license
# (https://github.com/kwotsin/mimicry/blob/master/LICENSE)
# ------------------------------------------------------------------------------
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
# ------------------------------------------------------------------------------
"""
MegEngine interface for computing Inception Score.
"""
import os
import random
import time

import numpy as np

from .inception_model import inception_utils
from .inception_score import inception_score_utils as tf_inception_score
from .utils import _normalize_images


def inception_score(netG,
                    device,
                    num_samples,
                    batch_size=50,
                    splits=10,
                    log_dir='./log',
                    seed=0,
                    print_every=20):
    """
    Computes the inception score of generated images.

    Args:
        netG (Module): The generator model to use for generating images.
        device (Device): Torch device object to send model and data to.
        num_samples (int): The number of samples to generate.
        batch_size (int): Batch size per feedforward step for inception model.
        splits (int): The number of splits to use for computing IS.
        log_dir (str): Path to store metric computation objects.
        seed (int): Random seed for generation.
    Returns:
        Mean and standard deviation of the inception score computed from using
        num_samples generated images.
    """
    # Make sure the random seeds are fixed
    random.seed(seed)
    np.random.seed(seed)

    # Build inception
    inception_path = os.path.join(log_dir, 'metrics/inception_model')
    inception_utils.create_inception_graph(inception_path)

    # Inference variables
    batch_size = min(batch_size, num_samples)
    num_batches = num_samples // batch_size

    # Get images
    images = []
    start_time = time.time()
    for idx in range(num_batches):
        fake_images = netG.generate_images(num_images=batch_size).numpy()

        fake_images = _normalize_images(fake_images)  # NCHW(BGR) -> NHWC(RGB)
        images.append(fake_images)

        if (idx + 1) % min(print_every, num_batches) == 0:
            end_time = time.time()
            print(
                "INFO: Generated image {}/{} [Random Seed {}] ({:.4f} sec/idx)"
                .format(
                    (idx + 1) * batch_size, num_samples, seed,
                    (end_time - start_time) / (print_every * batch_size)))
            start_time = end_time

    images = np.concatenate(images, axis=0)

    IS_score = tf_inception_score.get_inception_score(images,
                                                      splits=splits,
                                                      device=device)
    print("INFO: IS Score: {}".format(IS_score))
    return IS_score
