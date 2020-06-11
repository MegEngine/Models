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
MegEngine interface for computing FID.
"""
import os
import random
import time

import numpy as np
import tensorflow as tf

from ..datasets.image_loader import get_dataset_images
from .fid import fid_utils
from .inception_model import inception_utils
from .utils import _normalize_images


def compute_real_dist_stats(num_samples,
                            sess,
                            dataset_name,
                            batch_size,
                            stats_file=None,
                            seed=0,
                            verbose=True,
                            log_dir='./log'):
    """
    Reads the image data and compute the FID mean and cov statistics
    for real images.

    Args:
        num_samples (int): Number of real images to compute statistics.
        sess (Session): TensorFlow session to use.
        dataset_name (str): The name of the dataset to load.
        batch_size (int): The batch size to feedforward for inference.
        stats_file (str): The statistics file to load from if there is already one.
        verbose (bool): If True, prints progress of computation.
        log_dir (str): Directory where feature statistics can be stored.

    Returns:
        ndarray: Mean features stored as np array.
        ndarray: Covariance of features stored as np array.
    """
    # Create custom stats file name
    if stats_file is None:
        stats_dir = os.path.join(log_dir, 'metrics', 'fid', 'statistics')
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

        stats_file = os.path.join(
            stats_dir,
            "fid_stats_{}_{}k_run_{}.npz".format(dataset_name,
                                                 num_samples // 1000, seed))

    if stats_file and os.path.exists(stats_file):
        print("INFO: Loading existing statistics for real images...")
        f = np.load(stats_file)
        m_real, s_real = f['mu'][:], f['sigma'][:]
        f.close()

    else:
        # Obtain the numpy format data
        print("INFO: Obtaining images...")
        images = get_dataset_images(dataset_name, num_samples=num_samples)
        images = images[:, :, :, ::-1]  # NOTE: opencv image convert to RGB

        # Compute the mean and cov
        print("INFO: Computing statistics for real images...")
        m_real, s_real = fid_utils.calculate_activation_statistics(
            images=images, sess=sess, batch_size=batch_size, verbose=verbose)

        if not os.path.exists(stats_file):
            print("INFO: Saving statistics for real images...")
            np.savez(stats_file, mu=m_real, sigma=s_real)

    return m_real, s_real


def compute_gen_dist_stats(netG,
                           num_samples,
                           sess,
                           device,
                           seed,
                           batch_size,
                           print_every=20,
                           verbose=True):
    """
    Directly produces the images and convert them into numpy format without
    saving the images on disk.

    Args:
        netG (Module): Torch Module object representing the generator model.
        num_samples (int): The number of fake images for computing statistics.
        sess (Session): TensorFlow session to use.
        device (str): Device identifier to use for computation.
        seed (int): The random seed to use.
        batch_size (int): The number of samples per batch for inference.
        print_every (int): Interval for printing log.
        verbose (bool): If True, prints progress.

    Returns:
        ndarray: Mean features stored as np array.
        ndarray: Covariance of features stored as np array.
    """
    # Set model to evaluation mode
    netG.eval() # NOTE: in MegEngine this may has no effect

    # Inference variables
    batch_size = min(num_samples, batch_size)

    # Collect all samples()
    images = []
    start_time = time.time()
    for idx in range(num_samples // batch_size):
        # Collect fake image
        fake_images = netG.generate_images(num_images=batch_size).numpy()
        images.append(fake_images)

        # Print some statistics
        if (idx + 1) % print_every == 0:
            end_time = time.time()
            print(
                "INFO: Generated image {}/{} [Random Seed {}] ({:.4f} sec/idx)"
                .format(
                    (idx + 1) * batch_size, num_samples, seed,
                    (end_time - start_time) / (print_every * batch_size)))
            start_time = end_time

    # Produce images in the required (N, H, W, 3) format for FID computation
    images = np.concatenate(images, 0)  # Gives (N, 3, H, W) BGR
    images = _normalize_images(images)  # Gives (N, H, W, 3) RGB

    # Compute the FID
    print("INFO: Computing statistics for fake images...")
    m_fake, s_fake = fid_utils.calculate_activation_statistics(
        images=images, sess=sess, batch_size=batch_size, verbose=verbose)

    return m_fake, s_fake


def fid_score(num_real_samples,
              num_fake_samples,
              netG,
              device,
              seed,
              dataset_name,
              batch_size=50,
              verbose=True,
              stats_file=None,
              log_dir='./log'):
    """
    Computes FID stats using functions that store images in memory for speed and fidelity.
    Fidelity since by storing images in memory, we don't subject the scores to different read/write
    implementations of imaging libraries.

    Args:
        num_real_samples (int): The number of real images to use for FID.
        num_fake_samples (int): The number of fake images to use for FID.
        netG (Module): Torch Module object representing the generator model.
        device (str): Device identifier to use for computation.
        seed (int): The random seed to use.
        dataset_name (str): The name of the dataset to load.
        batch_size (int): The batch size to feedforward for inference.
        verbose (bool): If True, prints progress.
        stats_file (str): The statistics file to load from if there is already one.
        log_dir (str): Directory where feature statistics can be stored.

    Returns:
        float: Scalar FID score.
    """
    start_time = time.time()

    # Make sure the random seeds are fixed
    # torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Setup directories
    inception_path = os.path.join(log_dir, 'metrics', 'inception_model')

    # Setup the inception graph
    inception_utils.create_inception_graph(inception_path)

    # Start producing statistics for real and fake images
    if device is not None:
        # Avoid unbounded memory usage
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True,
                                    per_process_gpu_memory_fraction=0.15,
                                    visible_device_list=str(device))
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

    else:
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        m_real, s_real = compute_real_dist_stats(num_samples=num_real_samples,
                                                 sess=sess,
                                                 dataset_name=dataset_name,
                                                 batch_size=batch_size,
                                                 verbose=verbose,
                                                 stats_file=stats_file,
                                                 log_dir=log_dir,
                                                 seed=seed)

        m_fake, s_fake = compute_gen_dist_stats(netG=netG,
                                                num_samples=num_fake_samples,
                                                sess=sess,
                                                device=device,
                                                seed=seed,
                                                batch_size=batch_size,
                                                verbose=verbose)

        FID_score = fid_utils.calculate_frechet_distance(mu1=m_real,
                                                         sigma1=s_real,
                                                         mu2=m_fake,
                                                         sigma2=s_fake)

        print("INFO: FID Score: {} [Time Taken: {:.4f} secs]".format(
            FID_score,
            time.time() - start_time))

        return float(FID_score)
