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
import megengine.functional as F
from megengine.core.tensor_factory import zeros


def ns_loss_gen(output_fake):
    r"""
    Non-saturating loss for generator.

    Args:
        output_fake (Tensor): Discriminator output logits for fake images.

    Returns:
        Tensor: A scalar tensor loss output.
    """
    output_fake = F.sigmoid(output_fake)

    return -F.log(output_fake + 1e-8).mean()


# def ns_loss_gen(output_fake):
#     """numerical stable version"""
#     return F.log(1 + F.exp(-output_fake)).mean()


def _bce_loss_with_logits(output, labels, **kwargs):
    r"""
    Sigmoid cross entropy with logits, see tensorflow
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    """
    loss = F.maximum(output, 0) - output * labels + F.log(1 + F.exp(-F.abs(output)))
    return loss.mean()


def minimax_loss_dis(output_fake,
                     output_real,
                     real_label_val=1.0,
                     fake_label_val=0.0,
                     **kwargs):
    r"""
    Standard minimax loss for GANs through the BCE Loss with logits fn.

    Args:
        output_fake (Tensor): Discriminator output logits for fake images.
        output_real (Tensor): Discriminator output logits for real images.
        real_label_val (int): Label for real images.
        fake_label_val (int): Label for fake images.
        device (torch.device): Torch device object for sending created data.

    Returns:
        Tensor: A scalar tensor loss output.
    """
    # Produce real and fake labels.
    fake_labels = zeros((output_fake.shape[0], 1)) + fake_label_val
    real_labels = zeros((output_real.shape[0], 1)) + real_label_val

    # FF, compute loss and backprop D
    errD_fake = _bce_loss_with_logits(output=output_fake,
                                      labels=fake_labels,
                                      **kwargs)

    errD_real = _bce_loss_with_logits(output=output_real,
                                      labels=real_labels,
                                      **kwargs)

    # Compute cumulative error
    loss = errD_real + errD_fake

    return loss


def wasserstein_loss_gen(output_fake):
    r"""
    Computes the wasserstein loss for generator.

    Args:
        output_fake (Tensor): Discriminator output logits for fake images.

    Returns:
        Tensor: A scalar tensor loss output.
    """
    loss = -output_fake.mean()

    return loss


def wasserstein_loss_dis(output_real, output_fake):
    r"""
    Computes the wasserstein loss for the discriminator.

    Args:
        output_real (Tensor): Discriminator output logits for real images.
        output_fake (Tensor): Discriminator output logits for fake images.

    Returns:
        Tensor: A scalar tensor loss output.
    """
    loss = -1.0 * output_real.mean() + output_fake.mean()

    return loss
