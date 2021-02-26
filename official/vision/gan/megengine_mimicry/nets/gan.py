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
Implementation of Base GAN models.
"""
import megengine
import megengine.functional as F
import megengine.module as M
import megengine.random as R
import numpy as np

from . import losses
from .basemodel import BaseModel


class BaseGenerator(BaseModel):
    r"""
    Base class for a generic unconditional generator model.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz, ngf, bottom_width, loss_type, **kwargs):
        super().__init__(**kwargs)
        self.nz = nz
        self.ngf = ngf
        self.bottom_width = bottom_width
        self.loss_type = loss_type

    def _train_step_implementation(
        self,
        real_batch,
        netD=None,
        optG=None):
        # Produce fake images
        fake_images = self._infer_step_implementation(real_batch)

        # Compute output logit of D thinking image real
        output = netD(fake_images)

        # Compute loss
        errG = self.compute_gan_loss(output=output)

        optG.zero_grad()
        optG.backward(errG)
        optG.step()
        return errG

    def _infer_step_implementation(self, batch):
        # Get only batch size from real batch
        batch_size = batch.shape[0]

        noise = R.gaussian(shape=[batch_size, self.nz])

        fake_images = self.forward(noise)
        return fake_images

    def compute_gan_loss(self, output):
        if self.loss_type == "ns":
            errG = losses.ns_loss_gen(output)

        elif self.loss_type == "wasserstein":
            errG = losses.wasserstein_loss_gen(output)

        else:
            raise ValueError("Invalid loss_type {} selected.".format(
                self.loss_type))

        return errG

    def generate_images(self, num_images):
        """Generate images of shape [`num_images`, C, H, W].

        Depending on the final activation function, pixel values are NOT guarenteed
        to be within [0, 1].
        """
        return self.infer_step(np.empty(num_images, dtype="float32"))


class BaseDiscriminator(BaseModel):
    r"""
    Base class for a generic unconditional discriminator model.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, ndf, loss_type, **kwargs):
        super().__init__(**kwargs)
        self.ndf = ndf
        self.loss_type = loss_type

    def _train_step_implementation(
        self,
        real_batch,
        netG=None,
        optD=None):
        # Produce logits for real images
        output_real = self._infer_step_implementation(real_batch)

        # Produce fake images
        fake_images = netG._infer_step_implementation(real_batch)
        fake_images = F.zero_grad(fake_images)

        # Produce logits for fake images
        output_fake = self._infer_step_implementation(fake_images)

        # Compute loss for D
        errD = self.compute_gan_loss(output_real=output_real,
                                     output_fake=output_fake)
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        # Backprop and update gradients
        optD.zero_grad()
        optD.backward(errD)
        optD.step()
        return errD, D_x, D_Gz

    def _infer_step_implementation(self, batch):
        return self.forward(batch)

    def compute_gan_loss(self, output_real, output_fake):
        r"""
        Computes GAN loss for discriminator.

        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real images.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake images.

        Returns:
            errD (Tensor): A batch of GAN losses for the discriminator.
        """
        # Compute loss for D
        if self.loss_type == "gan" or self.loss_type == "ns":
            errD = losses.minimax_loss_dis(output_fake=output_fake,
                                           output_real=output_real)

        elif self.loss_type == "wasserstein":
            errD = losses.wasserstein_loss_dis(output_fake=output_fake,
                                               output_real=output_real)

        else:
            raise ValueError("Invalid loss_type selected.")

        return errD

    def compute_probs(self, output_real, output_fake):
        r"""
        Computes probabilities from real/fake images logits.

        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real images.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake images.

        Returns:
            tuple: Average probabilities of real/fake image considered as real for the batch.
        """
        D_x = F.sigmoid(output_real).mean()
        D_Gz = F.sigmoid(output_fake).mean()

        return D_x, D_Gz
