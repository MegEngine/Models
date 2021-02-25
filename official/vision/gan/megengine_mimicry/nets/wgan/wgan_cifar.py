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
import megengine.module as M

from ..blocks import GBlock
from . import wgan_base
from .wgan_base import WGANDBlockOptimized as DBlockOptimized
from .wgan_base import WGANDBlockWithLayerNorm as DBlock


class WGANGeneratorCIFAR(wgan_base.WGANBaseGenerator):
    r"""
    ResNet backbone generator for ResNet WGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz=128, ngf=256, bottom_width=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        # Build the layers
        self.l1 = M.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block3 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block4 = GBlock(self.ngf, self.ngf, upsample=True)
        self.b5 = M.BatchNorm2d(self.ngf)
        self.c5 = M.Conv2d(self.ngf, 3, 3, 1, padding=1)
        self.activation = M.ReLU()

        # Initialise the weights
        M.init.xavier_uniform_(self.l1.weight, 1.0)
        M.init.xavier_uniform_(self.c5.weight, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.

        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).

        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        h = self.l1(x)
        h = h.reshape(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = F.tanh(self.c5(h))

        return h


class WGANDiscriminatorCIFAR(wgan_base.WGANBaseDiscriminator):
    r"""
    ResNet backbone discriminator for ResNet WGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, ndf=128, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        # Build layers
        self.block1 = DBlockOptimized(3, self.ndf)
        self.block2 = DBlock(self.ndf,
                             self.ndf,
                             downsample=True)
        self.block3 = DBlock(self.ndf,
                             self.ndf,
                             downsample=False)
        self.block4 = DBlock(self.ndf,
                             self.ndf,
                             downsample=False)
        self.l5 = M.Linear(self.ndf, 1)
        self.activation = M.ReLU()

        # Initialise the weights
        M.init.xavier_uniform_(self.l5.weight, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)

        # Global average pooling
        h = h.mean(3).mean(2)

        output = self.l5(h)

        return output
