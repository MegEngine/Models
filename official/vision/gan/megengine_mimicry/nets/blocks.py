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
import math

import megengine.functional as F
import megengine.module as M


class GBlock(M.Module):
    r"""
    Residual block for generator.

    Uses bilinear (rather than nearest) interpolation, and align_corners
    set to False. This is as per how torchvision does upsampling, as seen in:
    https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        upsample (bool): If True, upsamples the input feature map.
        num_classes (int): If more than 0, uses conditional batch norm instead.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 upsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.learnable_sc = in_channels != out_channels or upsample
        self.upsample = upsample

        self.c1 = M.Conv2d(self.in_channels,
                           self.hidden_channels,
                           3,
                           1,
                           padding=1)
        self.c2 = M.Conv2d(self.hidden_channels,
                           self.out_channels,
                           3,
                           1,
                           padding=1)

        self.b1 = M.BatchNorm2d(self.in_channels)
        self.b2 = M.BatchNorm2d(self.hidden_channels)

        self.activation = M.ReLU()

        M.init.xavier_uniform_(self.c1.weight, math.sqrt(2.0))
        M.init.xavier_uniform_(self.c2.weight, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_sc:
            self.c_sc = M.Conv2d(in_channels,
                                 out_channels,
                                 1,
                                 1,
                                 padding=0)
            M.init.xavier_uniform_(self.c_sc.weight, 1.0)

    def _upsample_conv(self, x, conv):
        r"""
        Helper function for performing convolution after upsampling.
        """
        return conv(
            F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=False))

    def _residual(self, x):
        r"""
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)

        return h

    def _shortcut(self, x):
        r"""
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self._upsample_conv(
                x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x):
        r"""
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)


class DBlock(M.Module):
    """
    Residual block for discriminator.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        downsample (bool): If True, downsamples the input feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 downsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample

        # Build the layers
        self.c1 = M.Conv2d(self.in_channels, self.hidden_channels, 3, 1,
                           1)
        self.c2 = M.Conv2d(self.hidden_channels, self.out_channels, 3, 1,
                           1)

        self.activation = M.ReLU()

        M.init.xavier_uniform_(self.c1.weight, math.sqrt(2.0))
        M.init.xavier_uniform_(self.c2.weight, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_sc:
            self.c_sc = M.Conv2d(in_channels, out_channels, 1, 1, 0)
            M.init.xavier_uniform_(self.c_sc.weight, 1.0)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self.c_sc(x)
            return F.avg_pool2d(x, 2) if self.downsample else x

        else:
            return x

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        # NOTE: to completely reproduce pytorch, we use F.relu(x) to replace x in shortcut
        # since pytorch use inplace relu in residual branch.
        return self._residual(x) + self._shortcut(F.relu(x))


class DBlockOptimized(M.Module):
    """
    Optimized residual block for discriminator. This is used as the first residual block,
    where there is a definite downsampling involved. Follows the official SNGAN reference implementation
    in chainer.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self, in_channels, out_channels, spectral_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spectral_norm = spectral_norm

        # Build the layers
        self.c1 = M.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
        self.c2 = M.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        self.c_sc = M.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        self.activation = M.ReLU()

        M.init.xavier_uniform_(self.c1.weight, math.sqrt(2.0))
        M.init.xavier_uniform_(self.c2.weight, math.sqrt(2.0))
        M.init.xavier_uniform_(self.c_sc.weight, 1.0)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        return self.c_sc(F.avg_pool2d(x, 2))

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)
