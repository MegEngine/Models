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
import megengine.functional as F
import megengine.jit as jit

from .. import gan
from ..blocks import DBlock, DBlockOptimized


class WGANBaseGenerator(gan.BaseGenerator):
    r"""
    ResNet backbone generator for ResNet WGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz, ngf, bottom_width, **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type="wasserstein",
                         **kwargs)


class WGANBaseDiscriminator(gan.BaseDiscriminator):
    r"""
    ResNet backbone discriminator for ResNet WGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, ndf, **kwargs):
        super().__init__(ndf=ndf, loss_type="wasserstein", **kwargs)

    def _reset_jit_graph(self, impl: callable):
        """We override this func to attach weight clipping after default training step"""
        traced_obj = jit.trace(impl)
        def _(*args, **kwargs):
            ret = traced_obj(*args, **kwargs)
            if self.training:
                self._apply_lipshitz_constraint()  # dynamically apply weight clipping
            return ret
        return _

    def _apply_lipshitz_constraint(self):
        """Weight clipping described in [Wasserstein GAN](https://arxiv.org/abs/1701.07875)"""
        for p in self.parameters():
            F.add_update(p, F.clamp(p, lower=-3e-2, upper=3e-2), alpha=0)


def layernorm(x):
    original_shape = x.shape
    x = x.reshape(original_shape[0], -1)
    m = F.mean(x, axis=1, keepdims=True)
    v = F.mean((x - m) ** 2, axis=1, keepdims=True)
    x = (x - m) / F.maximum(F.sqrt(v), 1e-6)
    x = x.reshape(original_shape)
    return x


class WGANDBlockWithLayerNorm(DBlock):
    def _residual(self, x):
        h = x
        h = layernorm(h)
        h = self.activation(h)
        h = self.c1(h)
        h = layernorm(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        return h


class WGANDBlockOptimized(DBlockOptimized):
    pass
