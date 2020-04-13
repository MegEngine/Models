# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import sys

import megengine as mge
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../classification"))
from resnet.model import Bottleneck, ResNet  # pylint: disable=import-error,wrong-import-position
sys.path.pop(0)


class ModifiedResNet(ResNet):
    def _make_layer(
        self, block, channels, blocks, stride=1, dilate=False, norm=M.BatchNorm2d
    ):
        if dilate:
            self.dilation *= stride  # pylint: disable=no-member
            stride = 1

        layers = []
        layers.append(
            block(
                self.in_channels,  # pylint: disable=access-member-before-definition
                channels,
                stride,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,  # pylint: disable=no-member
                norm=norm,
            )
        )
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,  # pylint: disable=no-member
                    norm=norm,
                )
            )

        return M.Sequential(*layers)


class ASPP(M.Module):
    def __init__(self, in_channels, out_channels, dr=1):
        super().__init__()

        self.conv1 = M.Sequential(
            M.Conv2d(
                in_channels, out_channels, 1, 1, padding=0, dilation=dr, bias=True
            ),
            M.BatchNorm2d(out_channels),
            M.ReLU(),
        )
        self.conv2 = M.Sequential(
            M.Conv2d(
                in_channels,
                out_channels,
                3,
                1,
                padding=6 * dr,
                dilation=6 * dr,
                bias=True,
            ),
            M.BatchNorm2d(out_channels),
            M.ReLU(),
        )
        self.conv3 = M.Sequential(
            M.Conv2d(
                in_channels,
                out_channels,
                3,
                1,
                padding=12 * dr,
                dilation=12 * dr,
                bias=True,
            ),
            M.BatchNorm2d(out_channels),
            M.ReLU(),
        )
        self.conv4 = M.Sequential(
            M.Conv2d(
                in_channels,
                out_channels,
                3,
                1,
                padding=18 * dr,
                dilation=18 * dr,
                bias=True,
            ),
            M.BatchNorm2d(out_channels),
            M.ReLU(),
        )
        self.convgp = M.Sequential(
            M.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True),
            M.BatchNorm2d(out_channels),
            M.ReLU(),
        )
        self.convout = M.Sequential(
            M.Conv2d(out_channels * 5, out_channels, 1, 1, padding=0, bias=True),
            M.BatchNorm2d(out_channels),
            M.ReLU(),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv31 = self.conv2(x)
        conv32 = self.conv3(x)
        conv33 = self.conv4(x)

        gp = F.mean(x, 2, True)
        gp = F.mean(gp, 3, True)
        gp = self.convgp(gp)
        gp = F.interpolate(gp, (x.shapeof(2), x.shapeof(3)))

        out = F.concat([conv1, conv31, conv32, conv33, gp], axis=1)
        out = self.convout(out)
        return out


class DeepLabV3Plus(M.Module):
    def __init__(self, class_num=21, pretrained=None):
        super().__init__()

        self.output_stride = 16
        self.sub_output_stride = self.output_stride // 4
        self.class_num = class_num

        self.aspp = ASPP(
            in_channels=2048, out_channels=256, dr=16 // self.output_stride
        )
        self.dropout = M.Dropout(0.5)

        self.upstage1 = M.Sequential(
            M.Conv2d(256, 48, 1, 1, padding=1 // 2, bias=True),
            M.BatchNorm2d(48),
            M.ReLU(),
        )

        self.upstage2 = M.Sequential(
            M.Conv2d(256 + 48, 256, 3, 1, padding=1, bias=True),
            M.BatchNorm2d(256),
            M.ReLU(),
            M.Dropout(0.5),
            M.Conv2d(256, 256, 3, 1, padding=1, bias=True),
            M.BatchNorm2d(256),
            M.ReLU(),
            M.Dropout(0.1),
        )
        self.convout = M.Conv2d(256, self.class_num, 1, 1, padding=0)

        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.msra_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, M.BatchNorm2d):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)

        self.backbone = ModifiedResNet(
            Bottleneck, [3, 4, 23, 3], replace_stride_with_dilation=[False, False, True]
        )
        if pretrained is not None:
            model_dict = mge.load(pretrained)
            self.backbone.load_state_dict(model_dict)

    def forward(self, x):
        layers = self.backbone.extract_features(x)

        up0 = self.aspp(layers["res5"])
        up0 = self.dropout(up0)
        up0 = F.interpolate(up0, scale_factor=self.sub_output_stride)

        up1 = self.upstage1(layers["res2"])
        up1 = F.concat([up0, up1], 1)

        up2 = self.upstage2(up1)

        out = self.convout(up2)
        out = F.interpolate(out, scale_factor=4)
        return out


def softmax_cross_entropy(pred, label, axis=1, ignore_index=255):
    offset = F.zero_grad(pred.max(axis=axis, keepdims=True))
    pred = pred - offset
    log_prob = pred - F.log(F.exp(pred).sum(axis=axis, keepdims=True))

    mask = 1 - F.equal(label, ignore_index)
    vlabel = label * mask
    loss = -(F.indexing_one_hot(log_prob, vlabel, axis) * mask).sum() / F.maximum(
        mask.sum(), 1
    )
    return loss


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/"
    "sematicseg_0f8e02aa_deeplabv3plus.pkl"
)
def deeplabv3plus_res101(**kwargs):
    r"""DeepLab v3+ model from
    `"Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" <https://arxiv.org/abs/1802.02611>`_
    """
    return DeepLabV3Plus(**kwargs)
