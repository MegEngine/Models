# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import math
from typing import List

import megengine.functional as F
import megengine.module as M

import official.quantization.models.resnet as resnet
from official.vision.detection import layers
from official.vision.detection.models import RetinaNet, RetinaNetConfig


class QuantFPN(layers.FPN):

    def __init__(
        self,
        bottom_up: M.Module,
        in_features: List[str],
        out_channels: int = 256,
        norm: str = None,
        top_block: M.Module = None,
        strides=[8, 16, 32],
        channels=[512, 1024, 2048],
    ):
        super(QuantFPN, self).__init__(
            bottom_up, in_features, out_channels, norm, top_block, strides, channels
        )

        # --------------- disable quant for FPN ------------ #
        use_bias = norm is None
        self.lateral_convs = list()
        self.output_convs = list()
        for idx, in_channels in enumerate(channels):

            lateral_conv = M.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=use_bias,
            )
            lateral_conv.disable_quantize()
            output_conv = M.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )
            output_conv.disable_quantize()
            M.init.msra_normal_(lateral_conv.weight, mode="fan_in")
            M.init.msra_normal_(output_conv.weight, mode="fan_in")

            if use_bias:
                M.init.fill_(lateral_conv.bias, 0)
                M.init.fill_(output_conv.bias, 0)

            stage = int(math.log2(strides[idx]))
            setattr(self, "fpn_lateral{}".format(stage), lateral_conv)
            setattr(self, "fpn_output{}".format(stage), output_conv)

            self.lateral_convs.insert(0, lateral_conv)
            self.output_convs.insert(0, output_conv)

        self.top_block.disable_quantize()
        self.add_module = M.Sequential(
            *[M.Elemwise("ADD") for _ in range(len(self.in_features))]
        )
        self.add_module.disable_quantize()
        self.quant = M.QuantStub()
        self.dequant = M.Sequential(*[M.DequantStub() for _ in range(5)])

    def forward(self, x):
        x = self.quant(x)
        bottom_up_features = self.bottom_up.extract_features(x)
        bottom_up_features = {
            key: dequant(bottom_up_features[key])
            for dequant, key in zip(self.dequant, bottom_up_features)
        }
        x = [bottom_up_features[f] for f in self.in_features[::-1]]

        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))

        for features, lateral_conv, output_conv, add in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:], self.add_module
        ):
            top_down_features = F.interpolate(
                prev_features, scale_factor=2, mode="BILINEAR"
            )
            lateral_features = lateral_conv(features)
            prev_features = add(lateral_features, top_down_features)
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(
                self.top_block.in_feature, None
            )
            if top_block_in_feature is None:
                top_block_in_feature = results[
                    self._out_features.index(self.top_block.in_feature)
                ]
            results.extend(self.top_block(top_block_in_feature))

        return dict(zip(self._out_features, results))


class QuantRetinaNet(RetinaNet):

    def __init__(self, cfg, batch_size):
        super(QuantRetinaNet, self).__init__(cfg, batch_size)
        self.bottom_up = getattr(resnet, cfg.backbone)(
            norm=layers.get_norm(cfg.resnet_norm)
        )
        out_channels = cfg.fpn_out_channels
        self.backbone = QuantFPN(
            bottom_up=self.bottom_up,
            in_features=cfg.backbone_in_features,
            out_channels=out_channels,
            norm=cfg.fpn_norm,
            top_block=layers.LastLevelP6P7(cfg.fpn_in_channels_p6p7, out_channels),
            strides=cfg.backbone_features_strides,
            channels=cfg.backbone_features_channels,
        )
        if self.cfg.backbone_freeze_at >= 1:
            for p in self.bottom_up.conv_bn_relu1.parameters():
                p.requires_grad = False
        if self.cfg.backbone_freeze_at >= 2:
            for p in self.bottom_up.layer1.parameters():
                p.requires_grad = False

        self.head.disable_quantize()


class QRetinaNetConfig(RetinaNetConfig):

    def __init__(self):
        super(QRetinaNetConfig, self).__init__()
        self.backbone = "resnet18"
        self.backbone_features_channels = [128, 256, 512]
        self.resnet_norm = "BN"
        self.fpn_in_channels_p6p7 = 512

        self.img_mean = [128, 128, 128]  # BGR
        self.img_std = [1, 1, 1]

        self.loss_normalizer_momentum = 0  # 0.0 means disable EMA normalizer

        self.train_image_short_size = 800
