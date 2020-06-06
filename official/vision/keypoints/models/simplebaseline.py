# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine as mge
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import official.vision.classification.resnet.model as resnet

import numpy as np


class DeconvLayers(M.Module):
    def __init__(self, nf1, nf2s, kernels, num_layers, bias=True, norm=M.BatchNorm2d):
        super(DeconvLayers, self).__init__()
        _body = []
        for i in range(num_layers):
            kernel = kernels[i]
            padding = (
                kernel // 3
            )  # padding=0 when kernel=2 and padding=1 when kernel=4 or kernel=3
            _body += [
                M.ConvTranspose2d(nf1, nf2s[i], kernel, 2, padding, bias=bias),
                norm(nf2s[i]),
                M.ReLU(),
            ]
            nf1 = nf2s[i]
        self.body = M.Sequential(*_body)

    def forward(self, x):
        return self.body(x)


class SimpleBaseline(M.Module):
    def __init__(self, backbone, cfg):
        super(SimpleBaseline, self).__init__()
        norm = M.BatchNorm2d
        self.backbone = getattr(resnet, backbone)(
            norm=norm, pretrained=cfg.backbone_pretrained
        )
        del self.backbone.fc

        self.cfg = cfg

        self.deconv_layers = DeconvLayers(
            cfg.initial_deconv_channels,
            cfg.deconv_channels,
            cfg.deconv_kernel_sizes,
            cfg.num_deconv_layers,
            cfg.deconv_with_bias,
            norm,
        )
        self.last_layer = M.Conv2d(cfg.deconv_channels[-1], cfg.keypoint_num, 3, 1, 1)

        self._initialize_weights()

        self.inputs = {
            "image": mge.tensor(dtype="float32"),
            "heatmap": mge.tensor(dtype="float32"),
            "heat_valid": mge.tensor(dtype="float32"),
        }

    def calc_loss(self):
        out = self.forward(self.inputs["image"])
        valid = self.inputs["heat_valid"][:, :, None, None]
        label = self.inputs["heatmap"][:, -1]
        loss = F.square_loss(out * valid, label * valid)
        return loss

    def predict(self):
        return self.forward(self.inputs["image"])

    def _initialize_weights(self):

        for k, m in self.deconv_layers.named_modules():
            if isinstance(m, M.ConvTranspose2d):
                M.init.normal_(m.weight, std=0.001)
                if self.cfg.deconv_with_bias:
                    M.init.zeros_(m.bias)
            if isinstance(m, M.BatchNorm2d):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)

        M.init.normal_(self.last_layer.weight, std=0.001)
        M.init.zeros_(self.last_layer.bias)

    def forward(self, x):
        f = self.backbone.extract_features(x)["res5"]
        f = self.deconv_layers(f)
        pred = self.last_layer(f)
        return pred


class SimpleBaseline_Config:
    initial_deconv_channels = 2048
    num_deconv_layers = 3
    deconv_channels = [256, 256, 256]
    deconv_kernel_sizes = [4, 4, 4]
    deconv_with_bias = False
    keypoint_num = 17
    backbone_pretrained = True


cfg = SimpleBaseline_Config()


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/simplebaseline50_256x192_0_255_71_2.pkl"
)
def simplebaseline_res50(**kwargs):

    model = SimpleBaseline(backbone="resnet50", cfg=cfg, **kwargs)
    return model


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/simplebaseline101_256x192_0_255_72_2.pkl"
)
def simplebaseline_res101(**kwargs):

    model = SimpleBaseline(backbone="resnet101", cfg=cfg, **kwargs)
    return model


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/simplebaseline152_256x192_0_255_72_4.pkl"
)
def simplebaseline_res152(**kwargs):

    model = SimpleBaseline(backbone="resnet152", cfg=cfg, **kwargs)
    return model

