# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M

import official.vision.classification.resnet.model as resnet


class DeconvLayers(M.Module):
    def __init__(self, nf1, nf2s, kernels, num_layers, bias=True, norm=M.BatchNorm2d):
        super().__init__()
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
        super().__init__()
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

        self.initialize_weights()

    def calc_loss(self, images, heatmaps, heat_valid):
        out = self(images)
        valid = F.expand_dims((heat_valid > 0.1), [2, 3])
        label = heatmaps[:, -1]
        loss = F.nn.square_loss(out * valid, label * valid)
        return loss

    def predict(self, images):
        return self(images)

    def initialize_weights(self):

        for k, m in self.named_modules():
            if self.cfg.backbone_pretrained and ("backbone" in k):
                continue
            if isinstance(m, M.Conv2d):
                M.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        M.init.zeros_(m.bias)
            if isinstance(m, M.ConvTranspose2d):
                M.init.normal_(m.weight, std=0.001)
                if self.cfg.deconv_with_bias:
                    M.init.zeros_(m.bias)
            if isinstance(m, M.BatchNorm2d):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)

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
    "https://data.megengine.org.cn/models/weights/keypoint_models/"
    "simplebaseline_res50_256x192_0to255_71dot1_2c0de7.pkl"
)
def simplebaseline_res50(**kwargs):

    model = SimpleBaseline(backbone="resnet50", cfg=cfg, **kwargs)
    return model


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/keypoint_models/"
    "simplebaseline_res101_256x192_0to255_71dot8_df6304.pkl"
)
def simplebaseline_res101(**kwargs):

    model = SimpleBaseline(backbone="resnet101", cfg=cfg, **kwargs)
    return model


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/keypoint_models/"
    "simplebaseline_res152_256x192_0to255_72dot3_18ba8e.pkl"
)
def simplebaseline_res152(**kwargs):

    model = SimpleBaseline(backbone="resnet152", cfg=cfg, **kwargs)
    return model
