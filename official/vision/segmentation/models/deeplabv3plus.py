# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine.functional as F
import megengine.module as M

import official.vision.classification.resnet.model as resnet


class ASPP(M.Module):
    def __init__(self, in_channels, out_channels, dr=1):
        super().__init__()

        self.conv1 = M.Sequential(
            M.Conv2d(
                in_channels, out_channels, 1, 1, padding=0, dilation=dr, bias=False
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
                bias=False,
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
                bias=False,
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
                bias=False,
            ),
            M.BatchNorm2d(out_channels),
            M.ReLU(),
        )
        self.conv_gp = M.Sequential(
            M.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            M.BatchNorm2d(out_channels),
            M.ReLU(),
        )
        self.conv_out = M.Sequential(
            M.Conv2d(out_channels * 5, out_channels, 1, 1, padding=0, bias=False),
            M.BatchNorm2d(out_channels),
            M.ReLU(),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv31 = self.conv2(x)
        conv32 = self.conv3(x)
        conv33 = self.conv4(x)

        gp = F.mean(x, [2, 3], True)
        gp = self.conv_gp(gp)
        gp = F.nn.interpolate(gp, x.shape[2:])

        out = F.concat([conv1, conv31, conv32, conv33, gp], axis=1)
        out = self.conv_out(out)
        return out


class DeepLabV3Plus(M.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.output_stride = 16
        self.num_classes = cfg.num_classes

        self.aspp = ASPP(
            in_channels=2048, out_channels=256, dr=16 // self.output_stride
        )
        self.dropout = M.Dropout(0.5)

        self.upstage1 = M.Sequential(
            M.Conv2d(256, 48, 1, 1, padding=0, bias=False),
            M.BatchNorm2d(48),
            M.ReLU(),
        )

        self.upstage2 = M.Sequential(
            M.Conv2d(256 + 48, 256, 3, 1, padding=1, bias=False),
            M.BatchNorm2d(256),
            M.ReLU(),
            M.Dropout(0.5),
            M.Conv2d(256, 256, 3, 1, padding=1, bias=False),
            M.BatchNorm2d(256),
            M.ReLU(),
            M.Dropout(0.1),
        )
        self.conv_out = M.Conv2d(256, self.num_classes, 1, 1, padding=0)

        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.msra_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, M.BatchNorm2d):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)

        self.backbone = getattr(resnet, cfg.backbone)(
            replace_stride_with_dilation=[False, False, True],
            pretrained=cfg.backbone_pretrained,
        )
        del self.backbone.fc

    def forward(self, x):
        layers = self.backbone.extract_features(x)

        up0 = self.aspp(layers["res5"])
        up0 = self.dropout(up0)
        up0 = F.nn.interpolate(up0, layers["res2"].shape[2:])

        up1 = self.upstage1(layers["res2"])
        up1 = F.concat([up0, up1], 1)

        up2 = self.upstage2(up1)

        out = self.conv_out(up2)
        out = F.nn.interpolate(out, x.shape[2:])
        return out
