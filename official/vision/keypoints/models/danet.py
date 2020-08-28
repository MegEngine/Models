# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M


class CAU(M.Module):
    def __init__(self, channel1, channel2):
        super().__init__()

        self.att_conv = M.Sequential(
            M.BatchNorm2d(channel1),
            M.ReLU(),
            M.Conv2d(channel1, channel2, 1, 1, 0),
            M.BatchNorm2d(channel2),
            M.ReLU(),
            M.Conv2d(channel2, channel1, 1, 1, 0),
            M.Sigmoid(),
        )

    def forward(self, x):
        att = x.mean([2, 3], keepdims=True)
        att = self.att_conv(att)
        return att * x


class MAU(M.Module):
    def __init__(self, channel1, channel2):
        super().__init__()

        self.att_conv = M.Sequential(
            M.BatchNorm2d(channel1),
            M.ReLU(),
            M.Conv2d(channel1, channel2, 1, 1, 0),
            M.BatchNorm2d(channel2),
            M.ReLU(),
            M.Conv2d(channel2, channel2, 9, 1, 4, groups=channel2),
            M.Sigmoid(),
        )

    def forward(self, x1, x2):
        att = 1 - self.att_conv(x1)
        return att * x2


class OAB(M.Module):
    def __init__(self, channel1, channel2, channel3):
        super().__init__()

        self.cau = CAU(channel1, channel2)
        self.mau = MAU(channel1, channel3)

        self.body_conv = M.Sequential(
            M.BatchNorm2d(channel1),
            M.ReLU(),
            M.Conv2d(channel1, channel2, 1, 1, 0),
            M.BatchNorm2d(channel2),
            M.ReLU(),
            M.Conv2d(channel2, channel3, 3, 1, 1),
        )

    def forward(self, x):
        f = self.cau(x)
        f = self.body_conv(f)

        f = self.mau(x, f)
        return F.concat([x, f], 1)


class SFU(M.Module):
    def __init__(self, channel):
        super().__init__()

        self.att_conv = M.Sequential(
            M.BatchNorm2d(channel * 2),
            M.ReLU(),
            M.Conv2d(channel * 2, channel, 1, 1, 0),
            M.BatchNorm2d(channel),
            M.ReLU(),
            M.Conv2d(channel, channel, 9, 1, 4, groups=channel),
            M.Sigmoid(),
        )

    def forward(self, x1, x2):
        att = self.att_conv(F.concat([x1, x2], 1))
        x = att * x1 + (1 - att) * x2
        return x


class FP(M.Module):
    def __init__(self, cfg):
        super().__init__()

        self.head = M.Sequential(
            M.Conv2d(3, 64, 7, 2, 3), M.BatchNorm2d(64), M.ReLU(), M.MaxPool2d(3, 2, 1)
        )

        self.body_layer = []
        self.transition_layer = []

        channel = 64
        for i in range(4):
            compre_channel = cfg.compre_channel[i]
            inre_channel = cfg.inre_channel[i]
            fusion_channel = cfg.fusion_channel[i]

            body_conv = []
            for _ in range(cfg.blocks[i]):
                body_conv.append(OAB(channel, compre_channel, inre_channel))
                channel += inre_channel
            self.body_layer.append(M.Sequential(*body_conv))

            transition_conv = [
                M.BatchNorm2d(channel),
                M.ReLU(),
                M.Conv2d(channel, fusion_channel, 1, 1, 0),
            ]
            self.transition_layer.append(M.Sequential(*transition_conv))
            channel = fusion_channel

    def forward(self, x):
        x = self.head(x)
        outs = []
        for i in range(4):
            x = self.body_layer[i](x)
            x = self.transition_layer[i](x)
            outs.append(x)

            if i < 3:
                x = F.max_pool2d(x, 2, 2, 0)
        return outs


class DANet(M.Module):
    def __init__(self, cfg):
        super().__init__()
        self.keypoint_num = cfg.keypoint_num

        self.fp = FP(cfg)

        mid_channel = cfg.mid_channel

        self.sfus = []
        self.up_convs = []
        self.deconvs = []
        self.out_convs = []

        for i in range(4):
            self.up_convs.append(
                M.Sequential(
                    M.BatchNorm2d(cfg.fusion_channel[3 - i]),
                    M.ReLU(),
                    M.Conv2d(cfg.fusion_channel[3 - i], mid_channel, 1, 1, 0),
                )
            )
            self.out_convs.append(
                M.Sequential(
                    M.BatchNorm2d(mid_channel),
                    M.ReLU(),
                    M.Conv2d(mid_channel, mid_channel, 1, 1, 0),
                    M.BatchNorm2d(mid_channel),
                    M.ReLU(),
                    M.Conv2d(mid_channel, cfg.keypoint_num, 3, 1, 1),
                )
            )

            if i > 0:
                self.sfus.append(SFU(mid_channel))

                self.deconvs.append(
                    M.Sequential(
                        M.BatchNorm2d(mid_channel),
                        M.ReLU(),
                        M.ConvTranspose2d(mid_channel, mid_channel, 4, 2, 1),
                    )
                )

    def calc_loss(self, images, heatmaps, heat_valid):
        outs = self(images)

        loss = 0
        for stage_out in outs:
            for ind, scale_out in enumerate(stage_out[:-1]):
                label = heatmaps[:, ind] * F.expand_dims((heat_valid > 1.1), [2, 3])
                scale_out = scale_out * F.expand_dims((heat_valid > 1.1), [2, 3])
                tmp = F.loss.square_loss(scale_out, label)
                loss += tmp / 4 / len(outs)

            # OHKM loss for the largest heatmap
            tmp = ((stage_out[-1] - heatmaps[:, -1]) ** 2).mean([2, 3]) * (
                heat_valid > 0.1
            )
            ohkm_loss = 0
            for i in range(tmp.shape[0]):
                selected_loss, _ = F.math.sort(tmp[i], descending=True)
                selected_loss = selected_loss[: self.keypoint_num // 2]
                ohkm_loss += selected_loss.mean()
            ohkm_loss /= tmp.shape[0]
            loss += ohkm_loss
        return loss

    def predict(self, images):
        outputs = self(images)
        pred = outputs[-1][-1]
        return pred

    def forward(self, x):
        outs = self.fp(x)
        outs = list(reversed(outs))

        heatmaps = []
        for ind, out in enumerate(outs):
            if ind == 0:
                f_up = self.up_convs[ind](out)
            else:
                f1 = self.up_convs[ind](out)
                f2 = self.deconvs[ind - 1](f_up)
                f_up = self.sfus[ind - 1](f1, f2)

            heatmap = self.out_convs[ind](f_up)
            if ind < 3:
                heatmap = F.nn.interpolate(heatmap, scale_factor=2 ** (3 - ind))
            heatmaps.append(heatmap)
        return [heatmaps]


class DANetConfig:
    compre_channel = [32, 32, 32, 32]
    inre_channel = [32, 32, 32, 32]
    fusion_channel = [128, 256, 512, 640]
    blocks = [7, 13, 17, 15]
    mid_channel = 128
    keypoint_num = 17


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/keypoint_models/danet72_256x192_69_8.pkl"
)
def danet72(**kwargs):
    cfg = DANetConfig()
    cfg.fusion_channel = [96, 192, 384, 512]
    cfg.blocks = [3, 6, 12, 12]

    model = DANet(cfg)
    return model


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/keypoint_models/danet88_256x192_70_4.pkl"
)
def danet88(**kwargs):
    cfg = DANetConfig()
    cfg.fusion_channel = [128, 192, 512, 640]
    cfg.blocks = [4, 5, 18, 14]

    model = DANet(cfg)
    return model


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/keypoint_models/danet98_256x192_71_3.pkl"
)
def danet98(**kwargs):
    cfg = DANetConfig()
    cfg.fusion_channel = [128, 256, 512, 640]
    cfg.blocks = [4, 12, 16, 14]

    model = DANet(cfg)
    return model


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/keypoint_models/danet102_256x192_71_7.pkl"
)
def danet102(**kwargs):
    cfg = DANetConfig()
    cfg.fusion_channel = [128, 256, 512, 640]
    cfg.blocks = [6, 12, 16, 14]

    model = DANet(cfg)
    return model
