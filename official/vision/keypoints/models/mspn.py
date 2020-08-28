# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import math

import megengine.functional as F
import megengine.hub as hub
import megengine.module as M

import official.vision.classification.resnet.model as resnet


class ResnetBody(M.Module):
    def __init__(
        self,
        block,
        init_channel,
        layers,
        channels,
        zero_init_residual=False,
        norm=M.BatchNorm2d,
    ):
        super().__init__()
        self.in_channels = init_channel
        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=1, norm=norm
        )

        self.layer2 = self._make_layer(
            block, channels[1], layers[1], stride=2, norm=norm
        )

        self.layer3 = self._make_layer(
            block, channels[2], layers[2], stride=2, norm=norm,
        )

        self.layer4 = self._make_layer(
            block, channels[3], layers[3], stride=2, norm=norm,
        )

        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.msra_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    M.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, M.BatchNorm2d):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)
            elif isinstance(m, M.Linear):
                M.init.msra_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    M.init.uniform_(m.bias, -bound, bound)

    def _make_layer(self, block, channels, blocks, stride=1, norm=M.BatchNorm2d):
        layers = []
        layers.append(block(self.in_channels, channels, stride, norm=norm))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels, norm=norm))

        return M.Sequential(*layers)

    def forward(self, x):
        outputs = []

        x = self.layer1(x)
        outputs.append(x)
        x = self.layer2(x)
        outputs.append(x)
        x = self.layer3(x)
        outputs.append(x)
        x = self.layer4(x)
        outputs.append(x)

        return outputs


class SingleStage(M.Module):
    def __init__(
        self, block, init_channel, layers, channels, mid_channel, norm=M.BatchNorm2d
    ):
        super().__init__()
        self.down = ResnetBody(block, init_channel, layers, channels, norm)
        channel = block.expansion * channels[-1]
        self.up1 = M.Sequential(
            M.Conv2d(channel, mid_channel, 1, 1, 0), norm(mid_channel)
        )
        self.deconv1 = M.Sequential(
            M.ConvTranspose2d(mid_channel, mid_channel, 4, 2, 1), norm(mid_channel)
        )

        channel = block.expansion * channels[-2]
        self.up2 = M.Sequential(
            M.Conv2d(channel, mid_channel, 1, 1, 0), norm(mid_channel)
        )
        self.deconv2 = M.Sequential(
            M.ConvTranspose2d(mid_channel, mid_channel, 4, 2, 1), norm(mid_channel)
        )

        channel = block.expansion * channels[-3]
        self.up3 = M.Sequential(
            M.Conv2d(channel, mid_channel, 1, 1, 0), norm(mid_channel)
        )
        self.deconv3 = M.Sequential(
            M.ConvTranspose2d(mid_channel, mid_channel, 4, 2, 1), norm(mid_channel)
        )

        channel = block.expansion * channels[-4]
        self.up4 = M.Sequential(
            M.Conv2d(channel, mid_channel, 1, 1, 0), norm(mid_channel)
        )

    def forward(self, x):
        branches = self.down(x)
        branches = list(reversed(branches))

        outputs = []
        f_up = F.relu(self.up1(branches[0]))

        outputs.append(f_up)

        f = self.up2(branches[1])
        f_up = F.relu(self.deconv1(f_up) + f)
        outputs.append(f_up)

        f = self.up3(branches[2])
        f_up = F.relu(self.deconv2(f_up) + f)
        outputs.append(f_up)

        f = self.up4(branches[3])
        f_up = F.relu(self.deconv3(f_up) + f)
        outputs.append(f_up)

        return outputs


class MSPN(M.Module):
    def __init__(
        self,
        block,
        layers,
        channels,
        mid_channel,
        keypoint_num,
        nr_stg,
        norm=M.BatchNorm2d,
    ):
        super().__init__()

        self.nr_stg = nr_stg
        self.keypoint_num = keypoint_num

        self.head = M.Sequential(
            M.Conv2d(3, 64, 3, 2, 1),
            norm(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, 1, 1),
            norm(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, 2, 1),
            norm(64),
            M.ReLU(),
        )

        self.stages = {}
        for i in range(nr_stg):
            init_channel = 64
            self.stages["Stage_{}_body".format(i)] = SingleStage(
                block, init_channel, layers, channels, mid_channel, norm
            )
            tail = {}
            for j in range(4):
                tail["tail_{}".format(j)] = M.Conv2d(mid_channel, keypoint_num, 3, 1, 1)
            self.stages["Stage_{}_tail".format(i)] = tail

            if i < nr_stg - 1:
                self.stages["Stage_{}_next".format(i)] = M.Sequential(
                    M.Conv2d(mid_channel, 64, 1, 1, 0), norm(64), M.ReLU()
                )

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        M.init.zeros_(m.bias)
            if isinstance(m, M.ConvTranspose2d):
                M.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        M.init.zeros_(m.bias)
            if isinstance(m, M.BatchNorm2d):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)

    def calc_loss(self, images, heatmaps, heat_valid):
        outs = self(images)

        loss = 0
        for stage_out in outs:
            for ind, scale_out in enumerate(stage_out[:-1]):
                thre = 0.1 if ind == 3 else 1.1
                loss_weight = 1 if ind == 3 else 1 / 4 / len(outs)
                label = heatmaps[:, ind] * F.expand_dims((heat_valid > thre), [2, 3])
                scale_out = scale_out * F.expand_dims((heat_valid > thre), [2, 3])
                tmp = F.loss.square_loss(scale_out, label)
                loss += tmp * loss_weight

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

        f = self.head(x)
        outputs = []
        for i in range(self.nr_stg):
            multi_scale_features = self.stages["Stage_{}_body".format(i)](f)

            multi_scale_heatmaps = []
            for j in range(4):
                out = self.stages["Stage_{}_tail".format(i)]["tail_{}".format(j)](
                    multi_scale_features[j]
                )
                out = F.nn.interpolate(
                    out, scale_factor=2 ** (3 - j), align_corners=True
                )
                multi_scale_heatmaps.append(out)

            if i < self.nr_stg - 1:
                f = self.stages["Stage_{}_next".format(i)](multi_scale_features[-1])

            outputs.append(multi_scale_heatmaps)
        return outputs


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/keypoint_models/mspn_4stage_0_255_75_2.pkl"
)
def mspn_4stage(**kwargs):
    block = getattr(resnet, "Bottleneck")
    norm = M.SyncBatchNorm
    model = MSPN(
        block=block,
        layers=[5, 5, 6, 3],
        channels=[64, 128, 192, 384],
        nr_stg=4,
        mid_channel=256,
        keypoint_num=17,
        norm=norm,
        **kwargs
    )
    return model
