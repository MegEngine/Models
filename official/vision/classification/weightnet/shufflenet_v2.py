# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2019 Megvii Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
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
import megengine.hub as hub
import megengine.module as M

from weightnet import WeightNet, WeightNet_DW

class ShuffleV2Block(M.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp


        self.reduce = M.Conv2d(inp, max(16, inp//16), 1, 1, 0, bias=True)

        self.wnet1 = WeightNet(inp, mid_channels, 1, 1)
        self.bn1 = M.BatchNorm2d(mid_channels)

        self.wnet2 = WeightNet_DW(mid_channels, ksize, stride)
        self.bn2 =M.BatchNorm2d(mid_channels)

        self.wnet3 = WeightNet(mid_channels, outputs, 1, 1)
        self.bn3 = M.BatchNorm2d(outputs)

        if stride == 2:
            self.wnet_proj_1 = WeightNet_DW(inp, ksize, stride)
            self.bn_proj_1 = M.BatchNorm2d(inp)

            self.wnet_proj_2 = WeightNet(inp, inp, 1, 1)
            self.bn_proj_2 = M.BatchNorm2d(inp)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x

        x_gap = x.mean(axis=2,keepdims=True).mean(axis=3,keepdims=True)
        x_gap = self.reduce(x_gap)

        x = self.wnet1(x, x_gap)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.wnet2(x, x_gap)
        x = self.bn2(x)

        x = self.wnet3(x, x_gap)
        x = self.bn3(x)
        x = F.relu(x)

        if self.stride == 2:
            x_proj = self.wnet_proj_1(x_proj, x_gap)
            x_proj = self.bn_proj_1(x_proj)
            x_proj = self.wnet_proj_2(x_proj, x_gap)
            x_proj = self.bn_proj_2(x_proj)
            x_proj = F.relu(x_proj)

        return F.concat((x_proj, x), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.shape
        # assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.dimshuffle(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(M.Module):
    def __init__(self, input_size=224, num_classes=1000, model_size="1.5x"):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        # We reduce the width slightly here to make WeightNet's FLOPs comparable to the baselines.
        if model_size == "0.5x":
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self.stage_out_channels = [-1, 24, 112, 224, 448, 1024]
        elif model_size == "1.5x":
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self.stage_out_channels = [-1, 24, 248, 496, 992, 1024]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = M.Sequential(
            M.Conv2d(3, input_channel, 3, 2, 1, bias=True), M.BatchNorm2d(input_channel), M.ReLU(),
        )

        self.maxpool = M.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(
                        ShuffleV2Block(
                            input_channel, output_channel, mid_channels=output_channel // 2, ksize=3, stride=2,
                        )
                    )
                else:
                    self.features.append(
                        ShuffleV2Block(
                            input_channel // 2, output_channel, mid_channels=output_channel // 2, ksize=3, stride=1,
                        )
                    )

                input_channel = output_channel

        self.features = M.Sequential(*self.features)

        self.conv_last = M.Sequential(
            M.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=True),
            M.BatchNorm2d(self.stage_out_channels[-1]),
            M.ReLU(),
        )
        self.globalpool = M.AvgPool2d(7)
        if self.model_size == "2.0x":
            self.dropout = M.Dropout(0.2)
        self.classifier = M.Sequential(M.Linear(self.stage_out_channels[-1], num_classes, bias=True))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)
        if self.model_size == "2.0x":
            x = self.dropout(x)
        x = x.reshape(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, M.Conv2d):
                if "first" in name:
                    M.init.normal_(m.weight, 0, 0.01)
                else:
                    M.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    M.init.fill_(m.bias, 0)
            elif isinstance(m, M.BatchNorm2d):
                M.init.fill_(m.weight, 1)
                if m.bias is not None:
                    M.init.fill_(m.bias, 0.0001)
                M.init.fill_(m.running_mean, 0)
            elif isinstance(m, M.BatchNorm1d):
                M.init.fill_(m.weight, 1)
                if m.bias is not None:
                    M.init.fill_(m.bias, 0.0001)
                M.init.fill_(m.running_mean, 0)
            elif isinstance(m, M.Linear):
                M.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    M.init.fill_(m.bias, 0)


@hub.pretrained("https://data.megengine.org.cn/models/weights/snetv2_x2_0_75115_497d4601.pkl")
def shufflenet_v2_x2_0(num_classes=1000):
    return ShuffleNetV2(num_classes=num_classes, model_size="2.0x")


@hub.pretrained("https://data.megengine.org.cn/models/weights/snetv2_x1_5_72775_38ac4273.pkl")
def shufflenet_v2_x1_5(num_classes=1000):
    return ShuffleNetV2(num_classes=num_classes, model_size="1.5x")


@hub.pretrained("https://data.megengine.org.cn/models/weights/snetv2_x1_0_69369_daf9dba0.pkl")
def shufflenet_v2_x1_0(num_classes=1000):
    return ShuffleNetV2(num_classes=num_classes, model_size="1.0x")


@hub.pretrained("https://data.megengine.org.cn/models/weights/snetv2_x0_5_60750_c28db1a2.pkl")
def shufflenet_v2_x0_5(num_classes=1000):
    return ShuffleNetV2(num_classes=num_classes, model_size="0.5x")

