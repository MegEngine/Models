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
from megengine.module import (
    BatchNorm2d,
    Conv2d,
    ConvBn2d,
    ConvBnRelu2d,
    AvgPool2d,
    MaxPool2d,
    DequantStub,
    Linear,
    Module,
    QuantStub,
    Sequential,
    MaxPool2d,
    Sequential,
    Elemwise,
)
from megengine.quantization import *


class ShuffleV1Block(Module):
    def __init__(self, inp, oup, *, group, first_group, mid_channels, ksize, stride):
        super(ShuffleV1Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.group = group

        branch_main_1 = [
            # pw
            ConvBnRelu2d(inp, mid_channels, 1, 1, 0, groups=1 if first_group else group, bias=False),
            # dw
            ConvBn2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False)
        ]
        branch_main_2 = [
            # pw-linear
            ConvBn2d(mid_channels, oup, 1, 1, 0, groups=group, bias=False)
        ]
        self.branch_main_1 = Sequential(*branch_main_1)
        self.branch_main_2 = Sequential(*branch_main_2)
        self.add = Elemwise('FUSE_ADD_RELU')

        if stride == 2:
            self.branch_proj = ConvBn2d(inp, oup, 1, 2, 0, bias=False)

    def forward(self, old_x):
        x = old_x
        x_proj = old_x
        x = self.branch_main_1(x)
        if self.group > 1:
            x = self.channel_shuffle(x)
        x = self.branch_main_2(x)
        if self.stride == 1:
            return self.add(x, x_proj)
        elif self.stride == 2:
            return self.add(self.branch_proj(x_proj), x)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.shape
        # assert num_channels.numpy() % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.dimshuffle(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)
        return x


class ShuffleNetV1(Module):
    def __init__(self, num_classes=1000, model_size='2.0x', group=None):
        super(ShuffleNetV1, self).__init__()
        print('model size is ', model_size)

        assert group is not None

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if group == 3:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 12, 120, 240, 480]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 240, 480, 960]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 360, 720, 1440]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 480, 960, 1920]
            else:
                raise NotImplementedError
        elif group == 8:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 16, 192, 384, 768]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 384, 768, 1536]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 576, 1152, 2304]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 768, 1536, 3072]
            else:
                raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = Sequential(
            ConvBnRelu2d(3, input_channel, 3, 2, 1, bias=False)
        )
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                first_group = idxstage == 0 and i == 0
                self.features.append(ShuffleV1Block(input_channel, output_channel,
                                                    group=group, first_group=first_group,
                                                    mid_channels=output_channel // 4, ksize=3, stride=stride))
                input_channel = output_channel

        self.features = Sequential(*self.features)
        self.quant = QuantStub()
        self.dequant = DequantStub()
        self.classifier = Sequential(Linear(self.stage_out_channels[-1], num_classes, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.quant(x)
        x = self.first_conv(x)
        x = self.maxpool(x)

        x = self.features(x)

        x = F.avg_pool2d(x, 7)
        x = F.flatten(x, 1)
        x = self.dequant(x)
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


def shufflenet_v1_x0_5_g3(num_classes=1000):
    net = ShuffleNetV1(num_classes=num_classes, model_size="0.5x", group=3)
    return net

def shufflenet_v1_x1_0_g3(num_classes=1000):
    net = ShuffleNetV1(num_classes=num_classes, model_size="1.0x", group=3)
    return net

def shufflenet_v1_x1_5_g3(num_classes=1000):
    net = ShuffleNetV1(num_classes=num_classes, model_size="1.5x", group=3)
    return net

def shufflenet_v1_x2_0_g3(num_classes=1000):
    net = ShuffleNetV1(num_classes=num_classes, model_size="2.0x", group=3)
    return net
