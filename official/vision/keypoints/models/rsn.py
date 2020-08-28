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

from official.vision.keypoints.models.mspn import MSPN


class RSB(M.Module):
    expansion = 1
    branch_ratio = 26 / 64
    num_branch = 4

    def __init__(
        self, channels1, channels2, stride=1, norm=M.BatchNorm2d,
    ):
        super().__init__()
        self.branch_channels = int(channels2 * self.branch_ratio)
        in_channels = self.branch_channels * self.num_branch

        self.layer1 = M.Conv2d(channels1, in_channels, 1, stride, 0)

        branches_conv = [[] for i in range(self.num_branch)]
        for i in range(self.num_branch):
            for j in range(i + 1):
                branches_conv[i].append(
                    M.Sequential(
                        M.Conv2d(self.branch_channels, self.branch_channels, 3, 1, 1),
                        norm(self.branch_channels),
                        M.ReLU(),
                    )
                )
        self.branches_conv = branches_conv

        out_channels = channels2 * self.expansion
        self.final_layer = M.Sequential(
            M.Conv2d(in_channels, out_channels, 1, 1, 0), norm(out_channels)
        )

        self.downsample = (
            M.Identity()
            if channels1 == out_channels and stride == 1
            else M.Sequential(
                M.Conv2d(channels1, out_channels, 1, stride, bias=False),
                norm(out_channels),
            )
        )

    def forward(self, x):
        f = self.layer1(x)
        mid_outs = [[] for i in range(self.num_branch)]
        for i in range(self.num_branch):
            mid_out = f[:, self.branch_channels * i: self.branch_channels * (i + 1)]
            for j in range(i + 1):
                skip = 0 if (i == 0 or j == i) else mid_outs[i - 1][j]
                mid_out = mid_out + skip
                mid_out = self.branches_conv[i][j](mid_out)
                mid_outs[i].append(mid_out)

        outs = [mid_outs[i][-1] for i in range(self.num_branch)]

        out = self.final_layer(F.concat(outs, 1))
        out = out + self.downsample(x)

        return F.relu(out)


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/keypoint_models/rsn18_256x192_72_1.pkl"
)
def rsn18(**kwargs):
    block = RSB
    RSB.expansion = 1
    RSB.branch_ratio = 26 / 64
    RSB.num_branch = 4

    norm = M.BatchNorm2d

    model = MSPN(
        block=block,
        layers=[3, 4, 6, 3],
        channels=[64, 128, 256, 512],
        nr_stg=1,
        mid_channel=256,
        keypoint_num=17,
        norm=norm,
        **kwargs
    )

    return model


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/keypoint_models/rsn50_256x192_73_2.pkl"
)
def rsn50(**kwargs):
    block = RSB
    RSB.expansion = 4
    RSB.branch_ratio = 26 / 64
    RSB.num_branch = 4

    norm = M.BatchNorm2d

    model = MSPN(
        block=block,
        layers=[3, 4, 6, 3],
        channels=[64, 128, 256, 512],
        nr_stg=1,
        mid_channel=256,
        keypoint_num=17,
        norm=norm,
        **kwargs
    )

    return model


def rsn50_4stage(**kwargs):
    block = RSB
    RSB.expansion = 4
    RSB.branch_ratio = 26 / 64
    RSB.num_branch = 4

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
