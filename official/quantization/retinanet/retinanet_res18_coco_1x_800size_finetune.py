# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
from official.quantization.retinanet.retinanet_res18 import RetinaNetConfig, RetinaNet


class CustomRetinaNetConfig(RetinaNetConfig):
    def __init__(self):
        super().__init__()

        self.basic_lr = 0.001 / 16.0  # The basic learning rate for single-image
        # since input image minus 128, img_mena also need minus 128.
        self.warm_iters = 0
        self.log_interval = 20
        self.lr_decay_sates = [1, 2]
        self.max_epoch = 3


def retinanet_res50_coco_1x_800size(batch_size=1, **kwargs):
    return RetinaNet(CustomRetinaNetConfig(), batch_size=batch_size, **kwargs)


Net = RetinaNet
Cfg = CustomRetinaNetConfig
