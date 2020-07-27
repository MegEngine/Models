# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine import hub

from official.vision.detection import models


class CustomRetinaNetConfig(models.RetinaNetConfig):
    def __init__(self):
        super().__init__()

        self.backbone = "resnet101"

        # ------------------------ training cfg ---------------------- #
        self.max_epoch = 36
        self.lr_decay_stages = [24, 32, 34]


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/"
    "retinanet_res101_coco_2x_800size_40dot8_661c3608.pkl"
)
def retinanet_res101_coco_2x_800size(batch_size=1, **kwargs):
    r"""
    RetinaNet trained from COCO dataset.
    `"RetinaNet" <https://arxiv.org/abs/1708.02002>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    cfg = CustomRetinaNetConfig()
    cfg.backbone_pretrained = False
    return models.RetinaNet(cfg, batch_size=batch_size, **kwargs)


Net = models.RetinaNet
Cfg = CustomRetinaNetConfig
