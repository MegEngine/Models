# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine import hub

from official.vision.detection import models


class CustomRetinaNetConfig(models.RetinaNetConfig):
    def __init__(self):
        super().__init__()

        self.backbone = "resnet34"
        self.fpn_in_channels = [128, 256, 512]
        self.fpn_top_in_channel = 512


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/"
    "retinanet_res34_coco_3x_800size_38dot4_3485f9ec.pkl"
)
def retinanet_res34_coco_3x_800size(**kwargs):
    r"""
    RetinaNet trained from COCO dataset.
    `"RetinaNet" <https://arxiv.org/abs/1708.02002>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    cfg = CustomRetinaNetConfig()
    cfg.backbone_pretrained = False
    return models.RetinaNet(cfg, **kwargs)


Net = models.RetinaNet
Cfg = CustomRetinaNetConfig
