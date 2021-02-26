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


class CustomFreeAnchorConfig(models.FreeAnchorConfig):
    def __init__(self):
        super().__init__()

        self.backbone = "resnext101_32x8d"
        self.max_epoch = 36
        self.lr_decay_stages = [24, 32]


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/"
    "freeanchor_resx101_coco_2x_800size_44dot9_5a23fca7.pkl"
)
def freeanchor_resx101_coco_2x_800size(**kwargs):
    r"""
    FreeAnchor trained from COCO dataset.
    `"FreeAnchor" <https://arxiv.org/abs/1909.02466>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    cfg = models.FreeAnchorConfig()
    cfg.backbone_pretrained = False
    return models.FreeAnchor(cfg, **kwargs)


Net = models.FreeAnchor
Cfg = CustomFreeAnchorConfig
