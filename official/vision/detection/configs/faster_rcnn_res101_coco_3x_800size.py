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


class CustomFasterRCNNConfig(models.FasterRCNNConfig):
    def __init__(self):
        super().__init__()

        self.backbone = "resnet101"


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/"
    "faster_rcnn_res101_coco_3x_800size_42dot6_2538b0ff.pkl"
)
def faster_rcnn_res101_coco_3x_800size(**kwargs):
    r"""
    Faster-RCNN FPN trained from COCO dataset.
    `"Faster-RCNN" <https://arxiv.org/abs/1506.01497>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    cfg = CustomFasterRCNNConfig()
    cfg.backbone_pretrained = False
    return models.FasterRCNN(cfg, **kwargs)


Net = models.FasterRCNN
Cfg = CustomFasterRCNNConfig
