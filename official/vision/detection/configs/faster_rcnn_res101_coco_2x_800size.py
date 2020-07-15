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


class CustomFasterRCNNConfig(models.FasterRCNNConfig):
    def __init__(self):
        super().__init__()

        self.backbone = "resnet101"

        # ------------------------ training cfg ---------------------- #
        self.max_epoch = 36
        self.lr_decay_stages = [24, 32, 34]


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/"
    "faster_rcnn_res101_coco_2x_800size_43dot0_ee249359.pkl"
)
def faster_rcnn_res101_coco_2x_800size(batch_size=1, **kwargs):
    r"""
    Faster-RCNN FPN trained from COCO dataset.
    `"Faster-RCNN" <https://arxiv.org/abs/1506.01497>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    return models.FasterRCNN(CustomFasterRCNNConfig(), batch_size=batch_size, **kwargs)


Net = models.FasterRCNN
Cfg = CustomFasterRCNNConfig
