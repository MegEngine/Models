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


class CustomFasterRCNNFPNConfig(models.FasterRCNNConfig):
    def __init__(self):
        super().__init__()

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="coco",
            root="train2017",
            ann_file="annotations/instances_train2017.json"
        )
        self.test_dataset = dict(
            name="coco",
            root="val2017",
            ann_file="annotations/instances_val2017.json"
        )


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/"
    "faster_rcnn_fpn_ec2e80_res50_1x_800szie_37dot3.pkl"
)
def faster_rcnn_fpn_res50_coco_1x_800size(batch_size=1, **kwargs):
    r"""
    Faster-RCNN FPN trained from COCO dataset.
    `"Faster-RCNN" <https://arxiv.org/abs/1506.01497>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    return models.FasterRCNN(CustomFasterRCNNFPNConfig(), batch_size=batch_size, **kwargs)


Net = models.FasterRCNN
Cfg = CustomFasterRCNNFPNConfig
