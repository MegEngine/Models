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

        self.resnet_norm = "SyncBN"
        self.fpn_norm = "SyncBN"
        self.backbone_freeze_at = 0

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="coco",
            root="train2017",
            ann_file="annotations/instances_train2017.json",
        )
        self.test_dataset = dict(
            name="coco",
            root="val2017",
            ann_file="annotations/instances_val2017.json",
        )


def retinanet_res50_coco_1x_800size_syncbn(batch_size=1, **kwargs):
    r"""
    RetinaNet with SyncBN trained from COCO dataset.
    `"RetinaNet" <https://arxiv.org/abs/1708.02002>`_
    """
    return models.RetinaNet(CustomRetinaNetConfig(), batch_size=batch_size, **kwargs)


Net = models.RetinaNet
Cfg = CustomRetinaNetConfig
