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

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="objects365",
            root="train",
            ann_file="annotations/objects365_train_20190423.json"
        )
        self.test_dataset = dict(
            name="objects365",
            root="val",
            ann_file="annotations/objects365_val_20190423.json"
        )

        # ------------------------ training cfg ---------------------- #
        self.nr_images_epoch = 400000


def retinanet_objects365_res50_1x_800size(batch_size=1, **kwargs):
    r"""ResNet-18 model from
    `"RetinaNet" <https://arxiv.org/abs/1708.02002>`_
    """
    return models.RetinaNet(RetinaNetConfig(), batch_size=batch_size, **kwargs)


Net = models.RetinaNet
Cfg = CustomRetinaNetConfig
