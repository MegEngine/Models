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
            name="voc", root="VOCdevkit/VOC2012", image_set="train",
        )
        self.test_dataset = dict(name="voc", root="VOCdevkit/VOC2012", image_set="val",)
        self.num_classes = 20

        # ------------------------ training cfg ---------------------- #
        self.nr_images_epoch = 16000


def retinanet_res50_voc_1x_800size(batch_size=1, **kwargs):
    r"""
    RetinaNet trained from VOC dataset.
    `"RetinaNet" <https://arxiv.org/abs/1708.02002>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    """
    return models.RetinaNet(CustomRetinaNetConfig(), batch_size=batch_size, **kwargs)


Net = models.RetinaNet
Cfg = CustomRetinaNetConfig
