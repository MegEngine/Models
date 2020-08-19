# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from official.vision.detection import models


class CustomFasterRCNNConfig(models.FasterRCNNConfig):
    def __init__(self):
        super().__init__()

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="objects365",
            root="train",
            ann_file="annotations/objects365_train_20190423.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="objects365",
            root="val",
            ann_file="annotations/objects365_val_20190423.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 365

        # ------------------------ training cfg ---------------------- #
        self.nr_images_epoch = 400000


def faster_rcnn_res50_objects365_1x_800size(batch_size=1, **kwargs):
    r"""
    Faster-RCNN FPN trained from Objects365 dataset.
    `"Faster-RCNN" <https://arxiv.org/abs/1506.01497>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    """
    cfg = CustomFasterRCNNConfig()
    cfg.backbone_pretrained = False
    return models.FasterRCNN(cfg, batch_size=batch_size, **kwargs)


Net = models.FasterRCNN
Cfg = CustomFasterRCNNConfig
