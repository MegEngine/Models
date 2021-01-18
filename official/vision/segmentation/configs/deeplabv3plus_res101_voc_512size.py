# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
from megengine import hub

from official.vision.segmentation import models


class VOCConfig:
    def __init__(self):
        self.dataset = "VOC2012"
        self.data_type = "trainaug"

        self.backbone = "resnet101"
        self.backbone_pretrained = True

        self.batch_size = 8
        self.learning_rate = 0.02
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.max_epoch = 40
        self.nr_images_epoch = 64000

        self.ignore_label = 255
        self.num_classes = 21
        self.img_height = 512
        self.img_width = 512
        self.img_mean = [103.530, 116.280, 123.675]  # BGR
        self.img_std = [57.375, 57.120, 58.395]

        self.val_height = 512
        self.val_width = 512
        self.val_multiscale = [1.0]  # [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        self.val_flip = False
        self.val_slip = False
        self.val_save_path = None

        self.log_interval = 20


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/"
    "deeplabv3plus_res101_voc_512size_79dot5_7856dc84.pkl"
)
def deeplabv3plus_res101_voc_512size(**kwargs):
    r"""DeepLab v3+ model from
    `"Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1802.02611>`_
    """
    cfg = VOCConfig()
    cfg.backbone_pretrained = False
    return models.DeepLabV3Plus(cfg, **kwargs)


Net = models.DeepLabV3Plus
Cfg = VOCConfig
