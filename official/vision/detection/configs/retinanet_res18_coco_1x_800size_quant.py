# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from official.vision.detection import models


def retinanet_res18_coco_1x_800size_quant(batch_size=1, **kwargs):
    r"""
    RetinaNet trained from COCO dataset.
    `"RetinaNet" <https://arxiv.org/abs/1708.02002>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    return models.QauntRetinaNet(models.QRetinaNetConfig(), batch_size=batch_size, **kwargs)


Net = models.QuantRetinaNet
Cfg = models.QRetinaNetConfig
