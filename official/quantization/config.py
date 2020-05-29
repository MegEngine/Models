# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""
Configurations to train/finetune quantized classification models
"""
class ShufflenetConfig:
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0625
    MOMENTUM = 0.9
    WEIGHT_DECAY = lambda self, n, p: \
        4e-5 if n.find("weight") >= 0 and len(p.shape) > 1 else 0
    EPOCHS = 240

    SCHEDULER = "Linear"


class ResnetConfig:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0125
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    EPOCHS = 90

    SCHEDULER = "Multistep"
    SCHEDULER_STEPS = [30, 60, 80]
    SCHEDULER_GAMMA = 0.1


def get_config(arch: str):
    if "resne" in arch:  # both resnet and resnext
        return ResnetConfig()
    elif "shufflenet" in arch or "mobilenet" in arch:
        return ShufflenetConfig()
    else:
        raise ValueError("config for {} not exists".format(arch))


class ShufflenetFinetuneConfig:
    BATCH_SIZE = 128 // 2
    LEARNING_RATE = 0.03125
    MOMENTUM = 0.9
    WEIGHT_DECAY = lambda self, n, p: \
        4e-5 if n.find("weight") >= 0 and len(p.shape) > 1 else 0
    EPOCHS = 120

    SCHEDULER = "Linear"


class ResnetFinetuneConfig:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.000125
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    EPOCHS = 9

    SCHEDULER = "Multistep"
    SCHEDULER_STEPS = [6,]
    SCHEDULER_GAMMA = 0.1


def get_finetune_config(arch: str):
    if "resne" in arch:  # both resnet and resnext
        return ResnetFinetuneConfig()
    elif "shufflenet" in arch or "mobilenet" in arch:
        return ShufflenetFinetuneConfig()
    else:
        raise ValueError("config for {} not exists".format(arch))
