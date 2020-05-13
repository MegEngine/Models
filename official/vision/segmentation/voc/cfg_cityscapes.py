# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
import os


class Config:
    DATASET = "Cityscapes"

    BATCH_SIZE = 4
    LEARNING_RATE = 0.0065
    EPOCHS = 200

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__")))
    MODEL_SAVE_DIR = os.path.join(ROOT_DIR, "log")
    LOG_DIR = MODEL_SAVE_DIR
    if not os.path.isdir(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    DATA_WORKERS = 4

    IGNORE_INDEX = 255
    NUM_CLASSES = 19
    IMG_HEIGHT = 800
    IMG_WIDTH = 800
    IMG_MEAN = [103.530, 116.280, 123.675]
    IMG_STD = [57.375, 57.120, 58.395]

    VAL_BATCHES = 1
    VAL_MULTISCALE = [1.0]  # [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    VAL_FLIP = False
    VAL_SLIP = True
    VAL_SAVE = None


cfg = Config()
