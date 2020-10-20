# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
import importlib
import math
from tabulate import tabulate

import numpy as np

from megengine.data import Sampler


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, record_len=1):
        self.record_len = record_len
        self.reset()

    def reset(self):
        self.sum = [0 for i in range(self.record_len)]
        self.cnt = 0

    def update(self, val):
        self.sum = [s + v for s, v in zip(self.sum, val)]
        self.cnt += 1

    def average(self):
        return [s / self.cnt for s in self.sum]


def import_from_file(cfg_file):
    spec = importlib.util.spec_from_file_location("config", cfg_file)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    return cfg_module


def get_config_info(config):
    config_table = []
    for c, v in config.__dict__.items():
        if not isinstance(v, (int, float, str, list, tuple, dict, np.ndarray)):
            if hasattr(v, "__name__"):
                v = v.__name__
            elif hasattr(v, "__class__"):
                v = v.__class__
            elif isinstance(v, functools.partial):
                v = v.func.__name__
        config_table.append((str(c), str(v)))
    config_table = tabulate(config_table)
    return config_table


class InferenceSampler(Sampler):
    def __init__(self, dataset, batch_size=1, world_size=None, rank=None):
        super().__init__(dataset, batch_size, False, None, world_size, rank)
        begin = self.num_samples * self.rank
        end = min(self.num_samples * (self.rank + 1), len(self.dataset))
        self.indices = list(range(begin, end))

    def sample(self):
        pass

    def batch(self):
        step, length = self.batch_size, len(self.indices)
        batch_index = [self.indices[i: i + step] for i in range(0, length, step)]
        return iter(batch_index)

    def __len__(self):
        return int(math.ceil(len(self.indices) / self.batch_size))


# pre-defined colors for at most 20 categories
class_colors = [
    [0, 0, 0],  # background
    [0, 0, 128],
    [0, 128, 0],
    [0, 128, 128],
    [128, 0, 0],
    [128, 0, 128],
    [128, 128, 0],
    [128, 128, 128],
    [0, 0, 64],
    [0, 0, 192],
    [0, 128, 64],
    [0, 128, 192],
    [128, 0, 64],
    [128, 0, 192],
    [128, 128, 64],
    [128, 128, 192],
    [0, 64, 0],
    [0, 64, 128],
    [0, 192, 0],
    [0, 192, 128],
    [128, 64, 0],
]
