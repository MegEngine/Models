# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .danet import danet72, danet88, danet98, danet102
from .hrnet import hrnet_w32, hrnet_w48
from .mspn import mspn_4stage
from .rsn import rsn18, rsn50, rsn50_4stage
from .simplebaseline import simplebaseline_res50, simplebaseline_res101, simplebaseline_res152