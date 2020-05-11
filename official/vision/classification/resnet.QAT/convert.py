# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import multiprocessing as mp
import time

import megengine as mge
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.quantization as Q
import megengine.jit as jit

import model as M

logger = mge.get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="resnet50", type=str)
    parser.add_argument("-m", "--model", default=None, type=str)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    model = getattr(M, args.arch)(pretrained=(args.model is None))
    Q.quantize_qat(model, qconfig=Q.ema_fakequant_qconfig)

    if args.model:
        logger.info("load weights from %s", args.model)
        model.load_state_dict(mge.load(args.model)["state_dict"])

    Q.quantize(model)
    mge.save(
        {
            "state_dict": model.state_dict(),
        },
        args.output,
    )
    logger.info("INT8 model save at: {}".format(args.output))


if __name__ == '__main__':
    main()
