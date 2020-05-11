# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import os

import megengine as mge
import megengine.jit as jit
import numpy as np
from megengine.quantization import *

import model as M


@jit.trace(symbolic=True)
def infer_func(inputs, model=None):
    if model is None:
        raise ValueError("should provide the model module")
    model.eval()
    logits = model(inputs)
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="shufflenet_v1_x0_5_g3_int8", type=str)
    parser.add_argument("-m", "--model", default=None, type=str)
    args = parser.parse_args()

    model = getattr(M, args.arch)(pretrained=(args.model is None))
    quantize_qat(model, qconfig=ema_fakequant_qconfig)

    if args.model:
        state_dict = mge.load(args.model)
        model.load_state_dict(state_dict, strict=False)

    quantize(model)

    data = mge.tensor(np.zeros((10, 3, 224, 224), dtype="float32"))
    infer_func(data, model=model)
    infer_func.dump(
        args.arch, arg_names=["data"], optimize_for_inference=True
    )


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    main()
