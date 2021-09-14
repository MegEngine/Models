# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import argparse
# pylint: disable=import-error
import resnet.model as resnet_model
# pylint: disable=import-error
import shufflenet.model as snet_model
import megengine as mge
from megengine import jit
import numpy as np
import sys

def dump_static_graph(model, graph_name, shape):
    model.eval()

    data = mge.Tensor(np.random.random(shape))

    @jit.trace(capture_as_const=True)
    def pred_func(data):
        outputs = model(data)
        return outputs

    pred_func(data)
    pred_func.dump(
        graph_name,
        arg_names=["data"],
        optimize_for_inference=True,
        enable_fuse_conv_bias_nonlinearity=True,
    )

def main():
    parser = argparse.ArgumentParser(description="MegEngine Classification Dump .mge")
    parser.add_argument(
        "-a",
        "--arch",
        default="resnet18",
        help="model architecture (default: resnet18)",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=int,
        nargs=4,
        default="1 3 224 224",
        help="input shape (default: 1 3 224 224)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="model.mge",
        help="output filename"
    )

    args = parser.parse_args()

    if 'resnet' in args.arch:
        model = getattr(resnet_model, args.arch)(pretrained=True)
    elif 'shufflenet' in args.arch:
        model = getattr(snet_model, args.arch)(pretrained=True)
    else:
        print(f'unavailable arch {args.arch}')
        sys.exit()
    dump_static_graph(model, args.output, tuple(args.shape))

if __name__ == "__main__":
    main()
