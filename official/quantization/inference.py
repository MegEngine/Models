# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2019 Megvii Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
# ------------------------------------------------------------------------------
"""Finetune a pretrained fp32 with int8 quantization aware training(QAT)"""
import argparse
import json

import cv2
import megengine as mge
import megengine.data.transform as T
import megengine.functional as F
import megengine.jit as jit
import megengine.quantization as Q
import numpy as np
from tqdm import tqdm

import models

logger = mge.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="resnet18", type=str)
        # choices=["resnet18", "shufflenet_v1_x1_0_g3"])
    parser.add_argument("-c", "--checkpoint", default=None, type=str)
    parser.add_argument("-i", "--image", default=None, type=str)

    parser.add_argument("-m", "--mode", default="quantized", type=str,
        choices=["normal", "qat", "quantized"],
        help="Quantization Mode\n"
             "normal: no quantization, using float32\n"
             "qat: quantization aware training, simulate int8\n"
             "quantized: convert mode to int8 quantized, inference only")
    args = parser.parse_args()

    if args.mode == "quantized":
        mge.set_default_device("cpux")

    model = models.__dict__[args.arch]()

    if args.mode != "normal":
        Q.quantize_qat(model, Q.ema_fakequant_qconfig)

    if args.checkpoint:
        logger.info("Load pretrained weights from %s", args.checkpoint)
        ckpt = mge.load(args.checkpoint)
        ckpt = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(ckpt, strict=False)

    if args.mode == "quantized":
        Q.quantize(model)

    if args.image is None:
        path = "../assets/cat.jpg"
    else:
        path = args.image
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=128),
            T.ToMode("CHW"),
        ]
    )

    @jit.trace(symbolic=False)
    def infer_func(processed_img):
        model.eval()
        logits = model(processed_img)
        probs = F.softmax(logits)
        return probs

    processed_img = transform.apply(image)[np.newaxis, :]

    if args.mode == "normal":
        processed_img = processed_img.astype("float32")

    # for _ in tqdm(range(100)):
    probs = infer_func(processed_img)

    top_probs, classes = F.top_k(probs, k=5, descending=True)

    with open("../assets/imagenet_class_info.json") as fp:
        imagenet_class_index = json.load(fp)

    for rank, (prob, classid) in enumerate(
        zip(top_probs.numpy().reshape(-1), classes.numpy().reshape(-1))
    ):
        print(
            "{}: class = {:20s} with probability = {:4.1f} %".format(
                rank, imagenet_class_index[str(classid)][1], 100 * prob
            )
        )
if __name__ == "__main__":
    main()
