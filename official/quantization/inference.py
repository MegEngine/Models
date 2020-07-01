# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
from megengine.quantization.quantize import quantize, quantize_qat

import models

logger = mge.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="resnet18", type=str)
    parser.add_argument("-c", "--checkpoint", default=None, type=str)
    parser.add_argument("-i", "--image", default=None, type=str)

    parser.add_argument("-m", "--mode", default="quantized", type=str,
        choices=["normal", "qat", "quantized"],
        help="Quantization Mode\n"
             "normal: no quantization, using float32\n"
             "qat: quantization aware training, simulate int8\n"
             "quantized: convert mode to int8 quantized, inference only")
    parser.add_argument("--dump", action="store_true",
        help="Dump quantized model")
    args = parser.parse_args()

    if args.mode == "quantized":
        mge.set_default_device("cpux")

    model = models.__dict__[args.arch]()

    if args.mode != "normal":
        quantize_qat(model, Q.ema_fakequant_qconfig)

    if args.mode == "quantized":
        quantize(model)

    if args.checkpoint:
        logger.info("Load pretrained weights from %s", args.checkpoint)
        ckpt = mge.load(args.checkpoint)
        ckpt = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(ckpt, strict=False)

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

    @jit.trace(symbolic=True)
    def infer_func(processed_img):
        model.eval()
        logits = model(processed_img)
        probs = F.softmax(logits)
        return probs

    processed_img = transform.apply(image)[np.newaxis, :]

    if args.mode == "normal":
        processed_img = processed_img.astype("float32")
    elif args.mode == "quantized":
        processed_img = processed_img.astype("int8")

    probs = infer_func(processed_img)

    top_probs, classes = F.top_k(probs, k=5, descending=True)

    if args.dump:
        output_file = ".".join([args.arch, args.mode, "megengine"])
        logger.info("Dump to {}".format(output_file))
        infer_func.dump(output_file, arg_names=["data"])
        mge.save(model.state_dict(), output_file.replace("megengine", "pkl"))

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
