# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import json

import cv2
import megengine as mge
import megengine.data.transform as T
import megengine.functional as F
import megengine.jit as jit
import numpy as np

import model as M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="resnet50_frelu", type=str)
    parser.add_argument("-m", "--model", default=None, type=str)
    parser.add_argument("-i", "--image", default=None, type=str)
    args = parser.parse_args()

    model = getattr(M, args.arch)(pretrained=(args.model is None))
    if args.model:
        state_dict = mge.load(args.model)
        model.load_state_dict(state_dict)

    if args.image is None:
        path = "../../../assets/cat.jpg" # please find the files in https://github.com/MegEngine/Models/tree/master/official/assets
    else:
        path = args.image
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(
                mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
            ),  # BGR
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
    probs = infer_func(processed_img)

    top_probs, classes = F.top_k(probs, k=5, descending=True)

    with open("../../../assets/imagenet_class_info.json") as fp: # please find the files in https://github.com/MegEngine/Models/tree/master/official/assets
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
