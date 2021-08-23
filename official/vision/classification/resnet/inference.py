# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import json

# pylint: disable=import-error
import model as resnet_model

import cv2
import numpy as np

import megengine
import megengine.data.transform as T
import megengine.functional as F

logging = megengine.logger.get_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="resnet18", type=str)
    parser.add_argument("-m", "--model", default=None, type=str)
    parser.add_argument("-i", "--image", default=None, type=str)
    args = parser.parse_args()

    model = resnet_model.__dict__[args.arch](pretrained=(args.model is None))
    if args.model is not None:
        logging.info("load from checkpoint %s", args.model)
        checkpoint = megengine.load(args.model)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)

    if args.image is None:
        path = "../../../assets/cat.jpg"
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

    def infer_func(processed_img):
        model.eval()
        logits = model(processed_img)
        probs = F.softmax(logits)
        return probs

    processed_img = transform.apply(image)[np.newaxis, :]
    processed_img = megengine.tensor(processed_img, dtype="float32")
    probs = infer_func(processed_img)

    top_probs, classes = F.topk(probs, k=5, descending=True)

    with open("../../../assets/imagenet_class_info.json") as fp:
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
