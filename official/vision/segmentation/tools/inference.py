# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse

import cv2
import numpy as np

import megengine as mge

from official.vision.segmentation.tools.utils import class_colors, import_from_file

logger = mge.get_logger(__name__)
logger.setLevel("INFO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", default="net.py", type=str, help="net description file"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument("-i", "--image", type=str)
    args = parser.parse_args()

    current_network = import_from_file(args.file)
    cfg = current_network.Cfg()
    cfg.backbone_pretrained = False
    model = current_network.Net(cfg)
    model.eval()

    state_dict = mge.load(args.weight_file)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

    img = cv2.imread(args.image)
    pred = inference(img, model)
    cv2.imwrite("results.jpg", pred)


def inference(img, model):
    def pred_func(data):
        pred = model(data)
        return pred

    img = (
        img.astype("float32") - np.array(model.cfg.img_mean)
    ) / np.array(model.cfg.img_std)
    ori_h, ori_w = img.shape[:2]
    img = cv2.resize(img, (model.cfg.val_height, model.cfg.val_width))
    img = img.transpose(2, 0, 1)[np.newaxis]

    pred = pred_func(mge.tensor(img))
    pred = pred.numpy().squeeze().argmax(0)
    pred = cv2.resize(
        pred.astype("uint8"), (ori_w, ori_h), interpolation=cv2.INTER_NEAREST
    )

    out = np.zeros((ori_h, ori_w, 3))
    nids = np.unique(pred)
    for t in nids:
        out[pred == t] = class_colors[t]
    return out


if __name__ == "__main__":
    main()
