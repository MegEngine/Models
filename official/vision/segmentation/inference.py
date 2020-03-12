# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse

import cv2
import megengine as mge
import megengine.data.dataset as dataset
import megengine.jit as jit
import numpy as np

from megengine.utils.http_download import download_from_url
from official.vision.segmentation.deeplabv3plus import DeepLabV3Plus


class Config:
    NUM_CLASSES = 21
    IMG_SIZE = 512
    IMG_MEAN = [103.530, 116.280, 123.675]
    IMG_STD = [57.375, 57.120, 58.395]


cfg = Config()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=None, help="inference image")
    parser.add_argument("--model_path", type=str, default=None, help="inference model")
    args = parser.parse_args()

    net = load_model(args.model_path)
    if args.image_path is None:
        download_from_url("https://data.megengine.org.cn/images/cat.jpg", "test.jpg")
        img = cv2.imread("test.jpg")
    else:
        img = cv2.imread(args.image_path)
    pred = inference(img, net)
    cv2.imwrite("out.jpg", pred)


def load_model(model_path):
    model_dict = mge.load(model_path)
    net = DeepLabV3Plus(class_num=cfg.NUM_CLASSES)
    net.load_state_dict(model_dict["state_dict"])
    print("load model %s" % (model_path))
    net.eval()
    return net


def inference(img, net):
    @jit.trace(symbolic=True, opt_level=2)
    def pred_fun(data, net=None):
        net.eval()
        pred = net(data)
        return pred

    img = (img.astype("float32") - np.array(cfg.IMG_MEAN)) / np.array(cfg.IMG_STD)
    orih, oriw = img.shape[:2]
    img = cv2.resize(img, (cfg.IMG_SIZE, cfg.IMG_SIZE))
    img = img.transpose(2, 0, 1)[np.newaxis]

    data = mge.tensor()
    data.set_value(img)
    pred = pred_fun(data, net=net)
    pred = pred.numpy().squeeze().argmax(0)
    pred = cv2.resize(
        pred.astype("uint8"), (oriw, orih), interpolation=cv2.INTER_NEAREST
    )

    class_colors = dataset.PascalVOC.class_colors
    out = np.zeros((orih, oriw, 3))
    nids = np.unique(pred)
    for t in nids:
        out[pred == t] = class_colors[t]
    return out


if __name__ == "__main__":
    main()
