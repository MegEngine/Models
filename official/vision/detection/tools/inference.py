# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import importlib
import os
import sys

import cv2
import numpy as np

import megengine as mge
from megengine import jit
from megengine.data.dataset import COCO

from official.vision.detection.tools.utils import DetEvaluator

logger = mge.get_logger(__name__)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", default="net.py", type=str, help="net description file"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument("-i", "--image", default="example.jpg", type=str)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    logger.info("Load Model : %s completed", args.weight_file)

    @jit.trace(symbolic=True)
    def val_func():
        pred = model(model.inputs)
        return pred

    sys.path.insert(0, os.path.dirname(args.file))
    current_network = importlib.import_module(os.path.basename(args.file).split(".")[0])
    cfg = current_network.Cfg()
    cfg.backbone_pretrained = False
    model = current_network.Net(cfg, batch_size=1)
    model.eval()
    state_dict = mge.load(args.weight_file)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

    evaluator = DetEvaluator(model)

    ori_img = cv2.imread(args.image)
    data, im_info = DetEvaluator.process_inputs(
        ori_img.copy(), model.cfg.test_image_short_size, model.cfg.test_image_max_size,
    )
    model.inputs["im_info"].set_value(im_info)
    model.inputs["image"].set_value(data.astype(np.float32))
    pred_res = evaluator.predict(val_func)
    res_img = DetEvaluator.vis_det(
        ori_img, pred_res, is_show_label=True, classes=COCO.class_names,
    )
    cv2.imwrite("results.jpg", res_img)


if __name__ == "__main__":
    main()
