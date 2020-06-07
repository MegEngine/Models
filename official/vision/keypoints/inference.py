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
import os

import cv2
import megengine as mge
import numpy as np
from megengine import jit
import math

import sys
sys.path.insert(0, '../../../../')
from transforms import get_affine_transform
from config import Config as cfg

import official.vision.keypoints.models as M
import official.vision.detection.retinanet_res50_1x_800size as Det
from official.vision.detection.tools.test import DetEvaluator
from official.vision.keypoints.test import find_keypoints

logger = mge.get_logger(__name__)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--arch",
        default="SimpleBaseline_Res50",
        type=str,
        choices=[
            "SimpleBaseline_Res50",
            "SimpleBaseline_Res101",
            "SimpleBaseline_Res152"
        ],
    )
    parser.add_argument(
        "-det",
        "--detector",
        default="retinanet_res50_1x_800size",
        type=str,
    )

    parser.add_argument(
        "-m", "--model", default='/data/simplebaseline_256x192_71_2.pkl', type=str)
    parser.add_argument(
        "-image", "--image", default='/data/test_keyoint2.jpeg', type=str)
    return parser

def vis_skeleton(img, all_keypoints):
    
    canvas = img.copy()
    for keypoints in all_keypoints:
        for ind, skeleton in enumerate(cfg.vis_skeletons):
            jotint1 = skeleton[0]
            jotint2 = skeleton[1]

            X = np.array([
                keypoints[jotint1, 0],
                keypoints[jotint2, 0]
            ])
            
            Y = np.array([
                keypoints[jotint1, 1],
                keypoints[jotint2, 1]
            ])

            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5

            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(
                length / 2), 4), int(angle), 0, 360, 1)

            cur_canvas = canvas.copy()
            cv2.fillConvexPoly(cur_canvas, polygon, cfg.vis_colors[ind])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas


def main():

    parser = make_parser()
    args = parser.parse_args()

    detector = getattr(Det, args.detector)(pretrained=True)
    detector.eval()
    logger.info("Load Model : %s completed", args.detector)

    keypoint_model = getattr(M, args.arch)()
    keypoint_model.load_state_dict(mge.load(args.model)["state_dict"])
    keypoint_model.eval()
    logger.info("Load Model : %s completed", args.arch)

    @jit.trace(symbolic=True)
    def det_func():
        pred = detector(detector.inputs)
        return pred

    @jit.trace(symbolic=True)
    def keypoint_func():
        pred = keypoint_model.predict()
        return pred

    ori_img = cv2.imread(args.image)
    data, im_info = DetEvaluator.process_inputs(
        ori_img.copy(), detector.cfg.test_image_short_size, detector.cfg.test_image_max_size,
    )
    detector.inputs["im_info"].set_value(im_info)
    detector.inputs["image"].set_value(data.astype(np.float32))

    logger.info("Detecting Humans")
    evaluator = DetEvaluator(detector)
    det_res = evaluator.predict(det_func)

    normalized_img = (ori_img - np.array(cfg.IMG_MEAN).reshape(1,1,3)) / np.array(cfg.IMG_STD).reshape(1, 1, 3)

    logger.info("Detecting Keypoints")
    all_keypoints = []
    for det in det_res:
        cls_id = int(det[5] + 1)
        if cls_id == 1:
            bbox = det[:4]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            center_x = (bbox[0] + bbox[2]) / 2 
            center_y = (bbox[1] + bbox[3]) / 2 

            extend_w = w * (1 + cfg.test_x_ext)
            extend_h = h * (1 + cfg.test_y_ext)

            w_h_ratio = cfg.input_shape[1] / cfg.input_shape[0]
            if extend_w / extend_h > w_h_ratio:
                extend_h = extend_w / w_h_ratio
            else:
                extend_w = extend_h * w_h_ratio

            trans = get_affine_transform(
                np.array([center_x, center_y]),
                np.array([extend_h, extend_w]),
                1, 0, cfg.input_shape
                )
            
            croped_img = cv2.warpAffine(
                normalized_img, 
                trans, 
                (int(cfg.input_shape[1]), int(cfg.input_shape[0])), 
                flags=cv2.INTER_LINEAR, borderValue=0)
            
            fliped_img = croped_img[:,::-1]
            keypoint_input = np.stack([croped_img, fliped_img],0)
            keypoint_input = keypoint_input.transpose(0, 3, 1, 2)
            keypoint_input = np.ascontiguousarray(keypoint_input).astype(np.float32)

            keypoint_model.inputs["image"].set_value(keypoint_input)

            outs = keypoint_func()
            outs = outs.numpy()
            pred = outs[0]
            fliped_pred = outs[1][cfg.keypoint_flip_order][:, :, ::-1]
            pred = (pred + fliped_pred) / 2

            keypoints = find_keypoints(pred, bbox)
            all_keypoints.append(keypoints)

    logger.info("Visualizing")
    canvas = vis_skeleton(ori_img, all_keypoints)
    cv2.imwrite('vis_skeleton2.jpg', canvas)  

if __name__ == "__main__":
    main()
