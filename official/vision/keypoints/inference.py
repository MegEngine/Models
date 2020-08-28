# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import math

import cv2
import numpy as np

import megengine as mge

import official.vision.detection.configs as Det
import official.vision.keypoints.models as kpm
from official.vision.detection.tools.nms import py_cpu_nms
from official.vision.detection.tools.utils import DetEvaluator
from official.vision.keypoints.config import Config as cfg
from official.vision.keypoints.test import find_keypoints
from official.vision.keypoints.transforms import get_affine_transform

logger = mge.get_logger(__name__)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--arch",
        default="simplebaseline_res50",
        type=str,
        choices=cfg.model_choices,
    )
    parser.add_argument(
        "-det", "--detector", default="retinanet_res50_coco_1x_800size", type=str,
    )

    parser.add_argument(
        "-m",
        "--model",
        default="/data/models/simplebaseline_res50_256x192/epoch_199.pkl",
        type=str,
    )
    parser.add_argument(
        "-image", "--image", default="/data/test_keypoint.jpeg", type=str
    )
    return parser


class KeypointEvaluator:
    def __init__(self, detect_model, keypoint_model):

        self.detector = detect_model

        self.keypoint_model = keypoint_model

    def detect_persons(self, image):

        data, im_info = DetEvaluator.process_inputs(
            image.copy(),
            self.detector.cfg.test_image_short_size,
            self.detector.cfg.test_image_max_size,
        )

        evaluator = DetEvaluator(self.detector)

        det_res = evaluator.predict(image=mge.tensor(data), im_info=mge.tensor(im_info))

        persons = []
        for d in det_res:
            cls_id = int(d[5] + 1)
            if cls_id == 1:
                bbox = d[:5]
                persons.append(bbox)
        persons = np.array(persons).reshape(-1, 5)
        keep = py_cpu_nms(persons, cfg.nms_thr)
        return persons[keep]

    def predict_single_person(self, image, bbox):
        bbox = bbox[:4]

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

        bbox = np.array(
            [
                center_x - extend_w / 2,
                center_y - extend_h / 2,
                center_x + extend_w / 2,
                center_y + extend_h / 2,
            ]
        ).reshape(4,)

        trans = get_affine_transform(
            np.array([center_x, center_y]),
            np.array([extend_h, extend_w]),
            1,
            0,
            cfg.input_shape,
        )

        croped_img = cv2.warpAffine(
            image,
            trans,
            (int(cfg.input_shape[1]), int(cfg.input_shape[0])),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )

        fliped_img = croped_img[:, ::-1]
        keypoint_input = np.stack([croped_img, fliped_img], 0)
        keypoint_input = keypoint_input.transpose(0, 3, 1, 2)
        keypoint_input = np.ascontiguousarray(keypoint_input).astype(np.float32)

        outs = self.keypoint_model.predict(mge.tensor(keypoint_input))
        outs = outs.numpy()
        pred = outs[0]
        fliped_pred = outs[1][cfg.keypoint_flip_order][:, :, ::-1]
        pred = (pred + fliped_pred) / 2

        keypoints = find_keypoints(pred, bbox)

        return keypoints

    def predict(self, image, bboxes):
        normalized_img = (image - np.array(cfg.img_mean).reshape(1, 1, 3)) / np.array(
            cfg.img_std
        ).reshape(1, 1, 3)
        all_keypoints = []
        for bbox in bboxes:
            keypoints = self.predict_single_person(normalized_img, bbox)
            all_keypoints.append(keypoints)
        return all_keypoints

    @staticmethod
    def vis_skeletons(img, all_keypoints):
        canvas = img.copy()
        for keypoints in all_keypoints:
            for ind, skeleton in enumerate(cfg.vis_skeletons):
                jotint1 = skeleton[0]
                jotint2 = skeleton[1]

                X = np.array([keypoints[jotint1, 0], keypoints[jotint2, 0]])

                Y = np.array([keypoints[jotint1, 1], keypoints[jotint2, 1]])

                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5

                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                polygon = cv2.ellipse2Poly(
                    (int(mX), int(mY)), (int(length / 2), 4), int(angle), 0, 360, 1
                )

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

    keypoint_model = getattr(kpm, args.arch)()
    keypoint_model.load_state_dict(mge.load(args.model)["state_dict"])
    keypoint_model.eval()
    logger.info("Load Model : %s completed", args.arch)

    evaluator = KeypointEvaluator(detector, keypoint_model)

    image = cv2.imread(args.image)

    logger.info("Detecting Humans")
    person_boxes = evaluator.detect_persons(image)

    logger.info("Detecting Keypoints")
    all_keypoints = evaluator.predict(image, person_boxes)

    logger.info("Visualizing")
    canvas = evaluator.vis_skeletons(image, all_keypoints)
    cv2.imwrite("vis_skeleton.jpg", canvas)


if __name__ == "__main__":
    main()
