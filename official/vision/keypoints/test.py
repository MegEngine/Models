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
import os
from multiprocessing import Queue
from tqdm import tqdm

import cv2
import numpy as np

import megengine as mge
import megengine.data.transform as T
import megengine.distributed as dist
from megengine.data import DataLoader, SequentialSampler

import official.vision.keypoints.models as kpm
from official.vision.keypoints.config import Config as cfg
from official.vision.keypoints.dataset import COCOJoints
from official.vision.keypoints.transforms import ExtendBoxes, RandomBoxAffine

logger = mge.get_logger(__name__)


def build_dataloader(rank, world_size, data_root, ann_file):
    val_dataset = COCOJoints(
        data_root, ann_file, image_set="val2017", order=("image", "boxes", "info")
    )
    val_sampler = SequentialSampler(
        val_dataset, cfg.batch_size, world_size=world_size, rank=rank
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        num_workers=4,
        transform=T.Compose(
            transforms=[
                T.Normalize(mean=cfg.img_mean, std=cfg.img_std),
                ExtendBoxes(
                    cfg.test_x_ext,
                    cfg.test_y_ext,
                    cfg.input_shape[1] / cfg.input_shape[0],
                    random_extend_prob=0,
                ),
                RandomBoxAffine(
                    degrees=0,
                    scale=0,
                    output_shape=cfg.input_shape,
                    rotate_prob=0,
                    scale_prob=0,
                ),
                T.ToMode(),
            ],
            order=("image", "boxes", "info"),
        ),
    )
    return val_dataloader


def find_keypoints(pred, bbox):

    heat_prob = pred.copy()
    heat_prob = heat_prob / cfg.heat_range + 1

    border = cfg.test_aug_border
    pred_aug = np.zeros(
        (pred.shape[0], pred.shape[1] + 2 * border, pred.shape[2] + 2 * border),
        dtype=np.float32,
    )
    pred_aug[:, border:-border, border:-border] = pred.copy()
    pred_aug = cv2.GaussianBlur(
        pred_aug.transpose(1, 2, 0),
        (cfg.test_gaussian_kernel, cfg.test_gaussian_kernel),
        0,
    ).transpose(2, 0, 1)

    results = np.zeros((pred_aug.shape[0], 3), dtype=np.float32)
    for i in range(pred_aug.shape[0]):
        lb = pred_aug[i].argmax()
        y, x = np.unravel_index(lb, pred_aug[i].shape)
        if cfg.second_value_aug:
            y -= border
            x -= border

            pred_aug[i, y, x] = 0
            lb = pred_aug[i].argmax()
            py, px = np.unravel_index(lb, pred_aug[i].shape)
            pred_aug[i, py, px] = 0

            py -= border + y
            px -= border + x
            ln = (px ** 2 + py ** 2) ** 0.5
            delta = 0.35
            if ln > 1e-3:
                x += delta * px / ln
                y += delta * py / ln

            lb = pred_aug[i].argmax()
            py, px = np.unravel_index(lb, pred_aug[i].shape)
            pred_aug[i, py, px] = 0

            py -= border + y
            px -= border + x
            ln = (px ** 2 + py ** 2) ** 0.5
            delta = 0.15
            if ln > 1e-3:
                x += delta * px / ln
                y += delta * py / ln

            lb = pred_aug[i].argmax()
            py, px = np.unravel_index(lb, pred_aug[i].shape)
            pred_aug[i, py, px] = 0

            py -= border + y
            px -= border + x
            ln = (px ** 2 + py ** 2) ** 0.5
            delta = 0.05
            if ln > 1e-3:
                x += delta * px / ln
                y += delta * py / ln
        else:
            y -= border
            x -= border
        x = max(0, min(x, cfg.output_shape[1] - 1))
        y = max(0, min(y, cfg.output_shape[0] - 1))
        skeleton_score = heat_prob[i, int(round(y)), int(round(x))]

        stride = cfg.input_shape[1] / cfg.output_shape[1]

        x = (x + 0.5) * stride - 0.5
        y = (y + 0.5) * stride - 0.5

        bbox_top_leftx, bbox_top_lefty, bbox_bottom_rightx, bbox_bottom_righty = bbox
        x = (
            x / cfg.input_shape[1] * (bbox_bottom_rightx - bbox_top_leftx)
            + bbox_top_leftx
        )
        y = (
            y / cfg.input_shape[0] * (bbox_bottom_righty - bbox_top_lefty)
            + bbox_top_lefty
        )

        results[i, 0] = x
        results[i, 1] = y
        results[i, 2] = skeleton_score

    return results


def worker(
    arch,
    model_file,
    data_root,
    ann_file,
    result_queue,
):
    """
    :param net_file: network description file
    :param model_file: file of dump weights
    :param data_dir: the dataset directory
    :param worker_id: the index of the worker
    :param total_worker: number of gpu for evaluation
    :param result_queue: processing queue
    """

    model = getattr(kpm, arch)()
    model.eval()
    weight = mge.load(model_file)
    weight = weight["state_dict"] if "state_dict" in weight.keys() else weight
    model.load_state_dict(weight)

    loader = build_dataloader(dist.get_rank(), dist.get_world_size(), data_root, ann_file)

    for data_dict in loader:
        img, bbox, info = data_dict

        fliped_img = img[:, :, :, ::-1] - np.zeros_like(img)
        data = np.concatenate([img, fliped_img], 0)
        data = np.ascontiguousarray(data).astype(np.float32)

        outs = model.predict(mge.tensor(data)).numpy()
        preds = outs[: img.shape[0]]
        preds_fliped = outs[img.shape[0]:, cfg.keypoint_flip_order, :, ::-1]
        preds = (preds + preds_fliped) / 2

        for i in range(preds.shape[0]):

            results = find_keypoints(preds[i], bbox[i, 0])

            final_score = float(results[:, -1].mean() * info[-1][i])
            image_id = int(info[-2][i])

            keypoints = results.copy()
            keypoints[:, -1] = 1
            keypoints = keypoints.reshape(-1,).tolist()
            instance = {
                "image_id": image_id,
                "category_id": 1,
                "score": final_score,
                "keypoints": keypoints,
            }

            result_queue.put(instance)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--ngpus", default=None, type=int)
    parser.add_argument("-b", "--batch_size", default=None, type=int)
    parser.add_argument(
        "-s",
        "--save_dir",
        default="/data/models/simplebaseline_res50/results/",
        type=str,
    )
    parser.add_argument(
        "-dt",
        "--dt_file",
        default="COCO_val2017_detections_AP_H_56_person.json",
        type=str,
    )
    parser.add_argument("-se", "--start_epoch", default=-1, type=int)
    parser.add_argument("-ee", "--end_epoch", default=-1, type=int)
    parser.add_argument(
        "-md",
        "--model_dir",
        default="/data/models/simplebaseline_res50_256x192/",
        type=str,
    )
    parser.add_argument("-tf", "--test_freq", default=1, type=int)

    parser.add_argument(
        "-a",
        "--arch",
        default="simplebaseline_res50",
        type=str,
        choices=cfg.model_choices,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="/data/models/simplebaseline_res50_256x192/epoch_199.pkl",
        type=str,
    )
    return parser


def main():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    parser = make_parser()
    args = parser.parse_args()
    model_name = "{}_{}x{}".format(args.arch, cfg.input_shape[0], cfg.input_shape[1])
    save_dir = os.path.join(args.save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mge.set_log_file(os.path.join(save_dir, "log.txt"))

    args.ngpus = (
        dist.helper.get_device_count_by_fork("gpu")
        if args.ngpus is None
        else args.ngpus
    )
    cfg.batch_size = cfg.batch_size if args.batch_size is None else args.batch_size

    dt_path = os.path.join(cfg.data_root, "person_detection_results", args.dt_file)
    dets = json.load(open(dt_path, "r"))

    gt_path = os.path.join(
        cfg.data_root, "annotations", "person_keypoints_val2017.json"
    )
    eval_gt = COCO(gt_path)
    gt = eval_gt.dataset

    dets = [
        i for i in dets if (i["image_id"] in eval_gt.imgs and i["category_id"] == 1)
    ]
    ann_file = {"images": gt["images"], "annotations": dets}

    if args.end_epoch == -1:
        args.end_epoch = args.start_epoch

    for epoch_num in range(args.start_epoch, args.end_epoch + 1, args.test_freq):
        if args.model:
            model_file = args.model
        else:
            model_file = "{}/epoch_{}.pkl".format(args.model_dir, epoch_num)
        logger.info("Load Model : %s completed", model_file)

        all_results = list()

        result_queue = Queue(5000)

        dist_worker = dist.launcher()(worker)
        dist_worker(
            args.arch,
            model_file,
            cfg.data_root,
            ann_file,
            result_queue,
        )

        for _ in tqdm(range(len(dets))):
            all_results.append(result_queue.get())

        json_name = "log-of-{}_epoch_{}.json".format(args.arch, epoch_num)
        json_path = os.path.join(save_dir, json_name)
        all_results = json.dumps(all_results)
        with open(json_path, "w") as fo:
            fo.write(all_results)
        logger.info("Save to %s finished, start evaluation!", json_path)

        eval_dt = eval_gt.loadRes(json_path)
        cocoEval = COCOeval(eval_gt, eval_dt, iouType="keypoints")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        metrics = [
            "AP",
            "AP@0.5",
            "AP@0.75",
            "APm",
            "APl",
            "AR",
            "AR@0.5",
            "AR@0.75",
            "ARm",
            "ARl",
        ]
        logger.info("mmAP".center(32, "-"))
        for i, m in enumerate(metrics):
            logger.info("|\t%s\t|\t%.03f\t|", m, cocoEval.stats[i])
        logger.info("-" * 32)


if __name__ == "__main__":
    main()
