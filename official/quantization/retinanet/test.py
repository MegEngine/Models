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
import json
import os
import random
import sys
from multiprocessing import Process, Queue

import cv2
import megengine as mge
import numpy as np
from megengine import jit
from megengine.data import DataLoader, SequentialSampler
import megengine.quantization.quantize as quantize
import megengine.quantization as Q
from tqdm import tqdm

from official.vision.detection.tools.data_mapper import data_mapper
from official.vision.detection.tools.nms import py_cpu_nms

logger = mge.get_logger(__name__)


class DetEvaluator:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def get_hw_by_short_size(im_height, im_width, short_size, max_size):
        """get height and width by short size

           Args:
               im_height(int): height of original image, e.g. 800
               im_width(int): width of original image, e.g. 1000
               short_size(int): short size of transformed image. e.g. 800
               max_size(int): max size of transformed image. e.g. 1333

           Returns:
               resized_height(int): height of transformed image
               resized_width(int): width of transformed image
        """

        im_size_min = np.min([im_height, im_width])
        im_size_max = np.max([im_height, im_width])
        scale = (short_size + 0.0) / im_size_min
        if scale * im_size_max > max_size:
            scale = (max_size + 0.0) / im_size_max

        resized_height, resized_width = (
            int(round(im_height * scale)),
            int(round(im_width * scale)),
        )
        return resized_height, resized_width

    @staticmethod
    def process_inputs(img, short_size, max_size, flip=False):
        original_height, original_width, _ = img.shape
        resized_height, resized_width = DetEvaluator.get_hw_by_short_size(
            original_height, original_width, short_size, max_size
        )
        resized_img = cv2.resize(
            img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR,
        )
        resized_img = cv2.flip(resized_img, 1) if flip else resized_img
        trans_img = np.ascontiguousarray(
            resized_img.transpose(2, 0, 1)[None, :, :, :], dtype=np.uint8
        )
        img_mean = np.array([128, 128, 128])
        trans_img = trans_img - img_mean[None, :, None, None]
        im_info = np.array(
            [(resized_height, resized_width, original_height, original_width)],
            dtype=np.float32,
        )
        return trans_img, im_info

    def predict(self, val_func):
        """
        Args:
            val_func(callable): model inference function

        Returns:
            results boxes: detection model output
        """
        model = self.model

        box_cls, box_delta = val_func()
        box_cls, box_delta = box_cls.numpy(), box_delta.numpy()
        dtboxes_all = list()
        all_inds = np.where(box_cls > model.cfg.test_cls_threshold)

        for c in range(0, model.cfg.num_classes):
            inds = np.where(all_inds[1] == c)[0]
            inds = all_inds[0][inds]
            scores = box_cls[inds, c]
            if model.cfg.class_aware_box:
                bboxes = box_delta[inds, c, :]
            else:
                bboxes = box_delta[inds, :]

            dtboxes = np.hstack((bboxes, scores[:, np.newaxis])).astype(np.float32)

            if dtboxes.size > 0:
                keep = py_cpu_nms(dtboxes, model.cfg.test_nms)
                dtboxes = np.hstack(
                    (dtboxes[keep], np.ones((len(keep), 1), np.float32) * c)
                ).astype(np.float32)
                dtboxes_all.extend(dtboxes)

        if len(dtboxes_all) > model.cfg.test_max_boxes_per_image:
            dtboxes_all = sorted(dtboxes_all, reverse=True, key=lambda i: i[4])[
                : model.cfg.test_max_boxes_per_image
            ]

        dtboxes_all = np.array(dtboxes_all, dtype=np.float)
        return dtboxes_all

    @staticmethod
    def format(results, cfg):
        dataset_class = data_mapper[cfg.test_dataset["name"]]

        all_results = []
        for record in results:
            image_filename = record["image_id"]
            boxes = record["det_res"]
            if len(boxes) <= 0:
                continue
            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
            for box in boxes:
                elem = dict()
                elem["image_id"] = image_filename
                elem["bbox"] = box[:4].tolist()
                elem["score"] = box[4]
                elem["category_id"] = dataset_class.classes_originID[
                    dataset_class.class_names[int(box[5])]
                ]
                all_results.append(elem)
        return all_results

    @staticmethod
    def vis_det(
        img,
        dets,
        is_show_label=True,
        classes=None,
        thresh=0.3,
        name="detection",
        return_img=True,
    ):
        img = np.array(img)
        colors = dict()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in dets:
            bb = det[:4].astype(int)
            if is_show_label:
                cls_id = int(det[5])
                score = det[4]

                if cls_id == 0:
                    continue
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (
                            random.random() * 255,
                            random.random() * 255,
                            random.random() * 255,
                        )

                    cv2.rectangle(
                        img, (bb[0], bb[1]), (bb[2], bb[3]), colors[cls_id], 3
                    )

                    if classes and len(classes) > cls_id:
                        cls_name = classes[cls_id]
                    else:
                        cls_name = str(cls_id)
                    cv2.putText(
                        img,
                        "{:s} {:.3f}".format(cls_name, score),
                        (bb[0], bb[1] - 2),
                        font,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
            else:
                cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)

        if return_img:
            return img
        cv2.imshow(name, img)
        while True:
            c = cv2.waitKey(100000)
            if c == ord("d"):
                return None
            elif c == ord("n"):
                break


def build_dataloader(rank, world_size, data_dir, cfg):
    val_dataset = data_mapper[cfg.test_dataset["name"]](
        os.path.join(data_dir, cfg.test_dataset["name"], cfg.test_dataset["root"]),
        os.path.join(data_dir, cfg.test_dataset["name"], cfg.test_dataset["ann_file"]),
        order=["image", "info"],
    )
    val_sampler = SequentialSampler(val_dataset, 1, world_size=world_size, rank=rank)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, num_workers=2)
    return val_dataloader


def worker(
    net_file, model_file, data_dir, worker_id, total_worker, result_queue, test_mode
):
    """
    :param net_file: network description file
    :param model_file: file of dump weights
    :param data_dir: the dataset directory
    :param worker_id: the index of the worker
    :param total_worker: number of gpu for evaluation
    :param result_queue: processing queue
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)

    @jit.trace(symbolic=True, opt_level=2)
    def val_func():
        pred = model(model.inputs)
        return pred

    sys.path.insert(0, os.path.dirname(net_file))
    current_network = importlib.import_module(os.path.basename(net_file).split(".")[0])
    model = current_network.Net(current_network.Cfg(), batch_size=1)

    if test_mode != "fp32":
        # QAT
        model.head.disable_quantize()
        quantize.quantize_qat(model, qconfig=Q.ema_fakequant_qconfig)

    model.load_state_dict(mge.load(model_file)["state_dict"], strict=True)

    if test_mode == "quantized":
        quantize.quantize(model)
    model.eval()
    evaluator = DetEvaluator(model)

    loader = build_dataloader(worker_id, total_worker, data_dir, model.cfg)
    for data_dict in loader:
        data, im_info = DetEvaluator.process_inputs(
            data_dict[0][0],
            model.cfg.test_image_short_size,
            model.cfg.test_image_max_size,
        )
        model.inputs["im_info"].set_value(im_info)
        model.inputs["image"].set_value(data.astype(np.float32))

        pred_res = evaluator.predict(val_func)
        result_queue.put_nowait(
            {
                "det_res": pred_res,
                "image_id": int(data_dict[1][2][0].split(".")[0].split("_")[-1]),
            }
        )


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=1, type=int)
    parser.add_argument("-n", "--ngpus", default=1, type=int)
    parser.add_argument(
        "-f", "--file", default="net.py", type=str, help="net description file"
    )
    parser.add_argument("-d", "--dataset_dir", default="/data/datasets", type=str)
    parser.add_argument("-se", "--start_epoch", default=-1, type=int)
    parser.add_argument("-ee", "--end_epoch", default=-1, type=int)
    parser.add_argument("-m", "--model", default=None, type=str)
    parser.add_argument("--mode", default="fp32", type=str)
    return parser


def main():
    # pylint: disable=import-outside-toplevel
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    parser = make_parser()
    args = parser.parse_args()

    assert args.mode in ["fp32", "qat", "quantized"], "{} mode not supported".format(args.mode)
    if args.end_epoch == -1:
        args.end_epoch = args.start_epoch

    for epoch_num in range(args.start_epoch, args.end_epoch + 1):
        if args.model:
            model_file = args.model
        else:
            model_file = "log-of-{}/epoch_{}.pkl".format(
                os.path.basename(args.file).split(".")[0], epoch_num
            )
        logger.info("Load Model : %s completed", model_file)

        if args.mode == "quantized":
            mge.set_default_device("cpux")
            logger.warning("quantized mode use cpu only")

        results_list = list()
        result_queue = Queue(2000)

        if args.ngpus > 0:
            procs = []
            for i in range(args.ngpus):
                proc = Process(
                    target=worker,
                    args=(
                        args.file,
                        model_file,
                        args.dataset_dir,
                        i,
                        args.ngpus,
                        result_queue,
                        args.mode,
                    ),
                )
                proc.start()
                procs.append(proc)

            for _ in tqdm(range(5000)):
                results_list.append(result_queue.get())
            for p in procs:
                p.join()

        else:
            worker(
                args.file, model_file, args.dataset_dir,
                0, 1, result_queue, args.mode,
            )

        sys.path.insert(0, os.path.dirname(args.file))
        current_network = importlib.import_module(
            os.path.basename(args.file).split(".")[0]
        )
        cfg = current_network.Cfg()
        all_results = DetEvaluator.format(results_list, cfg)
        json_path = "log-of-{}/epoch_{}.json".format(
            os.path.basename(args.file).split(".")[0], epoch_num
        )
        all_results = json.dumps(all_results)

        with open(json_path, "w") as fo:
            fo.write(all_results)
        logger.info("Save to %s finished, start evaluation!", json_path)

        eval_gt = COCO(
            os.path.join(
                args.dataset_dir, cfg.test_dataset["name"], cfg.test_dataset["ann_file"]
            )
        )
        eval_dt = eval_gt.loadRes(json_path)
        cocoEval = COCOeval(eval_gt, eval_dt, iouType="bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        metrics = [
            "AP",
            "AP@0.5",
            "AP@0.75",
            "APs",
            "APm",
            "APl",
            "AR@1",
            "AR@10",
            "AR@100",
            "ARs",
            "ARm",
            "ARl",
        ]
        logger.info("mmAP".center(32, "-"))
        for i, m in enumerate(metrics):
            logger.info("|\t%s\t|\t%.03f\t|", m, cocoEval.stats[i])
        logger.info("-" * 32)


if __name__ == "__main__":
    main()
