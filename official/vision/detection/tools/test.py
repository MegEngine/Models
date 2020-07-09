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
import sys
from multiprocessing import Process, Queue
from tqdm import tqdm

import numpy as np

import megengine as mge
from megengine import jit
from megengine.data import DataLoader, SequentialSampler

from official.vision.detection.tools.data_mapper import data_mapper
from official.vision.detection.tools.utils import DetEvaluator

logger = mge.get_logger(__name__)


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
    return parser


def main():
    # pylint: disable=import-outside-toplevel
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    parser = make_parser()
    args = parser.parse_args()

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

        results_list = list()
        result_queue = Queue(2000)
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
                ),
            )
            proc.start()
            procs.append(proc)

        sys.path.insert(0, os.path.dirname(args.file))
        current_network = importlib.import_module(
            os.path.basename(args.file).split(".")[0]
        )
        cfg = current_network.Cfg()
        num_imgs = dict(coco=5000, objects365=30000)

        for _ in tqdm(range(num_imgs[cfg.test_dataset["name"]])):
            results_list.append(result_queue.get())
        for p in procs:
            p.join()

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


def worker(
    net_file, model_file, data_dir, worker_id, total_worker, result_queue,
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

    @jit.trace(symbolic=True)
    def val_func():
        pred = model(model.inputs)
        return pred

    sys.path.insert(0, os.path.dirname(net_file))
    current_network = importlib.import_module(os.path.basename(net_file).split(".")[0])
    model = current_network.Net(current_network.Cfg(), batch_size=1)
    model.eval()
    evaluator = DetEvaluator(model)
    state_dict = mge.load(model_file)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

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


def build_dataloader(rank, world_size, data_dir, cfg):
    val_dataset = data_mapper[cfg.test_dataset["name"]](
        os.path.join(data_dir, cfg.test_dataset["name"], cfg.test_dataset["root"]),
        os.path.join(data_dir, cfg.test_dataset["name"], cfg.test_dataset["ann_file"]),
        order=["image", "info"],
    )
    val_sampler = SequentialSampler(val_dataset, 1, world_size=world_size, rank=rank)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, num_workers=2)
    return val_dataloader


if __name__ == "__main__":
    main()
