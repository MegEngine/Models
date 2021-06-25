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
from tqdm import tqdm

import megengine as mge
import megengine.distributed as dist
from megengine.data import DataLoader

from official.vision.detection.tools.utils import (
    DetEvaluator,
    InferenceSampler,
    PseudoDetectionDataset,
    import_from_file
)

logger = mge.get_logger(__name__)
logger.setLevel("INFO")
mge.device.set_prealloc_config(1024, 1024, 256 * 1024 * 1024, 4.0)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", default="net.py", type=str, help="net description file"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument(
        "-n", "--devices", default=1, type=int, help="total number of gpus for testing",
    )
    parser.add_argument(
        "-d", "--dataset_dir", default="/data/datasets", type=str,
    )
    parser.add_argument("-se", "--start_epoch", default=-1, type=int)
    parser.add_argument("-ee", "--end_epoch", default=-1, type=int)
    return parser


def main():
    # pylint: disable=import-outside-toplevel,too-many-branches,too-many-statements
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    parser = make_parser()
    args = parser.parse_args()

    current_network = import_from_file(args.file)
    cfg = current_network.Cfg()

    if args.weight_file:
        args.start_epoch = args.end_epoch = -1
    else:
        if args.start_epoch == -1:
            args.start_epoch = cfg.max_epoch - 1
        if args.end_epoch == -1:
            args.end_epoch = args.start_epoch
        assert 0 <= args.start_epoch <= args.end_epoch < cfg.max_epoch

    for epoch_num in range(args.start_epoch, args.end_epoch + 1):
        if args.weight_file:
            weight_file = args.weight_file
        else:
            weight_file = "log-of-{}/epoch_{}.pkl".format(
                os.path.basename(args.file).split(".")[0], epoch_num
            )

        if args.devices > 1:
            dist_worker = dist.launcher(n_gpus=args.devices)(worker)
            result_list = dist_worker(current_network, weight_file, args.dataset_dir)
            result_list = sum(result_list, [])
        else:
            result_list = worker(current_network, weight_file, args.dataset_dir)

        all_results = DetEvaluator.format(result_list, cfg)
        if args.weight_file:
            json_path = "{}_{}.json".format(
                os.path.basename(args.file).split(".")[0],
                os.path.basename(args.weight_file).split(".")[0],
            )
        else:
            json_path = "log-of-{}/epoch_{}.json".format(
                os.path.basename(args.file).split(".")[0], epoch_num
            )
        all_results = json.dumps(all_results)

        with open(json_path, "w") as fo:
            fo.write(all_results)
        logger.info("Save results to %s, start evaluation!", json_path)

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


def worker(current_network, weight_file, dataset_dir):
    cfg = current_network.Cfg()
    cfg.backbone_pretrained = False

    model = current_network.Net(cfg)
    model.eval()

    state_dict = mge.load(weight_file)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

    evaluator = DetEvaluator(model)

    test_loader = build_dataloader(dataset_dir, model.cfg)
    if dist.get_rank() == 0:
        test_loader = tqdm(test_loader)

    result_list = []
    for data in test_loader:
        image, im_info = DetEvaluator.process_inputs(
            data[0][0],
            model.cfg.test_image_short_size,
            model.cfg.test_image_max_size,
        )
        pred_res = evaluator.predict(
            image=mge.tensor(image),
            im_info=mge.tensor(im_info)
        )
        result = {
            "pred_boxes": pred_res,
            "image_id": int(data[1][2][0].split(".")[0].split("_")[-1]),
        }
        result_list.append(result)
    return result_list


# pylint: disable=unused-argument
def build_dataloader(dataset_dir, cfg):
    val_dataset = PseudoDetectionDataset(length=5000, order=["image", "info"])
    val_sampler = InferenceSampler(val_dataset, 1)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, num_workers=2)
    return val_dataloader


if __name__ == "__main__":
    main()
