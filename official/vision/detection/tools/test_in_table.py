#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
from tabulate import tabulate

import megengine as mge

from official.vision.detection.tools.utils import import_from_file

logger = mge.get_logger(__name__)
logger.setLevel("INFO")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--files", nargs="+", default=None, help="all config file"
    )
    parser.add_argument(
        "-j", "--jsons", nargs="+", default=None, help="all json file"
    )
    parser.add_argument(
        "-d", "--dataset_dir", default="/data/Datasets", type=str,
    )
    return parser


def main():
    # pylint: disable=import-outside-toplevel,too-many-branches,too-many-statements
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    parser = make_parser()
    args = parser.parse_args()
    assert len(args.files) == len(args.jsons), "length of config and json mismatch"
    table_content = []

    for cfg_file, json_path in zip(args.files, args.jsons):
        current_network = import_from_file(cfg_file)
        cfg = current_network.Cfg()

        logger.info(f"load json from {json_path}, start evaluation!")

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
        cfg_name = cfg_file.split(".")[0]
        table_content.append([cfg_name, *["{:.3f}".format(v) for v in cocoEval.stats]])

    headers = [
        "name", "AP", "AP@0.5", "AP@0.75", "APs", "APm", "APl",
        "AR@1", "AR@10", "AR@100", "ARs", "ARm", "ARl",
    ]
    table = tabulate(table_content, headers=headers, tablefmt="pipe")
    logger.info("\n" + table)


if __name__ == "__main__":
    main()
