# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Test int8 quantizated model on ImageNet.

Note:
    * QAT simulate int8 with fp32, gpu only.
    * Quantized use real int8, cpu only, a bit slow.
    * Results may be slightly different between qat and quantized mode.
"""
import argparse
import time

# pylint: disable=import-error
import models

import megengine as mge
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.quantization as Q
from megengine.quantization.quantize import quantize, quantize_qat

logger = mge.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="resnet18", type=str)
    parser.add_argument("-d", "--data", default=None, type=str)
    parser.add_argument("-s", "--save", default="/data/models", type=str)
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        type=str,
        help="pretrained model to finetune",
    )

    parser.add_argument(
        "-m",
        "--mode",
        default="qat",
        type=str,
        choices=["normal", "qat", "quantized"],
        help="Quantization Mode\n"
        "normal: no quantization, using float32\n"
        "qat: quantization aware training, simulate int8\n"
        "quantized: convert mode to int8 quantized, inference only",
    )

    parser.add_argument("-n", "--ngpus", default=None, type=int)
    parser.add_argument("-w", "--workers", default=4, type=int)
    parser.add_argument("--report-freq", default=50, type=int)
    args = parser.parse_args()

    world_size = (
        dist.helper.get_device_count_by_fork("gpu")
        if args.ngpus is None
        else args.ngpus
    )
    world_size = 1 if world_size == 0 else world_size
    test_proc = dist.launcher(worker) if world_size > 1 else worker
    test_proc(world_size, args)


def worker(world_size, args):
    # pylint: disable=too-many-statements

    rank = dist.get_rank()
    if world_size > 1:
        # Initialize distributed process group
        logger.info("init distributed process group {} / {}".format(rank, world_size))

    model = models.__dict__[args.arch]()

    if args.mode != "normal":
        quantize_qat(model, qconfig=Q.ema_fakequant_qconfig)

    if args.checkpoint:
        logger.info("Load pretrained weights from %s", args.checkpoint)
        ckpt = mge.load(args.checkpoint)
        ckpt = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(ckpt, strict=False)

    if args.mode == "quantized":
        quantize(model)

    # Define valid graph
    def valid_func(image, label):
        model.eval()
        logits = model(image)
        loss = F.loss.cross_entropy(logits, label, label_smooth=0.1)
        acc1, acc5 = F.topk_accuracy(logits, label, (1, 5))
        if dist.is_distributed():  # all_reduce_mean
            loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
            acc1 = dist.functional.all_reduce_sum(acc1) / dist.get_world_size()
            acc5 = dist.functional.all_reduce_sum(acc5) / dist.get_world_size()
        return loss, acc1, acc5

    # Build valid datasets
    logger.info("preparing dataset..")
    valid_dataset = data.dataset.ImageNet(args.data, train=False)
    valid_sampler = data.SequentialSampler(
        valid_dataset, batch_size=100, drop_last=False
    )
    valid_queue = data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        transform=T.Compose(
            [T.Resize(256), T.CenterCrop(224), T.Normalize(mean=128), T.ToMode("CHW")]
        ),
        num_workers=args.workers,
    )

    _, valid_acc, valid_acc5 = infer(valid_func, valid_queue, args)
    if rank == 0:
        logger.info("TEST %f, %f", valid_acc, valid_acc5)


def infer(model, data_queue, args):
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    total_time = AverageMeter("Time")

    t = time.time()
    for step, (image, label) in enumerate(data_queue):
        n = image.shape[0]
        image = mge.tensor(image, dtype="float32")
        label = mge.tensor(label, dtype="int32")

        loss, acc1, acc5 = model(image, label)

        objs.update(loss.item(), n)
        top1.update(100 * acc1.item(), n)
        top5.update(100 * acc5.item(), n)
        total_time.update(time.time() - t)
        t = time.time()

        if step % args.report_freq == 0 and dist.get_rank() == 0:
            logger.info("Step %d, %s %s %s %s", step, objs, top1, top5, total_time)

    return objs.avg, top1.avg, top5.avg


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":.3f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    main()
