# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2019 Megvii Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
# ------------------------------------------------------------------------------
"""Finetune a pretrained fp32 with int8 quantization aware training(QAT)"""
import argparse
import collections
import multiprocessing as mp
import numbers
import os
import bisect
import time

import megengine as mge
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.jit as jit
import megengine.optimizer as optim
import megengine.quantization as Q

import config
import models
from imagenet_nori_dataset import ImageNetNoriDataset

logger = mge.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="resnet18", type=str,
        choices=["resnet18", "shufflenet_v1_x1_0_g3"])
    parser.add_argument("-d", "--data", default=None, type=str)
    parser.add_argument("-s", "--save", default="/data/models", type=str)
    parser.add_argument("-c", "--checkpoint", default=None, type=str,
        help="pretrained model to finetune")

    parser.add_argument("-m", "--mode", default="qat", type=str,
        choices=["normal", "qat", "quantized"],
        help="Quantization Mode\n"
             "normal: no quantization, using float32\n"
             "qat: quantization aware training, simulate int8\n"
             "quantized: convert mode to int8 quantized, inference only")

    parser.add_argument("-n", "--ngpus", default=None, type=int)
    parser.add_argument("-w", "--workers", default=4, type=int)
    parser.add_argument("--report-freq", default=50, type=int)
    args = parser.parse_args()

    world_size = mge.get_device_count("gpu") if args.ngpus is None else args.ngpus

    if world_size > 1:
        # start distributed training, dispatch sub-processes
        mp.set_start_method("spawn")
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=worker, args=(rank, world_size, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        worker(0, 1, args)


def get_parameters(model, cfg):
    if isinstance(cfg.WEIGHT_DECAY, numbers.Number):
        return {"params": model.parameters(requires_grad=True),
                "weight_decay": cfg.WEIGHT_DECAY}

    groups = collections.defaultdict(list)  # weight_decay -> List[param]
    for pname, p in model.named_parameters(requires_grad=True):
        wd = cfg.WEIGHT_DECAT(pname, p)
        groups[wd].append(p)
    groups = [
        {"params": params, "weight_decay": wd}
        for wd, params in groups.items()
    ]  # List[{param, weight_decay}]
    return groups


def worker(rank, world_size, args):
    # pylint: disable=too-many-statements

    if world_size > 1:
        # Initialize distributed process group
        logger.info("init distributed process group {} / {}".format(rank, world_size))
        dist.init_process_group(
            master_ip="localhost",
            master_port=23456,
            world_size=world_size,
            rank=rank,
            dev=rank,
        )

    save_dir = os.path.join(args.save, args.arch + "." + args.mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    mge.set_log_file(os.path.join(save_dir, "log.txt"))

    model = models.__dict__[args.arch]()
    cfg = config.get_finetune_config(args.arch)

    cfg.LEARNING_RATE *= world_size  # scale learning rate in distributed training
    total_batch_size = cfg.BATCH_SIZE * world_size
    steps_per_epoch = 1280000 // total_batch_size
    total_steps = steps_per_epoch * cfg.EPOCHS

    if args.mode != "normal":
        Q.quantize_qat(model, Q.ema_fakequant_qconfig)

    if args.checkpoint:
        logger.info("Load pretrained weights from %s", args.checkpoint)
        ckpt = mge.load(args.checkpoint)
        ckpt = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(ckpt, strict=False)

    if args.mode == "quantized":
        raise ValueError("mode = quantized only used during inference")
        Q.quantize(model)

    optimizer = optim.SGD(
        get_parameters(model, cfg),
        lr=cfg.LEARNING_RATE,
        momentum=cfg.MOMENTUM,
    )

    # Define train and valid graph
    @jit.trace(symbolic=True)
    def train_func(image, label):
        model.train()
        logits = model(image)
        loss = F.cross_entropy_with_softmax(logits, label, label_smooth=0.1)
        acc1, acc5 = F.accuracy(logits, label, (1, 5))
        optimizer.backward(loss)  # compute gradients
        if dist.is_distributed():  # all_reduce_mean
            loss = dist.all_reduce_sum(loss, "train_loss") / dist.get_world_size()
            acc1 = dist.all_reduce_sum(acc1, "train_acc1") / dist.get_world_size()
            acc5 = dist.all_reduce_sum(acc5, "train_acc5") / dist.get_world_size()
        return loss, acc1, acc5

    @jit.trace(symbolic=True)
    def valid_func(image, label):
        model.eval()
        logits = model(image)
        loss = F.cross_entropy_with_softmax(logits, label, label_smooth=0.1)
        acc1, acc5 = F.accuracy(logits, label, (1, 5))
        if dist.is_distributed():  # all_reduce_mean
            loss = dist.all_reduce_sum(loss, "valid_loss") / dist.get_world_size()
            acc1 = dist.all_reduce_sum(acc1, "valid_acc1") / dist.get_world_size()
            acc5 = dist.all_reduce_sum(acc5, "valid_acc5") / dist.get_world_size()
        return loss, acc1, acc5

    # Build train and valid datasets
    logger.info("preparing dataset..")
    train_dataset = ImageNetNoriDataset("/data/imagenet.train.nori.list") #data.dataset.ImageNet(args.data, train=True)
    train_sampler = data.Infinite(data.RandomSampler(
        train_dataset, batch_size=cfg.BATCH_SIZE, drop_last=True
    ))
    train_queue = data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.Normalize(mean=128),
                T.ToMode("CHW"),
            ]
        ),
        num_workers=args.workers,
    )
    train_queue = iter(train_queue)
    valid_dataset = ImageNetNoriDataset("/data/imagenet.val.nori.list")#data.dataset.ImageNet(args.data, train=False)
    valid_sampler = data.SequentialSampler(
        valid_dataset, batch_size=100, drop_last=False
    )
    valid_queue = data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        transform=T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.Normalize(mean=128),
                T.ToMode("CHW"),
            ]
        ),
        num_workers=args.workers,
    )

    def adjust_learning_rate(step, epoch):
        learning_rate = cfg.LEARNING_RATE
        if cfg.SCHEDULER == "Linear":
            learning_rate *= 1 - float(step) / total_steps
        elif cfg.SCHEDULER == "Multistep":
            learning_rate *= cfg.SCHEDULER_GAMMA ** bisect.bisect_right(cfg.SCHEDULER_STEPS, epoch)
        else:
            raise ValueError(cfg.SCHEDULER)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
        return learning_rate

    # Start training
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    total_time = AverageMeter("Time")

    t = time.time()
    for step in range(0, 1):
        # Linear learning rate decay
        epoch = step // steps_per_epoch
        learning_rate = adjust_learning_rate(step, epoch)

        image, label = next(train_queue)
        image = image.astype("float32")
        label = label.astype("int32")

        n = image.shape[0]

        optimizer.zero_grad()
        loss, acc1, acc5 = train_func(image, label)
        optimizer.step()

        top1.update(100 * acc1.numpy()[0], n)
        top5.update(100 * acc5.numpy()[0], n)
        objs.update(loss.numpy()[0], n)
        total_time.update(time.time() - t)
        t = time.time()
        if step % args.report_freq == 0 and rank == 0:
            logger.info(
                "TRAIN e%d %06d %f %s %s %s %s",
                epoch, step, learning_rate,
                objs, top1, top5, total_time
            )
            objs.reset()
            top1.reset()
            top5.reset()
            total_time.reset()
        if step % 10000 == 0 and rank == 0:
            logger.info("SAVING %06d", step)
            mge.save(
                {"step": step, "state_dict": model.state_dict()},
                os.path.join(save_dir, "checkpoint.pkl"),
            )
        if step % 10000 == 0 and step != 0:
            _, valid_acc, valid_acc5 = infer(valid_func, valid_queue, args)
            logger.info("TEST %06d %f, %f", step, valid_acc, valid_acc5)

    mge.save(
        {"step": step, "state_dict": model.state_dict()},
        os.path.join(save_dir, "checkpoint-final.pkl")
    )
    _, valid_acc, valid_acc5 = infer(valid_func, valid_queue, args)
    logger.info("TEST %06d %f, %f", step, valid_acc, valid_acc5)


def infer(model, data_queue, args):
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    total_time = AverageMeter("Time")

    t = time.time()
    for step, (image, label) in enumerate(data_queue):
        n = image.shape[0]
        image = image.astype("float32")  # convert np.uint8 to float32
        label = label.astype("int32")

        loss, acc1, acc5 = model(image, label)

        objs.update(loss.numpy()[0], n)
        top1.update(100 * acc1.numpy()[0], n)
        top5.update(100 * acc5.numpy()[0], n)
        total_time.update(time.time() - t)
        t = time.time()

        if step % args.report_freq == 0 and dist.get_rank() == 0:
            logger.info("Step %d, %s %s %s %s",
                        step, objs, top1, top5, total_time)

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
