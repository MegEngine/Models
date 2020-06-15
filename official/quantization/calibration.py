# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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

logger = mge.get_logger(__name__)
# from imagenet_nori_dataset import ImageNetNoriDataset
from megengine.quantization.quantize import enable_observer, quantize, quantize_qat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="resnet18", type=str)
    parser.add_argument("-d", "--data", default=None, type=str)
    parser.add_argument("-s", "--save", default="/data/models", type=str)
    parser.add_argument("-c", "--checkpoint", default=None, type=str,
        help="pretrained model to finetune")

    parser.add_argument("-m", "--mode", default="qat", type=str,
        choices=["normal", "qat", "quantized", "calibration"],
        help="Quantization Mode\n"
             "normal: no quantization, using float32\n"
             "qat: quantization aware training, simulate int8\n"
             "calibration: calibration\n"
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
        wd = cfg.WEIGHT_DECAY(pname, p)
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
    
    # load calibration model
    assert args.checkpoint
    logger.info("Load pretrained weights from %s", args.checkpoint)
    ckpt = mge.load(args.checkpoint)
    ckpt = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(ckpt, strict=False)

    # Build valid datasets
    valid_dataset = data.dataset.ImageNet(args.data, train=False)
    # valid_dataset = ImageNetNoriDataset(args.data)
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

    # calibration
    model.fc.disable_quantize()
    model = quantize_qat(model, qconfig=Q.calibration_qconfig)
    
    # calculate scale
    @jit.trace(symbolic=True)
    def calculate_scale(image, label):
        model.eval()
        enable_observer(model)
        logits = model(image)
        loss = F.cross_entropy_with_softmax(logits, label, label_smooth=0.1)
        acc1, acc5 = F.accuracy(logits, label, (1, 5))
        if dist.is_distributed():  # all_reduce_mean
            loss = dist.all_reduce_sum(loss, "valid_loss") / dist.get_world_size()
            acc1 = dist.all_reduce_sum(acc1, "valid_acc1") / dist.get_world_size()
            acc5 = dist.all_reduce_sum(acc5, "valid_acc5") / dist.get_world_size()
        return loss, acc1, acc5
    
    # model.fc.disable_quantize()
    infer(calculate_scale, valid_queue, args)

    # quantized
    model = quantize(model)

    # eval quantized model
    @jit.trace(symbolic=True)
    def eval_func(image, label):
        model.eval()
        logits = model(image)
        loss = F.cross_entropy_with_softmax(logits, label, label_smooth=0.1)
        acc1, acc5 = F.accuracy(logits, label, (1, 5))
        if dist.is_distributed():  # all_reduce_mean
            loss = dist.all_reduce_sum(loss, "valid_loss") / dist.get_world_size()
            acc1 = dist.all_reduce_sum(acc1, "valid_acc1") / dist.get_world_size()
            acc5 = dist.all_reduce_sum(acc5, "valid_acc5") / dist.get_world_size()
        return loss, acc1, acc5
        
    _, valid_acc, valid_acc5 = infer(eval_func, valid_queue, args)
    logger.info("TEST %f, %f", valid_acc, valid_acc5)

    # save quantized model
    mge.save(
        {"step": -1, "state_dict": model.state_dict()},
        os.path.join(save_dir, "checkpoint-calibration.pkl")
    )
    logger.info("save in {}".format(os.path.join(save_dir, "checkpoint-calibration.pkl")))

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
        
            # break
            if step == args.report_freq:
                break

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
