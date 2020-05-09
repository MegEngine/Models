# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import multiprocessing as mp
import time

import megengine as mge
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.quantization as Q
import megengine.jit as jit
from imagenet_nori_dataset import ImageNetNoriDataset

import model as M

logger = mge.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="resnet50", type=str)
    parser.add_argument("-d", "--data", default=None, type=str)
    parser.add_argument("-m", "--model", default=None, type=str)
    parser.add_argument("--mode", default="quantized", type=str, help="normal qat quantized")

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


def worker(rank, world_size, args):
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

    model = getattr(M, args.arch)(pretrained=(args.model is None))
    
    if args.mode != "normal":
        Q.quantize_qat(model, qconfig=Q.ema_fakequant_qconfig)

    if args.model:
        logger.info("load weights from %s", args.model)
        model.load_state_dict(mge.load(args.model)["state_dict"])
    
    if args.mode == "quantized":
        Q.quantize(model)
        logger.info("Test Q model")

        

    @jit.trace(symbolic=True)
    def valid_func(image, label):
        model.eval()
        logits = model(image)
        loss = F.cross_entropy_with_softmax(logits, label)
        acc1, acc5 = F.accuracy(logits, label, (1, 5))
        if dist.is_distributed():  # all_reduce_mean
            loss = dist.all_reduce_sum(loss, "valid_loss") / dist.get_world_size()
            acc1 = dist.all_reduce_sum(acc1, "valid_acc1") / dist.get_world_size()
            acc5 = dist.all_reduce_sum(acc5, "valid_acc5") / dist.get_world_size()
        return loss, acc1, acc5

    logger.info("preparing dataset..")
    if args.data and args.data.endswith(".list"):
        valid_dataset = ImageNetNoriDataset(args.data)
    else:
        valid_dataset = data.dataset.ImageNet(args.data, train=False)
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
                T.Normalize(
                    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
                ),  # BGR
                T.ToMode("CHW"),
            ]
        ),
        num_workers=args.workers,
    )
    _, valid_acc, valid_acc5 = infer(valid_func, valid_queue, args)
    logger.info("Valid %.3f / %.3f", valid_acc, valid_acc5)


def infer(model, data_queue, args, epoch=0):
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
            logger.info(
                "Epoch %d Step %d, %s %s %s %s",
                epoch,
                step,
                objs,
                top1,
                top5,
                total_time,
            )

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
