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
import os
import shutil
import time

import megengine as mge
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.jit as jit
import megengine.optimizer as optim

import model as M

logger = mge.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--arch",
        default="resnet50",
        type=str,
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50_32x4d",
            "resnext101_32x8d",
        ],
    )
    parser.add_argument("-d", "--data", default=None, type=str)
    parser.add_argument("-s", "--save", default="/data/models", type=str)

    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("--learning-rate", default=0.0125, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=90, type=int)

    parser.add_argument("-n", "--ngpus", default=None, type=int)
    parser.add_argument("-w", "--workers", default=4, type=int)
    parser.add_argument("--report-freq", default=50, type=int)
    args = parser.parse_args()

    save_dir = os.path.join(args.save, args.arch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mge.set_log_file(os.path.join(save_dir, "log.txt"))

    world_size = mge.get_device_count("gpu") if args.ngpus is None else args.ngpus

    if world_size > 1:
        # scale learning rate by number of gpus
        args.learning_rate *= world_size
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

    save_dir = os.path.join(args.save, args.arch)

    model = getattr(M, args.arch)()

    optimizer = optim.SGD(
        model.parameters(requires_grad=True),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.MultiStepLR(optimizer, [30, 60, 80])

    # Define train and valid graph
    @jit.trace(symbolic=True)
    def train_func(image, label):
        model.train()
        logits = model(image)
        loss = F.cross_entropy_with_softmax(logits, label)
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
        loss = F.cross_entropy_with_softmax(logits, label)
        acc1, acc5 = F.accuracy(logits, label, (1, 5))
        if dist.is_distributed():  # all_reduce_mean
            loss = dist.all_reduce_sum(loss, "valid_loss") / dist.get_world_size()
            acc1 = dist.all_reduce_sum(acc1, "valid_acc1") / dist.get_world_size()
            acc5 = dist.all_reduce_sum(acc5, "valid_acc5") / dist.get_world_size()
        return loss, acc1, acc5

    # Build train and valid datasets
    logger.info("preparing dataset..")
    train_dataset = data.dataset.ImageNet(args.data, train=True)
    train_sampler = data.RandomSampler(
        train_dataset, batch_size=args.batch_size, drop_last=True
    )
    train_queue = data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose(
            [  # Baseline Augmentation for small models
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.Normalize(
                    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
                ),  # BGR
                T.ToMode("CHW"),
            ]
        )
        if args.arch in ("resnet18", "resnet34")
        else T.Compose(
            [  # Facebook Augmentation for large models
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.Lighting(0.1),
                T.Normalize(
                    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
                ),  # BGR
                T.ToMode("CHW"),
            ]
        ),
        num_workers=args.workers,
    )
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

    # Start training
    top1_acc = 0
    for epoch in range(0, args.epochs):
        logger.info("Epoch %d LR %.3e", epoch, scheduler.get_lr()[0])
        _, train_acc, train_acc5 = train(
            train_func, train_queue, optimizer, args, epoch=epoch
        )
        logger.info("Epoch %d Train %.3f / %.3f", epoch, train_acc, train_acc5)
        _, valid_acc, valid_acc5 = infer(valid_func, valid_queue, args, epoch=epoch)
        logger.info("Epoch %d Valid %.3f / %.3f", epoch, valid_acc, valid_acc5)
        scheduler.step()
        if rank == 0:  # save checkpoint
            mge.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "accuracy": valid_acc,
                },
                os.path.join(save_dir, "checkpoint.pkl"),
            )
            if valid_acc > top1_acc:
                top1_acc = valid_acc
                shutil.copy(
                    os.path.join(save_dir, "checkpoint.pkl"),
                    os.path.join(save_dir, "model_best.pkl"),
                )


def train(model, data_queue, optimizer, args, epoch=0):
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    total_time = AverageMeter("Time")

    t = time.time()
    for step, (image, label) in enumerate(data_queue):
        n = image.shape[0]
        image = image.astype("float32")  # convert np.uint8 to float32
        label = label.astype("int32")

        optimizer.zero_grad()
        loss, acc1, acc5 = model(image, label)
        optimizer.step()

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
