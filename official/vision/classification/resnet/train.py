# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import bisect
import os
import time

# pylint: disable=import-error
import model as resnet_model

import megengine
import megengine.autodiff as autodiff
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.optimizer as optim

logging = megengine.logger.get_logger()


def main():
    parser = argparse.ArgumentParser(description="MegEngine ImageNet Training")
    parser.add_argument("-d", "--data", metavar="DIR", help="path to imagenet dataset")
    parser.add_argument(
        "-a",
        "--arch",
        default="resnet50",
        help="model architecture (default: resnet50)",
    )
    parser.add_argument(
        "-n",
        "--ngpus",
        default=None,
        type=int,
        help="number of GPUs per node (default: None, use all available GPUs)",
    )
    parser.add_argument(
        "--save",
        metavar="DIR",
        default="output",
        help="path to save checkpoint and log",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        help="number of total epochs to run (default: 90)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="SIZE",
        default=64,
        type=int,
        help="batch size for single GPU (default: 64)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        metavar="LR",
        default=0.025,
        type=float,
        help="learning rate for single GPU (default: 0.025)",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, help="momentum (default: 0.9)"
    )
    parser.add_argument(
        "--weight-decay", default=1e-4, type=float, help="weight decay (default: 1e-4)"
    )

    parser.add_argument("-j", "--workers", default=2, type=int)
    parser.add_argument(
        "-p",
        "--print-freq",
        default=20,
        type=int,
        metavar="N",
        help="print frequency (default: 20)",
    )

    parser.add_argument("--dist-addr", default="localhost")
    parser.add_argument("--dist-port", default=23456, type=int)
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)

    args = parser.parse_args()

    if args.ngpus is None:
        args.ngpus = dist.helper.get_device_count_by_fork("gpu")

    if args.world_size * args.ngpus > 1:
        dist_worker = dist.launcher(
            master_ip=args.dist_addr,
            port=args.dist_port,
            world_size=args.world_size * args.ngpus,
            rank_start=args.rank * args.ngpus,
            n_gpus=args.ngpus
        )(worker)
        dist_worker(args)
    else:
        worker(args)


def worker(args):
    # pylint: disable=too-many-statements
    if dist.get_rank() == 0:
        os.makedirs(os.path.join(args.save, args.arch), exist_ok=True)
        megengine.logger.set_log_file(os.path.join(args.save, args.arch, "log.txt"))

    # build dataset
    train_dataloader, valid_dataloader = build_dataset(args)
    train_queue = iter(train_dataloader)  # infinite
    steps_per_epoch = 1280000 // (dist.get_world_size() * args.batch_size)

    # build model
    model = resnet_model.__dict__[args.arch]()

    # Sync parameters and buffers
    if dist.get_world_size() > 1:
        dist.bcast_list_(model.parameters())
        dist.bcast_list_(model.buffers())

    # Autodiff gradient manager
    gm = autodiff.GradManager().attach(
        model.parameters(),
        callbacks=dist.make_allreduce_cb("mean") if dist.get_world_size() > 1 else None,
    )

    # Optimizer
    opt = optim.SGD(
        model.parameters(),
        lr=args.lr * dist.get_world_size(),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # train and valid func
    def train_step(image, label):
        with gm:
            logits = model(image)
            loss = F.nn.cross_entropy(logits, label)
            acc1, acc5 = F.topk_accuracy(logits, label, topk=(1, 5))
            gm.backward(loss)
            opt.step().clear_grad()
        return loss, acc1, acc5

    def valid_step(image, label):
        logits = model(image)
        loss = F.nn.cross_entropy(logits, label)
        acc1, acc5 = F.topk_accuracy(logits, label, topk=(1, 5))
        # calculate mean values
        if dist.get_world_size() > 1:
            loss = F.distributed.all_reduce_sum(loss) / dist.get_world_size()
            acc1 = F.distributed.all_reduce_sum(acc1) / dist.get_world_size()
            acc5 = F.distributed.all_reduce_sum(acc5) / dist.get_world_size()
        return loss, acc1, acc5

    # multi-step learning rate scheduler with warmup
    def adjust_learning_rate(step):
        lr = args.lr * 0.1 ** bisect.bisect_right(
            [30 * steps_per_epoch, 60 * steps_per_epoch, 80 * steps_per_epoch], step
        )
        if step < 5 * steps_per_epoch:  # warmup
            lr = args.lr * (step / (5 * steps_per_epoch))
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        return lr

    # start training
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    clck = AverageMeter("Time")

    for step in range(0, args.epochs * steps_per_epoch):
        lr = adjust_learning_rate(step)

        t = time.time()

        image, label = next(train_queue)
        image = megengine.tensor(image, dtype="float32")
        label = megengine.tensor(label, dtype="int32")

        loss, acc1, acc5 = train_step(image, label)

        objs.update(loss.item())
        top1.update(100 * acc1.item())
        top5.update(100 * acc5.item())
        clck.update(time.time() - t)

        if step % args.print_freq == 0 and dist.get_rank() == 0:
            logging.info(
                "Epoch %d Step %d, LR %.4f, %s %s %s %s",
                step // steps_per_epoch,
                step,
                lr,
                objs,
                top1,
                top5,
                clck,
            )
            objs.reset()
            top1.reset()
            top5.reset()
            clck.reset()

        if (step + 1) % steps_per_epoch == 0:
            model.eval()
            _, valid_acc1, valid_acc5 = valid(valid_step, valid_dataloader, args)
            model.train()
            logging.info(
                "Epoch %d Test Acc@1 %.3f, Acc@5 %.3f",
                (step + 1) // steps_per_epoch,
                valid_acc1,
                valid_acc5,
            )
            if dist.get_rank() == 0:
                megengine.save(
                    {
                        "epoch": (step + 1) // steps_per_epoch,
                        "state_dict": model.state_dict(),
                    },
                    os.path.join(args.save, args.arch, "checkpoint.pkl"),
                )


def valid(func, data_queue, args):
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    clck = AverageMeter("Time")

    t = time.time()
    for step, (image, label) in enumerate(data_queue):
        image = megengine.tensor(image, dtype="float32")
        label = megengine.tensor(label, dtype="int32")

        n = image.shape[0]

        loss, acc1, acc5 = func(image, label)

        objs.update(loss.item(), n)
        top1.update(100 * acc1.item(), n)
        top5.update(100 * acc5.item(), n)
        clck.update(time.time() - t, n)
        t = time.time()

        if step % args.print_freq == 0 and dist.get_rank() == 0:
            logging.info("Test step %d, %s %s %s %s", step, objs, top1, top5, clck)

    return objs.avg, top1.avg, top5.avg


def build_dataset(args):
    train_dataset = data.dataset.ImageNet(args.data, train=True)
    train_sampler = data.Infinite(
        data.RandomSampler(train_dataset, batch_size=args.batch_size, drop_last=True)
    )
    train_dataloader = data.DataLoader(
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
    valid_dataloader = data.DataLoader(
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
    return train_dataloader, valid_dataloader


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
