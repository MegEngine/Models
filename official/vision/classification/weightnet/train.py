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
import argparse
import multiprocessing as mp
import os
import time

import megengine as mge
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.jit as jit
import megengine.optimizer as optim

import shufflenet_v2 as M

logger = mge.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="shufflenet_v2_x0_5", type=str)
    parser.add_argument("-d", "--data", default=None, type=str)
    parser.add_argument("-s", "--save", default="./models", type=str)
    parser.add_argument("-m", "--model", default=None, type=str)

    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument("--learning-rate", default=0.0625, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=4e-5, type=float)
    parser.add_argument("--steps", default=300000, type=int)

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


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters(requires_grad=True):
        if pname.find("weight") >= 0 and len(p.shape) > 1:
            # print("include ", pname, p.shape)
            group_weight_decay.append(p)
        else:
            # print("not include ", pname, p.shape)
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(
        group_no_weight_decay
    )
    groups = [
        dict(params=group_weight_decay),
        dict(params=group_no_weight_decay, weight_decay=0.0),
    ]
    return groups


def worker(rank, world_size, args):
    # pylint: disable=too-many-statements
    mge.set_log_file(os.path.join(args.save, args.arch, "log.txt"))

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
    step_start = 0
    if args.model:
        logger.info("load weights from %s", args.model)
        model.load_state_dict(mge.load(args.model))
        step_start = int(args.model.split("-")[1].split(".")[0])

    optimizer = optim.SGD(
        get_parameters(model),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
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
    train_dataset = data.dataset.ImageNet(args.data, train=True)
    train_sampler = data.Infinite(data.RandomSampler(
        train_dataset, batch_size=args.batch_size, drop_last=True
    ))
    train_queue = data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose(
            [
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
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    total_time = AverageMeter("Time")

    t = time.time()
    for step in range(step_start, args.steps + 1):
        # Linear learning rate decay
        decay = 1.0
        decay = 1 - float(step) / args.steps if step < args.steps else 0
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate * decay

        image, label = next(train_queue)
        time_data=time.time()-t
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
        time_iter=time.time()-t
        t = time.time()
        if step % args.report_freq == 0 and rank == 0:
            logger.info(
                "TRAIN Iter %06d: lr = %f,\tloss = %f,\twc_loss = 1,\tTop-1 err = %f,\tTop-5 err = %f,\tdata_time = %f,\ttrain_time = %f,\tremain_hours=%f",
                step,
                args.learning_rate * decay,
                float(objs.__str__().split()[1]),
                1-float(top1.__str__().split()[1])/100,
                1-float(top5.__str__().split()[1])/100,
                time_data,
                time_iter - time_data,
                time_iter * (args.steps - step) / 3600,
            )
            objs.reset()
            top1.reset()
            top5.reset()
            total_time.reset()
        if step % 10000 == 0 and rank == 0 and step != 0:
            logger.info("SAVING %06d", step)
            mge.save(
                model.state_dict(),
                os.path.join(save_dir, "checkpoint-{:06d}.pkl".format(step)),
            )
        if step % 50000 == 0 and step != 0:
            _, valid_acc, valid_acc5 = infer(valid_func, valid_queue, args)
            logger.info("TEST Iter %06d: loss = %f,\tTop-1 err = %f,\tTop-5 err = %f", step, _, 1-valid_acc/100, 1-valid_acc5/100)

    mge.save(
        model.state_dict(), os.path.join(save_dir, "checkpoint-{:06d}.pkl".format(step))
    )
    _, valid_acc, valid_acc5 = infer(valid_func, valid_queue, args)
    logger.info("TEST Iter %06d: loss=%f,\tTop-1 err = %f,\tTop-5 err = %f", step, _, 1-valid_acc/100, 1-valid_acc5/100)


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
            logger.info(
                "Step %d, %s %s %s %s",
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
