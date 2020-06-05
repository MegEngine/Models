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
import json
import numpy as np
import cv2

import megengine as mge
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.jit as jit
import megengine.optimizer as optim

<<<<<<< HEAD
import official.vision.keypoints.models as M
from official.vision.keypoints.transforms import (
    RandomBoxAffine,
    RandomHorizontalFlip,
    HalfBodyTransform,
    ExtendBoxes,
)
from official.vision.keypoints.dataset import COCOJoints, HeatmapCollator
from official.vision.keypoints.config import Config as cfg
=======
import sys
sys.path.insert(0, '../../../')
import models as M
from transforms import RandomAffine, RandomHorizontalFlip, HalfBodyTransform, ExtendBoxes
from dataset import COCOJoints, HeatmapCollator
from config import Config as cfg
>>>>>>> fix code style

logger = mge.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--arch",
<<<<<<< HEAD
        default="simplebaseline_res50",
        type=str,
        choices=[
            "simplebaseline_res50",
            "simplebaseline_res101",
            "simplebaseline_res152",
        ],
    )
    parser.add_argument("--pretrained", default=True, type=bool)
    parser.add_argument("-s", "--save", default="/data/models", type=str)
    parser.add_argument("--data_root", default="/data/coco/images/", type=str)
    parser.add_argument(
        "--ann_file",
        default="/data/coco/annotations/person_keypoints_train2017.json",
        type=str,
    )
    parser.add_argument("--continue", default=None, type=str)

    parser.add_argument("-b", "--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=6e-4, type=float)
    parser.add_argument("--epochs", default=200, type=int)

    parser.add_argument("--multi_scale_supervision", default=True, type=bool)

    parser.add_argument("-n", "--ngpus", default=8, type=int)
    parser.add_argument("-w", "--workers", default=8, type=int)
=======
        default="SimpleBaseline_Res50",
        type=str,
        choices=[
            "SimpleBaseline_Res50",
            "SimpleBaseline_Res101",
            "SimpleBaseline_Res152"
        ],
    )
    parser.add_argument("--pretrained", default=True, type=bool)
    parser.add_argument("-d", "--data", default=None, type=str)
    parser.add_argument("-s", "--save", default="/data/models", type=str)
    parser.add_argument("--data_root", default='/data/coco_data', type=str)
    parser.add_argument(
        "--ann_root", default='/data/coco_data/person_keypoints_trainvalminusminival2014.json', type=str
    )
    parser.add_argument(
        "--c", default=None, type=str
    )

    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--initial_lr", default=5e-4, type=float)
    parser.add_argument("--lr_ratio", default=0.1, type=float)
    parser.add_argument("--warm_epochs", default=1, type=float)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--epochs", default=200, type=int)

    parser.add_argument("--multi_scale_supervision", default=True, type=bool)
    parser.add_argument("--half_body_transform", default=True, type=bool)
    parser.add_argument("--extend_boxes", default=True, type=bool)

    parser.add_argument("-n", "--ngpus", default=8, type=int)
    parser.add_argument("-w", "--workers", default=12, type=int)
>>>>>>> fix code style
    parser.add_argument("--report-freq", default=10, type=int)

    args = parser.parse_args()

<<<<<<< HEAD
    model_name = "{}_{}x{}".format(args.arch, cfg.input_shape[0], cfg.input_shape[1])
=======
    model_name = '{}_{}x{}'.format(
        args.arch, cfg.input_shape[0], cfg.input_shape[1])
>>>>>>> fix code style
    save_dir = os.path.join(args.save, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mge.set_log_file(os.path.join(save_dir, "log.txt"))

<<<<<<< HEAD
    world_size = mge.get_device_count("gpu") if args.ngpus is None else args.ngpus

    if world_size > 1:
        # scale learning rate by number of gpus
        args.lr *= world_size
=======
    world_size = mge.get_device_count(
        "gpu") if args.ngpus is None else args.ngpus

    if world_size > 1:
        # scale learning rate by number of gpus
        args.initial_lr *= world_size
>>>>>>> fix code style
        # start distributed training, dispatch sub-processes
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
<<<<<<< HEAD
        logger.info("init distributed process group {} / {}".format(rank, world_size))
=======
        logger.info(
            "init distributed process group {} / {}".format(rank, world_size))
>>>>>>> fix code style
        dist.init_process_group(
            master_ip="localhost",
            master_port=23456,
            world_size=world_size,
            rank=rank,
            dev=rank,
        )

<<<<<<< HEAD
    model_name = "{}_{}x{}".format(args.arch, cfg.input_shape[0], cfg.input_shape[1])
=======
    model_name = '{}_{}x{}'.format(
        args.arch, cfg.input_shape[0], cfg.input_shape[1])
>>>>>>> fix code style
    save_dir = os.path.join(args.save, model_name)

    model = getattr(M, args.arch)(pretrained=args.pretrained)
    model.train()
    start_epoch = 0
    if args.c is not None:
        file = mge.load(args.c)
<<<<<<< HEAD
        model.load_state_dict(file["state_dict"])
        start_epoch = file["epoch"]

    optimizer = optim.Adam(
        model.parameters(requires_grad=True), lr=args.lr, weight_decay=cfg.weight_decay,
=======
        model.load_state_dict(file['state_dict'])
        start_epoch = file['epoch']

    optimizer = optim.Adam(
        model.parameters(requires_grad=True),
        lr=args.initial_lr,
        weight_decay=args.weight_decay,
>>>>>>> fix code style
    )
    # Build train datasets
    logger.info("preparing dataset..")
    train_dataset = COCOJoints(
<<<<<<< HEAD
        args.data_root,
        args.ann_file,
        image_set="train",
        order=("image", "keypoints", "boxes", "info"),
    )
=======
        args.data_root, args.ann_root, image_set='train', order=("image", "keypoints", "boxes", "info"))
>>>>>>> fix code style
    train_sampler = data.RandomSampler(
        train_dataset, batch_size=args.batch_size, drop_last=True
    )

    transforms = [T.Normalize(mean=cfg.IMG_MEAN, std=cfg.IMG_STD)]
<<<<<<< HEAD
    if cfg.half_body_transform:
        transforms.append(
            HalfBodyTransform(
                cfg.upper_body_ids, cfg.lower_body_ids, cfg.prob_half_body
            )
        )
    if cfg.extend_boxes:
        transforms.append(
            ExtendBoxes(cfg.x_ext, cfg.y_ext, cfg.input_shape[1] / cfg.input_shape[0])
        )
    transforms += [
        RandomHorizontalFlip(0.5, keypoint_flip_order=cfg.keypoint_flip_order)
    ]
    transforms += [
        RandomBoxAffine(
            degrees=cfg.rotate_range,
            scale=cfg.scale_range,
            output_shape=cfg.input_shape,
            rotate_prob=cfg.rotation_prob,
            scale_prob=cfg.scale_prob,
        )
    ]
=======
    if args.half_body_transform:
        transforms.append(
            HalfBodyTransform(
                cfg.upper_body_ids,
                cfg.lower_body_ids,
                cfg.prob_half_body
                )
            )
    if args.extend_boxes:
        transforms.append(
           ExtendBoxes(
                cfg.x_ext,
                cfg.y_ext,
                cfg.w_h_ratio
                )
            )
    transforms +=  [RandomHorizontalFlip(
                    0.5,
                    keypoint_flip_order=cfg.keypoint_flip_order)]
    transforms +=  [RandomAffine(
                    degrees=cfg.rotate_range,
                    scale=cfg.scale_range,
                    output_shape=cfg.input_shape,
                    rotate_prob=cfg.rotation_prob,
                    scale_prob=cfg.scale_prob
                )]
>>>>>>> fix code style
    transforms += [T.ToMode()]

    train_queue = data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        num_workers=args.workers,
<<<<<<< HEAD
        transform=T.Compose(transforms=transforms, order=train_dataset.order,),
=======
        transform=T.Compose(
            transforms=transforms,
            order=train_dataset.order,
        ),
>>>>>>> fix code style
        collator=HeatmapCollator(
            cfg.input_shape,
            cfg.output_shape,
            cfg.keypoint_num,
            cfg.heat_thre,
            cfg.heat_kernel if args.multi_scale_supervision else cfg.heat_kernel[-1:],
<<<<<<< HEAD
            cfg.heat_range,
        ),
=======
            cfg.heat_range
        )
>>>>>>> fix code style
    )

    # Start training
    for epoch in range(start_epoch, args.epochs):
<<<<<<< HEAD
        loss = train(model, train_queue, optimizer, args, epoch=epoch)
=======
        loss = train(
            model, train_queue, optimizer, args, epoch=epoch
        )
>>>>>>> fix code style
        logger.info("Epoch %d Train %.6f ", epoch, loss)

        if rank == 0:  # save checkpoint
            mge.save(
<<<<<<< HEAD
                {"epoch": epoch + 1, "state_dict": model.state_dict(),},
                os.path.join(save_dir, "epoch_{}.pkl".format(epoch)),
            )


def train(model, data_queue, optimizer, args, epoch=0):
    @jit.trace(symbolic=True, opt_level=2)
    def train_func():
        loss = model.calc_loss()
        optimizer.backward(loss)  # compute gradients
        if dist.is_distributed():  # all_reduce_mean
            loss = dist.all_reduce_sum(loss, "train_loss") / dist.get_world_size()
=======
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                },
                os.path.join(save_dir, "epoch_{}.pkl".format(epoch)),
            )

def train(model, data_queue, optimizer, args, epoch=0):
    @jit.trace(symbolic=True, opt_level=2)
    def train_func():
        loss = model.cal_loss()
        optimizer.backward(loss)  # compute gradients
        if dist.is_distributed():  # all_reduce_mean
            loss = dist.all_reduce_sum(
                loss, "train_loss") / dist.get_world_size()
>>>>>>> fix code style
        return loss

    avg_loss = 0
    total_time = 0

    t = time.time()
    for step, mini_batch in enumerate(data_queue):

        for param_group in optimizer.param_groups:
            current_step = epoch * len(data_queue) + step
<<<<<<< HEAD
            if current_step < cfg.warm_epochs * len(data_queue):
                lr_factor = cfg.lr_ratio + (
                    1 - cfg.lr_ratio
                ) * current_step / cfg.warm_epochs / len(data_queue)
            else:
                lr_factor = 1 - (current_step - len(data_queue) * cfg.warm_epochs) / (
                    len(data_queue) * (args.epochs - cfg.warm_epochs)
                )
=======
            if current_step < args.warm_epochs * len(data_queue):
                lr_factor = (args.lr_ratio + (
                    1 - args.lr_ratio) * current_step / args.warm_epochs / len(data_queue))
            else:
                lr_factor = 1 - (
                    current_step - len(data_queue)*args.warm_epochs
                ) / (len(data_queue) * (args.epochs - args.warm_epochs))
>>>>>>> fix code style

            lr = args.initial_lr * lr_factor
            param_group["lr"] = lr

        lr = optimizer.param_groups[0]["lr"]
        model.inputs["image"].set_value(mini_batch["data"])
        model.inputs["heatmap"].set_value(mini_batch["heatmap"])
        model.inputs["heat_valid"].set_value(mini_batch["heat_valid"])

        optimizer.zero_grad()
        loss = train_func()
        optimizer.step()

        avg_loss = (avg_loss * step + loss.numpy().item()) / (step + 1)
        total_time += time.time() - t
        t = time.time()

        if step % args.report_freq == 0 and dist.get_rank() == 0:
            logger.info(
                "Epoch {} Step {}, LR {:.6f} Loss {:.6f} Elapsed Time {:.3f}s".format(
<<<<<<< HEAD
                    epoch, step, lr, loss.numpy().item(), total_time
=======
                    epoch,
                    step,
                    lr,
                    loss.numpy().item(),
                    total_time
>>>>>>> fix code style
                )
            )

    return avg_loss

<<<<<<< HEAD

if __name__ == "__main__":
    main()
=======
if __name__ == "__main__":
    main()
>>>>>>> fix code style
