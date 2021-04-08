# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import os
import time

import megengine as mge
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.optimizer as optim
from megengine.autodiff import GradManager

import official.vision.keypoints.models as kpm
from official.vision.keypoints.config import Config as cfg
from official.vision.keypoints.dataset import COCOJoints, HeatmapCollator
from official.vision.keypoints.transforms import (
    ExtendBoxes,
    HalfBodyTransform,
    RandomBoxAffine,
    RandomHorizontalFlip
)

logger = mge.get_logger(__name__)
logger.setLevel("INFO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--arch",
        default="simplebaseline_res50",
        type=str,
        choices=cfg.model_choices,
    )
    parser.add_argument("-s", "--save", default="/data/models", type=str)
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("-lr", "--initial_lr", default=3e-4, type=float)

    parser.add_argument("--resume", default=None, type=str)

    parser.add_argument("--multi_scale_supervision", action="store_true")

    parser.add_argument("-n", "--ngpus", default=8, type=int)
    parser.add_argument("-w", "--workers", default=8, type=int)

    args = parser.parse_args()

    model_name = "{}_{}x{}".format(args.arch, cfg.input_shape[0], cfg.input_shape[1])
    save_dir = os.path.join(args.save, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mge.set_log_file(os.path.join(save_dir, "log.txt"))

    if args.batch_size != cfg.batch_size:
        cfg.batch_size = args.batch_size
    if args.initial_lr != cfg.initial_lr:
        cfg.initial_lr = args.initial_lr

    if args.ngpus is None:
        args.ngpus = dist.helper.get_device_count_by_fork("gpu")

    if args.ngpus > 1:
        # scale learning rate by number of gpus
        cfg.weight_decay *= args.ngpus
        dist_worker = dist.launcher(n_gpus=args.ngpus)(worker)
        dist_worker(args)
    else:
        worker(args)


def worker(args):
    model_name = "{}_{}x{}".format(args.arch, cfg.input_shape[0], cfg.input_shape[1])
    save_dir = os.path.join(args.save, model_name)

    model = getattr(kpm, args.arch)()
    model.train()
    start_epoch = 0
    if args.resume is not None:
        file = mge.load(args.resume)
        model.load_state_dict(file["state_dict"])
        start_epoch = file["epoch"]

    optimizer = optim.Adam(
        model.parameters(), lr=cfg.initial_lr, weight_decay=cfg.weight_decay
    )

    gm = GradManager()
    if dist.get_world_size() > 1:
        gm.attach(
            model.parameters(), callbacks=[dist.make_allreduce_cb("SUM", dist.WORLD)],
        )
    else:
        gm.attach(model.parameters())

    if dist.get_world_size() > 1:
        dist.bcast_list_(model.parameters(), dist.WORLD)  # sync parameters

    # Build train datasets
    logger.info("preparing dataset..")
    ann_file = os.path.join(
        cfg.data_root, "annotations", "person_keypoints_train2017.json"
    )
    train_dataset = COCOJoints(
        cfg.data_root,
        ann_file,
        image_set="train2017",
        order=("image", "keypoints", "boxes", "info"),
    )
    logger.info("Num of Samples: {}".format(len(train_dataset)))
    train_sampler = data.RandomSampler(
        train_dataset, batch_size=cfg.batch_size, drop_last=True
    )

    transforms = [
        T.Normalize(mean=cfg.img_mean, std=cfg.img_std),
        RandomHorizontalFlip(0.5, keypoint_flip_order=cfg.keypoint_flip_order)
    ]

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
        RandomBoxAffine(
            degrees=cfg.rotate_range,
            scale=cfg.scale_range,
            output_shape=cfg.input_shape,
            rotate_prob=cfg.rotation_prob,
            scale_prob=cfg.scale_prob,
        )
    ]
    transforms += [T.ToMode()]

    train_queue = data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        num_workers=args.workers,
        transform=T.Compose(transforms=transforms, order=train_dataset.order,),
        collator=HeatmapCollator(
            cfg.input_shape,
            cfg.output_shape,
            cfg.keypoint_num,
            cfg.heat_thr,
            cfg.heat_kernels if args.multi_scale_supervision else cfg.heat_kernels[-1:],
            cfg.heat_range,
        ),
    )

    # Start training
    for epoch in range(start_epoch, cfg.epochs):
        loss = train(model, train_queue, optimizer, gm, epoch=epoch)
        logger.info("Epoch %d Train %.6f ", epoch, loss)

        if dist.get_rank() == 0 and epoch % cfg.save_freq == 0:  # save checkpoint
            mge.save(
                {"epoch": epoch + 1, "state_dict": model.state_dict()},
                os.path.join(save_dir, "epoch_{}.pkl".format(epoch)),
            )


def train(model, data_queue, optimizer, gm, epoch=0):
    def train_func(images, heatmaps, heat_valid):
        with gm:
            loss = model.calc_loss(images, heatmaps, heat_valid)
            gm.backward(loss)  # compute gradients
        optimizer.step().clear_grad()
        return loss

    avg_loss = 0
    total_time = 0

    t = time.time()
    for step, mini_batch in enumerate(data_queue):

        current_step = epoch * len(data_queue) + step
        if current_step < cfg.warm_epochs * len(data_queue):
            lr_factor = cfg.lr_ratio + (
                1 - cfg.lr_ratio
            ) * current_step / cfg.warm_epochs / len(data_queue)
        else:
            lr_factor = 1 - (current_step - len(data_queue) * cfg.warm_epochs) / (
                len(data_queue) * (cfg.epochs - cfg.warm_epochs)
            )

        lr = cfg.initial_lr * lr_factor

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        lr = optimizer.param_groups[0]["lr"]

        loss = train_func(
            mge.tensor(mini_batch["data"]),
            mge.tensor(mini_batch["heatmap"]),
            mge.tensor(mini_batch["heat_valid"]),
        )

        avg_loss = (avg_loss * step + loss.numpy().item()) / (step + 1)
        total_time += time.time() - t
        t = time.time()

        if step % cfg.report_freq == 0 and dist.get_rank() == 0:
            logger.info(
                "Epoch {} Step {}, LR {:.6f} Loss {:.6f}({:.6f}) Elapsed Time {:.3f}s".format(
                    epoch, step, lr, loss.numpy().item(), avg_loss, total_time
                )
            )

    return avg_loss


if __name__ == "__main__":
    main()
