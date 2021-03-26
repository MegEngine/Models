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

import numpy as np

import megengine as mge
import megengine.distributed as dist
import megengine.functional as F
from megengine.autodiff import GradManager
from megengine.data import DataLoader, Infinite, RandomSampler, dataset
from megengine.data import transform as T
from megengine.optimizer import SGD

from official.vision.segmentation.tools.utils import AverageMeter, get_config_info, import_from_file

logger = mge.get_logger(__name__)
logger.setLevel("INFO")
mge.device.set_prealloc_config(1024, 1024, 256 * 1024 * 1024, 4.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", default="net.py", type=str, help="net description file"
    )
    parser.add_argument(
        "-n", "--devices", type=int, default=8, help="batch size for training"
    )
    parser.add_argument(
        "-d", "--dataset_dir", type=str, default="/data/datasets",
    )
    parser.add_argument(
        "-r", "--resume", type=str, default=None, help="resume model file"
    )
    args = parser.parse_args()

    # ------------------------ begin training -------------------------- #
    logger.info("Device Count = %d", args.devices)

    log_dir = "log-of-{}".format(os.path.basename(args.file).split(".")[0])
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    if args.devices > 1:
        trainer = dist.launcher(worker, n_gpus=args.devices)
        trainer(args)
    else:
        worker(args)


# pylint: disable=too-many-branches
def worker(args):
    current_network = import_from_file(args.file)

    model = current_network.Net(current_network.Cfg())
    model.train()

    if dist.get_rank() == 0:
        logger.info(get_config_info(model.cfg))
        logger.info(repr(model))

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    opt = SGD(
        [
            {
                "params": backbone_params,
                "lr": model.cfg.learning_rate * dist.get_world_size() * 0.1,
            },
            {"params": head_params},
        ],
        lr=model.cfg.learning_rate * dist.get_world_size(),
        momentum=model.cfg.momentum,
        weight_decay=model.cfg.weight_decay,
    )

    gm = GradManager()
    if dist.get_world_size() > 1:
        gm.attach(
            model.parameters(),
            callbacks=[dist.make_allreduce_cb("mean", dist.WORLD)]
        )
    else:
        gm.attach(model.parameters())

    cur_epoch = 0
    if args.resume is not None:
        pretrained = mge.load(args.resume)
        cur_epoch = pretrained["epoch"] + 1
        model.load_state_dict(pretrained["state_dict"])
        opt.load_state_dict(pretrained["opt"])
        if dist.get_rank() == 0:
            logger.info("load success: epoch %d", cur_epoch)

    if dist.get_world_size() > 1:
        dist.bcast_list_(model.parameters())  # sync parameters
        dist.bcast_list_(model.buffers())  # sync buffers

    if dist.get_rank() == 0:
        logger.info("Prepare dataset")
    train_loader = iter(
        build_dataloader(model.cfg.batch_size, args.dataset_dir, model.cfg)
    )

    for epoch in range(cur_epoch, model.cfg.max_epoch):
        train_one_epoch(model, train_loader, opt, gm, epoch)
        if dist.get_rank() == 0:
            save_path = "log-of-{}/epoch_{}.pkl".format(
                os.path.basename(args.file).split(".")[0], epoch
            )
            mge.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "opt": opt.state_dict()
            }, save_path)
            logger.info("dump weights to %s", save_path)


def train_one_epoch(model, data_queue, opt, gm, epoch):
    def train_func(data, label):
        with gm:
            pred = model(data)
            loss = cross_entropy(
                pred, label, ignore_label=model.cfg.ignore_label
            )
            gm.backward(loss)
        opt.step().clear_grad()
        return loss

    meter = AverageMeter(record_len=1)
    time_meter = AverageMeter(record_len=2)
    log_interval = model.cfg.log_interval
    tot_step = model.cfg.nr_images_epoch // (
        model.cfg.batch_size * dist.get_world_size()
    )
    for step in range(tot_step):
        adjust_learning_rate(opt, epoch, step, tot_step, model.cfg)

        data_tik = time.time()
        inputs, labels = next(data_queue)
        labels = np.squeeze(labels, axis=1).astype(np.int32)
        data_tok = time.time()

        tik = time.time()
        loss = train_func(mge.tensor(inputs), mge.tensor(labels))
        tok = time.time()

        time_meter.update([tok - tik, data_tok - data_tik])

        if dist.get_rank() == 0:
            info_str = "e%d, %d/%d, lr:%f, "
            loss_str = ", ".join(["{}:%f".format(loss) for loss in ["loss"]])
            time_str = ", train_time:%.3fs, data_time:%.3fs"
            log_info_str = info_str + loss_str + time_str
            meter.update([loss.numpy() for loss in [loss]])
            if step % log_interval == 0:
                logger.info(
                    log_info_str,
                    epoch,
                    step,
                    tot_step,
                    opt.param_groups[1]["lr"],
                    *meter.average(),
                    *time_meter.average()
                )
                meter.reset()
                time_meter.reset()


def adjust_learning_rate(optimizer, epoch, step, tot_step, cfg):
    max_iter = cfg.max_epoch * tot_step
    cur_iter = epoch * tot_step + step
    cur_lr = cfg.learning_rate * (1 - cur_iter / (max_iter + 1)) ** 0.9
    optimizer.param_groups[0]["lr"] = cur_lr * 0.1
    optimizer.param_groups[1]["lr"] = cur_lr


def cross_entropy(pred, label, axis=1, ignore_label=255):
    mask = label != ignore_label
    pred = pred.transpose(0, 2, 3, 1)
    return F.loss.cross_entropy(pred[mask], label[mask], axis)


def build_dataloader(batch_size, dataset_dir, cfg):
    if cfg.dataset == "VOC2012":
        train_dataset = dataset.PascalVOC(
            dataset_dir,
            cfg.data_type,
            order=["image", "mask"]
        )
    elif cfg.dataset == "Cityscapes":
        train_dataset = dataset.Cityscapes(
            dataset_dir,
            "train",
            mode='gtFine',
            order=["image", "mask"]
        )
    else:
        raise ValueError("Unsupported dataset {}".format(cfg.dataset))

    train_sampler = Infinite(RandomSampler(train_dataset, batch_size, drop_last=True))
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose(
            transforms=[
                T.RandomHorizontalFlip(0.5),
                T.RandomResize(scale_range=(0.5, 2)),
                T.RandomCrop(
                    output_size=(cfg.img_height, cfg.img_width),
                    padding_value=[0, 0, 0],
                    padding_maskvalue=255,
                ),
                T.Normalize(mean=cfg.img_mean, std=cfg.img_std),
                T.ToMode(),
            ],
            order=["image", "mask"],
        ),
        num_workers=2,
    )
    return train_dataloader


if __name__ == "__main__":
    main()
