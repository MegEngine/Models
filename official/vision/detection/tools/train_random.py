# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import bisect
import copy
import functools
import importlib
import multiprocessing as mp
import os
import sys
import time
from tabulate import tabulate

import numpy as np

import megengine as mge
from megengine import optimizer as optim
from megengine.data import DataLoader, Infinite, RandomSampler
from megengine.data import transform as T
from megengine.distributed.group import get_default_group, init_process_group
from megengine.distributed.server import Server

from official.vision.detection.tools.data_mapper import data_mapper
from official.vision.detection.tools.utils import (
    AverageMeter,
    DetectionPadCollator,
    GroupedRandomSampler
)

from megengine import logger
logger.set_mgb_log_level("ERROR")

logger = mge.get_logger(__name__)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", default="net.py", type=str, help="net description file"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument(
        "-n", "--ngpus", default=-1, type=int, help="total number of gpus for training",
    )
    parser.add_argument(
        "-b", "--batch_size", default=2, type=int, help="batchsize for training",
    )
    parser.add_argument(
        "-d", "--dataset_dir", default="/data/datasets", type=str,
    )

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    # ------------------------ begin training -------------------------- #
    valid_nr_dev = mge.get_device_count("gpu")
    if args.ngpus == -1:
        world_size = valid_nr_dev
    else:
        if args.ngpus > valid_nr_dev:
            logger.error("do not have enough gpus for training")
            sys.exit(1)
        else:
            world_size = args.ngpus

    logger.info("Device Count = %d", world_size)

    log_dir = "log-of-{}".format(os.path.basename(args.file).split(".")[0])
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    if world_size > 1:
        server = Server()
        server.serve_in_thread()
        addr, port = server.server_address
        mp.set_start_method("spawn")
        processes = list()
        for i in range(world_size):
            process = mp.Process(target=worker, args=(i, world_size, addr, port, args))
            process.start()
            processes.append(process)

        for p in processes:
            p.join()
    else:
        worker(0, 1, 0, 0, args)


def worker(rank, world_size, addr, port, args):
    if world_size > 1:
        init_process_group(
            addr=addr,
            port=port,
            world_size=world_size,
            rank=rank,
        )
        group = get_default_group()
        mge.device.set_default_device("gpu{}".format(group.rank))
        logger.info("Init process group for gpu{} done".format(group.rank))

    sys.path.insert(0, os.path.dirname(args.file))
    current_network = importlib.import_module(os.path.basename(args.file).split(".")[0])

    model = current_network.Net(current_network.Cfg(), batch_size=args.batch_size)
    params = model.parameters(requires_grad=True)
    model.train()

    if rank == 0:
        logger.info(get_config_info(model.cfg))
    opt = optim.SGD(
        params,
        lr=model.cfg.basic_lr * world_size * model.batch_size,
        momentum=model.cfg.momentum,
        weight_decay=model.cfg.weight_decay,
    )

    if args.weight_file is not None:
        weights = mge.load(args.weight_file)
        model.backbone.bottom_up.load_state_dict(weights)

    if rank == 0:
        logger.info("Prepare dataset")
    train_loader = iter(build_dataloader(model.batch_size, args.dataset_dir, model.cfg))

    for epoch_id in range(model.cfg.max_epoch):
        for param_group in opt.param_groups:
            param_group["lr"] = (
                model.cfg.basic_lr
                * world_size
                * model.batch_size
                * (
                    model.cfg.lr_decay_rate
                    ** bisect.bisect_right(model.cfg.lr_decay_stages, epoch_id)
                )
            )

        tot_steps = model.cfg.nr_images_epoch // (model.batch_size * world_size)
        train_one_epoch(
            model,
            train_loader,
            opt,
            tot_steps,
            rank,
            epoch_id,
            world_size,
        )
        if rank == 0:
            save_path = "log-of-{}/epoch_{}.pkl".format(
                os.path.basename(args.file).split(".")[0], epoch_id
            )
            mge.save(
                {"epoch": epoch_id, "state_dict": model.state_dict()}, save_path,
            )
            logger.info("dump weights to %s", save_path)


def train_one_epoch(
    model,
    data_queue,
    opt,
    tot_steps,
    rank,
    epoch_id,
    world_size,
):
    meter = AverageMeter(record_len=model.cfg.num_losses)
    time_meter = AverageMeter(record_len=2)
    log_interval = model.cfg.log_interval
    for step in range(tot_steps):
        adjust_learning_rate(opt, epoch_id, step, model, world_size)

        data_tik = time.time()
        mini_batch = next(data_queue)
        data_tok = time.time()

        tik = time.time()
        opt.zero_grad()
        with opt.record():
            loss_dict = model(
                image=mge.tensor(mini_batch["data"]),
                gt_boxes=mge.tensor(mini_batch["gt_boxes"]),
                im_info=mge.tensor(mini_batch["im_info"])
            )
            opt.backward(loss_dict["total_loss"])
            loss_list = list(loss_dict.values())
        opt.step()
        tok = time.time()

        time_meter.update([tok - tik, data_tok - data_tik])

        if rank == 0:
            info_str = "e%d, %d/%d, lr:%f, "
            loss_str = ", ".join(
                ["{}:%f".format(loss) for loss in model.cfg.losses_keys]
            )
            time_str = ", train_time:%.3fs, data_time:%.3fs"
            log_info_str = info_str + loss_str + time_str
            meter.update([loss.numpy() for loss in loss_list])
            if step % log_interval == 0:
                average_loss = meter.average()
                logger.info(
                    log_info_str,
                    epoch_id,
                    step,
                    tot_steps,
                    opt.param_groups[0]["lr"],
                    *average_loss,
                    *time_meter.average()
                )
                meter.reset()
                time_meter.reset()


def get_config_info(config):
    config_table = []
    for c, v in config.__dict__.items():
        if not isinstance(v, (int, float, str, list, tuple, dict, np.ndarray)):
            if hasattr(v, "__name__"):
                v = v.__name__
            elif hasattr(v, "__class__"):
                v = v.__class__
            elif isinstance(v, functools.partial):
                v = v.func.__name__
        config_table.append((str(c), str(v)))
    config_table = tabulate(config_table)
    return config_table


def adjust_learning_rate(optimizer, epoch_id, step, model, world_size):
    base_lr = (
        model.cfg.basic_lr
        * world_size
        * model.batch_size
        * (
            model.cfg.lr_decay_rate
            ** bisect.bisect_right(model.cfg.lr_decay_stages, epoch_id)
        )
    )
    # Warm up
    if epoch_id == 0 and step < model.cfg.warm_iters:
        lr_factor = (step + 1.0) / model.cfg.warm_iters
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr * lr_factor


def build_dataset():
    from megengine.data.dataset import VisionDataset

    class PseudoDetectionDataset(VisionDataset):
        supported_order = ("image", "boxes", "boxes_category", "info")

        def __init__(self, length=256, *, order=None):
            super().__init__(None, order=order, supported_order=self.supported_order)
            self.length = length
            self.image = []
            self.boxes = []
            self.boxes_category = []
            self.info = []
            for i in range(self.length):
                self.image.append(np.random.randint(256, size=(320, 480, 3), dtype=np.uint8))
                b = []
                c = []
                for i in range(np.random.randint(1, 10)):
                    x, y, w, h = np.random.uniform(320, size=4)
                    b.append(np.array([x, y, x + w, y + h], dtype=np.float32))
                    c.append(np.random.randint(1, 81, dtype=np.int32))
                self.boxes.append(np.concatenate(b))
                self.boxes_category.append(np.stack(c))
                self.info.append({"height": 320, "width": 480, "file_name": str(i)})

        def __getitem__(self, index):
            target = []
            for k in self.order:
                if k == "image":
                    target.append(self.image[index])
                elif k == "boxes":
                    target.append(self.boxes[index])
                elif k == "boxes_category":
                    target.append(self.boxes_category[index])
                elif k == "info":
                    target.append([
                        self.info[index]["height"],
                        self.info[index]["width"],
                        self.info[index]["file_name"]
                    ])
            return tuple(target)

        def __len__(self):
            return self.length

        def get_img_info(self, index):
            return self.info[index]

    return PseudoDetectionDataset(order=["image", "boxes", "boxes_category", "info"])


def build_sampler(train_dataset, batch_size, aspect_grouping=[1]):
    def _compute_aspect_ratios(dataset):
        aspect_ratios = []
        for i in range(len(dataset)):
            info = dataset.get_img_info(i)
            aspect_ratios.append(info["height"] / info["width"])
        return aspect_ratios

    def _quantize(x, bins):
        return list(map(lambda y: bisect.bisect_right(sorted(bins), y), x))

    if len(aspect_grouping) == 0:
        return Infinite(RandomSampler(train_dataset, batch_size, drop_last=True))

    aspect_ratios = _compute_aspect_ratios(train_dataset)
    group_ids = _quantize(aspect_ratios, aspect_grouping)
    return Infinite(GroupedRandomSampler(train_dataset, batch_size, group_ids))


def build_dataloader(batch_size, data_dir, cfg):
    train_dataset = build_dataset()
    train_sampler = build_sampler(train_dataset, batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose(
            transforms=[
                T.ShortestEdgeResize(
                    cfg.train_image_short_size,
                    cfg.train_image_max_size,
                    sample_style="choice",
                ),
                T.RandomHorizontalFlip(),
                T.ToMode(),
            ],
            order=["image", "boxes", "boxes_category"],
        ),
        collator=DetectionPadCollator(),
        num_workers=2,
    )
    return train_dataloader


if __name__ == "__main__":
    main()
