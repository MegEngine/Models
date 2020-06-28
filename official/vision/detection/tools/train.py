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
from collections import defaultdict

import megengine as mge
import numpy as np
from megengine import distributed as dist
from megengine import jit
from megengine import optimizer as optim
from megengine.data import Collator, DataLoader, Infinite, RandomSampler
from megengine.data import transform as T
from tabulate import tabulate

from official.vision.detection.tools.data_mapper import data_mapper

logger = mge.get_logger(__name__)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, record_len=1):
        self.record_len = record_len
        self.sum = [0 for i in range(self.record_len)]
        self.cnt = 0

    def reset(self):
        self.sum = [0 for i in range(self.record_len)]
        self.cnt = 0

    def update(self, val):
        self.sum = [s + v for s, v in zip(self.sum, val)]
        self.cnt += 1

    def average(self):
        return [s / self.cnt for s in self.sum]


def worker(rank, world_size, args):
    if world_size > 1:
        dist.init_process_group(
            master_ip="localhost",
            master_port=23456,
            world_size=world_size,
            rank=rank,
            dev=rank,
        )
        logger.info("Init process group for gpu%d done", rank)

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

    logger.info("Prepare dataset")
    loader = build_dataloader(model.batch_size, args.dataset_dir, model.cfg)
    train_loader = iter(loader["train"])

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
            args.enable_sublinear,
        )
        if rank == 0:
            save_path = "log-of-{}/epoch_{}.pkl".format(
                os.path.basename(args.file).split(".")[0], epoch_id
            )
            mge.save(
                {"epoch": epoch_id, "state_dict": model.state_dict()}, save_path,
            )
            logger.info("dump weights to %s", save_path)


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


def train_one_epoch(
    model,
    data_queue,
    opt,
    tot_steps,
    rank,
    epoch_id,
    world_size,
    enable_sublinear=False,
):
    sublinear_cfg = jit.SublinearMemoryConfig() if enable_sublinear else None

    @jit.trace(symbolic=True, opt_level=2, sublinear_memory_config=sublinear_cfg)
    def propagate():
        loss_dict = model(model.inputs)
        opt.backward(loss_dict["total_loss"])
        losses = list(loss_dict.values())
        return losses

    meter = AverageMeter(record_len=model.cfg.num_losses)
    log_interval = model.cfg.log_interval
    for step in range(tot_steps):
        adjust_learning_rate(opt, epoch_id, step, model, world_size)
        mini_batch = next(data_queue)
        model.inputs["image"].set_value(mini_batch["data"])
        model.inputs["gt_boxes"].set_value(mini_batch["gt_boxes"])
        model.inputs["im_info"].set_value(mini_batch["im_info"])

        opt.zero_grad()
        loss_list = propagate()
        opt.step()

        if rank == 0:
            loss_str = ", ".join(["{}:%f".format(loss) for loss in model.cfg.losses_keys])
            log_info_str = "e%d, %d/%d, lr:%f, " + loss_str
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
                )
                meter.reset()


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", default="net.py", type=str, help="net description file"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="pre-train weights file",
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
    parser.add_argument("--enable_sublinear", action="store_true")

    return parser


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
        mp.set_start_method("spawn")
        processes = list()
        for i in range(world_size):
            process = mp.Process(target=worker, args=(i, world_size, args))
            process.start()
            processes.append(process)

        for p in processes:
            p.join()
    else:
        worker(0, 1, args)


def build_dataset(data_dir, cfg):
    data_cfg = copy.deepcopy(cfg.train_dataset)
    data_name = data_cfg.pop("name")

    data_cfg["root"] = os.path.join(data_dir, data_name, data_cfg["root"])

    if "ann_file" in data_cfg:
        data_cfg["ann_file"] = os.path.join(data_dir, data_name, data_cfg["ann_file"])

    data_cfg["order"] = ["image", "boxes", "boxes_category", "info"]

    return data_mapper[data_name](**data_cfg)


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
    train_dataset = build_dataset(data_dir, cfg)
    train_sampler = build_sampler(train_dataset, batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose(
            transforms=[
                T.ShortestEdgeResize(
                    cfg.train_image_short_size, cfg.train_image_max_size
                ),
                T.RandomHorizontalFlip(),
                T.ToMode(),
            ],
            order=["image", "boxes", "boxes_category"],
        ),
        collator=DetectionPadCollator(),
        num_workers=2,
    )
    return {"train": train_dataloader}


class GroupedRandomSampler(RandomSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        group_ids,
        indices=None,
        world_size=None,
        rank=None,
        seed=None,
    ):
        super().__init__(dataset, batch_size, False, indices, world_size, rank, seed)
        self.group_ids = group_ids
        assert len(group_ids) == len(dataset)
        groups = np.unique(self.group_ids).tolist()

        # buffer the indices of each group until batch size is reached
        self.buffer_per_group = {k: [] for k in groups}

    def batch(self):
        indices = list(self.sample())
        if self.world_size > 1:
            indices = self.scatter(indices)

        batch_index = []
        for ind in indices:
            group_id = self.group_ids[ind]
            group_buffer = self.buffer_per_group[group_id]
            group_buffer.append(ind)
            if len(group_buffer) == self.batch_size:
                batch_index.append(group_buffer)
                self.buffer_per_group[group_id] = []

        return iter(batch_index)

    def __len__(self):
        raise NotImplementedError("len() of GroupedRandomSampler is not well-defined.")


class DetectionPadCollator(Collator):
    def __init__(self, pad_value: float = 0.0):
        super().__init__()
        self.pad_value = pad_value

    def apply(self, inputs):
        """
        assume order = ["image", "boxes", "boxes_category", "info"]
        """
        batch_data = defaultdict(list)

        for image, boxes, boxes_category, info in inputs:
            batch_data["data"].append(image)
            batch_data["gt_boxes"].append(
                np.concatenate([boxes, boxes_category[:, np.newaxis]], axis=1).astype(
                    np.float32
                )
            )

            _, current_height, current_width = image.shape
            assert len(boxes) == len(boxes_category)
            num_instances = len(boxes)
            info = [
                current_height,
                current_width,
                info[0],
                info[1],
                num_instances,
            ]
            batch_data["im_info"].append(np.array(info, dtype=np.float32))

        for key, value in batch_data.items():
            pad_shape = list(max(s) for s in zip(*[x.shape for x in value]))
            pad_value = [
                np.pad(
                    v,
                    self._get_padding(v.shape, pad_shape),
                    constant_values=self.pad_value,
                )
                for v in value
            ]
            batch_data[key] = np.ascontiguousarray(pad_value)

        return batch_data

    def _get_padding(self, original_shape, target_shape):
        assert len(original_shape) == len(target_shape)
        shape = []
        for o, t in zip(original_shape, target_shape):
            shape.append((0, t - o))
        return tuple(shape)


if __name__ == "__main__":
    main()
