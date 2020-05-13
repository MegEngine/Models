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

import megengine as mge
import megengine.data as data
import megengine.data.dataset as dataset
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.jit as jit
import megengine.optimizer as optim
import numpy as np

from official.vision.segmentation.deeplabv3plus import (
    DeepLabV3Plus,
    softmax_cross_entropy,
)
from official.vision.segmentation.utils import import_config_from_file

logger = mge.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="configuration file"
    )
    parser.add_argument(
        "-d", "--dataset_dir", type=str, default="/data/datasets/VOC2012",
    )
    parser.add_argument(
        "-w", "--weight_file", type=str, default=None, help="pre-train weights file",
    )
    parser.add_argument(
        "-n", "--ngpus", type=int, default=8, help="batchsize for training"
    )
    parser.add_argument(
        "-r", "--resume", type=str, default=None, help="resume model file"
    )
    args = parser.parse_args()

    world_size = args.ngpus
    logger.info("Device Count = %d", world_size)
    if world_size > 1:
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
    cfg = import_config_from_file(args.config)

    if world_size > 1:
        dist.init_process_group(
            master_ip="localhost",
            master_port=23456,
            world_size=world_size,
            rank=rank,
            dev=rank,
        )
        logger.info("Init process group done")

    logger.info("Prepare dataset")
    train_loader, epoch_size = build_dataloader(cfg.BATCH_SIZE, args.dataset_dir, cfg)
    batch_iter = epoch_size // (cfg.BATCH_SIZE * world_size)

    net = DeepLabV3Plus(class_num=cfg.NUM_CLASSES, pretrained=args.weight_file)
    base_lr = cfg.LEARNING_RATE * world_size
    optimizer = optim.SGD(
        net.parameters(requires_grad=True),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.00004,
    )

    @jit.trace(symbolic=True, opt_level=2)
    def train_func(data, label, net=None, optimizer=None):
        net.train()
        pred = net(data)
        loss = softmax_cross_entropy(pred, label, ignore_index=cfg.IGNORE_INDEX)
        optimizer.backward(loss)
        return pred, loss

    begin_epoch = 0
    end_epoch = cfg.EPOCHS
    if args.resume is not None:
        pretrained = mge.load(args.resume)
        begin_epoch = pretrained["epoch"] + 1
        net.load_state_dict(pretrained["state_dict"])
        logger.info("load success: epoch %d", begin_epoch)

    itr = begin_epoch * batch_iter
    max_itr = end_epoch * batch_iter

    image = mge.tensor(
        np.zeros([cfg.BATCH_SIZE, 3, cfg.IMG_HEIGHT, cfg.IMG_WIDTH]).astype(np.float32),
        dtype="float32",
    )
    label = mge.tensor(
        np.zeros([cfg.BATCH_SIZE, cfg.IMG_HEIGHT, cfg.IMG_WIDTH]).astype(np.int32),
        dtype="int32",
    )
    exp_name = os.path.abspath(os.path.dirname(__file__)).split("/")[-1]

    for epoch in range(begin_epoch, end_epoch):
        for i_batch, sample_batched in enumerate(train_loader):

            def adjust_lr(optimizer, itr, max_itr):
                now_lr = base_lr * (1 - itr / (max_itr + 1)) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group["lr"] = now_lr
                return now_lr

            now_lr = adjust_lr(optimizer, itr, max_itr)
            inputs_batched, labels_batched = sample_batched
            labels_batched = np.squeeze(labels_batched, axis=1).astype(np.int32)
            image.set_value(inputs_batched)
            label.set_value(labels_batched)

            optimizer.zero_grad()
            _, loss = train_func(image, label, net=net, optimizer=optimizer)
            optimizer.step()
            running_loss = loss.numpy()[0]

            if rank == 0:
                logger.info(
                    "%s epoch:%d/%d\tbatch:%d/%d\titr:%d\tlr:%g\tloss:%g",
                    exp_name,
                    epoch,
                    end_epoch,
                    i_batch,
                    batch_iter,
                    itr + 1,
                    now_lr,
                    running_loss,
                )
            itr += 1

        if rank == 0:
            save_path = os.path.join(cfg.MODEL_SAVE_DIR, "epoch%d.pkl" % (epoch))
            mge.save({"epoch": epoch, "state_dict": net.state_dict()}, save_path)
            logger.info("save epoch%d", epoch)


def build_dataloader(batch_size, dataset_dir, cfg):
    if cfg.DATASET == "VOC2012":
        train_dataset = dataset.PascalVOC(
            dataset_dir,
            cfg.DATA_TYPE,
            order=["image", "mask"]
        )
    elif cfg.DATASET == "Cityscapes":
        train_dataset = dataset.Cityscapes(
            dataset_dir,
            "train",
            mode='gtFine',
            order=["image", "mask"]
        )
    else:
        raise ValueError("Unsupported dataset {}".format(cfg.DATASET))
    train_sampler = data.RandomSampler(train_dataset, batch_size, drop_last=True)
    train_dataloader = data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose(
            transforms=[
                T.RandomHorizontalFlip(0.5),
                T.RandomResize(scale_range=(0.5, 2)),
                T.RandomCrop(
                    output_size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
                    padding_value=[0, 0, 0],
                    padding_maskvalue=255,
                ),
                T.Normalize(mean=cfg.IMG_MEAN, std=cfg.IMG_STD),
                T.ToMode(),
            ],
            order=["image", "mask"],
        ),
        num_workers=0,
    )
    return train_dataloader, train_dataset.__len__()


if __name__ == "__main__":
    main()
