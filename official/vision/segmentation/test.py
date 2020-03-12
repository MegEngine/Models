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

import cv2
import megengine as mge
import megengine.data as data
import megengine.data.dataset as dataset
import megengine.data.transform as T
import megengine.jit as jit
import numpy as np
from tqdm import tqdm

from official.vision.segmentation.deeplabv3plus import DeepLabV3Plus


class Config:
    DATA_WORKERS = 4

    NUM_CLASSES = 21
    IMG_SIZE = 512
    IMG_MEAN = [103.530, 116.280, 123.675]
    IMG_STD = [57.375, 57.120, 58.395]

    VAL_BATCHES = 1
    VAL_MULTISCALE = [1.0]  # [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    VAL_FLIP = False
    VAL_SLIP = False
    VAL_SAVE = None


cfg = Config()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset_dir", type=str, default="/data/datasets/VOC2012",
    )
    parser.add_argument(
        "-m", "--model_path", type=str, default=None, help="eval model file"
    )
    args = parser.parse_args()

    test_loader, test_size = build_dataloader(args.dataset_dir)
    print("number of test images: %d" % (test_size))
    net = DeepLabV3Plus(class_num=cfg.NUM_CLASSES)
    model_dict = mge.load(args.model_path)

    net.load_state_dict(model_dict["state_dict"])
    print("load model %s" % (args.model_path))
    net.eval()

    result_list = []
    for sample_batched in tqdm(test_loader):
        img = sample_batched[0].squeeze()
        label = sample_batched[1].squeeze()
        pred = evaluate(net, img)
        result_list.append({"pred": pred, "gt": label})
    if cfg.VAL_SAVE:
        save_results(result_list, cfg.VAL_SAVE)
    compute_metric(result_list)


def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0
    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2
    img = cv2.copyMakeBorder(
        img, margin[0], margin[1], margin[2], margin[3], border_mode, value=value
    )
    return img, margin


def eval_single(net, img, is_flip):
    @jit.trace(symbolic=True, opt_level=2)
    def pred_fun(data, net=None):
        net.eval()
        pred = net(data)
        return pred

    data = mge.tensor()
    data.set_value(img.transpose(2, 0, 1)[np.newaxis])
    pred = pred_fun(data, net=net)
    if is_flip:
        img_flip = img[:, ::-1, :]
        data.set_value(img_flip.transpose(2, 0, 1)[np.newaxis])
        pred_flip = pred_fun(data, net=net)
        pred = (pred + pred_flip[:, :, :, ::-1]) / 2.0
        del pred_flip
    pred = pred.numpy().squeeze().transpose(1, 2, 0)
    del data
    return pred


def evaluate(net, img):
    ori_h, ori_w, _ = img.shape
    pred_all = np.zeros((ori_h, ori_w, cfg.NUM_CLASSES))
    for rate in cfg.VAL_MULTISCALE:
        if cfg.VAL_SLIP:
            img_scale = cv2.resize(
                img, None, fx=rate, fy=rate, interpolation=cv2.INTER_LINEAR
            )
            val_size = (cfg.IMG_SIZE, cfg.IMG_SIZE)
        else:
            out_h, out_w = int(cfg.IMG_SIZE * rate), int(cfg.IMG_SIZE * rate)
            img_scale = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            val_size = (out_h, out_w)

        new_h, new_w, _ = img_scale.shape
        if (new_h <= val_size[0]) and (new_h <= val_size[1]):
            img_pad, margin = pad_image_to_shape(
                img_scale, val_size, cv2.BORDER_CONSTANT, value=0
            )
            pred = eval_single(net, img_pad, cfg.VAL_FLIP)
            pred = pred[
                margin[0] : (pred.shape[0] - margin[1]),
                margin[2] : (pred.shape[1] - margin[3]),
                :,
            ]
        else:
            stride_rate = 2 / 3
            stride = [int(np.ceil(i * stride_rate)) for i in val_size]
            print(img_scale.shape, stride, val_size)
            img_pad, margin = pad_image_to_shape(
                img_scale, val_size, cv2.BORDER_CONSTANT, value=0
            )
            pad_h, pad_w = img_pad.shape[:2]
            r_grid, c_grid = [
                int(np.ceil((ps - cs) / stride)) + 1
                for ps, cs, stride in zip(img_pad.shape, val_size, stride)
            ]

            pred_scale = np.zeros((pad_h, pad_w, cfg.NUM_CLASSES))
            count_scale = np.zeros((pad_h, pad_w, cfg.NUM_CLASSES))
            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride[1]
                    s_y = grid_yidx * stride[0]
                    e_x = min(s_x + val_size[1], pad_w)
                    e_y = min(s_y + val_size[0], pad_h)
                    s_x = e_x - val_size[1]
                    s_y = e_y - val_size[0]
                    img_sub = img_pad[s_y:e_y, s_x:e_x, :]
                    timg_pad, tmargin = pad_image_to_shape(
                        img_sub, val_size, cv2.BORDER_CONSTANT, value=0
                    )
                    print(tmargin, timg_pad.shape)
                    tpred = eval_single(net, timg_pad, cfg.VAL_FLIP)
                    tpred = tpred[
                        margin[0] : (tpred.shape[0] - margin[1]),
                        margin[2] : (tpred.shape[1] - margin[3]),
                        :,
                    ]
                    count_scale[s_y:e_y, s_x:e_x, :] += 1
                    pred_scale[s_y:e_y, s_x:e_x, :] += tpred
            pred_scale = pred_scale / count_scale
            pred = pred_scale[
                margin[0] : (pred_scale.shape[0] - margin[1]),
                margin[2] : (pred_scale.shape[1] - margin[3]),
                :,
            ]

        pred = cv2.resize(pred, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
        pred_all = pred_all + pred

    pred_all = pred_all / len(cfg.VAL_MULTISCALE)
    result = np.argmax(pred_all, axis=2).astype(np.uint8)
    return result


def save_results(result_list, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx, sample in enumerate(result_list):
        file_path = os.path.join(save_dir, "%d.png" % idx)
        cv2.imwrite(file_path, sample["pred"])
        file_path = os.path.join(save_dir, "%d.gt.png" % idx)
        cv2.imwrite(file_path, sample["gt"])


def compute_metric(result_list):
    """
    modified from https://github.com/YudeWang/deeplabv3plus-pytorch
    """
    TP, P, T = [], [], []
    for i in range(cfg.NUM_CLASSES):
        TP.append(mp.Value("i", 0, lock=True))
        P.append(mp.Value("i", 0, lock=True))
        T.append(mp.Value("i", 0, lock=True))

    def compare(start, step, TP, P, T):
        for idx in tqdm(range(start, len(result_list), step)):
            pred = result_list[idx]["pred"]
            gt = result_list[idx]["gt"]
            cal = gt < 255
            mask = (pred == gt) * cal
            for i in range(cfg.NUM_CLASSES):
                P[i].acquire()
                P[i].value += np.sum((pred == i) * cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt == i) * cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt == i) * mask)
                TP[i].release()

    p_list = []
    for i in range(8):
        p = mp.Process(target=compare, args=(i, 8, TP, P, T))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    class_names = dataset.PascalVOC.class_names
    IoU = []
    for i in range(cfg.NUM_CLASSES):
        IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
    for i in range(cfg.NUM_CLASSES):
        if i == 0:
            print("%11s:%7.3f%%" % ("backbound", IoU[i] * 100), end="\t")
        else:
            if i % 2 != 1:
                print("%11s:%7.3f%%" % (class_names[i - 1], IoU[i] * 100), end="\t")
            else:
                print("%11s:%7.3f%%" % (class_names[i - 1], IoU[i] * 100))
    miou = np.mean(np.array(IoU))
    print("\n======================================================")
    print("%11s:%7.3f%%" % ("mIoU", miou * 100))
    return miou


def build_dataloader(dataset_dir):
    val_dataset = dataset.PascalVOC(dataset_dir, "val", order=["image", "mask"])
    val_sampler = data.SequentialSampler(val_dataset, cfg.VAL_BATCHES)
    val_dataloader = data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        transform=T.Normalize(
            mean=cfg.IMG_MEAN, std=cfg.IMG_STD, order=["image", "mask"]
        ),
        num_workers=cfg.DATA_WORKERS,
    )
    return val_dataloader, val_dataset.__len__()


if __name__ == "__main__":
    main()
