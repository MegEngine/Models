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
from multiprocessing import Queue
from tqdm import tqdm

import cv2
import numpy as np

import megengine as mge
import megengine.distributed as dist
from megengine.data import DataLoader, dataset
from megengine.data import transform as T

from official.vision.segmentation.tools.utils import (
    InferenceSampler,
    class_colors,
    import_from_file
)

logger = mge.get_logger(__name__)
logger.setLevel("INFO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", default="net.py", type=str, help="net description file"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument(
        "-n", "--devices", default=1, type=int, help="total number of gpus for testing",
    )
    parser.add_argument(
        "-d", "--dataset_dir", default="/data/datasets", type=str,
    )
    args = parser.parse_args()

    current_network = import_from_file(args.file)
    cfg = current_network.Cfg()

    result_list = []
    if args.devices > 1:
        result_queue = Queue(500)

        dist_worker = dist.launcher(n_gpus=args.devices)(worker)
        dist_worker(
            current_network,
            args.weight_file,
            args.dataset_dir,
            result_queue,
        )

        num_imgs = dict(VOC2012=1449, Cityscapes=500)

        for _ in tqdm(range(num_imgs[cfg.dataset])):
            result_list.append(result_queue.get())

    else:
        worker(current_network, args.weight_file, args.dataset_dir, result_list)

    if cfg.val_save_path is not None:
        save_results(result_list, cfg.val_save_path, cfg)
    logger.info("Start evaluation!")
    compute_metric(result_list, cfg)


def worker(
    current_network, weight_file, dataset_dir, result_list,
):

    cfg = current_network.Cfg()
    cfg.backbone_pretrained = False
    model = current_network.Net(cfg)
    model.eval()

    state_dict = mge.load(weight_file)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

    def pred_func(data):
        pred = model(data)
        return pred

    test_loader = build_dataloader(dataset_dir, model.cfg)
    if dist.get_world_size() == 1:
        test_loader = tqdm(test_loader)

    for data in test_loader:
        img = data[0].squeeze()
        label = data[1].squeeze()
        im_info = data[2]
        pred = evaluate(pred_func, img, model.cfg)
        result = {"pred": pred, "gt": label, "name": im_info[2]}
        if dist.get_world_size() > 1:
            result_list.put_nowait(result)
        else:
            result_list.append(result)


# inference one image
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


def eval_single(pred_func, img, is_flip):
    pred = pred_func(mge.tensor(img.transpose(2, 0, 1)[np.newaxis]))
    if is_flip:
        pred_flip = pred_func(mge.tensor(img[:, ::-1].transpose(2, 0, 1)[np.newaxis]))
        pred = (pred + pred_flip[:, :, :, ::-1]) / 2.0
        del pred_flip
    pred = pred.numpy().squeeze().transpose(1, 2, 0)
    return pred


def evaluate(pred_func, img, cfg):
    ori_h, ori_w, _ = img.shape
    pred_all = np.zeros((ori_h, ori_w, cfg.num_classes))
    for rate in cfg.val_multiscale:
        if cfg.val_slip:
            new_h, new_w = int(ori_h * rate), int(ori_w * rate)
            val_size = (cfg.val_height, cfg.val_width)
        else:
            new_h, new_w = int(cfg.val_height * rate), int(cfg.val_width * rate)
            val_size = (new_h, new_w)
        img_scale = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        if (new_h <= val_size[0]) and (new_h <= val_size[1]):
            img_pad, margin = pad_image_to_shape(
                img_scale, val_size, cv2.BORDER_CONSTANT, value=0
            )
            pred = eval_single(pred_func, img_pad, cfg.val_flip)
            pred = pred[
                margin[0]:(pred.shape[0] - margin[1]),
                margin[2]:(pred.shape[1] - margin[3]),
            ]
        else:
            stride_rate = 2 / 3
            stride = [int(np.ceil(i * stride_rate)) for i in val_size]
            img_pad, margin = pad_image_to_shape(
                img_scale, val_size, cv2.BORDER_CONSTANT, value=0
            )
            pad_h, pad_w = img_pad.shape[:2]
            r_grid, c_grid = [
                int(np.ceil((ps - cs) / stride)) + 1
                for ps, cs, stride in zip(img_pad.shape, val_size, stride)
            ]

            pred_scale = np.zeros((pad_h, pad_w, cfg.num_classes))
            count_scale = np.zeros((pad_h, pad_w, cfg.num_classes))
            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride[1]
                    s_y = grid_yidx * stride[0]
                    e_x = min(s_x + val_size[1], pad_w)
                    e_y = min(s_y + val_size[0], pad_h)
                    s_x = e_x - val_size[1]
                    s_y = e_y - val_size[0]
                    img_sub = img_pad[s_y:e_y, s_x:e_x]
                    tpred = eval_single(pred_func, img_sub, cfg.val_flip)
                    count_scale[s_y:e_y, s_x:e_x] += 1
                    pred_scale[s_y:e_y, s_x:e_x] += tpred
            # pred_scale = pred_scale / count_scale
            pred = pred_scale[
                margin[0]:(pred_scale.shape[0] - margin[1]),
                margin[2]:(pred_scale.shape[1] - margin[3]),
            ]

        pred_all += cv2.resize(pred, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)

    # pred_all = pred_all / len(cfg.val_multiscale)
    result = np.argmax(pred_all, axis=2).astype(np.uint8)
    return result


def save_results(result_list, save_dir, cfg):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for sample in result_list:
        if cfg.dataset == "Cityscapes":
            name = sample["name"].split("/")[-1][:-4]
        else:
            name = sample["name"]
        file_path = os.path.join(save_dir, "%s.png" % name)
        cv2.imwrite(file_path, sample["pred"])
        file_path = os.path.join(save_dir, "%s.gt.png" % name)
        cv2.imwrite(file_path, sample["gt"])


# voc cityscapes metric
def compute_metric(result_list, cfg):
    num_classes = cfg.num_classes
    hist = np.zeros((num_classes, num_classes))
    correct = 0
    labeled = 0
    count = 0
    for result in result_list:
        pred = result["pred"]
        gt = result["gt"]
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < num_classes)
        labeled += np.sum(k)
        correct += np.sum((pred[k] == gt[k]))
        # pylint: disable=no-member
        hist += np.bincount(
            num_classes * gt[k].astype(int) + pred[k].astype(int),
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)
        count += 1

    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    # mean_IU_no_back = np.nanmean(iu[1:])
    # freq = hist.sum(1) / hist.sum()
    # freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    if cfg.dataset == "VOC2012":
        class_names = ("background", ) + dataset.PascalVOC.class_names
    elif cfg.dataset == "Cityscapes":
        class_names = dataset.Cityscapes.class_names
    else:
        raise ValueError("Unsupported dataset {}".format(cfg.dataset))

    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = "Class %d:" % (i + 1)
        else:
            cls = "%d %s" % (i + 1, class_names[i])
        lines.append("%-8s\t%.3f%%" % (cls, iu[i] * 100))
    lines.append(
        "----------------------------     %-8s\t%.3f%%\t%-8s\t%.3f%%" % (
            "mean_IU", mean_IU * 100, "mean_pixel_ACC", mean_pixel_acc * 100
        )
    )
    line = "\n".join(lines)
    logger.info(line)


class EvalPascalVOC(dataset.PascalVOC):
    def _trans_mask(self, mask):
        label = np.ones(mask.shape[:2]) * 255
        for i, (b, g, r) in enumerate(class_colors):
            label[
                (mask[:, :, 0] == b) & (mask[:, :, 1] == g) & (mask[:, :, 2] == r)
            ] = i
        return label.astype(np.uint8)


def build_dataloader(dataset_dir, cfg):
    if cfg.dataset == "VOC2012":
        val_dataset = EvalPascalVOC(
            dataset_dir,
            "val",
            order=["image", "mask", "info"]
        )
    elif cfg.dataset == "Cityscapes":
        val_dataset = dataset.Cityscapes(
            dataset_dir,
            "val",
            mode="gtFine",
            order=["image", "mask", "info"]
        )
    else:
        raise ValueError("Unsupported dataset {}".format(cfg.dataset))

    val_sampler = InferenceSampler(val_dataset, 1)
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        transform=T.Normalize(
            mean=cfg.img_mean, std=cfg.img_std, order=["image", "mask"]
        ),
        num_workers=2,
    )
    return val_dataloader


if __name__ == "__main__":
    main()
