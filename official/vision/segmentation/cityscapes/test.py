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

from official.vision.segmentation.cityscapes.deeplabv3plus import DeepLabV3Plus


class Config:
    DATA_WORKERS = 4

    NUM_CLASSES = 19
    VAL_HEIGHT = 800
    VAL_WIDTH = 800
    IMG_MEAN = [103.530, 116.280, 123.675]
    IMG_STD = [57.375, 57.120, 58.395]
    VAL_BATCHES = 1
    VAL_MULTISCALE = [1.0]  # [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    VAL_FLIP = False
    VAL_SLIP = True
    VAL_SAVE = None


cfg = Config()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset_dir", type=str, default="/data/datasets/Cityscapes",
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
        im_info = sample_batched[2]
        pred = evaluate(net, img)
        result_list.append({"pred": pred, "gt": label, "name":im_info[2]})
    if cfg.VAL_SAVE:
        save_results(result_list, cfg.VAL_SAVE)
    compute_metric(result_list)


## inference one image
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
            new_h, new_w = int(ori_h*rate), int(ori_w*rate)
            val_size = (cfg.VAL_HEIGHT, cfg.VAL_WIDTH)
        else:
            new_h, new_w = int(cfg.VAL_HEIGHT*rate), int(cfg.VAL_WIDTH*rate)
            val_size = (new_h, new_w)
        img_scale = cv2.resize(
			img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
		)

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
                    tpred = eval_single(net, img_sub, cfg.VAL_FLIP)
                    count_scale[s_y:e_y, s_x:e_x, :] += 1
                    pred_scale[s_y:e_y, s_x:e_x, :] += tpred
            #pred_scale = pred_scale / count_scale
            pred = pred_scale[
                margin[0] : (pred_scale.shape[0] - margin[1]),
                margin[2] : (pred_scale.shape[1] - margin[3]),
                :,
            ]

        pred = cv2.resize(pred, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
        pred_all = pred_all + pred

    #pred_all = pred_all / len(cfg.VAL_MULTISCALE)
    result = np.argmax(pred_all, axis=2).astype(np.uint8)
    return result


def save_results(result_list, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx, sample in enumerate(result_list):
        name = sample["name"].split('/')[-1][:-4]
        file_path = os.path.join(save_dir, "%s.png"%name)
        cv2.imwrite(file_path, sample["pred"])
        file_path = os.path.join(save_dir, "%s.gt.png"%name)
        cv2.imwrite(file_path, sample["gt"])

# voc cityscapes metric
def compute_metric(result_list):
    class_num = cfg.NUM_CLASSES
    hist = np.zeros((class_num, class_num))
    correct = 0
    labeled = 0
    count = 0
    for idx in range(len(result_list)):
        pred = result_list[idx]['pred']
        gt = result_list[idx]['gt']
        assert(pred.shape == gt.shape)
        k = (gt>=0) & (gt<class_num)
        labeled += np.sum(k)
        correct += np.sum((pred[k]==gt[k]))
        hist += np.bincount(class_num * gt[k].astype(int) + pred[k].astype(int), minlength=class_num**2).reshape(class_num, class_num)
        count += 1
    
    iu        = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU   = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU   = (iu[freq > 0] * freq[freq >0]).sum()
    mean_pixel_acc = correct / labeled
    
    class_names = dataset.Cityscapes.class_names
    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i+1)
        else:
            cls = '%d %s' % (i+1, class_names[i])
        lines.append('%-8s\t%.3f%%' % (cls, iu[i] * 100))
    lines.append('----------------------------     %-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IU', mean_IU * 100,'mean_pixel_ACC',mean_pixel_acc*100))
    line = "\n".join(lines)
    print(line)
    return mean_IU




def build_dataloader(dataset_dir):
    val_dataset = dataset.Cityscapes(
        dataset_dir,
        "val",
        mode='gtFine',
        order=["image", "mask", "info"]
    )
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
