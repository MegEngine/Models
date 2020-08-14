# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import random
from collections import defaultdict

import cv2
import numpy as np

from megengine.data import Collator, RandomSampler

from official.vision.detection.tools.data_mapper import data_mapper
from official.vision.detection.tools.nms import py_cpu_nms


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
            batch_data["data"].append(image.astype(np.float32))
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


class DetEvaluator:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def get_hw_by_short_size(im_height, im_width, short_size, max_size):
        """get height and width by short size

           Args:
               im_height(int): height of original image, e.g. 800
               im_width(int): width of original image, e.g. 1000
               short_size(int): short size of transformed image. e.g. 800
               max_size(int): max size of transformed image. e.g. 1333

           Returns:
               resized_height(int): height of transformed image
               resized_width(int): width of transformed image
        """

        im_size_min = np.min([im_height, im_width])
        im_size_max = np.max([im_height, im_width])
        scale = (short_size + 0.0) / im_size_min
        if scale * im_size_max > max_size:
            scale = (max_size + 0.0) / im_size_max

        resized_height, resized_width = (
            int(round(im_height * scale)),
            int(round(im_width * scale)),
        )
        return resized_height, resized_width

    @staticmethod
    def process_inputs(img, short_size, max_size, flip=False):
        original_height, original_width, _ = img.shape
        resized_height, resized_width = DetEvaluator.get_hw_by_short_size(
            original_height, original_width, short_size, max_size
        )
        resized_img = cv2.resize(
            img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR,
        )
        resized_img = cv2.flip(resized_img, 1) if flip else resized_img
        trans_img = np.ascontiguousarray(
            resized_img.transpose(2, 0, 1)[None, :, :, :], dtype=np.float32
        )
        im_info = np.array(
            [(resized_height, resized_width, original_height, original_width)],
            dtype=np.float32,
        )
        return trans_img, im_info

    def predict(self, val_func):
        """
        Args:
            val_func(callable): model inference function

        Returns:
            results boxes: detection model output
        """
        model = self.model

        box_cls, box_delta = val_func()
        box_cls, box_delta = box_cls.numpy(), box_delta.numpy()
        dtboxes_all = list()
        all_inds = np.where(box_cls > model.cfg.test_cls_threshold)

        for c in range(0, model.cfg.num_classes):
            inds = np.where(all_inds[1] == c)[0]
            inds = all_inds[0][inds]
            scores = box_cls[inds, c]
            if model.cfg.class_aware_box:
                bboxes = box_delta[inds, c, :]
            else:
                bboxes = box_delta[inds, :]

            dtboxes = np.hstack((bboxes, scores[:, np.newaxis])).astype(np.float32)

            if dtboxes.size > 0:
                keep = py_cpu_nms(dtboxes, model.cfg.test_nms)
                dtboxes = np.hstack(
                    (dtboxes[keep], np.ones((len(keep), 1), np.float32) * c)
                ).astype(np.float32)
                dtboxes_all.extend(dtboxes)

        if len(dtboxes_all) > model.cfg.test_max_boxes_per_image:
            dtboxes_all = sorted(dtboxes_all, reverse=True, key=lambda i: i[4])[
                : model.cfg.test_max_boxes_per_image
            ]

        dtboxes_all = np.array(dtboxes_all, dtype=np.float)
        return dtboxes_all

    @staticmethod
    def format(results, cfg):
        dataset_class = data_mapper[cfg.test_dataset["name"]]

        all_results = []
        for record in results:
            image_filename = record["image_id"]
            boxes = record["det_res"]
            if len(boxes) <= 0:
                continue
            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
            for box in boxes:
                elem = dict()
                elem["image_id"] = image_filename
                elem["bbox"] = box[:4].tolist()
                elem["score"] = box[4]
                if hasattr(dataset_class, "classes_originID"):
                    elem["category_id"] = dataset_class.classes_originID[
                        dataset_class.class_names[int(box[5])]
                    ]
                else:
                    elem["category_id"] = int(box[5])
                all_results.append(elem)
        return all_results

    @staticmethod
    def vis_det(
        img,
        dets,
        is_show_label=True,
        classes=None,
        thresh=0.3,
        name="detection",
        return_img=True,
    ):
        img = np.array(img)
        colors = dict()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in dets:
            bb = det[:4].astype(int)
            if is_show_label:
                cls_id = int(det[5])
                score = det[4]

                if cls_id == 0:
                    continue
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (
                            random.random() * 255,
                            random.random() * 255,
                            random.random() * 255,
                        )

                    cv2.rectangle(
                        img, (bb[0], bb[1]), (bb[2], bb[3]), colors[cls_id], 3
                    )

                    if classes and len(classes) > cls_id:
                        cls_name = classes[cls_id]
                    else:
                        cls_name = str(cls_id)
                    cv2.putText(
                        img,
                        "{:s} {:.3f}".format(cls_name, score),
                        (bb[0], bb[1] - 2),
                        font,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
            else:
                cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)

        if return_img:
            return img
        cv2.imshow(name, img)
        while True:
            c = cv2.waitKey(100000)
            if c == ord("d"):
                return None
            elif c == ord("n"):
                break
