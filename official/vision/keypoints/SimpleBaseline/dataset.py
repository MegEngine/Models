import megengine as mge
from megengine.data.dataset.vision.meta_vision import VisionDataset
from megengine.data import Collator

import numpy as np
import cv2
import os.path as osp
import json
from collections import defaultdict, OrderedDict

class COCOJoints(VisionDataset):

    supported_order = ("image", "keypoints", "boxes", "info")

    keypoint_names = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )

    min_bbox_h = 0
    min_bbox_w = 0
    min_box_size = 1500.
    min_bbox_score = 1e-10

    def __init__(self, root, ann_file, order, image_set = 'train', remove_untypical_ann=True):

        super(COCOJoints, self).__init__(
            root, order=order, supported_order=self.supported_order
        )

        self.keypoint_num = len(self.keypoint_names)
        self.root = root
        self.image_set = image_set
        self.order = order


        if isinstance(ann_file, str):
            with open(ann_file, "r") as f:
                dataset = json.load(f)
        else:
            dataset = ann_file

        self.imgs = OrderedDict()

        for img in dataset["images"]:
            # for saving memory
            if "license" in img:
                del img["license"]
            if "coco_url" in img:
                del img["coco_url"]
            if "date_captured" in img:
                del img["date_captured"]
            if "flickr_url" in img:
                del img["flickr_url"]
            self.imgs[img["id"]] = img

        self.ids = list(sorted(self.imgs.keys()))

        selected_anns = []
        for ann in dataset["annotations"]:
            if "image_id" in ann.keys() and ann["image_id"] not in self.ids:
                continue

            if "iscrowd" in ann.keys() and ann["iscrowd"]:
                continue

            if remove_untypical_ann:
                if "keypoints" in ann.keys() and "keypoints" in self.order:
                    joints = np.array(ann["keypoints"]).reshape(
                        self.keypoint_num, 3)
                    if np.sum(joints[:, -1]) == 0 or ann['num_keypoints'] == 0:
                        continue

                if "bbox" in ann.keys() and "bbox" in self.order:
                    x, y, h, w = ann["bbox"]
                    if (
                            h < self.min_bbox_h or
                            w < self.min_bbox_w or
                            h*w < self.min_bbox_area):
                        continue

                if "score" in ann.keys() and "score" in self.order:
                    if ann["score"] < self.min_bbox_score:
                        continue

            selected_anns.append(ann)
        self.anns = selected_anns
      
    def __len__(self):
        return len(self.anns)

    def get_image_info(self, index):
        img_id = self.anns[index]["image_id"]
        img_info = self.imgs[img_id]
        return img_info

    def __getitem__(self, index):

        ann = self.anns[index]
        img_id = ann["image_id"]

        if not self.has_fetcher:
            self.fetcher = nr.Fetcher()
            self.has_fetcher = True

        target = []
        for k in self.order:
            if k == "image":

                file_name = self.imgs[img_id]["file_name"]
                img_path = osp.join(self.root, self.image_set, file_name)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                target.append(image)

            elif k == "keypoints":
                joints = np.array(ann["keypoints"]).reshape(
                    len(self.keypoint_names), 3).astype(np.float)
                joints = joints[np.newaxis]
                target.append(joints)

            elif k == "boxes":
                x, y, w, h = np.array(ann["bbox"]).reshape(4)
                bbox = [
                    x,
                    y,
                    x + w,
                    y + h
                ]
                bbox = np.array(bbox, dtype=np.float32)
                target.append(bbox[np.newaxis])

            elif k == "info":
                info = self.imgs[img_id]
                info = [info["height"], info["width"],
                        info["file_name"], ann["image_id"]]
                if "score" in ann.keys():
                    info.append(ann["score"])
                target.append(info)

        return tuple(target)


class HeatmapCollator(Collator):
    def __init__(self, image_shape, heatmap_shape, keypoint_num, heat_thre, heat_kernel, heat_range=255):
        super().__init__()
        self.image_shape = image_shape
        self.heatmap_shape = heatmap_shape
        self.keypoint_num = keypoint_num
        self.heat_thre = heat_thre
        self.heat_kernel = heat_kernel
        self.heat_range = heat_range

        self.stride = image_shape[1] // heatmap_shape[1]

        x = np.arange(0, heatmap_shape[1], 1)
        y = np.arange(0, heatmap_shape[0], 1)

        grid_x, grid_y = np.meshgrid(x, y)

        self.grid_x = grid_x[None].repeat(keypoint_num, 0)
        self.grid_y = grid_y[None].repeat(keypoint_num, 0)

    def apply(self, inputs):
        """
        assume order = ("images, keypoints, bboxes, info")
        """
        batch_data = defaultdict(list)

        for image, keypoints, _, info in inputs:

            batch_data["data"].append(image)

            joints = (keypoints[0, :, :2] + 0.5) / self.stride - 0.5
            heat_valid = np.array(keypoints[0, :, -1] > 0.1).astype(np.float32)
            dis = (
                self.grid_x - joints[:, 0, np.newaxis, np.newaxis])**2 + \
                (self.grid_y - joints[:, 1, np.newaxis, np.newaxis])**2
            heatmap = np.exp(
                -dis / 2 / self.heat_kernel**2
            )
            heatmap[heatmap < self.heat_thre] = 0
            heatmap[heat_valid == 0] = 0
            sum_for_norm = heatmap.sum((1, 2))
            heatmap[sum_for_norm > 0] = heatmap[sum_for_norm > 0] / \
                sum_for_norm[sum_for_norm > 0][:, None, None]
            maxi = np.max(heatmap, (1, 2))
            # heatmap *= self.heat_range
            heatmap[maxi > 1e-5] = heatmap[maxi > 1e-5] / \
                maxi[:, None, None][maxi > 1e-5] * self.heat_range

            batch_data["heatmap"].append(heatmap)
            batch_data["heat_valid"].append(heat_valid)
            batch_data["info"].append(info)

        for key, v in batch_data.items():
            if key != "info":
                batch_data[key] = np.ascontiguousarray(v).astype(np.float32)
        return batch_data
