from megengine.data.transform import VisionTransform
from megengine.data.transform.vision import functional as F
import cv2
import numpy as np


def get_dir(src_point, rot_rad):
    # browed from SimpleBaseline
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    # browed from SimpleBaseline
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, bbox_shape, scale, rot, output_shape, inv=0):

    dst_w = output_shape[1]
    dst_h = output_shape[0]
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)

    scale = dst_w / (bbox_shape[1] * scale)

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, 1 * -0.5], rot_rad)
    dst_dir = np.array([0, scale * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv == 0:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    else:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))

    return trans


class HalfBodyTransform(VisionTransform):

    def __init__(self, upper_body_ids, lower_body_ids, prob=0.3, order=None):
        super(HalfBodyTransform, self).__init__()

        self.prob = prob
        self.upper_body_ids = upper_body_ids
        self.lower_body_ids = lower_body_ids
        self.order = order

    def apply(self, input: tuple):

        self.joints = input[self.order.index("keypoints")][0]
        self.keypoint_num = self.joints.shape[0]

        self._is_transform = False
        if np.random.random() < self.prob:
            self._is_transform = True

        return super().apply(input)

    def _apply_image(self, image):
        return image

    def _apply_keypoints(self, keypoints):
        return keypoints

    def _apply_boxes(self, boxes):

        if self._is_transform:
            upper_joints = []
            lower_joints = []
            for joint_id in range(self.keypoint_num):
                if self.joints[joint_id, -1] > 0:
                    if joint_id in self.upper_body_ids:
                        upper_joints.append(self.joints[joint_id])
                    else:
                        lower_joints.append(self.joints[joint_id])

            if np.random.randn() < 0.5 and len(upper_joints) > 3:
                selected_joints = upper_joints
            else:
                selected_joints = lower_joints \
                    if len(lower_joints) > 3 else upper_joints

            selected_joints = np.array(selected_joints, np.float32)
            if len(selected_joints) < 3:
                return boxes
            else:
                left_top = np.amin(selected_joints[:, :2], axis=0)
                right_bottom = np.amax(selected_joints[:, :2], axis=0)

                center = (left_top + right_bottom) / 2

                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]

                boxes[0] = np.array([
                    center[0] - w/2,
                    center[1] - h/2,
                    center[0] + w/2,
                    center[1] + h/2
                ], dtype=np.float32)
                return boxes
        else:
            return boxes


class ExtendBoxes(VisionTransform):
    def __init__(self, extend_x, extend_y, w_h_ratio, random_extend_prob=1, order=None):
        super(ExtendBoxes, self).__init__()
        self.extend_x = extend_x
        self.extend_y = extend_y
        self.w_h_ratio = w_h_ratio
        self.random_extend_prob = random_extend_prob
        self.order = order

    def apply(self, input: tuple):
        self._rand = 1
        if np.random.random() < self.random_extend_prob:
            self._rand = np.random.random()
        return super().apply(input)

    def _apply_image(self, image):
        return image

    def _apply_keypoints(self, keypoints):
        return keypoints

    def _apply_boxes(self, boxes):
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            h = y2 - y1
            w = x2 - x1
            extend_h = (1 + self._rand * self.extend_y) * h
            extend_w = (1 + self._rand * self.extend_x) * w

            if extend_w > self.w_h_ratio * extend_h:
                extend_h = extend_w * 1.0 / self.w_h_ratio
            else:
                extend_w = extend_h * 1.0 * self.w_h_ratio

            boxes[i] = np.array([
                center_x - extend_w / 2,
                center_y - extend_h / 2,
                center_x + extend_w / 2,
                center_y + extend_h / 2
            ], dtype=np.float32)
        return boxes


class RandomAffine(VisionTransform):
    def __init__(self,
                 degrees, scale,  output_shape, rotate_prob=1,
                 scale_prob=1, bordervalue=0, order=None
                 ):
        super(RandomAffine, self).__init__(order)

        self.degrees_range = degrees
        self.scale_range = scale
        self.output_shape = output_shape
        self.rotate_prob = rotate_prob
        self.scale_prob = scale_prob
        self.bordervalue = bordervalue
        self.order = order

    def apply(self, input: tuple):
        _scale = 1
        is_scale = np.random.random() < self.scale_prob
        if is_scale:
            _scale = np.random.uniform(
                self.scale_range[0], self.scale_range[1])

        _degree = 0
        is_rotate = np.random.random() < self.rotate_prob
        if is_rotate:
            _degree = np.random.uniform(
                self.degrees_range[0], self.degrees_range[1])

        _bbox = input[self.order.index("boxes")][0]

        _center = np.array([
            (_bbox[0] + _bbox[2]) / 2,
            (_bbox[1] + _bbox[3]) / 2
        ], dtype=np.float32)
        _bbox_shape = np.array([
            _bbox[3] - _bbox[1],
            _bbox[2] - _bbox[0]
        ], dtype=np.float32)

        self.trans = get_affine_transform(
            _center, _bbox_shape,  _scale,
            _degree, self.output_shape
        )
        return super().apply(input)

    def _apply_image(self, image):
        img = cv2.warpAffine(
            image,
            self.trans,
            (int(self.output_shape[1]), int(self.output_shape[0])),
            flags=cv2.INTER_LINEAR, borderValue=self.bordervalue)
        return img

    def _apply_keypoints(self, keypoints):
        _keypoints = np.zeros_like(keypoints, dtype=np.float32)
        for i in range(keypoints.shape[0]):
            for j in range(keypoints.shape[1]):
                pt = np.array([keypoints[i, j, 0], keypoints[i, j, 1], 1.]).T
                pt = np.dot(self.trans, pt)[:2]
                if (
                    pt[0] < 0 or
                    pt[0] > self.output_shape[1] - 1 or
                    pt[1] < 0 or
                    pt[1] > self.output_shape[0] - 1 or
                    keypoints[i, j, 2] == 0
                ):
                    continue
                _keypoints[i, j, :2] = pt
                _keypoints[i, j, -1] = keypoints[i, j, -1]
        return _keypoints

    def _apply_boxes(self, boxes):
        return boxes


class RandomHorizontalFlip(VisionTransform):
    """Horizontally flip the input data randomly with a given probability.
    :param p: probability of the input data being flipped. Default: 0.5
    :param order: The same with :class:`VisionTransform`
    """

    def __init__(self, prob: float = 0.5, *, keypoint_flip_order, order=None):
        super().__init__(order)
        self.prob = prob
        self.keypoint_flip_order = keypoint_flip_order

    def apply(self, input: tuple):
        self._flipped = np.random.random() < self.prob
        self._w = self._get_image(input).shape[1]
        return super().apply(input)

    def _apply_image(self, image):
        if self._flipped:
            image = image.astype("float32")
            return F.flip(image, flipCode=1)
        return image

    def _apply_coords(self, coords):
        if self._flipped:
            coords[:, 0] = self._w - coords[:, 0]
        return coords

    def _apply_boxes(self, boxes):
        _boxes = boxes.copy()
        if self._flipped:
            _boxes[:, 0] = self._w - boxes[:, 2]
            _boxes[:, 2] = self._w - boxes[:, 0]
        return _boxes

    def _apply_mask(self, mask):
        if self._flipped:
            return F.flip(mask, flipCode=1)
        return mask

    def _apply_keypoints(self, keypoints):
        if self._flipped:
            for i in range(len(keypoints)):
                keypoints[i, :, 0] = self._w - keypoints[i, :, 0] - 1
                keypoints[i] = keypoints[i][self.keypoint_flip_order]
        return keypoints
