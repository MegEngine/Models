#!/usr/bin/env mdl
# This file will seal the nms opr within a better way than lib_nms
import ctypes
import os
import struct

import numpy as np
import megengine as mge
import megengine.functional as F
from megengine._internal.craniotome import CraniotomeBase
from megengine.core.tensor import wrap_io_tensor

_so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib_nms.so')
_so_lib = ctypes.CDLL(_so_path)

_TYPE_POINTER = ctypes.c_void_p
_TYPE_POINTER = ctypes.c_void_p
_TYPE_INT = ctypes.c_int32
_TYPE_FLOAT = ctypes.c_float

_so_lib.NMSForwardGpu.argtypes = [
    _TYPE_POINTER,
    _TYPE_POINTER,
    _TYPE_POINTER,
    _TYPE_POINTER,
    _TYPE_FLOAT,
    _TYPE_INT,
    _TYPE_POINTER,
]
_so_lib.NMSForwardGpu.restype = _TYPE_INT

_so_lib.CreateHostDevice.restype = _TYPE_POINTER


class NMSCran(CraniotomeBase):
    __nr_inputs__ = 1
    __nr_outputs__ = 3

    def setup(self, iou_threshold, max_output):
        self._iou_threshold = iou_threshold
        self._max_output = max_output
        # Load the necessary host device
        self._host_device = _so_lib.CreateHostDevice()

    def execute(self, inputs, outputs):
        box_tensor_ptr = inputs[0].pubapi_dev_tensor_ptr
        output_tensor_ptr = outputs[0].pubapi_dev_tensor_ptr
        output_num_tensor_ptr = outputs[1].pubapi_dev_tensor_ptr
        mask_tensor_ptr = outputs[2].pubapi_dev_tensor_ptr

        _so_lib.NMSForwardGpu(
            box_tensor_ptr, mask_tensor_ptr,
            output_tensor_ptr, output_num_tensor_ptr,
            self._iou_threshold, self._max_output,
            self._host_device
        )

    def grad(self, wrt_idx, inputs, outputs, out_grad):
        return 0

    def init_output_dtype(self, input_dtypes):
        return [np.int32, np.int32, np.int32]

    def get_serialize_params(self):
        return ('nms', struct.pack('fi', self._iou_threshold, self._max_output))

    def infer_shape(self, inp_shapes):
        nr_box = inp_shapes[0][0]
        threadsPerBlock = 64
        output_size = nr_box
        # here we compute the number of int32 used in mask_outputs.
        # In original version, we compute the bytes only.
        mask_size = int(
            nr_box * (
                nr_box // threadsPerBlock + int((nr_box % threadsPerBlock) > 0)
            ) * 8 / 4
        )
        return [[output_size], [1], [mask_size]]


@wrap_io_tensor
def gpu_nms(box, iou_threshold, max_output):
    keep, num, _ = NMSCran.make(box, iou_threshold=iou_threshold, max_output=max_output)
    return keep[:num]


def batched_nms(boxes, scores, idxs, iou_threshold, num_keep, use_offset=False):
    if use_offset:
        boxes_offset = mge.tensor(
            [0, 0, 1, 1], device=boxes.device
        ).reshape(1, 4).broadcast(boxes.shapeof(0), 4)
        boxes = boxes - boxes_offset
    max_coordinate = boxes.max()
    offsets = idxs * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets.reshape(-1, 1).broadcast(boxes.shapeof(0), 4)
    boxes_with_scores = F.concat([boxes_for_nms, scores.reshape(-1, 1)], axis=1)
    keep_inds = gpu_nms(boxes_with_scores, iou_threshold, num_keep)
    return keep_inds
