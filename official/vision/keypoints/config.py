# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
class Config:
    ##############3## train ##############################################
    lr_ratio = 0.1
    warm_epochs = 1
    weight_decay = 1e-5

    half_body_transform = True
    extend_boxes = True

    ################## data ###############################################
    # basic
    # normalize
    IMG_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    IMG_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # shape
    input_shape = (256, 192)
    output_shape = (64, 48)

    # heat maps
    keypoint_num = 17
    heat_kernel = 1.5
    heat_thr = 1e-2
    heat_range = 255

    ##################### augumentation #####################################
    # extend
    x_ext = 0.6
    y_ext = 0.6

    # half body
    num_keypoints_half_body = 3
    prob_half_body = 0.3
    upper_body_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    lower_body_ids = [11, 12, 13, 14, 15, 16]

    keypoint_flip_order = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    # scale
    scale_prob = 1
    scale_range = [0.7, 1.3]

    # rorate
    rotation_prob = 0.6
    rotate_range = [-45, 45]

    ############## testing settings ##########################################
    test_aug_border = 10
    test_x_ext = 0.10
    test_y_ext = 0.10
    test_gaussian_kernel = 17
    second_value_aug = True
