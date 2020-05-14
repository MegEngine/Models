#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved.
import io

import numpy as np
import cv2
import nori2 as nori

from megengine.data.dataset import MapDataset


class NoriDataset(MapDataset):
    def __init__(self, nori_list):
        self.nori_fetcher = None

        self.nori_list = nori_list
        self.decode_nori_list()

    def __getitem__(self, index):
        self._check_nori_fetcher()
        nori_id, target = self.samples[index]

        img_bytes = self.nori_fetcher.get(nori_id)
        sample = cv2.imdecode(
            np.fromstring(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        return sample, target
    
    def _check_nori_fetcher(self):
        """Lazy initialize nori fetcher. In this way, `NoriDataset` can be pickled and used
            in multiprocessing.
        """
        if self.nori_fetcher is None:
            self.nori_fetcher = nori.Fetcher()

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Nori List: {}\n'.format(self.nori_list)

    def decode_nori_list(self, nori_list):
        raise NotImplementedError


class ImageNetNoriDataset(NoriDataset):
    def __init__(self, nori_list):
        super().__init__(nori_list)

    def decode_nori_list(self):
        self.samples = []
        with open(self.nori_list, 'r') as f:
            for line in f:
                nori_id, target, _ = line.strip().split()
                self.samples.append((nori_id, int(target)))