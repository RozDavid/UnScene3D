###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import sys
import numpy as np
import random
import math
from PIL import Image, ImageOps, ImageFilter
import pandas as pd

import torch
import torch.utils.data as data
import torchvision.transforms as transform

from encoding.datasets.base import BaseDataset

from data.constants.scannet_constants import *

class ScanNet2DSegmentationDataset(BaseDataset):

    NUM_CLASS = len(VALID_CLASS_IDS_200)
    VALID_LABELS = VALID_CLASS_IDS_200
    CLASS_LABELS = CLASS_LABELS_200

    def __init__(self, root='', split='train',
                 mode=None, transform=None, target_transform=None, ignore_index=-1, **kwargs):
        super(ScanNet2DSegmentationDataset, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists and prepare dataset automatically
        assert os.path.exists(root), f"Please setup the dataset correctly, data was not found at path: {root}"
        self.images, self.masks = _get_scannet_pairs(root, split)
        if split != 'test':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

        # Create label map
        label_map = {}
        n_used = 0
        for l in range(max(SCANNET_COLOR_MAP_LONG.keys()) + 1):
            if l in VALID_CLASS_IDS_200:
                label_map[l] = n_used
                n_used += 1
            else:
                label_map[l] = ignore_index
        label_map[ignore_index] = ignore_index
        self.label_map = label_map
        self.label_mapper = np.vectorize(lambda x: self.label_map[x])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])

        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def get_classnames(self):
        return list(self.CLASS_LABELS)

    def _mask_transform(self, mask):
        target = np.array(mask)
        target = self.label_mapper(target).astype('int64')  # map to outputids
        return torch.from_numpy(target)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1


class ScanNet20_2DSegmentationDataset(ScanNet2DSegmentationDataset):

    NUM_CLASS = len(VALID_CLASS_IDS_20)
    VALID_LABELS = VALID_CLASS_IDS_20
    CLASS_LABELS = CLASS_LABELS_20

    def __init__(self, root='', split='train', mode=None, transform=None, target_transform=None, ignore_index=-1, **kwargs):

        super(ScanNet20_2DSegmentationDataset, self).__init__(root, split, mode, transform, target_transform, **kwargs)

        # Load dataframe with label map
        labels_pd = pd.read_csv('data/constants/scannetv2-labels.combined.tsv', sep='\t', header=0)
        labels_pd.loc[labels_pd.raw_category == 'stick', ['category']] = 'object'
        labels_pd.loc[labels_pd.category == 'wardrobe ', ['category']] = 'wardrobe'

        # Create label map
        label_map = {}
        for index, row in labels_pd.iterrows():
            id = row['id']
            nyu40id = row['nyu40id']
            if nyu40id in self.VALID_LABELS:
                scannet20_index = self.VALID_LABELS.index(nyu40id)
                label_map[id] = scannet20_index
            else:
                label_map[id] = ignore_index

        # Add ignore
        label_map[0] = ignore_index
        label_map[ignore_index] = ignore_index
        self.label_map = label_map
        self.label_mapper = np.vectorize(lambda x: self.label_map[x])


def _get_scannet_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'labels/training')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        print('len(img_paths):', len(img_paths))
    elif split == 'val':
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'labels/validation')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    else:
        assert split == 'trainval'
        train_img_folder = os.path.join(folder, 'images/training')
        train_mask_folder = os.path.join(folder, 'annotations/training')
        val_img_folder = os.path.join(folder, 'images/validation')
        val_mask_folder = os.path.join(folder, 'annotations/validation')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths