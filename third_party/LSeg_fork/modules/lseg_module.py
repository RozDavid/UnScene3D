import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from argparse import ArgumentParser
import pytorch_lightning as pl
from .lsegmentation_module import LSegmentationModule
from .models.lseg_net import LSegNet
from encoding.models.sseg.base import up_kwargs

from data.constants.scannet_constants import CLASS_LABELS_200

import os
import clip
import numpy as np

from scipy import signal
import glob

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


class LSegModule(LSegmentationModule):
    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, **kwargs):
        super(LSegModule, self).__init__(
            data_path, dataset, batch_size, base_lr, max_epochs, **kwargs
        )

        if dataset == "citys":
            self.base_size = 2048
            self.crop_size = 768
        elif 'scannet' in dataset.lower() and not kwargs['high_res']:
            self.base_size = int(520 * kwargs['res_scale'])
            self.crop_size = int(480 * kwargs['res_scale'])

        use_pretrained = True
        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]

        print('** Use norm {}, {} as the mean and std **'.format(norm_mean, norm_std))

        train_transform = [
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.1),
            transforms.Normalize(norm_mean, norm_std),
        ]

        val_transform = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]

        self.train_transform = transforms.Compose(train_transform)
        self.val_transform = transforms.Compose(val_transform)

        self.trainset = self.get_trainset(
            dataset,
            augment=kwargs["augment"],
            base_size=self.base_size,
            crop_size=self.crop_size
        )
        
        self.valset = self.get_valset(
            dataset,
            augment=False,
            base_size=self.base_size,
            crop_size=self.crop_size
        )

        use_batchnorm = (
            (not kwargs["no_batchnorm"]) if "no_batchnorm" in kwargs else True
        )
        # print(kwargs)

        labels = self.get_labels(dataset)

        self.net = LSegNet(
            labels=labels,
            backbone=kwargs["backbone"],
            features=kwargs["num_features"],
            crop_size=self.crop_size,
            arch_option=kwargs["arch_option"],
            block_depth=kwargs["block_depth"],
            activation=kwargs["activation"],
        )

        self.net.pretrained.model.patch_embed.img_size = (
            self.crop_size,
            self.crop_size,
        )

        self._up_kwargs = up_kwargs
        self.mean = norm_mean
        self.std = norm_std

        self.criterion = self.get_criterion(**kwargs)

    def get_labels(self, dataset):
        if 'scannet' in dataset.lower():
            return list(self.dataset.CLASS_LABELS)
        else:
            labels = []
            path = 'label_files/{}_objectInfo150.txt'.format(dataset)
            assert os.path.exists(path), '*** Error : {} not exist !!!'.format(path)
            f = open(path, 'r')
            lines = f.readlines()
            for line in lines:
                label = line.strip().split(',')[-1].split(';')[0]
                labels.append(label)
            f.close()
            if dataset in ['ade20k']:
                labels = labels[1:]
            return labels


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LSegmentationModule.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser])

        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone network",
        )

        parser.add_argument(
            "--num_features",
            type=int,
            default=256,
            help="number of featurs that go from encoder to decoder",
        )

        parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")

        parser.add_argument(
            "--finetune_weights", type=str, help="load weights to finetune from"
        )

        parser.add_argument(
            "--no-scaleinv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )

        parser.add_argument(
            "--no-batchnorm",
            default=False,
            action="store_true",
            help="turn off batchnorm",
        )

        parser.add_argument(
            "--widehead", default=False, action="store_true", help="wider output head"
        )

        parser.add_argument(
            "--no_wandb", default=False, action="store_true", help="If want to sync to wandb"
        )

        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )

        parser.add_argument(
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )

        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )

        parser.add_argument(
            "--activation",
            choices=['lrelu', 'tanh'],
            default="lrelu",
            help="use which activation to activate the block",
        )

        return parser
