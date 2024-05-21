import types
import time
import random
import clip
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import sys
import os
import logging
from argparse import ArgumentParser

import pytorch_lightning as pl
from torchmetrics import Precision  # Precision@1
from torchmetrics import JaccardIndex  # IoU
from torchmetrics import AveragePrecision  #mAP
from torchmetrics import Recall   # we call it class accuracy instead
from torchmetrics import Accuracy

from modules.utils.utils import MetricAverageMeter as AverageMeter, colorize_image
from modules.utils.utils import print_info, nanmean_t

from data import get_dataset, get_available_datasets

from encoding.models import get_segmentation_model
from encoding.nn import SegmentationLosses

from encoding.utils import batch_pix_accuracy, batch_intersection_union

# add mixed precision
import torch.cuda.amp as amp
import numpy as np

from encoding.utils import SegmentationMetric

class LSegmentationModule(pl.LightningModule):
    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, **kwargs):
        super().__init__()
        self.setup_logging()

        self.data_path = data_path
        self.batch_size = batch_size
        self.base_lr = base_lr / 16 * batch_size
        self.lr = self.base_lr

        self.epochs = max_epochs
        self.other_kwargs = kwargs
        self.other_kwargs['batch_size'] = batch_size
        self.other_kwargs['lr'] = self.lr
        self.ignore_index = self.other_kwargs['ignore_index']
        self.log_frequency = self.other_kwargs['log_frequency']
        self.enabled = False    # True mixed precision will make things complicated and leading to NAN error
        self.scaler = amp.GradScaler(enabled=self.enabled)

        self.save_hyperparameters(self.other_kwargs)

        # Initialize dataset for hyperparameter reads from object
        self.dataset = get_dataset(dataset, root=self.data_path)
        self.num_classes = self.dataset.num_class

        ### Initialize metrics ###
        self.losses = AverageMeter(ignore_index=self.ignore_index)
        self.scores = Precision(num_classes=self.num_classes, average='macro')
        self.accuracy = Recall(num_classes=self.num_classes, average='macro')  # this is the class accuracy
        self.pixel_accuracy = Accuracy()  # this is the global pixel accuracy
        self.iou_scores = JaccardIndex(num_classes=self.num_classes, reduction='none')

        # Other accumulators for logging statistics
        self.val_iteration = 0


    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=lambda x: random.seed(time.time() + x),
        )

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4)
        self.validation_max_iter = len(val_dataloader)
        return val_dataloader


    def get_trainset(self, dset, augment=False, **kwargs):
        print(kwargs)
        mode = "train"

        print(mode)
        dset = get_dataset(
            dset,
            root=self.data_path,
            split="train",
            mode=mode,
            transform=self.train_transform,
            **kwargs
        )

        return dset

    def get_valset(self, dset, augment=False, **kwargs):
        self.val_accuracy = Recall(num_classes=self.num_classes, average='macro')
        self.val_iou = SegmentationMetric(self.num_classes)

        if augment == True:
            mode = "val_x"
        else:
            mode = "val"

        print(mode)
        return get_dataset(
            dset,
            root=self.data_path,
            split="val",
            mode=mode,
            transform=self.val_transform,
            **kwargs
        )

    def forward(self, x):
        return self.net(x)

    def evaluate(self, x, target=None):
        pred = self.net.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)

        return correct, labeled, inter, union

    def evaluate_random(self, x, labelset, target=None):
        pred = self.net.forward(x, labelset)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)

        return correct, labeled, inter, union
    

    def training_step(self, batch, batch_nb):
        img, target = batch
        with amp.autocast(enabled=self.enabled):
            out = self(img)
            multi_loss = isinstance(out, tuple)
            if multi_loss:
                loss = self.criterion(*out, target)
            else:
                loss = self.criterion(out, target)
            loss = self.scaler.scale(loss)
        final_output = out[0] if multi_loss else out
        train_pred, train_gt = self._filter_invalid(final_output, target)

        # Update metrics
        if train_gt.nelement() != 0:
            self.accuracy(train_pred, train_gt)
            self.pixel_accuracy(train_pred, train_gt)
            self.iou_scores(train_pred, train_gt)
            self.scores(train_pred, train_gt)
            self.losses(loss.item())

        self.log("training/step_loss", loss.item())

        # Log after every N iteration
        if self.global_step % self.log_frequency == 0:
            self.loop_log('training_mid_epoch')

        return loss

    def on_validation_epoch_start(self):
        # We have to log the training scores here due to the order of hooks
        if self.global_step > 0:
            self.loop_log('training')

        # Reset accumulators
        self.reset_accumulators()


    def validation_step(self, batch, batch_nb):
        img, target = batch
        out = self(img) 
        multi_loss = isinstance(out, tuple)
        if multi_loss:
            val_loss = self.criterion(*out, target)
        else:
            val_loss = self.criterion(out, target)
        final_output = out[0] if multi_loss else out
        valid_pred, valid_gt = self._filter_invalid(final_output, target)

        # Update metrics
        if valid_gt.nelement() != 0:
            self.accuracy(valid_pred, valid_gt)
            self.pixel_accuracy(valid_pred, valid_gt)
            self.iou_scores(valid_pred, valid_gt)
            self.scores(valid_pred, valid_gt)
            self.losses(val_loss.item())

        self.log("validation/step_loss", val_loss.item())

        # Log after every N iteration
        if self.val_iteration % self.log_frequency == 0:

            self.loop_log('validation_mid_epoch')
            ious = self.iou_scores.compute()
            class_names = self.dataset.get_classnames() if hasattr(self.dataset, 'get_classnames') else None
            print_info(
                self.val_iteration,
                self.validation_max_iter,
                self.losses.compute(),
                self.scores.compute() * 100.,
                ious.cpu().numpy() * 100.,
                self.iou_scores.confmat.cpu().numpy(),
                class_names=class_names)

            if self.other_kwargs['visualize']:
                _, preds = torch.max(final_output, dim=1)
                pred_color = colorize_image(preds[0].cpu().numpy())
                target_color = colorize_image(target[0].cpu().numpy())
                color_color = ((img[0].permute(1,2,0) + 1.0) * 255. / 2.).cpu().numpy().astype(int)
                concat_img = np.concatenate((pred_color, target_color, color_color), axis=1).astype(np.uint8)

                concat_img = Image.fromarray(concat_img)
                concat_img.save(self.other_kwargs['visualize_path'] + f"{self.val_iteration}.jpg")

        self.val_iteration += 1


    def validation_epoch_end(self, outs):
        self.loop_log('validation')

        ious = self.iou_scores.compute()
        class_names = self.dataset.get_classnames()
        print_info(
            self.val_iteration,
            self.validation_max_iter,
            self.losses.compute(),
            self.scores.compute() * 100.,
            ious.cpu().numpy() * 100.,
            self.iou_scores.confmat.cpu().numpy(),
            class_names=class_names)

        self.reset_accumulators()

    def reset_accumulators(self):
        self.losses.reset()
        self.iou_scores.reset()
        self.scores.reset()
        self.accuracy.reset()
        self.pixel_accuracy.reset()
        self.val_iteration = 0

    def _filter_invalid(self, pred, target):
        valid = target != self.other_kwargs["ignore_index"]
        _, mx = torch.max(pred, dim=1)
        return mx[valid], target[valid]

    def configure_optimizers(self):
        params_list = [
            {"params": self.net.pretrained.parameters(), "lr": self.base_lr},
        ]
        if hasattr(self.net, "scratch"):
            print("Found output scratch")
            params_list.append(
                {"params": self.net.scratch.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.net, "auxlayer"):
            print("Found auxlayer")
            params_list.append(
                {"params": self.net.auxlayer.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.net, "scale_inv_conv"):
            print(self.net.scale_inv_conv)
            print("Found scaleinv layers")
            params_list.append(
                {
                    "params": self.net.scale_inv_conv.parameters(),
                    "lr": self.base_lr * 10,
                }
            )
            params_list.append(
                {"params": self.net.scale2_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.net.scale3_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.net.scale4_conv.parameters(), "lr": self.base_lr * 10}
            )

        if self.other_kwargs["midasproto"]:
            print("Using midas optimization protocol")
            
            opt = torch.optim.Adam(
                params_list,
                lr=self.base_lr,
                betas=(0.9, 0.999),
                weight_decay=self.other_kwargs["weight_decay"],
            )
            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
            )

        else:
            opt = torch.optim.SGD(
                params_list,
                lr=self.base_lr,
                momentum=0.9,
                weight_decay=self.other_kwargs["weight_decay"],
            )
            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
            )
        return [opt], [sch]


    def get_criterion(self, **kwargs):
        return SegmentationLosses(
            se_loss=kwargs["se_loss"], 
            aux=kwargs["aux"], 
            nclass=self.num_classes, 
            se_weight=kwargs["se_weight"], 
            aux_weight=kwargs["aux_weight"], 
            ignore_index=kwargs["ignore_index"], 
        )

    def loop_log(self, phase='training'):

        ious = self.iou_scores.compute() * 100.

        # Write to logger
        self.log(f'{phase}/loss', self.losses.compute())
        self.log(f'{phase}/precision', self.scores.compute() * 100.)
        self.log(f'{phase}/accuracy', self.accuracy.compute() * 100.)
        self.log(f'{phase}/mIoU', nanmean_t(ious))

        if self.pixel_accuracy.total > 0:  # this fails before first values
            self.log(f'{phase}/pixel_accuracy', self.pixel_accuracy.compute() * 100.)

        if 'train' in phase:
            self.log(phase + '/learning_rate', self.optimizers().param_groups[0]['lr'])

    def test_dataloader(self):
        return self.val_dataloader()

    def on_test_start(self):
        self.target_epoch_freqs = {}
        return self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, outputs):
        return self.validation_step_end(outputs)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data_path", type=str, help="path where dataset is stored"
        )
        parser.add_argument(
            "--dataset",
            choices=get_available_datasets(),
            default="ade20k",
            help="dataset to train on",
        )
        parser.add_argument(
            "--batch_size", type=int, default=16, help="size of the batches"
        )
        parser.add_argument(
            "--base_lr", type=float, default=0.004, help="learning rate"
        )
        parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
        parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="weight_decay"
        )
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--aux-weight",
            type=float,
            default=0.2,
            help="Auxilary loss weight (default: 0.2)",
        )
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument(
            "--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)"
        )

        parser.add_argument(
            "--midasproto", action="store_true", default=False, help="midasprotocol"
        )

        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )
        parser.add_argument(
            "--log_frequency",
            type=int,
            default=100,
            help="For intermediate logging before epoch ends",
        )
        parser.add_argument(
            "--augment",
            action="store_true",
            default=False,
            help="Use extended augmentations",
        )

        return parser

    def setup_logging(self):
        ch = logging.StreamHandler(sys.stdout)
        logging.getLogger().setLevel(logging.INFO)
        logging.basicConfig(
            format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
            datefmt='%m/%d %H:%M:%S',
            handlers=[ch])