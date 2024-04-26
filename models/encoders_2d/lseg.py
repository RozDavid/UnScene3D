import os
import argparse
import torch
from pytorch_lightning.core.lightning import LightningModule

from ext.LSeg_fork.modules.models.lseg_net import LSegNet as LSegBaseNet

from lib.utils.utils import load_state_with_same_shape

class LSegNet(LightningModule):

    def __init__(self, config, dataset, **kwargs):
        super().__init__()

        local_args = self.parse_configs()
        self.config = config

        local_args.backbone = config.image_data.image_backbone
        local_args.weights = os.path.join(config.data.scannet_path, config.image_data.model_checkpoint)

        assert os.path.isfile(local_args.weights), f'Cannot find LSeg weights at {local_args.weights}'
        self.crop_size = dataset.depth_shape[0]
        model = LSegBaseNet(
            labels=list(dataset.CLASS_LABELS),
            backbone=local_args.backbone,
            features=local_args.num_features,
            crop_size=self.crop_size,
            arch_option=local_args.arch_option,
            block_depth=local_args.block_depth,
            activation=local_args.activation)

        # Load dict
        state = torch.load(local_args.weights)
        matched_weights = load_state_with_same_shape(model, state['state_dict'], prefix='net.')
        model_dict = model.state_dict()
        model_dict.update(matched_weights)
        model.load_state_dict(model_dict)

        model.pretrained.model.patch_embed.img_size = (
            self.crop_size,
            self.crop_size,
        )
        self.model = model

        # Output dimension
        self.feature_dim = 512

    def forward(self, input_images):
        batch_num = input_images.shape[0]
        image_num = input_images.shape[1]
        c = input_images.shape[2]
        h = input_images.shape[3]
        w = input_images.shape[4]

        # flatten batches
        out, half_res, quart_res, eightht_res = self.model.forward_image(input_images.reshape(-1, *input_images.shape[2:]))

        # reshape to batch and chunk frames
        out = out.view(batch_num, image_num, h, w, -1).contiguous()
        half_res = half_res.view(batch_num, image_num, h // 2, w // 2, -1)
        quart_res = quart_res.view(batch_num, image_num, h // 4, w // 4, -1)
        eightht_res = eightht_res.view(batch_num, image_num, h // 8, w // 8, -1)

        return out, half_res, quart_res, eightht_res


    def parse_configs(self):

        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
        # model and dataset
        parser.add_argument(
            "--model", type=str, default="encnet", help="model name (default: encnet)"
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone name (default: resnet50)",
        )

        parser.add_argument(
            "--num_features",
            type=int,
            default=256,
            help="number of features that go from encoder to decoder",
        )

        parser.add_argument(
            "--dataset",
            type=str,
            default="ade20k",
            help="dataset name (default: pascal12)",
        )
        parser.add_argument(
            "--workers", type=int, default=16, metavar="N", help="dataloader threads"
        )
        parser.add_argument(
            "--base-size", type=int, default=640, help="base image size"
        )
        parser.add_argument(
            "--crop-size", type=int, default=480, help="crop image size"
        )
        parser.add_argument(
            "--train-split",
            type=str,
            default="train",
            help="dataset train split (default: train)",
        )
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
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
            "--batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                               training (default: auto)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                               testing (default: same as batch size)",
        )
        # cuda, seed and logging
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        # checking point
        parser.add_argument(
            "--weights", type=str, default='', help="checkpoint to test"
        )
        # evaluation option
        parser.add_argument(
            "--eval", action="store_true", default=False, help="evaluating mIoU"
        )
        parser.add_argument(
            "--export",
            type=str,
            default=None,
            help="put the path to resuming file if needed",
        )
        parser.add_argument(
            "--acc-bn",
            action="store_true",
            default=False,
            help="Re-accumulate BN statistics",
        )
        parser.add_argument(
            "--test-val",
            action="store_true",
            default=False,
            help="generate masks on val set",
        )
        parser.add_argument(
            "--no-val",
            action="store_true",
            default=False,
            help="skip validation during training",
        )

        parser.add_argument(
            "--module",
            default='lseg',
            help="select model definition",
        )

        # test option
        parser.add_argument(
            "--data-path", type=str, default='../datasets/', help="path to test image folder"
        )

        parser.add_argument(
            "--no-scaleinv",
            dest="scale_inv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )

        parser.add_argument(
            "--widehead", default=False, action="store_true", help="wider output head"
        )

        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )
        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )

        parser.add_argument(
            "--label_src",
            type=str,
            default="default",
            help="how to get the labels",
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

        args = parser.parse_args(args=[])
        args.scale_inv = False
        args.widehead = True

        return args

