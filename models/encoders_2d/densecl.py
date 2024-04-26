import torch
import torch.nn.functional as F
import os


from detectron2.config import get_cfg
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.utils.registry import Registry
from detectron2.checkpoint import DetectionCheckpointer

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

from ext.detectron2.detectron2.modeling.backbone import Backbone


from pytorch_lightning.core.lightning import LightningModule

def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)
    return backbone

class DenseCLNet(LightningModule):

    def __init__(self, config, dataset, **kwargs):
        super().__init__()

        local_path = os.path.dirname(__file__)
        config_path = os.path.join(local_path, 'config', 'densecl.yaml')

        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = os.path.join(config.data.scannet_path, config.image_data.model_checkpoint)

        self.model = build_backbone(cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)


        self.feature_dim = 256

    def forward(self, input_images):

        batch_num = input_images.shape[0]
        image_num = input_images.shape[1]
        c = input_images.shape[2]
        h = input_images.shape[3]
        w = input_images.shape[4]

        outputs = self.model(input_images.reshape(-1, *input_images.shape[2:]))

        # interpolate features to correct resolution
        out = F.interpolate(outputs['res2'], (h, w), mode='bilinear', align_corners=True).permute(0, 2, 3, 1).contiguous()
        half_res = F.interpolate(outputs['res3'], (h // 2, w // 2), mode='bilinear', align_corners=True).permute(0, 2, 3, 1).contiguous()
        quart_res = F.interpolate(outputs['res4'], (h // 4, w // 4), mode='bilinear', align_corners=True).permute(0, 2, 3, 1).contiguous()
        eightht_res = F.interpolate(outputs['res5'], (h // 8, w // 8), mode='bilinear', align_corners=True).permute(0, 2, 3, 1).contiguous()
        
        # reshape everything as batch/image
        out = out.view(batch_num, image_num, *out.shape[1:])
        half_res = half_res.view(batch_num, image_num, *half_res.shape[1:])
        quart_res = quart_res.view(batch_num, image_num, *quart_res.shape[1:])
        eightht_res = eightht_res.view(batch_num, image_num, *eightht_res.shape[1:])
        
        return out, half_res, quart_res, eightht_res


if __name__ == '__main__':

    model = DenseCLNet(None, None)


