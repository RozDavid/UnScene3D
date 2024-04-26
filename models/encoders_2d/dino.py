import torch
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule

from ext.dino_vit.extractor import ViTExtractor

class DinoNet(LightningModule):

    def __init__(self, config, dataset, **kwargs):
        super().__init__()

        self.config = config
        self.dataset = dataset
        self.backbone = config.image_data.image_backbone
        self.image_shape = dataset.depth_shape

        # VITExtractor params
        stride = config.image_data.dino_vit_stride
        self.vit_feature = config.image_data.dino_vit_feature
        self.layer = config.image_data.dino_vit_layer
        self.facet = 'key'
        self.bin = False
        self.vit_extractor = ViTExtractor(self.backbone, stride)


        # Output dimension fixed for DINO VIT smalls
        self.feature_dim = 384

    def forward_descriptor(self, input_images):

        if input_images.device != self.vit_extractor.device:
            self.vit_extractor.model.to(input_images.device)

        encoded_dino_feats = []
        with torch.no_grad():
            for img in input_images.view(-1, *input_images.shape[2:]):
                # Extract descriptors
                descs = self.vit_extractor.extract_descriptors(img.unsqueeze(0), self.layer, self.facet, self.bin, include_cls=False)

                # reshape image
                curr_num_patches, curr_load_size = self.vit_extractor.num_patches, self.vit_extractor.load_size
                descs = descs.view(*curr_num_patches, -1)

                # Add to batch
                encoded_dino_feats += [descs]

        # Stack to make a tensor
        encoded_dino_feats = torch.stack(encoded_dino_feats)

        # Rearrange to grid
        encoded_dino_feats = encoded_dino_feats.view(-1, *encoded_dino_feats.shape[1:])

        # Permute for interpolation
        encoded_dino_feats = encoded_dino_feats.permute(0, 3, 1, 2).contiguous()

        # Interpolate to orig resolution
        encoded_dino_feats = F.interpolate(encoded_dino_feats, size=input_images.shape[3:], mode='bilinear')

        # Permute to channel last
        encoded_dino_feats = encoded_dino_feats.permute(0, 2, 3, 1).contiguous()

        # Reshape to batches
        encoded_dino_feats = encoded_dino_feats.view(*input_images.shape[:2], *input_images.shape[3:], -1)

        return encoded_dino_feats, None

    def forward_attention(self, input_images):
        '''
            Extracts both key and query features from the last block, scale them to output resolution and return them
        '''

        # Push model to correct device
        if input_images.device != self.vit_extractor.device:
            self.vit_extractor.model.to(input_images.device)

        # Register hooks and shape input
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        self.vit_extractor.model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

        encoded_dino_keys, encoded_dino_queries = [], []
        with torch.no_grad():
            for img in input_images.view(-1, *input_images.shape[2:]):
                descs = self.vit_extractor.extract_descriptors(img.unsqueeze(0), self.layer, self.facet, self.bin, include_cls=True)
                curr_num_patches, curr_load_size = self.vit_extractor.num_patches, self.vit_extractor.load_size
                bs, nb_head, nb_token = descs.shape[0], self.vit_extractor._feats[0].shape[1], curr_num_patches[0] * curr_num_patches[1] + 1
                qkv = (
                    feat_out["qkv"]
                    .reshape(bs, nb_token, 3, nb_head, -1)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]

                k = k.transpose(1, 2).reshape(bs, nb_token, -1)
                q = q.transpose(1, 2).reshape(bs, nb_token, -1)
                v = v.transpose(1, 2).reshape(bs, nb_token, -1)  # don't need this haha

                # Modality selection
                feats_k = k[:, 1:].reshape(*curr_num_patches, self.feature_dim)
                feats_q = q[:, 1:].reshape(*curr_num_patches, self.feature_dim)

                encoded_dino_keys += [feats_k]
                encoded_dino_queries += [feats_q]

        # Stack to make a tensor
        encoded_dino_keys = torch.stack(encoded_dino_keys)
        encoded_dino_queries = torch.stack(encoded_dino_queries)

        # Permute for interpolation
        encoded_dino_keys = encoded_dino_keys.permute(0, 3, 1, 2).contiguous()
        encoded_dino_queries = encoded_dino_queries.permute(0, 3, 1, 2).contiguous()

        # Interpolate to orig resolution
        encoded_dino_keys = F.interpolate(encoded_dino_keys, size=input_images.shape[3:], mode='bilinear')
        encoded_dino_queries = F.interpolate(encoded_dino_queries, size=input_images.shape[3:], mode='bilinear')

        # Permute to channel last
        encoded_dino_keys = encoded_dino_keys.permute(0, 2, 3, 1).contiguous()
        encoded_dino_queries = encoded_dino_queries.permute(0, 2, 3, 1).contiguous()

        # Reshape to batches
        encoded_dino_keys = encoded_dino_keys.view(*input_images.shape[:2], *input_images.shape[3:], -1)
        encoded_dino_queries = encoded_dino_queries.view(*input_images.shape[:2], *input_images.shape[3:], -1)

        return encoded_dino_keys, encoded_dino_queries

    def forward(self, input_images):
        if self.vit_feature == "attention":
            return self.forward_attention(input_images)
        else:
            return self.forward_descriptor(input_images)
