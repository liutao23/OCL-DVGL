"""Offline timm DINOv3 ViT-B/16, matching the project's existing wrapper."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
from torch import nn
from common import guard


class DinoBackbone(nn.Module):
    def __init__(self, checkpoint=None):
        super().__init__()
        import timm
        from safetensors.torch import load_file
        from timm.data import resolve_model_data_config, create_transform
        self.model = timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=False, num_classes=0)
        if checkpoint is not None:
            state = load_file(str(guard(checkpoint)), device='cpu')
            if 'storage_tokens' in state:
                from timm.models.eva import checkpoint_filter_fn
                state = checkpoint_filter_fn(state, self.model)
            self.model.load_state_dict(state, strict=True)
        self.prefix = int(self.model.num_prefix_tokens)
        self.dim = int(self.model.num_features)
        self.patch_size = tuple(self.model.patch_embed.patch_size)
        if self.dim != 768 or self.patch_size != (16, 16):
            raise ValueError('Required architecture is DINOv3 ViT-B/16, dim=768')
        cfg = resolve_model_data_config(self.model)
        cfg.update(input_size=(3, 256, 256), crop_pct=1.0)
        self.preprocess_config = cfg
        self.transform = create_transform(**cfg, is_training=False)

    def forward(self, images):
        tokens = self.model.forward_features(images)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError('Expected timm forward_features [B,prefix+patches,D]')
        patches = tokens[:, self.prefix:]
        expected = (images.shape[-2]//self.patch_size[0])*(images.shape[-1]//self.patch_size[1])
        if patches.shape[1:] != (expected, 768):
            raise ValueError(f'Unexpected last-layer patch shape {patches.shape}; prefix={self.prefix}')
        return patches

    def global_descriptor(self, images):
        """Historical DINOv3LVD.extract_global_descriptors convention: CLS/prelogits.

        Matches 主实验/retrieval_core.py: forward_features -> forward_head(pre_logits=True).
        Uses this wrapper's unchanged 256px preprocessing; no learned OCL modules.
        """
        tokens = self.model.forward_features(images)
        features = self.model.forward_head(tokens, pre_logits=True)
        if not isinstance(features, torch.Tensor) or features.shape != (images.shape[0], 768):
            raise ValueError('Expected DINO global/prelogits [B,768]')
        return torch.nn.functional.normalize(features.float(), dim=-1)
