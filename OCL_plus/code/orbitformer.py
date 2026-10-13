"""Permutation-invariant latent aggregation; identical path for m=1..70."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from view_tokenizer import ViewTokenizer


class LatentBlock(nn.Module):
    def __init__(self, dim, heads, ratio):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*ratio), nn.GELU(), nn.Linear(dim*ratio, dim))

    def forward(self, x):
        n = self.norm1(x)
        x = x+self.attn(n, n, n, need_weights=False)[0]
        return x+self.ffn(self.norm2(x))


class OrbitFormer(nn.Module):
    def __init__(self, dim=768, num_orbit_latents=8, depth=3, heads=8, mlp_ratio=4):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, 1+num_orbit_latents, dim))
        nn.init.trunc_normal_(self.queries, std=.02)
        self.qnorm, self.vnorm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.layers = nn.ModuleList([LatentBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, views, valid=None, return_norm=False):
        # views [B,m,4,768]; True means a real observation, False means padding.
        if views.ndim != 4 or views.shape[2:] != (4, 768):
            raise ValueError('Expected [B,m,4,768] View Tokens')
        if valid is None:
            valid = torch.ones(views.shape[:2], dtype=torch.bool, device=views.device)
        if valid.shape != views.shape[:2] or valid.dtype != torch.bool or not valid.any(1).all():
            raise ValueError('Each place requires at least one valid observation')
        padding = ~valid.repeat_interleave(4, dim=1)
        # Zero before LayerNorm as well: padding cannot propagate arbitrary NaNs.
        tokens = views.masked_fill(~valid[:, :, None, None], 0).flatten(1, 2)
        q = self.queries.expand(views.shape[0], -1, -1)
        values = self.vnorm(tokens)
        q = q+self.cross(self.qnorm(q), values, values, key_padding_mask=padding, need_weights=False)[0]
        for block in self.layers:
            q = block(q)
        place = self.norm(q[:, 0]).float()
        descriptor = F.normalize(place, dim=-1)
        return (descriptor, place.norm(dim=-1)) if return_norm else descriptor


class OCLModel(nn.Module):
    def __init__(self, backbone, freeze=False):
        super().__init__()
        self.backbone, self.tokenizer, self.orbit = backbone, ViewTokenizer(), OrbitFormer()
        self.freeze_backbone = freeze
        self.gradient_checkpointing = True
        self.backbone.requires_grad_(not freeze)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def image_tokens(self, images):
        if self.freeze_backbone:
            with torch.no_grad():
                patches = self.backbone(images)
        else:
            patches = self.backbone(images)
        return self.tokenizer(patches)

    def encode(self, data, selections, device, chunk_size, return_norm=False):
        # Flatten images across places, then reconstruct exactly the recorded boundaries.
        groups = [data.images(sid, indices) for sid, indices in selections]
        return self.encode_preloaded(groups,device,chunk_size,return_norm)

    def encode_preloaded(self, groups, device, chunk_size, return_norm=False):
        """Encode already transformed per-place image tensors without re-reading files."""
        images = torch.cat(groups)
        chunks = []
        for start in range(0, len(images), chunk_size):
            x = images[start:start+chunk_size].to(device,non_blocking=True)
            if self.training and torch.is_grad_enabled() and self.gradient_checkpointing:
                # Checkpoint the entire image encoder: chunk size also bounds saved activations.
                v = checkpoint(self.image_tokens, x, use_reentrant=False)
            else:
                v = self.image_tokens(x)
            chunks.append(v)
        tokens = torch.cat(chunks)
        lengths = [len(group) for group in groups]
        token_groups = tokens.split(lengths)
        max_m = max(lengths)
        padded = torch.stack([F.pad(group, (0, 0, 0, 0, 0, max_m-len(group))) for group in token_groups])
        valid = torch.arange(max_m, device=device)[None] < torch.tensor(lengths, device=device)[:, None]
        return self.orbit(padded, valid, return_norm=return_norm)
