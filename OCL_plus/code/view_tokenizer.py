"""Four learned queries cross-attend to unpooled final-layer DINO patches."""
import torch
from torch import nn


class CrossBlock(nn.Module):
    def __init__(self, dim=768, heads=8, mlp_ratio=4):
        super().__init__()
        self.qnorm, self.knorm, self.fnorm = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Linear(dim*mlp_ratio, dim))

    def forward(self, q, x, padding=None):
        k = self.knorm(x)
        q = q + self.attn(self.qnorm(q), k, k, key_padding_mask=padding, need_weights=False)[0]
        return q+self.ffn(self.fnorm(q))


class ViewTokenizer(nn.Module):
    def __init__(self, dim=768, num_queries=4, depth=2, heads=8, mlp_ratio=4):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_queries, dim))
        nn.init.trunc_normal_(self.queries, std=.02)
        self.layers = nn.ModuleList([CrossBlock(dim, heads, mlp_ratio) for _ in range(depth)])

    def forward(self, patches):
        q = self.queries.expand(patches.shape[0], -1, -1)
        for layer in self.layers:
            q = layer(q, patches)
        return q
