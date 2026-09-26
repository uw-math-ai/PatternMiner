"""Attention biases from exact comparisons of inverse-permutation positions."""
import math
import torch
from torch import nn
import torch.nn.functional as F
import campaign as c


class OrderSelector(c.Selector):
    def __init__(self, multiplier=1, dropout=0., full_ranks=False, extra_depth=0):
        if dropout:
            raise ValueError('This experiment uses dropout zero')
        super().__init__(multiplier, dropout, full_ranks)
        for _ in range(extra_depth):
            layer = nn.TransformerEncoderLayer(96*multiplier, 4*multiplier, 192*multiplier,
                        0., batch_first=True, activation='gelu', norm_first=True)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            self.encoder.layers.append(layer)
        self.order_bias = nn.Parameter(torch.zeros(2+extra_depth, 4*multiplier, 12))

    @staticmethod
    def comparisons(x):
        n = (1/x[:, 0, 18]).round().long()
        endpoint = (x[:, :, 3:9]*n[:, None, None]).round()
        supplied = (x[:, :, 9:15]*n[:, None, None]).round()
        candidate_order = (endpoint[:, :, None, :]-endpoint[:, None, :, :]).sign()
        supplied_order = (supplied[:, :, None, :]-endpoint[:, None, :, :]).sign()
        real = x[:, :, 18] != 0
        return torch.cat([candidate_order, supplied_order], dim=-1)*(real[:, :, None, None]&real[:, None, :, None])

    def forward(self, x):
        relations = self.comparisons(x)
        padding = x.abs().sum(-1) == 0
        h = self.input(self.feature_map(x))
        for number, layer in enumerate(self.encoder.layers):
            batch, tokens, dim = h.shape
            heads = layer.self_attn.num_heads
            dh = dim//heads
            attn_input = layer.norm1(h) if layer.norm_first else h
            qkv = F.linear(attn_input, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
            q, k, v = qkv.reshape(batch, tokens, 3, heads, dh).permute(2, 0, 3, 1, 4)
            bias = torch.einsum('bqkr,hr->bhqk', relations, self.order_bias[number])
            logits = (q @ k.transpose(-2, -1))/math.sqrt(dh)+bias
            pattern = logits.masked_fill(padding[:, None, None, :], -torch.inf).softmax(-1)
            transported = (pattern @ v).transpose(1, 2).reshape(batch, tokens, dim)
            h = h+layer.self_attn.out_proj(transported)
            if not layer.norm_first:
                h = layer.norm1(h)
            mlp_input = layer.norm2(h) if layer.norm_first else h
            hidden = F.gelu(layer.linear1(mlp_input), approximate='tanh')
            h = h+layer.linear2(hidden)
            if not layer.norm_first:
                h = layer.norm2(h)
        return self.head(h).squeeze(-1)
