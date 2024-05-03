# This code is based on https://github.com/microsoft/Cream/tree/main/AutoFormer.

import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_module.Linear_super import LinearSuper
from attention_module.layernorm_super import LayerNormSuper
from attention_module.multihead_super import AttentionSuper
from attention_module.utils import DropPath


class TransformerEncoderLayer(nn.Module):

    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, dropout=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, pre_norm=True, scale=False, relative_position=True, change_qkv=True, max_relative_position=14):
        super().__init__()

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm
        self.super_dropout = attn_drop
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale = scale
        self.relative_position = relative_position

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        # self.sample_dropout = None
        # self.sample_attn_dropout = None

        self.attn = AttentionSuper(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=dropout, scale=self.scale, relative_position=self.relative_position, change_qkv=change_qkv,
            max_relative_position=max_relative_position
        )

        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.activation_fn = gelu

        self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer)
        self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim)

    def set_sample_config(self):

        sample_mlp_ratio = int(random.choice([0.7, 1.0, 1.3]) * self.super_mlp_ratio)
        sample_num_heads = int(random.choice([0.7, 1.0, 1.3]) * self.super_num_heads)
        # sample_dropout =
        # sample_attn_dropout =

        self.sample_embed_dim = self.super_embed_dim
        self.sample_out_dim = self.super_embed_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(self.sample_embed_dim*self.sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads

        # self.sample_dropout = sample_dropout
        # self.sample_attn_dropout = sample_attn_dropout
        self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        self.attn.set_sample_config(sample_q_embed_dim=64, sample_num_heads=self.sample_num_heads_this_layer, sample_in_embed_dim=self.sample_embed_dim)

        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_out_dim)

        self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        self.set_sample_config()

        residual = x
        x = self._maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        # x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self._maybe_layer_norm(self.attn_layer_norm, x, after=True)

        residual = x
        x = self._maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        # x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = self.fc2(x)
        # x = F.dropout(x, p=self.sample_dropout, training=self.training)
        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + x
        x = self._maybe_layer_norm(self.ffn_layer_norm, x, after=True)

        return x

    def _maybe_layer_norm(self, layer_norm, x, before=False, after=False):

        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Transform(nn.Module):

    def __init__(self, to_attn=True):
        super().__init__()

        if to_attn:
            self.transform = self.cnn_to_attn
        else:
            self.transform = self.attn_to_cnn

    def cnn_to_attn(self, x):
        "B, C, H, W ---> B, S, C"

        B, C, H, W = x.shape
        return x.reshape(B, C, H*W).permute(0, 2, 1)

    def attn_to_cnn(self, x):
        "B, S, C ---> B, C, H, W"

        B, S, C = x.shape
        H = int(math.sqrt(S))
        assert H == math.sqrt(S), "The sequence dimension of tensor x cannot be transformed to image."

        return x.permute(0, 2, 1).reshape(B, C, H, H)

    def forward(self, x):
        return self.transform(x)
