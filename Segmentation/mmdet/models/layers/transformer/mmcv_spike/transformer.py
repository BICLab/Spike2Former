# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional
import torch
from torch import nn, Tensor
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.registry import MODELS
from mmengine.utils import deprecated_api_warning, to_2tuple
from .SNN_core import RepConv

from mmcv.cnn import (Linear, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmdet.models.utils.Qtrick import MultiSpike_norm4, MultiSpike_4
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from Qtrick_architecture.clock_driven.neuron import Q_IFNode
from Qtrick_architecture.clock_driven.surrogate import Quant,Quant4

# Avoid BC-breaking of importing MultiScaleDeformableAttention from this file
try:
    from mmcv.ops.multi_scale_deform_attn import \
        MultiScaleDeformableAttention  # noqa F401

    warnings.warn(
        ImportWarning(
            '``MultiScaleDeformableAttention`` has been moved to '
            '``mmcv.ops.multi_scale_deform_attn``, please change original path '  # noqa E501
            '``from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention`` '  # noqa E501
            'to ``from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention`` '  # noqa E501
        ))

except ImportError:
    warnings.warn('Fail to import ``MultiScaleDeformableAttention`` from '
                  '``mmcv.ops.multi_scale_deform_attn``, '
                  'You should install ``mmcv`` rather than ``mmcv-lite`` '
                  'if you need this module. ')


def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return MODELS.build(cfg, default_args=default_args)


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return MODELS.build(cfg, default_args=default_args)


def build_feedforward_network(cfg, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return MODELS.build(cfg, default_args=default_args)


def build_transformer_layer(cfg, default_args=None):
    """Builder for transformer layer."""
    return MODELS.build(cfg, default_args=default_args)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return MODELS.build(cfg, default_args=default_args)


class LocalRepresentation(nn.Module):
    """
    Local Representation module for generating feature vectors from input features.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors (default: 256).

    Attributes:
        to_query_3x3 (nn.Conv2d): 3x3 depth-wise convolutional layer for local feature extraction.
        bn (nn.BatchNorm2d): Batch normalization layer.
        out (nn.Linear): Linear transformation layer.
        d_model (int): The dimensionality of the input and output feature vectors.

    Methods:
        forward(self, x): Forward pass through the LocalRepresentation module.
    """

    def __init__(self, d_model=256):
        super().__init__()

        self.to_query_3x3 = nn.Conv2d(d_model, d_model, 3, groups=d_model, padding=1)
        self.bn = nn.SyncBatchNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

        self.d_model = d_model

    def forward(self, x):
        # Retrieve input tensor shape
        # import pdb; pdb.set_trace()
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        # Apply pre-normalisation followed by 3x3 local convolution to extract local features
        x = self.bn(x)
        x_3x3 = self.to_query_3x3(x)

        # Reshape the local features and permute dimensions for linear transformation
        return self.out(x_3x3.view(B, self.d_model, H * W).permute(0, 2, 1))


class PEM_CA(nn.Module):
    """
    Prototype-based Masked Cross-Attention module.

    This module implements a variant of the cross-attention mechanism for use in segmentation heads.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors (default: 256).
        nhead (int): The number of attention heads (default: 8).

    Attributes:
        to_query (LocalRepresentation): Module for converting input to query representations.
        to_key (nn.Sequential): Sequential module for transforming input to key representations.
        proj (nn.Linear): Linear transformation layer.
        final (nn.Linear): Final linear transformation layer.
        alpha (nn.Parameter): Parameter for scaling in the attention mechanism.
        num_heads (int): Number of attention heads.

    Methods:
        with_pos_embed(self, tensor, pos): Adds positional embeddings to the input tensor.
        most_similar_tokens(self, x, q, mask=None): Finds the most similar tokens based on content-based attention.
        forward(self, q, x, memory_mask, pos, query_pos): Forward pass through the PEM_CA module.
    """

    def __init__(self, d_model=256, nhead=8):
        super().__init__()

        self.feature_proj = LocalRepresentation(d_model)
        self.query_proj = nn.Sequential(nn.LayerNorm(d_model),
                                        nn.Linear(d_model, d_model))

        self.proj = nn.Linear(d_model, d_model)
        self.final = nn.Linear(d_model, d_model)

        self.alpha = nn.Parameter(torch.ones(1, 1, d_model))
        self.num_heads = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def most_similar_tokens(self, x, q, mask=None):
        # Retrieve input tensors shapes
        B, N, C = x.shape
        q = q.permute(1, 0, 2)
        Q, D = q.shape[1], C // self.num_heads

        # Reshape tensors in multi-head fashion
        x = x.view(B, N, self.num_heads, D).permute(0, 2, 1, 3)
        q = q.view(B, Q, self.num_heads, D).permute(0, 2, 1, 3)

        # Compute similarity scores between features and queries
        sim = torch.einsum('bhnc, bhqc -> bhnq', x, q)

        # Apply mask to similarity scores if provided
        if mask is not None:
            mask = (mask.flatten(2).permute(0, 2, 1).detach() < 0.0).bool()
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            mask[torch.all(mask.sum(2) == mask.shape[2], dim=2)] = False
            sim.masked_fill_(mask, float('-inf'))

        # Find indices of most similar tokens
        most_similar_indices = torch.argmax(sim, dim=2)

        # Gather most similar tokens
        return torch.gather(x, 2, most_similar_indices.unsqueeze(-1).expand(-1, -1, -1, D)).permute(0, 2, 1, 3).reshape(
            B, Q, C)

    def forward(self, tgt, memory, memory_mask, pos, query_pos):
        res = tgt

        # Add positional embeddings to input tensors
        memory, tgt = self.with_pos_embed(memory, pos), self.with_pos_embed(tgt, query_pos)

        # Project input tensors
        memory = self.feature_proj(memory)  # BxDxHxW
        tgt = self.query_proj(tgt.permute(1, 0, 2))  # BxQxD

        # Normalize input tensors
        memory = torch.nn.functional.normalize(memory, dim=-1)
        tgt = torch.nn.functional.normalize(tgt, dim=-1)

        # Find the most similar feature token to each query
        memory = self.most_similar_tokens(memory, tgt, memory_mask).permute(1, 0, 2)  # BxQxD
        # import pdb; pdb.set_trace()
        # Perform attention mechanism with projection and scaling
        out = nn.functional.normalize(self.proj(memory * tgt), dim=1) * self.alpha + memory  # BxQxD

        # Final linear transformation
        out = self.final(out)  # BxQxD

        return out.permute(1, 0, 2)


# NOTE: 改这里
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 attn_drop=0.0,
                 dropout=0.0,
                 proj_drop=0.0,
                 batch_first=True,
                 dropout_layer=None,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dims
        self.scale = (embed_dims // num_heads) ** -0.5

        self.q_conv_spike = Q_IFNode(surrogate_function=Quant())
        self.q_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims),
                                    )

        self.k_conv_spike = Q_IFNode(surrogate_function=Quant())
        self.k_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims),
                                    )

        self.v_conv_spike = Q_IFNode(surrogate_function=Quant())
        self.v_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims),
                                    )

        self.q_spike = Q_IFNode(surrogate_function=Quant())
        self.k_spike = Q_IFNode(surrogate_function=Quant())
        self.v_spike = Q_IFNode(surrogate_function=Quant())
        # self.q_spike = Q_IFNode(surrogate_function=Quant())
        # self.k_spike = Q_IFNode(surrogate_function=Quant())
        # self.v_spike = Q_IFNode(surrogate_function=Quant())

        self.attn_spike = Q_IFNode(surrogate_function=Quant())
        self.out_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                      nn.BatchNorm1d(embed_dims))

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # Check input dim
        # query: [bs, nq, dim]  key/value: [bs, nq', dim]
        t, b, nq, dim = query.shape
        t, b, nk, dim = key.shape
        query = self.q_conv_spike(query).permute(0, 1, 3, 2)  # t,b,nq,dim -> t,b,dim,nq
        query = self.q_conv(query.flatten(0, 1))  # [bs, NQ, embed_dim]
        query = self.q_spike(query.permute(0, 2, 1).view(t, b, nq, dim).contiguous())

        key = self.k_conv_spike(key).permute(0, 1, 3, 2)
        key = self.k_conv(key.flatten(0, 1))       # [bs, NK, embed_dim]
        key = self.k_spike(key.permute(0, 2, 1).view(t, b, nk, dim).contiguous())

        value = self.v_conv_spike(value).permute(0, 1, 3, 2)
        value = self.v_conv(value.flatten(0, 1))
        value = self.v_spike(value.permute(0, 2, 1).view(t, b, nk, dim).contiguous())

        split_size = self.embed_dim // self.num_heads  # embed_dim // num_heads = 32
        querys = torch.stack(torch.split(query, split_size, dim=3), dim=3) \
            .permute(0, 1, 3, 2, 4).contiguous()  # [bs, num_heads, NQ, embed_dim//num_heads]
        keys = torch.stack(torch.split(key, split_size, dim=3), dim=3) \
            .permute(0, 1, 3, 2, 4).contiguous()  # [bs, num_heads, NK,  embed_dim//num_heads]
        values = torch.stack(torch.split(value, split_size, dim=3), dim=3) \
            .permute(0, 1, 3, 2, 4).contiguous()  # [bs, num_heads, NK,  embed_dim//num_heads]

        scores = torch.matmul(querys, keys.transpose(3, 4))  # [h, N, T_q, T_k]
        scores = scores / (self.embed_dim ** 0.5) # For D>=2

        ## mask
        if attn_mask is not None:
            ## mask:  [bs*heads, NQ, NK] --> [bs, heads, NQ, NK]
            mask = attn_mask.reshape(querys.shape[0], self.num_heads, querys.shape[2], keys.shape[2]).contiguous()
            scores = scores.masked_fill(mask, 0)

        out = torch.matmul(scores, values)  # [T, bs, num_heads, NQ, embed_dim//num_heads]
        out = torch.cat(torch.split(out, 1, dim=2), dim=4).squeeze(2)  # [bs, NQ, embed_dims]

        out = self.attn_spike(out).permute(0, 1, 3, 2).contiguous()

        out = self.out_conv(out.flatten(0, 1)).permute(0, 2, 1).view(t, b, nq, dim).contiguous()

        return out, scores

class CrossMultiHeadAttentionBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 attn_drop=0.0,
                 dropout=0.0,
                 proj_drop=0.0,
                 batch_first=True,
                 dropout_layer=None,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dims
        self.scale = (embed_dims // num_heads) ** -0.5

        self.q_conv_spike = Q_IFNode(surrogate_function=Quant())
        self.q_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims),
                                    )

        self.k_conv_spike = Q_IFNode(surrogate_function=Quant())
        self.k_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims),
                                    )

        self.v_conv_spike = Q_IFNode(surrogate_function=Quant())
        self.v_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims),
                                    )

        self.q_spike = Q_IFNode(surrogate_function=Quant())
        self.k_spike = Q_IFNode(surrogate_function=Quant())
        self.v_spike = Q_IFNode(surrogate_function=Quant())

        self.attn_spike = Q_IFNode(surrogate_function=Quant())
        self.out_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                      nn.BatchNorm1d(embed_dims))

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # Check input dim
        # query: [bs, nq, dim]  key/value: [bs, nq', dim]
        t, b, nq, dim = query.shape
        t, b, nk, dim = key.shape
        # import pdb; pdb.set_trace()
        query = self.q_conv_spike(query).permute(0, 1, 3, 2)  # t,b,nq,dim -> t,b,dim,nq
        query = self.q_conv(query.flatten(0, 1))  # [bs, NQ, embed_dim]
        query = self.q_spike(query.permute(0, 2, 1).view(t, b, nq, dim).contiguous())

        key = self.k_conv_spike(key).permute(0, 1, 3, 2)
        key = self.k_conv(key.flatten(0, 1))       # [bs, NK, embed_dim]
        key = self.k_spike(key.permute(0, 2, 1).view(t, b, nk, dim).contiguous())

        value = self.v_conv_spike(value).permute(0, 1, 3, 2)
        value = self.v_conv(value.flatten(0, 1))
        value = self.v_spike(value.permute(0, 2, 1).view(t, b, nk, dim).contiguous())

        split_size = self.embed_dim // self.num_heads  # embed_dim // num_heads = 32
        querys = torch.stack(torch.split(query, split_size, dim=3), dim=3) \
            .permute(0, 1, 3, 2, 4).contiguous()  # [bs, num_heads, NQ, embed_dim//num_heads]
        keys = torch.stack(torch.split(key, split_size, dim=3), dim=3) \
            .permute(0, 1, 3, 2, 4).contiguous()  # [bs, num_heads, NK,  embed_dim//num_heads]
        values = torch.stack(torch.split(value, split_size, dim=3), dim=3) \
            .permute(0, 1, 3, 2, 4).contiguous()  # [bs, num_heads, NK,  embed_dim//num_heads]

        # import pdb; pdb.set_trace()
        scores = torch.matmul(querys, keys.transpose(3, 4))  # [h, N, T_q, T_k]
        scores = scores / (self.embed_dim ** 0.5)
        # Add Identity Here
        ## mask
        if attn_mask is not None:
            ## mask:  [bs*heads, NQ, NK] --> [bs, heads, NQ, NK]
            mask = attn_mask.reshape(querys.shape[0], self.num_heads, querys.shape[2], keys.shape[2]).contiguous()
            scores = scores.masked_fill(mask, 0)
        # import pdb; pdb.set_trace()
        out = torch.matmul(scores, values)  # [bs, num_heads, NQ, embed_dim//num_heads]
        out = torch.cat(torch.split(out, 1, dim=2), dim=4).squeeze(2)  # [bs, NQ, embed_dims]

        out = self.attn_spike(out).permute(0, 1, 3, 2).contiguous()

        out = self.out_conv(out.flatten(0, 1)).permute(0, 2, 1).view(t, b, nq, dim).contiguous()

        return out, scores


class MultiHeadCrossAttentionBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 attn_drop=0.0,
                 dropout=0.0,
                 proj_drop=0.0,
                 batch_first=True,
                 dropout_layer=None,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dims
        self.scale = (embed_dims // num_heads) ** -0.5

        self.q_conv_spike = Q_IFNode(surrogate_function=Quant())
        self.q_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims))

        self.k_conv_spike = Q_IFNode(surrogate_function=Quant())
        self.k_conv = RepConv(in_channel=embed_dims, out_channel=embed_dims, bias=False)

        self.v_conv_spike = Q_IFNode(surrogate_function=Quant())
        self.v_conv = RepConv(in_channel=embed_dims, out_channel=embed_dims, bias=False)

        self.q_spike = Q_IFNode(surrogate_function=Quant())
        self.k_spike = Q_IFNode(surrogate_function=Quant())
        self.v_spike = Q_IFNode(surrogate_function=Quant())

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.channel_conv = nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1, bias=False)
        self.alpha = nn.Parameter(torch.ones(1, 1, embed_dims))

        self.attn_spike = Q_IFNode(surrogate_function=Quant())
        self.out_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                      nn.BatchNorm1d(embed_dims))

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # import pdb; pdb.set_trace()
        _, H, W, _ = key.shape
        # query: [bs, nq, dim]  key/value: [bs, nq', dim]
        query = self.q_conv_spike(query).permute(0, 2, 1)
        tgt = query
        query = self.q_conv(query)  # [bs, NQ, embed_dim]
        query = self.q_spike(query).permute(0, 2, 1)

        key = self.k_conv_spike(key).permute(0, 3, 1, 2)  # [bs, h, w, c] -> [bs, C, h, w]
        key = self.k_conv(key)  # [bs, NQ, embed_dim]
        key = self.k_spike(key).flatten(2).permute(0, 2, 1)

        value = self.v_conv_spike(value).permute(0, 3, 1, 2)
        value = self.v_conv(value)
        value = self.v_spike(value).flatten(2).permute(0, 2, 1)

        split_size = self.embed_dim // self.num_heads  # embed_dim // num_heads = 32
        querys = torch.stack(torch.split(query, split_size, dim=2), dim=2) \
            .permute(0, 2, 1, 3).contiguous()  # [bs, num_heads, NQ, embed_dim//num_heads]
        keys = torch.stack(torch.split(key, split_size, dim=2), dim=2) \
            .permute(0, 2, 1, 3).contiguous()  # [bs, num_heads, NK,  embed_dim//num_heads]
        values = torch.stack(torch.split(value, split_size, dim=2), dim=2) \
            .permute(0, 2, 1, 3).contiguous()  # [bs, num_heads, NK,  embed_dim//num_heads]

        scores = torch.matmul(keys.transpose(2, 3), values)
        scores = scores / (self.embed_dim ** 0.5)
        # out = torch.matmul(scores, values)  # [bs, num_heads, NQ, embed_dim//num_heads]
        out = torch.matmul(querys, scores)
        out = torch.cat(torch.split(out, 1, dim=1), dim=3).squeeze(1)  # [bs, NQ, embed_dims]

        # Build Channel attention
        tgt = self.pool(self.channel_conv(tgt)).permute(0, 2, 1)
        out = out * self.alpha + tgt

        out = self.attn_spike(out).permute(0, 2, 1).contiguous()

        return self.out_conv(out).permute(0, 2, 1).contiguous(), scores


# NOTE: 改这里
class MSMultiHeadAttentionBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 operation='SA',
                 batch_first=True,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dims
        self.scale = (embed_dims // num_heads) ** -0.5

        self.q_conv_spike = Q_IFNode(surrogate_function=Quant())
        self.q_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims))

        self.k_conv_spike = Q_IFNode(surrogate_function=Quant())
        self.k_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims))

        self.v_conv_spike = Q_IFNode(surrogate_function=Quant())
        self.v_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims))

        self.q_spike = Q_IFNode(surrogate_function=Quant())
        self.k_spike = Q_IFNode(surrogate_function=Quant())
        self.v_spike = Q_IFNode(surrogate_function=Quant())

        self.attn_spike = Q_IFNode(surrogate_function=Quant())
        self.out_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                      nn.BatchNorm1d(embed_dims))

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # query: [bs, nq, dim]  key/value: [bs, nq', dim]
        query = self.q_conv_spike(query).permute(0, 2, 1)
        query = self.q_conv(query)  # [bs, NQ, embed_dim]
        query = self.q_spike(query).permute(0, 2, 1)

        key = self.k_conv_spike(key).permute(0, 2, 1)
        key = self.k_conv(key)  # [bs, NQ, embed_dim]
        key = self.k_spike(key).permute(0, 2, 1)

        value = self.v_conv_spike(value).permute(0, 2, 1)
        value = self.v_conv(value)
        value = self.v_spike(value).permute(0, 2, 1)

        split_size = self.embed_dim // self.num_heads  # embed_dim // num_heads = 32
        querys = torch.stack(torch.split(query, split_size, dim=2), dim=2) \
            .permute(0, 2, 1, 3).contiguous()  # [bs, num_heads, NQ, embed_dim//num_heads]
        keys = torch.stack(torch.split(key, split_size, dim=2), dim=2) \
            .permute(0, 2, 1, 3).contiguous()  # [bs, num_heads, NK,  embed_dim//num_heads]
        values = torch.stack(torch.split(value, split_size, dim=2), dim=2) \
            .permute(0, 2, 1, 3).contiguous()  # [bs, num_heads, NK,  embed_dim//num_heads]

        scores = torch.matmul(keys.transpose(-2, -1), values)
        out = torch.matmul(querys, scores) * self.scale

        out = torch.cat(torch.split(out, 1, dim=1), dim=3).squeeze(1)  # [bs, NQ, embed_dims]
        out = self.attn_spike(out).permute(0, 2, 1).contiguous()

        return self.out_conv(out).permute(0, 2, 1).contiguous(), scores


class MultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_type='SA',
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        if attn_type == 'SA':
            self.attn = MultiHeadAttentionBlock(embed_dims, num_heads, attn_drop,
                                                **kwargs)
        elif attn_type == 'CA':
            self.attn = CrossMultiHeadAttentionBlock(embed_dims, num_heads, attn_drop,
                                                **kwargs)
        elif attn_type == 'LinearCA':
            self.attn = MultiHeadCrossAttentionBlock(embed_dims, num_heads, attn_drop,
                                                     **kwargs)
        elif attn_type == 'LinearSA':
            self.attn = MSMultiHeadAttentionBlock(embed_dims, num_heads, attn_drop,
                                                  **kwargs)
        else:
            self.attn = MultiHeadAttentionBlock(embed_dims, num_heads, attn_drop,
                                                **kwargs)

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        # import pdb; pdb.set_trace()
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        # import pdb; pdb.set_trace()
        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        return out


# @MODELS.register_module()
class FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        layer_scale_init_value (float): Initial value of scale factor in
            LayerScale. Default: 1.0
    """

    @deprecated_api_warning(
        {
            'dropout': 'ffn_drop',
            'add_residual': 'add_identity'
        },
        cls_name='FFN')
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=2048,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 layer_scale_init_value=0.):
        super().__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
                             f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        in_channels = embed_dims
        self.fc1 = Linear(in_channels, feedforward_channels, bias=True)
        self.act_1 = build_activation_layer(act_cfg)
        self.dp_1 = nn.Dropout(ffn_drop)
        self.fc2 = Linear(feedforward_channels, embed_dims)
        self.dp_2 = nn.Dropout(ffn_drop)

    @deprecated_api_warning({'residual': 'identity'}, cls_name='FFN')
    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        # import pdb; pdb.set_trace()
        out1 = self.dp_1(self.act_1(self.fc1(x)))
        out = self.dp_2(self.fc2(out1))
        if identity is None:
            identity = x
        return identity + out


class MSDA_FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        layer_scale_init_value (float): Initial value of scale factor in
            LayerScale. Default: 1.0
    """

    @deprecated_api_warning(
        {
            'dropout': 'ffn_drop',
            'add_residual': 'add_identity'
        },
        cls_name='FFN')
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=2048,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 T=4,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 layer_scale_init_value=0.):
        super().__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
                             f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.T = T

        in_channels = embed_dims
        self.fc1_spike = Q_IFNode(surrogate_function=Quant())
        self.fc1 = nn.Conv1d(in_channels, feedforward_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(feedforward_channels)
        self.fc2_spike = Q_IFNode(surrogate_function=Quant())
        self.fc2 = nn.Conv1d(feedforward_channels, embed_dims, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(embed_dims)

    @deprecated_api_warning({'residual': 'identity'}, cls_name='FFN')
    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        # NOTE: Lif -> Conv -> Bn -> Sigmoid -> Lif -> Conv -> Bn
        # import pdb;
        # pdb.set_trace()
        t, bs, N, C = x.shape
        out1 = self.fc1_spike(x).reshape(t, bs, C, N)
        out1 = self.bn1(self.fc1(out1.flatten(0, 1))).reshape(t, bs, self.feedforward_channels, N)

        out = self.fc2_spike(out1)
        out = self.bn2(self.fc2(out.flatten(0, 1))).reshape(t, bs, N, C)
        if identity is None:
            identity = x
        return out


class MS_MLP(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            feedforward_channels=2048,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
            ffn_drop=0.,
            T=4,
            dropout_layer=None,
            add_identity=True,
            init_cfg=None,
            layer_scale_init_value=0.

    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.T = T

        in_channels = embed_dims
        self.fc1_spike = Q_IFNode(surrogate_function=Quant())
        self.fc1_conv = nn.Conv1d(in_channels, feedforward_channels, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(feedforward_channels)
        self.fc2_spike = Q_IFNode(surrogate_function=Quant())
        self.fc2_conv = nn.Conv1d(feedforward_channels, embed_dims, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(embed_dims)

    @deprecated_api_warning({'residual': 'identity'}, cls_name='FFN')
    def forward(self, x):

        T, B, H, W, C = x.shape
        N = H * W
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.flatten(3)
        x = self.fc1_spike(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.feedforward_channels, N)

        x = self.fc2_spike(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, H, W, C)

        return x
