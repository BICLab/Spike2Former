# from visualizer import get_local
import torchinfo
from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)
import warnings
from collections import OrderedDict
from copy import deepcopy
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from mmengine.model import BaseModule, ModuleList
import torch
import torch.nn.functional as F
from torch import nn

from mmseg.registry import MODELS
from mmengine.logging import print_log
from mmengine.runner import CheckpointLoader
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from spikingjelly.clock_driven.functional import reset_net
from typing import Optional, Tuple
from torch import Tensor
from mmdet.models.utils.Qtrick import MultiSpike_norm4


class BNAndPadLayer(nn.Module):
    def __init__(
            self,
            pad_pixels,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                        self.bn.bias.detach()
                        - self.bn.running_mean
                        * self.bn.weight.detach()
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class RepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
            self,
            dim,
            expansion_ratio=2,
            act2_layer=nn.Identity,
            bias=False,
            kernel_size=7,
            padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        # self.lif1 = MultiSpike_norm4(T=4)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        # self.lif2 = MultiSpike_norm4(T=4)
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.dwconv(x.flatten(0, 1))
        x = self.bn2(self.pwconv2(x)).reshape(T, B, -1, H, W)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.0,
    ):
        super().__init__()

        self.Conv = SepConv(dim=dim)
        # self.Conv = MHMC(dim=dim)

        # self.lif1 = MultiSpike_norm4(T=4)
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv1 = RepConv(dim, dim*mlp_ratio)
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.lif2 = MultiSpike_norm4(T=4)
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv2 = RepConv(dim*mlp_ratio, dim)
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, 4 * C, H, W)
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class SelfAttention(nn.Module):
    def __init__(
            self,
            embed_dims,
            num_heads=8,
            attn_drop=0.0,
            dropout=0.0,
            proj_drop=0.0,
            batch_first=True,
            dropout_layer=None,
    ):
        super().__init__()
        assert (
                embed_dims % num_heads == 0
        ), f"dim {embed_dims} should be divided by num_heads {num_heads}."
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.scale = 0.1

        self.head_lif = MultiSpike_norm4(T=4)

        # self.q_conv = nn.Sequential(RepConv(embed_dims, embed_dims, bias=False), nn.BatchNorm2d(embed_dims))
        self.q_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims))

        # self.k_conv = nn.Sequential(RepConv(embed_dims, embed_dims, bias=False), nn.BatchNorm2d(embed_dims))
        self.k_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims))

        # self.v_conv = nn.Sequential(RepConv(embed_dims, embed_dims, bias=False), nn.BatchNorm2d(embed_dims))
        self.v_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims))

        self.q_lif = MultiSpike_norm4(T=4)
        self.k_lif = MultiSpike_norm4(T=4)
        self.v_lif = MultiSpike_norm4(T=4)

        self.attn_lif = MultiSpike_norm4(T=4)

        # self.proj_conv = nn.Sequential(
        #     RepConv(embed_dims, embed_dims, bias=False), nn.BatchNorm2d(embed_dims)
        # )
        self.proj_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                       nn.BatchNorm1d(embed_dims))

    def forward(self, query, key, value,
                query_pos=None, key_pos=None, attn_mask=None, query_key_padding_mask=None):
        # query:[bs, 100, 256] key=query=value, query_pos = key_pos
        tmp = x = query + query_pos
        T, B, N, C = x.shape
        x = self.head_lif(x.reshape(T, B, C, N))

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, N)  # 用三组深度可分离卷积生成qkv
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, N)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, N)

        q = self.q_lif(q).reshape(T, B, C, N)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k = self.k_lif(k).flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v).flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = q @ k.transpose(-2, -1) * self.scale
        x = (x @ v)

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.reshape(T, B, C, N)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, N, C)

        return x + tmp


class CrossAttention(nn.Module):
    def __init__(
            self,
            embed_dims,
            num_heads=8,
            attn_drop=0.0,
            dropout=0.0,
            proj_drop=0.0,
            batch_first=True,
            dropout_layer=None,
    ):
        super().__init__()
        assert (
                embed_dims % num_heads == 0
        ), f"dim {embed_dims} should be divided by num_heads {num_heads}."
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.scale = 0.1
        # self.scale = embed_dims**-2

        self.head_lif_q = MultiSpike_norm4(T=4)
        self.head_lif_k = MultiSpike_norm4(T=4)
        # self.head_lif_v = MultiSpike_norm4(T=4)

        # self.q_conv = nn.Sequential(RepConv(embed_dims, embed_dims, bias=False), nn.BatchNorm2d(embed_dims))
        self.q_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims))

        # self.k_conv = nn.Sequential(RepConv(embed_dims, embed_dims, bias=False), nn.BatchNorm2d(embed_dims))
        self.k_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims))

        # self.v_conv = nn.Sequential(RepConv(embed_dims, embed_dims, bias=False), nn.BatchNorm2d(embed_dims))
        self.v_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                    nn.BatchNorm1d(embed_dims))


        self.q_lif = MultiSpike_norm4(T=4)
        self.k_lif = MultiSpike_norm4(T=4)
        self.v_lif = MultiSpike_norm4(T=4)
        self.attn_lif = MultiSpike_norm4(T=4)
        # self.proj_conv = nn.Sequential(
        #     RepConv(embed_dims, embed_dims, bias=False), nn.BatchNorm2d(embed_dims)
        # )
        self.proj_conv = nn.Sequential(nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1),
                                       nn.BatchNorm1d(embed_dims))


    def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
        """
        Determine mask type and combine masks if necessary. If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=query.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            mask_type = 0
            merged_mask = attn_mask
        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask
        if (attn_mask is not None) and (key_padding_mask is not None):
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2
            key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len) \
                .expand(-1, self.num_heads, -1, -1)
            attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded + key_padding_mask_expanded
        return merged_mask, mask_type

    def forward(self, query, key, value,
                query_pos=None, key_pos=None, attn_mask=None, key_padding_mask=None, query_key_padding_mask=None):
        """
            Q: Using linear to replace the Conv

        """
        # merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)
        # query:[bs, 100, 256] key=value: [bs, 1024, 256], query_pos, key_pos, attn_mask:[bs*num_heads, 100, 1024]
        T, B, NQ, C = query.shape
        T, B, NK, C = key.shape

        q_t = query = query + query_pos
        key = key + key_pos
        # attn_mask = attn_mask.reshape(T, B, N, C)

        query = self.head_lif_q(query.reshape(T, B, C, NQ))
        query = MultiSpike4.quant4.apply(query.reshape(T, B, C, NQ))
        key = self.head_lif_k(key.reshape(T, B, C, NK))
        value = key

        q = self.q_conv(query.flatten(0, 1)).reshape(T, B, C, NQ)  # 用三组深度可分离卷积生成qkv
        k = self.k_conv(key.flatten(0, 1)).reshape(T, B, C, NK)
        v = self.v_conv(value.flatten(0, 1)).reshape(T, B, C, NK)

        q = self.q_lif(q).reshape(T, B, C, NQ)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, NQ, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k = self.k_lif(k).reshape(T, B, C, NK)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, NK, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v).reshape(T, B, C, NK)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, NK, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        # NOTE: CHANGE HERE
        x = q @ k.transpose(-2, -1) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.reshape(B, self.num_heads, NQ, NK).contiguous()
                x = x.masked_fill_(attn_mask, float('0'))
                # 原本填充-inf是为了softmax后attention_weights变为0， 先由于已经稀疏，因此直接填零
            else:
                x += attn_mask

        x = (x @ v)

        x = x.transpose(3, 4).reshape(T, B, C, NQ).contiguous()
        x = self.attn_lif(x)
        x = x.reshape(T, B, C, NQ)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, NQ, C)

        return x + q_t


class MLP(nn.Module):
    def __init__(
            self, embed_dims,
            feedforward_channels,
            num_fcs=None,
            ffn_drop=0.0,
            layer=0,
            dropout_layer=None,
            act_cfg=None,
            add_identity=None,
    ):
        super().__init__()
        out_features = embed_dims
        hidden_features = feedforward_channels
        in_features = embed_dims
        # self.fc1 = linear_unit(in_features, hidden_features)
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiSpike_norm4(T=4)

        # self.fc2 = linear_unit(hidden_features, out_features)
        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiSpike_norm4(T=4)
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, N, C = x.shape
        x_t = x
        x = x.reshape(T, B, C, N).contiguous()

        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, N, C).contiguous()

        return x + x_t


class MSTransformerDecoder(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            sr_ratio=1,
    ):
        super().__init__()
        self.crossattn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.crossattn(x)
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


# MARK:
"""
lzx2
6,7改动： 更改了shortcut路径，增大学习率和weightdecay
# 确定forwardhead中没有问题，
NEXT：更换所有的repconv -> linear
"""
