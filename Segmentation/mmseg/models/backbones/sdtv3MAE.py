from functools import partial
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block

import copy
from torchvision import transforms
import matplotlib.pyplot as plt

import os
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmengine.logging import print_log
from mmengine.runner import CheckpointLoader


@torch.jit.script
def jit_mul(x, y):
    return x.mul(y)


@torch.jit.script
def jit_sum(x):
    return x.sum(dim=[-1, -2], keepdim=True)


# class Quant(torch.autograd.Function):
#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def forward(ctx, i, min_value=0, max_value=8):
#         ctx.min = min_value
#         ctx.max = max_value
#         ctx.save_for_backward(i)
#         return torch.round(torch.clamp(i, min=min_value, max=max_value))

#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         i, = ctx.saved_tensors
#         grad_input[i < ctx.min] = 0
#         grad_input[i > ctx.max] = 0
#         return grad_input, None, None
class SepConv_Spike(nn.Module):
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
        self.spike1 = Multispike()
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike2 = Multispike()
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels,
                      bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike3 = Multispike()
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.spike1(x)

        x = self.pwconv1(x.flatten(0, 1)).reshape(T, B, -1, H, W)

        x = self.spike2(x)

        x = self.dwconv(x.flatten(0, 1)).reshape(T, B, -1, H, W)

        x = self.spike3(x)

        x = self.pwconv2(x.flatten(0, 1)).reshape(T, B, C, H, W)
        return x


# class Multispike(nn.Module):
#     def __init__(
#             self,
#             Norm=8,
#     ):
#         super().__init__()
#         self.spike = Quant()
#         self.Norm = Norm

#     def forward(self, x):
#         return self.spike.apply(x) / (self.Norm)

#     def __repr__(self):
#         return f"Multispike(Norm={self.Norm})"
class ReLUX(nn.Module):
    def __init__(self, thre=8):
        super(ReLUX, self).__init__()
        self.thre = thre

    def forward(self, input):
        return torch.clamp(input, 0, self.thre)


relu8 = ReLUX(thre=8)

import torch


class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens=8):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(relu8(input) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None


class Multispike(nn.Module):
    def __init__(self, lens=8, spike=multispike):
        super().__init__()
        self.lens = lens
        self.spike = spike

    def forward(self, inputs):
        return self.spike.apply(inputs) / 8


class Multispike_head(nn.Module):
    def __init__(self, lens=8, spike=multispike):
        super().__init__()
        self.lens = lens
        self.spike = spike

    def forward(self, inputs):
        return self.spike.apply(inputs)


class MS_ConvBlock(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.0,

    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim)

        self.mlp_ratio = mlp_ratio

        self.spike1 = Multispike()
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = Multispike()
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)

        x = self.bn1(self.conv1(x.flatten(0, 1))).reshape(T, B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)

        x = self.bn2(self.conv2(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MS_MLP(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = Multispike()

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = Multispike()

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_spike(x)

        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
        x = self.fc2_spike(x)

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x


# def norm_mem(input_tensor):

#     mean = input_tensor.mean()
#     std = input_tensor.std()

#     # 通过线性变换和偏移量来重新缩放张量
#     scaled_tensor = (input_tensor - mean) / std * 0.5 + 0.5
#     return scaled_tensor
#     # return 0.5 * (input_tensor + 1)
class LePEAttention(nn.Module):
    def __init__(self, dim=None, resolution=None, idx=-1, split_num=2, dim_out=None, num_heads=8, attn_drop=0.,
                 proj_drop=0., qk_scale=None):
        super().__init__()
        self.split_num = split_num
        self.num_heads = num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale
        self.idx = idx
        if idx == -1:
            H_num, W_num = 1, 1
        elif idx == 0:
            H_num, W_num = 1, self.split_num
        elif idx == 1:
            H_num, W_num = self.split_num, 1
        else:
            print("ERROR MODE", idx)
            exit(0)

        self.H_num = H_num
        self.W_num = W_num

    def forward(self, q, k, v, v_lamda):
        """
        q,k,v: T, B C H W
        """
        # Add Padding here

        T, B, C, H, W = q.shape
        H_, W_ = H, W
        # import pdb; pdb.set_trace()
        H_num, W_num = self.H_num, self.W_num

        # Calculate the required padding for H and W
        H_pad = (H_num - H % H_num) % H_num if H % H_num != 0 else 0
        W_pad = (W_num - W % W_num) % W_num if W % W_num != 0 else 0

        # Add padding on left and bottom
        q = F.pad(q, (0, W_pad, 0, H_pad), "constant", 0)
        k = F.pad(k, (0, W_pad, 0, H_pad), "constant", 0)
        v = F.pad(v, (0, W_pad, 0, H_pad), "constant", 0)

        H += H_pad
        W += W_pad

        self.H_sp = H // self.H_num
        self.W_sp = W // self.W_num

        q = (
            q.reshape(T, B, C, H // self.H_sp, self.H_sp, W // self.W_sp, self.W_sp)
            .permute(0, 1, 3, 5, 4, 6, 2)
            .reshape(T, -1, self.H_sp * self.W_sp, C)
            .reshape(T, -1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k = (
            k.reshape(T, B, C, H // self.H_sp, self.H_sp, W // self.W_sp, self.W_sp)
            .permute(0, 1, 3, 5, 4, 6, 2)
            .reshape(T, -1, self.H_sp * self.W_sp, C)
            .reshape(T, -1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = (
            v.reshape(T, B, C * v_lamda, H // self.H_sp, self.H_sp, W // self.W_sp, self.W_sp)
            .permute(0, 1, 3, 5, 4, 6, 2)
            .reshape(T, -1, self.H_sp * self.W_sp, C * v_lamda)
            .reshape(T, -1, self.H_sp * self.W_sp, self.num_heads, C * v_lamda // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = q @ k.transpose(-2, -1)
        x = (x @ v) * (self.scale * 2)

        x = (
            x.transpose(2, 3).reshape(T, -1, self.H_sp * self.W_sp, C * v_lamda)
            .reshape(T, B, H // self.H_sp, W // self.W_sp, self.H_sp, self.W_sp, -1)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(T, B, H, W, -1)
            .permute(0, 1, 4, 2, 3)
            .contiguous()# T, B, C, H, W
        )
        # import pdb;
        # pdb.set_trace()
        x = x[:, :, :, :H_, :W_]  # [H, W]
        return x


class MS_Attention_linear_cswin(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            sr_ratio=1,
            last_stage=False,
            lamda_ratio=1,
    ):
        super().__init__()
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        if last_stage == False:
            self.branch_num = 2
        else:
            self.branch_num = 1

        self.dim = dim
        self.num_heads = num_heads // self.branch_num
        self.scale = (dim // num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio

        self.head_spike = Multispike()

        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = Multispike()

        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.k_spike = Multispike()

        self.v_conv = nn.Sequential(nn.Conv2d(dim, int(dim * lamda_ratio), 1, 1, bias=False),
                                    nn.BatchNorm2d(int(dim * lamda_ratio)))

        self.v_spike = Multispike()

        self.attn_spike = Multispike()

        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim * lamda_ratio, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    idx=-1, num_heads=num_heads, qk_scale=self.scale)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    idx=i, num_heads=num_heads // 2, qk_scale=self.scale)
                for i in range(self.branch_num)])

    def forward(self, x):
        T, B, C, H, W = x.shape
        C_v = int(C * self.lamda_ratio)

        x = self.head_spike(x)

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C_v, H, W)

        q = self.q_spike(q)
        k = self.q_spike(k)
        v = self.q_spike(v)

        if self.branch_num == 2:
            x1 = self.attns[0](q[:, :, :C // 2, :, :], k[:, :, :C // 2, :, :], v[:, :, :C_v // 2, :, :],
                               self.lamda_ratio)
            x2 = self.attns[1](q[:, :, C // 2:, :, :], k[:, :, C // 2:, :, :], v[:, :, C_v // 2:, :, :],
                               self.lamda_ratio)
            x = torch.cat([x1, x2], dim=2)
        else:
            x = self.attns[0](q, k, v, self.lamda_ratio)

        x = self.attn_spike(x)

        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        return x


class MS_Block_cswin(nn.Module):
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
            init_values=1e-6,
            last_stage=False,
            resolution=14,
            T=None
    ):
        super().__init__()
        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1)
        self.attn = MS_Attention_linear_cswin(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            lamda_ratio=4,
            last_stage=last_stage,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.layer_scale1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.layer_scale2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.layer_scale3 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.conv(x) * self.layer_scale1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        x = x + self.drop_path(self.attn(x) * self.layer_scale2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        x = x + self.drop_path(self.mlp(x) * self.layer_scale3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        return x



class MS_DownSampling(nn.Module):
    def __init__(
            self,
            in_channels=2,
            embed_dims=256,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=True,

    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = Multispike()

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W)

        return x


@MODELS.register_module()
class Spiking_vit_MetaFormerv3(BaseModule):
    def __init__(self,
                 T=1,
                 img_size_h=224,
                 img_size_w=224,
                 patch_size=16,
                 embed_dim=[128, 256, 512, 640],
                 num_heads=8,
                 mlp_ratios=4,
                 in_channels=3,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 num_classes=1000,
                 qkv_bias=False,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),  # norm_layer=nn.LayerNorm shaokun
                 depths=8,
                 sr_ratios=1,
                 nb_classes=1000,
                 decode_mode='QTrick',
                 init_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 pretrained=None,
                 kd=False):
        super().__init__(init_cfg=init_cfg)

        ### MAE encoder spikformer
        self.T = T
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.depths = depths
        # embed_dim = [64, 128, 256, 512]

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,

        )

        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,

        )

        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,

        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,

        )

        self.block3 = nn.ModuleList(
            [
                MS_Block_cswin(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,

                )
                for j in range(int(depths * 0.75))
            ]
        )

        self.block4 = nn.ModuleList(
            [
                MS_Block_cswin(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j + int(depths * 0.75)],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    last_stage=True,
                )
                for j in range(int(depths * 0.25))
            ]
        )
        self.downsample_raito = 16

        self.initialize_weights()

    def init_weights(self):
        # logger = MMlogger.get_current_instance()
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
            # self.apply(self._init_weights)

            print_log("init_weighting.....")
            self.apply(self._init_weights)
            print_log("Time step: {:}".format(self.T))
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            # import pdb; pdb.set_trace()
            # state_dict = self.state_dict()
            # import pdb; pdb.set_trace()
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            # import pdb; pdb.set_trace()
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                # 使用mmseg保存的checkpoint中包含backbone, neck, decode_head三个部分
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v
            # import pdb; pdb.set_trace()
            self.load_state_dict(state_dict, strict=False)

            print_log("--------------Successfully load checkpoint for BACKNONE------------")
            print_log("Time step: {:}".format(self.T))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            for name, param in m.named_parameters():
                param.requires_grad = True

    def forward_encoder(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x1 = x
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)
        x2 = x
        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)
        x3 = x
        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        for blk in self.block4:
            x = blk(x)
        x4 = x
        # import pdb; pdb.set_trace()
        return [x1.mean(0), x2.mean(0), x3.mean(0), x4.mean(0)]

    def forward(self, imgs):
        x = self.forward_encoder(imgs)
        return x

# def spikformer_12_768(**kwargs):
#     model = spikeformer(
#         T=1,
#         img_size_h=224,
#         img_size_w=224,
#         patch_size=16,
#         embed_dim=[192, 384, 768],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=1000,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=12,
#         sr_ratios=1,
#         **kwargs)
#     return model
#
#
# def spikformer_12_512(**kwargs):
#     model = spikeformer(
#         T=1,
#         img_size_h=224,
#         img_size_w=224,
#         patch_size=16,
#         embed_dim=[128, 256, 512],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=1000,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=12,
#         sr_ratios=1,
#         **kwargs)
#     return model
#
#
# def spikformer_12_1024(**kwargs):
#     model = spikeformer(
#         T=1,
#         img_size_h=224,
#         img_size_w=224,
#         patch_size=16,
#         embed_dim=[256, 512, 1024],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=1000,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=12,
#         sr_ratios=1,
#         **kwargs)
#     return model
#
#
# def spikformer_16_640(**kwargs):
#     model = spikeformer(
#         T=1,
#         img_size_h=224,
#         img_size_w=224,
#         patch_size=16,
#         embed_dim=[160, 320, 640],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=1000,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=16,
#         sr_ratios=1,
#         **kwargs)
#     return model
#
#
# def spikformer_16_512(**kwargs):
#     model = spikeformer(
#         T=1,
#         img_size_h=224,
#         img_size_w=224,
#         patch_size=16,
#         embed_dim=[128, 256, 512],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=1000,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=16,
#         sr_ratios=1,
#         **kwargs)
#     return model

#
# import json
#
# import json
# import pdb
#
#
# def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.95):
#     """
#     Parameter groups for layer-wise lr decay
#     Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
#     """
#     param_group_names = {}
#     param_groups = {}
#     # num_layers = len(model.block3) + 1
#     num_layers = model.depths + 1
#     layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
#     print(layer_scales, len(layer_scales))
#     for n, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#
#         # no decay: all 1D parameters and model specific ones
#         if p.ndim == 1 or n in no_weight_decay_list:
#             g_decay = "no_decay"
#             this_decay = 0.
#         else:
#             g_decay = "decay"
#             this_decay = weight_decay
#
#         layer_id = get_layer_id_for_vit(n, num_layers, model.depths)
#         group_name = "layer_%d_%s" % (layer_id, g_decay)
#
#         if group_name not in param_group_names:
#             this_scale = layer_scales[layer_id]
#
#             param_group_names[group_name] = {
#                 "lr_scale": this_scale,
#                 "weight_decay": this_decay,
#                 "params": [],
#             }
#             param_groups[group_name] = {
#                 "lr_scale": this_scale,
#                 "weight_decay": this_decay,
#                 "params": [],
#             }
#
#         param_group_names[group_name]["params"].append(n)
#         param_groups[group_name]["params"].append(p)
#
#     print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
#
#     return list(param_groups.values())
#
#
# def get_layer_id_for_vit(name, num_layers, depth):
#     """
#     Assign a parameter with its layer id
#     Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
#     """
#     if name in ['cls_token', 'pos_embed']:
#         return 0
#     elif name.startswith('downsample1_1'):
#         return 0
#     elif name.startswith('ConvBlock1_1'):
#         return 0
#     elif name.startswith('downsample1_2'):
#         return 0
#     elif name.startswith('ConvBlock1_2'):
#         return 0
#     elif name.startswith('downsample2'):
#         return 0
#     elif name.startswith('ConvBlock2_1'):
#         return 0
#     elif name.startswith('ConvBlock2_2'):
#         return 0
#     elif name.startswith('downsample3'):
#         return 0
#     elif name.startswith('block3'):
#         return int(name.split('.')[1]) + 1
#     elif name.startswith('block4'):
#         return int(name.split('.')[1]) + depth - 2
#     else:
#         return num_layers
#
# #
# if __name__ == "__main__":
#     # from encoder import SparseEncoder,nn.Conv2d
#
#     # model = SparseEncoder(spikformer_8_512_CAFormer(), 224)
#     # #     # print(model)
#     # import torchsummary
#     model = spikformer_12_768()
#     # model = spikformer_8_512_CAFormer()
#     x = torch.randn(1, 3, 224, 224)
#     # print(f"number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
#     # loss = model(x,mask_ratio=0.75)
#     # print('loss',loss)
#     # out = model(x)
#     print(f"number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
#     # print(out.shape)
#     # out = param_groups_lrd(model)
