# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# import pdb; pdb.set_trace()
# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from ..functions import dcnv3_core_pytorch
from mmdet.models.utils.Qtrick import MultiSpike_norm4, MultiSpike_4
from mmdet.models.layers.transformer.mmcv_spike.SNN_core import SepConv_Spike
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from Qtrick_architecture.clock_driven.neuron import Q_IFNode
from Qtrick_architecture.clock_driven.surrogate import Quant


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


class DCNv3_pytorch(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            expension_ratio=4,
            T=4,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.T = T

        self.dw_spike = Q_IFNode(surrogate_function=Quant())  # without normal
        self.offset_spike = Q_IFNode(surrogate_function=Quant())
        self.mask_spike = Q_IFNode(surrogate_function=Quant())
        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels,
                      channels,
                      kernel_size=dw_kernel_size,
                      padding=(dw_kernel_size - 1) // 2,
                      groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels)
        )

        self.offset = nn.Sequential(
            nn.Conv2d(channels, group * kernel_size * kernel_size * 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(group * kernel_size * kernel_size * 2)
        )

        self.mask = nn.Sequential(
            nn.Conv2d(channels, group * kernel_size * kernel_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(group * kernel_size * kernel_size)
        )

        self.input_proj = SepConv_Spike(dim=channels,
                                        kernel_size=dw_kernel_size,
                                        padding=(dw_kernel_size - 1) // 2,
                                        expansion_ratio=expension_ratio,
                                        )

        self.output_proj = SepConv_Spike(dim=channels,
                                         kernel_size=dw_kernel_size,
                                         padding=(dw_kernel_size - 1) // 2,
                                         expansion_ratio=expension_ratio)

        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset[0].weight.data, 0.)
        constant_(self.offset[0].bias.data, 0.)
        constant_(self.mask[0].weight.data, 0.)
        constant_(self.mask[0].bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        # import pdb; pdb.set_trace()
        T, N, H, W, C = input.shape
        x = self.input_proj(input)
        x_proj = x

        x1 = input.permute(0, 1, 4, 2, 3).contiguous()
        x1 = self.dw_spike(x1)
        x1 = self.dw_conv(x1.flatten(0, 1)).reshape(T, N, C, H, W)
        x1 = self.offset_spike(x1)

        offset = self.offset(x1.flatten(0, 1)).reshape(T, N, H, W, -1)  # [256 -> 576]
        mask = self.mask(x1.flatten(0, 1)).reshape(T, N, H, W, self.group, -1).reshape(T, N, H, W, -1)  # [256 -> 288]
        mask = self.mask_spike(mask)

        # import pdb; pdb.set_trace()
        # x:[bs, H, W, dim]
        # offset:[bs, H, W, group * kernel_size * kernel_size * 2]
        # mask: [bs, H, W, group * kernel_size * kernel_size]
        # import pdb; pdb.set_trace()
        x = dcnv3_core_pytorch(
            x.flatten(0, 1), offset.flatten(0, 1), mask.flatten(0, 1),
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale)

        x = self.output_proj(x.reshape(T, N, H, W, C))

        return x

# class DCNv3_pytorch(nn.Module):
#     def __init__(
#             self,
#             channels=64,
#             kernel_size=3,
#             dw_kernel_size=None,
#             stride=1,
#             pad=1,
#             dilation=1,
#             group=4,
#             offset_scale=1.0,
#             act_layer='GELU',
#             norm_layer='LN',
#             center_feature_scale=False):
#         """
#         DCNv3 Module
#         :param channels
#         :param kernel_size
#         :param stride
#         :param pad
#         :param dilation
#         :param group
#         :param offset_scale
#         :param act_layer
#         :param norm_layer
#         """
#         super().__init__()
#         if channels % group != 0:
#             raise ValueError(
#                 f'channels must be divisible by group, but got {channels} and {group}')
#         _d_per_group = channels // group
#         dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
#         # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
#         if not _is_power_of_2(_d_per_group):
#             warnings.warn(
#                 "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
#                 "which is more efficient in our CUDA implementation.")
#
#         self.offset_scale = offset_scale
#         self.channels = channels
#         self.kernel_size = kernel_size
#         self.dw_kernel_size = dw_kernel_size
#         self.stride = stride
#         self.dilation = dilation
#         self.pad = pad
#         self.group = group
#         self.group_channels = channels // group
#         self.offset_scale = offset_scale
#         self.center_feature_scale = center_feature_scale
#
#         self.dw_conv = nn.Sequential(
#             nn.Conv2d(
#                 channels,
#                 channels,
#                 kernel_size=dw_kernel_size,
#                 stride=1,
#                 padding=(dw_kernel_size - 1) // 2,
#                 groups=channels),
#             build_norm_layer(
#                 channels,
#                 norm_layer,
#                 'channels_first',
#                 'channels_last'),
#             build_act_layer(act_layer))
#         self.offset = nn.Linear(
#             channels,
#             group * kernel_size * kernel_size * 2)
#         self.mask = nn.Linear(
#             channels,
#             group * kernel_size * kernel_size)
#         self.input_proj = nn.Linear(channels, channels)
#         self.output_proj = nn.Linear(channels, channels)
#         self._reset_parameters()
#
#         if center_feature_scale:
#             self.center_feature_scale_proj_weight = nn.Parameter(
#                 torch.zeros((group, channels), dtype=torch.float))
#             self.center_feature_scale_proj_bias = nn.Parameter(
#                 torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
#             self.center_feature_scale_module = CenterFeatureScaleModule()
#
#     def _reset_parameters(self):
#         constant_(self.offset.weight.data, 0.)
#         constant_(self.offset.bias.data, 0.)
#         constant_(self.mask.weight.data, 0.)
#         constant_(self.mask.bias.data, 0.)
#         xavier_uniform_(self.input_proj.weight.data)
#         constant_(self.input_proj.bias.data, 0.)
#         xavier_uniform_(self.output_proj.weight.data)
#         constant_(self.output_proj.bias.data, 0.)
#
#     def forward(self, input):
#         """
#         :param query                       (N, H, W, C)
#         :return output                     (N, H, W, C)
#         """
#         N, H, W, _ = input.shape
#
#         x = self.input_proj(input)
#         x_proj = x
#
#         x1 = input.permute(0, 3, 1, 2)
#         x1 = self.dw_conv(x1)
#         offset = self.offset(x1)
#         mask = self.mask(x1).reshape(N, H, W, self.group, -1)
#         mask = F.softmax(mask, -1).reshape(N, H, W, -1)
#
#         import pdb; pdb.set_trace()
#         x = dcnv3_core_pytorch(
#             x, offset, mask,
#             self.kernel_size, self.kernel_size,
#             self.stride, self.stride,
#             self.pad, self.pad,
#             self.dilation, self.dilation,
#             self.group, self.group_channels,
#             self.offset_scale)
#         if self.center_feature_scale:
#             center_feature_scale = self.center_feature_scale_module(
#                 x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
#             # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
#             center_feature_scale = center_feature_scale[..., None].repeat(
#                 1, 1, 1, 1, self.channels // self.group).flatten(-2)
#             x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
#         x = self.output_proj(x)
#
#         return x

#
# class DCNv3(nn.Module):
#     def __init__(
#             self,
#             channels=64,
#             kernel_size=3,
#             dw_kernel_size=None,
#             stride=1,
#             pad=1,
#             dilation=1,
#             group=4,
#             offset_scale=1.0,
#             act_layer='GELU',
#             norm_layer='LN',
#             center_feature_scale=False,
#             use_dcn_v4_op=False,
#             ):
#         """
#         DCNv3 Module
#         :param channels
#         :param kernel_size
#         :param stride
#         :param pad
#         :param dilation
#         :param group
#         :param offset_scale
#         :param act_layer
#         :param norm_layer
#         """
#         super().__init__()
#         if channels % group != 0:
#             raise ValueError(
#                 f'channels must be divisible by group, but got {channels} and {group}')
#         _d_per_group = channels // group
#         dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
#         # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
#         if not _is_power_of_2(_d_per_group):
#             warnings.warn(
#                 "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
#                 "which is more efficient in our CUDA implementation.")
#
#         self.offset_scale = offset_scale
#         self.channels = channels
#         self.kernel_size = kernel_size
#         self.dw_kernel_size = dw_kernel_size
#         self.stride = stride
#         self.dilation = dilation
#         self.pad = pad
#         self.group = group
#         self.group_channels = channels // group
#         self.offset_scale = offset_scale
#         self.center_feature_scale = center_feature_scale
#
#         self.use_dcn_v4_op = use_dcn_v4_op
#
#         self.dw_conv = nn.Sequential(
#             nn.Conv2d(
#                 channels,
#                 channels,
#                 kernel_size=dw_kernel_size,
#                 stride=1,
#                 padding=(dw_kernel_size - 1) // 2,
#                 groups=channels),
#             build_norm_layer(
#                 channels,
#                 norm_layer,
#                 'channels_first',
#                 'channels_last'),
#             build_act_layer(act_layer))
#         self.offset = nn.Linear(
#             channels,
#             group * kernel_size * kernel_size * 2)
#         self.mask = nn.Linear(
#             channels,
#             group * kernel_size * kernel_size)
#         self.input_proj = nn.Linear(channels, channels)
#         self.output_proj = nn.Linear(channels, channels)
#         self._reset_parameters()
#
#         if center_feature_scale:
#             self.center_feature_scale_proj_weight = nn.Parameter(
#                 torch.zeros((group, channels), dtype=torch.float))
#             self.center_feature_scale_proj_bias = nn.Parameter(
#                 torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
#             self.center_feature_scale_module = CenterFeatureScaleModule()
#
#     def _reset_parameters(self):
#         constant_(self.offset.weight.data, 0.)
#         constant_(self.offset.bias.data, 0.)
#         constant_(self.mask.weight.data, 0.)
#         constant_(self.mask.bias.data, 0.)
#         xavier_uniform_(self.input_proj.weight.data)
#         constant_(self.input_proj.bias.data, 0.)
#         xavier_uniform_(self.output_proj.weight.data)
#         constant_(self.output_proj.bias.data, 0.)
#
#     def forward(self, input):
#         """
#         :param query                       (N, H, W, C)
#         :return output                     (N, H, W, C)
#         """
#         N, H, W, _ = input.shape
#
#         x = self.input_proj(input)
#         x_proj = x
#         dtype = x.dtype
#
#         x1 = input.permute(0, 3, 1, 2)
#         x1 = self.dw_conv(x1)
#         offset = self.offset(x1)
#         mask = self.mask(x1).reshape(N, H, W, self.group, -1)
#
#         if not self.use_dcn_v4_op:
#             mask = F.softmax(mask, -1).reshape(N, H, W, -1).type(dtype)
#             x = DCNv3Function.apply(
#                 x, offset, mask,
#                 self.kernel_size, self.kernel_size,
#                 self.stride, self.stride,
#                 self.pad, self.pad,
#                 self.dilation, self.dilation,
#                 self.group, self.group_channels,
#                 self.offset_scale,
#                 256)
#         else:
#             # DCNv4 combines offset and weight mask into one tensor `offset_mask`.
#             # The following code is to align DCNv3 and DCNv4
#             offset = offset.view(N, H, W, self.group, -1)
#             mask = F.softmax(mask, -1)
#             mask = mask.view(N, H, W, self.group, -1)
#             offset_mask = torch.cat([offset, mask], -1).view(N, H, W, -1).contiguous()
#
#             # For efficiency, the last dimension of the offset_mask tensor in dcnv4 is a multiple of 8.
#             K3 = offset_mask.size(-1)
#             K3_pad = int(math.ceil(K3/8)*8)
#             pad_dim = K3_pad - K3
#             offset_mask = torch.cat([offset_mask, offset_mask.new_zeros([*offset_mask.size()[:3], pad_dim])], -1)
#
#             x = DCNv4Function.apply(
#                 x, offset_mask,
#                 self.kernel_size, self.kernel_size,
#                 self.stride, self.stride,
#                 self.pad, self.pad,
#                 self.dilation, self.dilation,
#                 self.group, self.group_channels,
#                 self.offset_scale,
#                 256,
#                 False
#             )
#
#         if self.center_feature_scale:
#             center_feature_scale = self.center_feature_scale_module(
#                 x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
#             # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
#             center_feature_scale = center_feature_scale[..., None].repeat(
#                 1, 1, 1, 1, self.channels // self.group).flatten(-2)
#             x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
#         x = self.output_proj(x)
#
#         return x
