# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule
from mmengine.model import BaseModule, ModuleList, caffe2_xavier_init
from torch import Tensor
from mmdet.models.utils.Qtrick import MultiSpike_norm4, MultiSpike_4
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from Qtrick_architecture.clock_driven.neuron import Q_IFNode
from Qtrick_architecture.clock_driven.surrogate import Quant,Quant4

from mmdet.registry import MODELS
# from mmseg.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from .positional_encoding import SinePositionalEncoding
from .transformer import DetrTransformerEncoder, DCNDetrTransformerEncoder


# NOTE: here we change the
@MODELS.register_module()
class PixelDecoder(BaseModule):
    """Pixel decoder with a structure like fpn.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        feat_channels (int): Number channels for feature.
        out_channels (int): Number channels for output.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`ConfigDict` or dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`ConfigDict` or dict): Config for transorformer
            encoder.Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer encoder position encoding. Defaults to
            dict(type='SinePositionalEncoding', num_feats=128,
            normalize=True).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: Union[List[int], Tuple[int]],
                 feat_channels: int,
                 out_channels: int,
                 T: int = 4,
                 norm_cfg: ConfigType = dict(type='GN', num_groups=32),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_inputs = len(in_channels)
        self.lateral_convs = ModuleList()
        self.lateral_convs_spike = ModuleList()
        self.output_convs = ModuleList()
        self.output_convs_spike = ModuleList()
        self.use_bias = norm_cfg is None
        self.T = T
        for i in range(0, self.num_inputs - 1):
            lateral_conv_spike = Q_IFNode(surrogate_function=Quant())
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels[i], feat_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(feat_channels)
            )
            output_conv_spike = Q_IFNode(surrogate_function=Quant())
            output_conv = nn.Sequential(
                nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(feat_channels)
            )

            self.lateral_convs.append(lateral_conv)
            self.lateral_convs_spike.append(lateral_conv_spike)
            self.output_convs.append(output_conv)
            self.output_convs_spike.append(output_conv_spike)

        self.last_feat_conv_spike = Q_IFNode(surrogate_function=Quant())
        self.last_feat_conv = nn.Sequential(
            nn.Conv2d(in_channels[-1], feat_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels)
        )
        self.mask_feature_spike = Q_IFNode(surrogate_function=Quant())
        self.mask_feature = nn.Conv2d(
            feat_channels, out_channels, kernel_size=1, stride=1, padding=1)

    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, self.num_inputs - 2):
            caffe2_xavier_init(self.lateral_convs[i][0], bias=0)
            caffe2_xavier_init(self.output_convs[i][0], bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)
        caffe2_xavier_init(self.last_feat_conv[0], bias=0)

    def forward(self, feats: List[Tensor],
                batch_img_metas: List[dict]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).
            batch_img_metas (list[dict]): List of image information.
                Pass in for creating more accurate padding mask. Not
                used here.

        Returns:
            tuple[Tensor, Tensor]: a tuple containing the following:

                - mask_feature (Tensor): Shape (batch_size, c, h, w).
                - memory (Tensor): Output of last stage of backbone.\
                        Shape (batch_size, c, h, w).
        """


        t, bs, c, h, w = feats[-1].shape
        y = self.last_feat_conv(self.last_feat_conv_spike(feats[-1]).flatten(0, 1))

        out = []
        for i in range(self.num_inputs - 2, -1, -1):
            x = feats[i].flatten(0, 1)
            x = self.lateral_convs_spike[i](x)
            cur_feat = self.lateral_convs[i](x)
            y = cur_feat + \
                F.interpolate(y, size=cur_feat.shape[-2:], mode='nearest')
            y = self.output_convs_spike[i](y)
            y = self.output_convs[i](y)
            _, C, H, W = y.shape
            out.append(y.reshape(t, bs, C, H, W))

        y = self.mask_feature_spike(y)
        mask_feature = self.mask_feature(y)
        BS, C, H, W = mask_feature.shape
        mask_feature = mask_feature.view(t, BS, C, H, W)
        memory = feats[-1].reshape(t, bs, c, h, w)
        # import pdb; pdb.set_trace()
        return mask_feature, memory, out[:3]


@MODELS.register_module()
class TransformerEncoderPixelDecoder(PixelDecoder):
    """Pixel decoder with transormer encoder inside.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        feat_channels (int): Number channels for feature.
        out_channels (int): Number channels for output.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`ConfigDict` or dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`ConfigDict` or dict): Config for transformer encoder.
            Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer encoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: Union[List[int], Tuple[int]],
                 feat_channels: int,
                 out_channels: int,
                 T: int = 4,
                 norm_cfg: ConfigType = dict(type='GN', num_groups=32),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 encoder: ConfigType = None,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.last_feat_conv = None
        self.in_channels = in_channels
        self.num_inputs = len(in_channels)
        self.lateral_convs = ModuleList()
        self.lateral_convs_spike = ModuleList()
        self.output_convs = ModuleList()
        self.output_convs_spike = ModuleList()
        self.use_bias = norm_cfg is None
        self.T = T
        for i in range(0, self.num_inputs - 1):
            lateral_conv_spike = Q_IFNode(surrogate_function=Quant())
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels[i], feat_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(feat_channels)
            )

            output_conv_spike = Q_IFNode(surrogate_function=Quant())
            output_conv = nn.Sequential(
                nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1,
                          groups=feat_channels, bias=False),
                nn.BatchNorm2d(feat_channels)
            )

            self.lateral_convs.append(lateral_conv)
            self.lateral_convs_spike.append(lateral_conv_spike)
            self.output_convs.append(output_conv)
            self.output_convs_spike.append(output_conv_spike)

        self.mask_feature_spike = Q_IFNode(surrogate_function=Quant())  # 这里的norm对结果影响很大，因为没有BN层
        # self.mask_feature = nn.Conv2d(feat_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.mask_feature = nn.Conv2d(feat_channels, out_channels, kernel_size=1, stride=1)

        self.encoder = DetrTransformerEncoder(**encoder)
        self.encoder_embed_dims = self.encoder.embed_dims
        assert self.encoder_embed_dims == feat_channels, 'embed_dims({}) of ' \
                                                         'tranformer encoder must equal to feat_channels({})'.format(
            feat_channels, self.encoder_embed_dims)
        self.positional_encoding = SinePositionalEncoding(
            **positional_encoding)
        self.encoder_in_proj_spike = Q_IFNode(surrogate_function=Quant())
        self.encoder_in_proj = nn.Sequential(
            nn.Conv2d(in_channels[-1], feat_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(feat_channels)
        )
        self.encoder_out_proj_spike = Q_IFNode(surrogate_function=Quant())
        self.encoder_out_proj = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(feat_channels)
        )

    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, self.num_inputs - 2):
            caffe2_xavier_init(self.lateral_convs[i][0], bias=0)
            caffe2_xavier_init(self.output_convs[i][0], bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)
        caffe2_xavier_init(self.encoder_in_proj[0], bias=0)
        caffe2_xavier_init(self.encoder_out_proj[0], bias=0)

    def forward(self, feats: List[Tensor],
                batch_img_metas: List[dict]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).
            batch_img_metas (list[dict]): List of image information. Pass in
                for creating more accurate padding mask.

        Returns:
            tuple: a tuple containing the following:

                - mask_feature (Tensor): shape (batch_size, c, h, w).
                - memory (Tensor): shape (batch_size, c, h, w).
        """

        feat_last = feats[-1]
        t, bs, c, h, w = feat_last.shape
        # import pdb;
        # pdb.set_trace()
        input_img_h, input_img_w = batch_img_metas[0]['ori_shape']  # default batch_input_shape
        padding_mask = feat_last.new_ones((bs, input_img_h, input_img_w),
                                          dtype=torch.float32)
        for i in range(bs):
            img_h, img_w = batch_img_metas[i]['img_shape']
            padding_mask[i, :img_h, :img_w] = 0
        padding_mask = F.interpolate(
            padding_mask.unsqueeze(1),
            size=feat_last.shape[-2:],
            mode='nearest').to(torch.bool).squeeze(1)

        pos_embed = self.positional_encoding(padding_mask)
        feat_last = self.last_feat_conv_spike(feat_last)
        feat_last = self.encoder_in_proj(feat_last.flatten(0, 1)).view(t, bs, self.encoder_embed_dims, h, w)
        # (batch_size, c, h, w) -> (batch_size, num_queries, c)
        feat_last = feat_last.flatten(3).permute(0, 1, 3, 2)
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)
        # (batch_size, h, w) -> (batch_size, h*w)
        padding_mask = padding_mask.flatten(1)
        # import pdb; pdb.set_trace()
        memory = self.encoder(
            query=feat_last,
            query_pos=pos_embed,
            key_padding_mask=padding_mask)
        # (batch_size, num_queries, c) -> (batch_size, c, h, w)
        memory = memory.permute(0, 1, 3, 2).view(t, bs, self.encoder_embed_dims, h, w).contiguous()
        y = self.encoder_out_proj(memory.flatten(0, 1)).view(t, bs, self.encoder_embed_dims, h, w)

        out = []
        out.append(y)
        for i in range(self.num_inputs - 2, -1, -1):
            # import pdb; pdb.set_trace()
            x = feats[i]
            x = self.lateral_convs_spike[i](x)
            cur_feat = self.lateral_convs[i](x.flatten(0, 1))
            y = cur_feat + F.interpolate(
                y.flatten(0, 1),
                size=cur_feat.shape[-2:],
                mode='bilinear',
                align_corners=False)
            BS, C, H, W = y.shape
            y = y.reshape(t, bs, C, H, W)
            y = self.output_convs_spike[i](y)
            y = self.output_convs[i](y.flatten(0, 1)).reshape(t, bs, C, H, W)
            # import pdb; pdb.set_trace()
            out.append(y)

        y = self.mask_feature_spike(y)
        mask_feature = self.mask_feature(y.flatten(0, 1))
        tbs, C, H, W = mask_feature.shape
        mask_feature = mask_feature.reshape(t, bs, C, H, W)

        return mask_feature, memory, out[:3]


@MODELS.register_module()
class DCNTransformerEncoderPixelDecoder(PixelDecoder):
    """Pixel decoder with transormer encoder inside.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        feat_channels (int): Number channels for feature.
        out_channels (int): Number channels for output.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`ConfigDict` or dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`ConfigDict` or dict): Config for transformer encoder.
            Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer encoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: Union[List[int], Tuple[int]],
                 feat_channels: int,
                 out_channels: int,
                 T: int = 4,
                 norm_cfg: ConfigType = dict(type='GN', num_groups=32),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 encoder: ConfigType = None,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.last_feat_conv = None
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_inputs = len(in_channels)
        self.lateral_convs = ModuleList()
        self.lateral_convs_spike = ModuleList()
        self.output_convs = ModuleList()
        self.output_convs_spike = ModuleList()
        self.use_bias = norm_cfg is None
        self.T = T
        for i in range(0, self.num_inputs - 1):
            lateral_conv_spike = Q_IFNode(surrogate_function=Quant())
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels[i], feat_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(feat_channels)
            )

            output_conv_spike = Q_IFNode(surrogate_function=Quant())
            output_conv = nn.Sequential(
                nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1,
                          groups=feat_channels, bias=False),
                nn.BatchNorm2d(feat_channels)
            )

            self.lateral_convs.append(lateral_conv)
            self.lateral_convs_spike.append(lateral_conv_spike)
            self.output_convs.append(output_conv)
            self.output_convs_spike.append(output_conv_spike)

        self.mask_feature_spike = Q_IFNode(surrogate_function=Quant())  # 这里的norm对结果影响很大，因为没有BN层
        self.mask_feature = nn.Conv2d(feat_channels, out_channels, kernel_size=1, stride=1)

        # Change to DCNvs Based module
        self.encoder = DCNDetrTransformerEncoder(**encoder)
        self.encoder_embed_dims = self.encoder.embed_dims
        assert self.encoder_embed_dims == feat_channels, 'embed_dims({}) of ' \
                                                         'tranformer encoder must equal to feat_channels({})'.format(
            feat_channels, self.encoder_embed_dims)
        self.positional_encoding = SinePositionalEncoding(
            **positional_encoding)
        self.encoder_in_proj_spike = Q_IFNode(surrogate_function=Quant())
        self.encoder_in_proj = nn.Sequential(
            nn.Conv2d(in_channels[-1], feat_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(feat_channels)
        )
        self.encoder_out_proj_spike = Q_IFNode(surrogate_function=Quant())
        self.encoder_out_proj = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(feat_channels)
        )

    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, self.num_inputs - 2):
            caffe2_xavier_init(self.lateral_convs[i][0], bias=0)
            caffe2_xavier_init(self.output_convs[i][0], bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)
        caffe2_xavier_init(self.encoder_in_proj[0], bias=0)
        caffe2_xavier_init(self.encoder_out_proj[0], bias=0)

    def forward(self, feats: List[Tensor],
                batch_img_metas: List[dict]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).
            batch_img_metas (list[dict]): List of image information. Pass in
                for creating more accurate padding mask.

        Returns:
            tuple: a tuple containing the following:

                - mask_feature (Tensor): shape (batch_size, c, h, w).
                - memory (Tensor): shape (batch_size, c, h, w).
        """

        feat_last = feats[-1]  # [embed_dim, 32, 32]  bolttem featuremap
        t, bs, c, h, w = feat_last.shape
        feat_last = self.last_feat_conv_spike(feat_last)
        feat_last = self.encoder_in_proj(feat_last.flatten(0, 1)).reshape(t, bs, self.encoder_embed_dims, h, w)
        # (batch_size, c, h, w) -> (batch_size, num_queries, c)
        feat_last = feat_last.permute(0, 1, 3, 4, 2)  # (t, bs, c, h, w) -> (t, bs, h, w, c)
        # (batch_size, h, w) -> (batch_size, h*w)
        # import pdb; pdb.set_trace()
        memory = self.encoder(query=feat_last)
        # [bs, h, w, c]
        # (batch_size, num_queries, c) -> (batch_size, c, h, w)
        memory = memory.permute(0, 1, 4, 2, 3).view(t, bs, self.encoder_embed_dims, h, w).contiguous()
        memory = self.encoder_out_proj_spike(memory)
        y = self.encoder_out_proj(memory.flatten(0, 1)).reshape(t, bs, self.encoder_embed_dims, h, w)

        # utilize the [128, 64, 64], [64, 128, 128], [32, 256, 256] featuremap to get mask_features
        out = []
        out.append(y)
        for i in range(self.num_inputs - 2, -1, -1):
            # import pdb; pdb.set_trace()
            x = feats[i]
            x = self.lateral_convs_spike[i](x)
            cur_feat = self.lateral_convs[i](x.flatten(0, 1))
            y = cur_feat + F.interpolate(
                y.flatten(0, 1),
                size=cur_feat.shape[-2:],
                mode='bilinear',
                align_corners=False)
            BS, C, H, W = y.shape
            y = y.reshape(t, bs, C, H, W)
            y = self.output_convs_spike[i](y)
            y = self.output_convs[i](y.flatten(0, 1)).reshape(t, bs, C, H, W)
            out.append(y)

        y = self.mask_feature_spike(y)
        mask_feature = self.mask_feature(y.flatten(0, 1))
        tbs, C, H, W = mask_feature.shape
        mask_feature = mask_feature.reshape(t, bs, C, H, W)

        return mask_feature, memory, out[:3]
