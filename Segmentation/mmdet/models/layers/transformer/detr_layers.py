# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmdet.models.layers.transformer.mmcv_spike.transformer \
    import MultiheadAttention, MSDA_FFN, MS_MLP, PEM_CA, MultiHeadCrossAttentionBlock
from mmengine import ConfigDict
from mmengine.model import BaseModule, ModuleList
from torch import Tensor
import torch.nn as nn

from mmdet.utils import ConfigType, OptConfigType
from mmdet.models.layers.transformer.ops_dcnv3.modules.dcnv3 import DCNv3_pytorch
from mmdet.models.layers.transformer.mmcv_spike.SNN_core import SepConv_Spike, DW1x1


class DCNDetrTransformerEncoder(BaseModule):
    """Encoder of DETR.

    Args:
        num_layers (int): Number of encoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DCNDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor) -> Tensor:
        """Forward function of encoder.

        Args:
            query (Tensor): Input queries of encoder, has shape
                (bs, num_queries, dim).

        Returns:
            Tensor: Has shape (bs, num_queries, dim) if `batch_first` is
            `True`, otherwise (num_queries, bs, dim).
        """
        for layer in self.layers:
            query = layer(query)
        return query


class DetrTransformerEncoder(BaseModule):
    """Encoder of DETR.

    Args:
        num_layers (int): Number of encoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of encoder.

        Args:
            query (Tensor): Input queries of encoder, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional embeddings of the queries, has
                shape (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).

        Returns:
            Tensor: Has shape (bs, num_queries, dim) if `batch_first` is
            `True`, otherwise (num_queries, bs, dim).
        """
        for layer in self.layers:
            query = layer(query, query_pos, key_padding_mask, **kwargs)
        return query


class DetrTransformerDecoder(BaseModule):
    """Decoder of DETR.

    Args:
        num_layers (int): Number of decoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        post_norm_cfg (:obj:`ConfigDict` or dict, optional): Config of the
            post normalization layer. Defaults to `LN`.
        return_intermediate (bool, optional): Whether to return outputs of
            intermediate layers. Defaults to `True`,
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 post_norm_cfg: OptConfigType = dict(type='LN'),
                 return_intermediate: bool = True,
                 init_cfg: Union[dict, ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.layer_cfg = layer_cfg
        self.num_layers = num_layers
        self.post_norm_cfg = post_norm_cfg
        self.return_intermediate = return_intermediate
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            DetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                query_pos: Tensor, key_pos: Tensor, key_padding_mask: Tensor,
                **kwargs) -> Tensor:
        """Forward function of decoder
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor): The input key, has shape (bs, num_keys, dim).
            value (Tensor): The input value with the same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).

        Returns:
            Tensor: The forwarded results will have shape
            (num_decoder_layers, bs, num_queries, dim) if
            `return_intermediate` is `True` else (1, bs, num_queries, dim).
        """
        intermediate = []
        for layer in self.layers:
            query = layer(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                **kwargs)
            if self.return_intermediate:
                intermediate.append(query)
        # query = self.post_norm(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query.unsqueeze(0)


# NOTE: Go This Branch
class DetrTransformerEncoderLayer(BaseModule):
    """Implements encoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = MSDA_FFN(**self.ffn_cfg)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = query + self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs)

        query = query + self.ffn(query)

        return query


# NOTE: Go This Branch
class DCNDetrTransformerEncoderLayer(BaseModule):
    """Implements encoder layer in DETR transformer. with DCNv3 alg

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self.layer_scale = 1e-6
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        # self.self_attn = MultiheadAttention(**self.self_attn_cfg) -> DCNv3
        self.embed_dims = self.self_attn_cfg.embed_dims
        self.Conv = SepConv_Spike(dim=self.embed_dims,
                                  kernel_size=3,
                                  padding=1,
                                  expansion_ratio=2)
        # self.Conv = DW1x1(dim=self.embed_dims)
        self.dcn = DCNv3_pytorch(
            channels=self.embed_dims,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=self.self_attn_cfg.group,
            offset_scale=1.0,
            expension_ratio=2,
            act_layer='GELU',
            norm_layer='BN',
            dw_kernel_size=self.self_attn_cfg.dw_kernel_size,  # for InternImage-H/G
            center_feature_scale=False,
        )
        self.ffn = MS_MLP(**self.ffn_cfg)

        if self.layer_scale:
            self.gamma1 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims), requires_grad=True)
            self.gamma2 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims), requires_grad=True)
            self.gamma3 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims), requires_grad=True)

    def forward(self, query: Tensor) -> Tensor:
        # import pdb; pdb.set_trace()
        query = query + self.gamma1 * self.Conv(query)
        query = query + self.gamma2 * self.dcn(query)
        query = query + self.gamma3 * self.ffn(query)
        return query


# NOTE: Go This Branch
class DetrTransformerEncoderLayer(BaseModule):
    """Implements encoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs)

        query = self.ffn(query)

        return query


# NOTE: Spike Drivern manner
class DetrTransformerDecoderLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 T: int = 4,
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.layer_scale = None
        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        # TODO: Change the MHSA to Spike former
        # import pdb; pdb.set_trace()
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiheadAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = MSDA_FFN(**self.ffn_cfg)
        if self.layer_scale:
            self.gamma1 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims), requires_grad=True)
            self.gamma2 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims), requires_grad=True)
            self.gamma3 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims), requires_grad=True)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query_ca = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        # query = query + query_ca * self.gamma2.unsqueeze(0).unsqueeze(0)
        query = query + query_ca

        query_sa = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        # query = query + query_sa * self.gamma1.unsqueeze(0).unsqueeze(0)

        query = query + query_sa

        query_ffn = self.ffn(query)
        # query = query + query_ffn * self.gamma3.unsqueeze(0).unsqueeze(0)
        query = query + query_ffn

        # import pdb; pdb.set_trace()
        return query


# NOTE: Go This Branch
class SpikeDetrTransformerEncoderLayer(BaseModule):
    """Implements encoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        # import pdb; pdb.set_trace()
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.ffn(query)

        return query

# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import Union
#
# import torch
# from mmcv.cnn import build_norm_layer
# from mmcv.cnn.bricks.transformer import FFN
# from mmdet.models.layers.transformer.mmcv_spike.transformer \
#     import MultiheadAttention, MSDA_FFN, MS_MLP, PEM_CA, MultiHeadCrossAttentionBlock
# from mmengine import ConfigDict
# from mmengine.model import BaseModule, ModuleList
# from torch import Tensor
# import torch.nn as nn
#
# from mmdet.utils import ConfigType, OptConfigType
# from mmdet.models.layers.transformer.ops_dcnv3.modules.dcnv3 import DCNv3_pytorch
# from mmdet.models.layers.transformer.mmcv_spike.SNN_core import SepConv_Spike, DW1x1
#
#
# class DCNDetrTransformerEncoder(BaseModule):
#     """Encoder of DETR.
#
#     Args:
#         num_layers (int): Number of encoder layers.
#         layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
#             layer. All the layers will share the same config.
#         init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
#             the initialization. Defaults to None.
#     """
#
#     def __init__(self,
#                  num_layers: int,
#                  layer_cfg: ConfigType,
#                  init_cfg: OptConfigType = None) -> None:
#         super().__init__(init_cfg=init_cfg)
#         self.num_layers = num_layers
#         self.layer_cfg = layer_cfg
#         self._init_layers()
#
#     def _init_layers(self) -> None:
#         """Initialize encoder layers."""
#         self.layers = ModuleList([
#             DCNDetrTransformerEncoderLayer(**self.layer_cfg)
#             for _ in range(self.num_layers)
#         ])
#         self.embed_dims = self.layers[0].embed_dims
#
#     def forward(self, query: Tensor, query_pos: Tensor,
#                 key_padding_mask: Tensor, **kwargs) -> Tensor:
#         """Forward function of encoder.
#
#         Args:
#             query (Tensor): Input queries of encoder, has shape
#                 (bs, num_queries, dim).
#
#         Returns:
#             Tensor: Has shape (bs, num_queries, dim) if `batch_first` is
#             `True`, otherwise (num_queries, bs, dim).
#         """
#         for layer in self.layers:
#             query = layer(query)
#         return query
#
#
# class DetrTransformerEncoder(BaseModule):
#     """Encoder of DETR.
#
#     Args:
#         num_layers (int): Number of encoder layers.
#         layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
#             layer. All the layers will share the same config.
#         init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
#             the initialization. Defaults to None.
#     """
#
#     def __init__(self,
#                  num_layers: int,
#                  layer_cfg: ConfigType,
#                  init_cfg: OptConfigType = None) -> None:
#         super().__init__(init_cfg=init_cfg)
#         self.num_layers = num_layers
#         self.layer_cfg = layer_cfg
#         self._init_layers()
#
#     def _init_layers(self) -> None:
#         """Initialize encoder layers."""
#         self.layers = ModuleList([
#             DetrTransformerEncoderLayer(**self.layer_cfg)
#             for _ in range(self.num_layers)
#         ])
#         self.embed_dims = self.layers[0].embed_dims
#
#     def forward(self, query: Tensor, query_pos: Tensor,
#                 key_padding_mask: Tensor, **kwargs) -> Tensor:
#         """Forward function of encoder.
#
#         Args:
#             query (Tensor): Input queries of encoder, has shape
#                 (bs, num_queries, dim).
#             query_pos (Tensor): The positional embeddings of the queries, has
#                 shape (bs, num_queries, dim).
#             key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
#                 input. ByteTensor, has shape (bs, num_queries).
#
#         Returns:
#             Tensor: Has shape (bs, num_queries, dim) if `batch_first` is
#             `True`, otherwise (num_queries, bs, dim).
#         """
#         for layer in self.layers:
#             query = layer(query, query_pos, key_padding_mask, **kwargs)
#         return query
#
#
# class DetrTransformerDecoder(BaseModule):
#     """Decoder of DETR.
#
#     Args:
#         num_layers (int): Number of decoder layers.
#         layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
#             layer. All the layers will share the same config.
#         post_norm_cfg (:obj:`ConfigDict` or dict, optional): Config of the
#             post normalization layer. Defaults to `LN`.
#         return_intermediate (bool, optional): Whether to return outputs of
#             intermediate layers. Defaults to `True`,
#         init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
#             the initialization. Defaults to None.
#     """
#
#     def __init__(self,
#                  num_layers: int,
#                  layer_cfg: ConfigType,
#                  post_norm_cfg: OptConfigType = dict(type='LN'),
#                  return_intermediate: bool = True,
#                  init_cfg: Union[dict, ConfigDict] = None) -> None:
#         super().__init__(init_cfg=init_cfg)
#         self.layer_cfg = layer_cfg
#         self.num_layers = num_layers
#         self.post_norm_cfg = post_norm_cfg
#         self.return_intermediate = return_intermediate
#         self._init_layers()
#
#     def _init_layers(self) -> None:
#         """Initialize decoder layers."""
#         self.layers = ModuleList([
#             DetrTransformerDecoderLayer(**self.layer_cfg)
#             for _ in range(self.num_layers)
#         ])
#         self.embed_dims = self.layers[0].embed_dims
#
#     def forward(self, query: Tensor, key: Tensor, value: Tensor,
#                 query_pos: Tensor, key_pos: Tensor, key_padding_mask: Tensor,
#                 **kwargs) -> Tensor:
#         """Forward function of decoder
#         Args:
#             query (Tensor): The input query, has shape (bs, num_queries, dim).
#             key (Tensor): The input key, has shape (bs, num_keys, dim).
#             value (Tensor): The input value with the same shape as `key`.
#             query_pos (Tensor): The positional encoding for `query`, with the
#                 same shape as `query`.
#             key_pos (Tensor): The positional encoding for `key`, with the
#                 same shape as `key`.
#             key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
#                 input. ByteTensor, has shape (bs, num_value).
#
#         Returns:
#             Tensor: The forwarded results will have shape
#             (num_decoder_layers, bs, num_queries, dim) if
#             `return_intermediate` is `True` else (1, bs, num_queries, dim).
#         """
#         intermediate = []
#         for layer in self.layers:
#             query = layer(
#                 query,
#                 key=key,
#                 value=value,
#                 query_pos=query_pos,
#                 key_pos=key_pos,
#                 key_padding_mask=key_padding_mask,
#                 **kwargs)
#             if self.return_intermediate:
#                 intermediate.append(query)
#         # query = self.post_norm(query)
#
#         if self.return_intermediate:
#             return torch.stack(intermediate)
#
#         return query.unsqueeze(0)
#
#
# # NOTE: Go This Branch
# class DetrTransformerEncoderLayer(BaseModule):
#     """Implements encoder layer in DETR transformer.
#
#     Args:
#         self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
#             attention.
#         ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
#         norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
#             normalization layers. All the layers will share the same
#             config. Defaults to `LN`.
#         init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
#             the initialization. Defaults to None.
#     """
#
#     def __init__(self,
#                  self_attn_cfg: OptConfigType = dict(
#                      embed_dims=256, num_heads=8, dropout=0.0),
#                  ffn_cfg: OptConfigType = dict(
#                      embed_dims=256,
#                      feedforward_channels=1024,
#                      num_fcs=2,
#                      ffn_drop=0.,
#                      act_cfg=dict(type='ReLU', inplace=True)),
#                  norm_cfg: OptConfigType = dict(type='LN'),
#                  init_cfg: OptConfigType = None) -> None:
#
#         super().__init__(init_cfg=init_cfg)
#
#         self.self_attn_cfg = self_attn_cfg
#         if 'batch_first' not in self.self_attn_cfg:
#             self.self_attn_cfg['batch_first'] = True
#         else:
#             assert self.self_attn_cfg['batch_first'] is True, 'First \
#             dimension of all DETRs in mmdet is `batch`, \
#             please set `batch_first` flag.'
#
#         self.ffn_cfg = ffn_cfg
#         self.norm_cfg = norm_cfg
#         self._init_layers()
#
#     def _init_layers(self) -> None:
#         """Initialize self-attention, FFN, and normalization."""
#         self.self_attn = MultiheadAttention(**self.self_attn_cfg)
#         self.embed_dims = self.self_attn.embed_dims
#         self.ffn = MSDA_FFN(**self.ffn_cfg)
#
#     def forward(self, query: Tensor, query_pos: Tensor,
#                 key_padding_mask: Tensor, **kwargs) -> Tensor:
#         """Forward function of an encoder layer.
#
#         Args:
#             query (Tensor): The input query, has shape (bs, num_queries, dim).
#             query_pos (Tensor): The positional encoding for query, with
#                 the same shape as `query`.
#             key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
#                 input. ByteTensor. has shape (bs, num_queries).
#         Returns:
#             Tensor: forwarded results, has shape (bs, num_queries, dim).
#         """
#         query = self.self_attn(
#             query=query,
#             key=query,
#             value=query,
#             query_pos=query_pos,
#             key_pos=query_pos,
#             key_padding_mask=key_padding_mask,
#             **kwargs)
#
#         query = self.ffn(query)
#
#         return query
#
#
# # NOTE: Go This Branch
# class DCNDetrTransformerEncoderLayer(BaseModule):
#     """Implements encoder layer in DETR transformer. with DCNv3 alg
#
#     Args:
#         self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
#             attention.
#         ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
#         norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
#             normalization layers. All the layers will share the same
#             config. Defaults to `LN`.
#         init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
#             the initialization. Defaults to None.
#     """
#
#     def __init__(self,
#                  self_attn_cfg: OptConfigType = dict(
#                      embed_dims=256, num_heads=8, dropout=0.0),
#                  ffn_cfg: OptConfigType = dict(
#                      embed_dims=256,
#                      feedforward_channels=1024,
#                      num_fcs=2,
#                      ffn_drop=0.,
#                      act_cfg=dict(type='ReLU', inplace=True)),
#                  norm_cfg: OptConfigType = dict(type='LN'),
#                  init_cfg: OptConfigType = None) -> None:
#
#         super().__init__(init_cfg=init_cfg)
#
#         self.self_attn_cfg = self_attn_cfg
#         if 'batch_first' not in self.self_attn_cfg:
#             self.self_attn_cfg['batch_first'] = True
#         else:
#             assert self.self_attn_cfg['batch_first'] is True, 'First \
#             dimension of all DETRs in mmdet is `batch`, \
#             please set `batch_first` flag.'
#
#         self.ffn_cfg = ffn_cfg
#         self.norm_cfg = norm_cfg
#         self.layer_scale = 1e-6
#         self._init_layers()
#
#     def _init_layers(self) -> None:
#         """Initialize self-attention, FFN, and normalization."""
#         # self.self_attn = MultiheadAttention(**self.self_attn_cfg) -> DCNv3
#         self.embed_dims = self.self_attn_cfg.embed_dims
#         self.Conv = SepConv_Spike(dim=self.embed_dims,
#                                   kernel_size=3,
#                                   padding=1,
#                                   expansion_ratio=2)
#         # self.Conv = DW1x1(dim=self.embed_dims)
#         self.dcn = DCNv3_pytorch(
#             channels=self.embed_dims,
#             kernel_size=3,
#             stride=1,
#             pad=1,
#             dilation=1,
#             group=self.self_attn_cfg.group,
#             offset_scale=1.0,
#             expension_ratio=2,
#             act_layer='GELU',
#             norm_layer='BN',
#             dw_kernel_size=self.self_attn_cfg.dw_kernel_size,  # for InternImage-H/G
#             center_feature_scale=False,
#         )
#         self.ffn = MS_MLP(**self.ffn_cfg)
#
#         if self.layer_scale:
#             self.gamma1 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims), requires_grad=True)
#             self.gamma2 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims), requires_grad=True)
#             self.gamma3 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims), requires_grad=True)
#
#     def forward(self, query: Tensor) -> Tensor:
#         query = query + self.gamma1 * self.Conv(query)
#         query = query + self.gamma2 * self.dcn(query)
#         query = query + self.gamma3 * self.ffn(query)
#         return query
#
#
# # NOTE: Go This Branch
# class DetrTransformerEncoderLayer(BaseModule):
#     """Implements encoder layer in DETR transformer.
#
#     Args:
#         self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
#             attention.
#         ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
#         norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
#             normalization layers. All the layers will share the same
#             config. Defaults to `LN`.
#         init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
#             the initialization. Defaults to None.
#     """
#
#     def __init__(self,
#                  self_attn_cfg: OptConfigType = dict(
#                      embed_dims=256, num_heads=8, dropout=0.0),
#                  ffn_cfg: OptConfigType = dict(
#                      embed_dims=256,
#                      feedforward_channels=1024,
#                      num_fcs=2,
#                      ffn_drop=0.,
#                      act_cfg=dict(type='ReLU', inplace=True)),
#                  norm_cfg: OptConfigType = dict(type='LN'),
#                  init_cfg: OptConfigType = None) -> None:
#
#         super().__init__(init_cfg=init_cfg)
#
#         self.self_attn_cfg = self_attn_cfg
#         if 'batch_first' not in self.self_attn_cfg:
#             self.self_attn_cfg['batch_first'] = True
#         else:
#             assert self.self_attn_cfg['batch_first'] is True, 'First \
#             dimension of all DETRs in mmdet is `batch`, \
#             please set `batch_first` flag.'
#
#         self.ffn_cfg = ffn_cfg
#         self.norm_cfg = norm_cfg
#         self._init_layers()
#
#     def _init_layers(self) -> None:
#         """Initialize self-attention, FFN, and normalization."""
#         self.self_attn = MultiheadAttention(**self.self_attn_cfg)
#         self.embed_dims = self.self_attn.embed_dims
#         self.ffn = FFN(**self.ffn_cfg)
#
#     def forward(self, query: Tensor, query_pos: Tensor,
#                 key_padding_mask: Tensor, **kwargs) -> Tensor:
#         """Forward function of an encoder layer.
#
#         Args:
#             query (Tensor): The input query, has shape (bs, num_queries, dim).
#             query_pos (Tensor): The positional encoding for query, with
#                 the same shape as `query`.
#             key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
#                 input. ByteTensor. has shape (bs, num_queries).
#         Returns:
#             Tensor: forwarded results, has shape (bs, num_queries, dim).
#         """
#         query = self.self_attn(
#             query=query,
#             key=query,
#             value=query,
#             query_pos=query_pos,
#             key_pos=query_pos,
#             key_padding_mask=key_padding_mask,
#             **kwargs)
#
#         query = self.ffn(query)
#
#         return query
#
#
# # NOTE: Spike Drivern manner
# class DetrTransformerDecoderLayer(BaseModule):
#     """Implements decoder layer in DETR transformer.
#
#     Args:
#         self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
#             attention.
#         cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
#             attention.
#         ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
#         norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
#             normalization layers. All the layers will share the same
#             config. Defaults to `LN`.
#         init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
#             the initialization. Defaults to None.
#     """
#
#     def __init__(self,
#                  self_attn_cfg: OptConfigType = dict(
#                      embed_dims=256,
#                      num_heads=8,
#                      dropout=0.0,
#                      batch_first=True),
#                  cross_attn_cfg: OptConfigType = dict(
#                      embed_dims=256,
#                      num_heads=8,
#                      dropout=0.0,
#                      batch_first=True),
#                  ffn_cfg: OptConfigType = dict(
#                      embed_dims=256,
#                      feedforward_channels=1024,
#                      num_fcs=2,
#                      ffn_drop=0.,
#                      act_cfg=dict(type='ReLU', inplace=True),
#                  ),
#                  T: int = 4,
#                  norm_cfg: OptConfigType = dict(type='LN'),
#                  init_cfg: OptConfigType = None) -> None:
#
#         super().__init__(init_cfg=init_cfg)
#
#         self.self_attn_cfg = self_attn_cfg
#         self.cross_attn_cfg = cross_attn_cfg
#         if 'batch_first' not in self.self_attn_cfg:
#             self.self_attn_cfg['batch_first'] = True
#         else:
#             assert self.self_attn_cfg['batch_first'] is True, 'First \
#             dimension of all DETRs in mmdet is `batch`, \
#             please set `batch_first` flag.'
#
#         if 'batch_first' not in self.cross_attn_cfg:
#             self.cross_attn_cfg['batch_first'] = True
#         else:
#             assert self.cross_attn_cfg['batch_first'] is True, 'First \
#             dimension of all DETRs in mmdet is `batch`, \
#             please set `batch_first` flag.'
#
#         self.layer_scale = None
#         self.ffn_cfg = ffn_cfg
#         self.norm_cfg = norm_cfg
#         self._init_layers()
#
#     def _init_layers(self) -> None:
#         # TODO: Change the MHSA to Spike former
#         # import pdb; pdb.set_trace()
#         """Initialize self-attention, FFN, and normalization."""
#         self.self_attn = MultiheadAttention(**self.self_attn_cfg)
#         self.cross_attn = MultiheadAttention(**self.cross_attn_cfg)
#         self.embed_dims = self.self_attn.embed_dims
#         self.ffn = MSDA_FFN(**self.ffn_cfg)
#         if self.layer_scale:
#             self.gamma1 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims), requires_grad=True)
#             self.gamma2 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims), requires_grad=True)
#             self.gamma3 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims), requires_grad=True)
#
#     def forward(self,
#                 query: Tensor,
#                 key: Tensor = None,
#                 value: Tensor = None,
#                 query_pos: Tensor = None,
#                 key_pos: Tensor = None,
#                 self_attn_mask: Tensor = None,
#                 cross_attn_mask: Tensor = None,
#                 key_padding_mask: Tensor = None,
#                 **kwargs) -> Tensor:
#         """
#         Args:
#             query (Tensor): The input query, has shape (bs, num_queries, dim).
#             key (Tensor, optional): The input key, has shape (bs, num_keys,
#                 dim). If `None`, the `query` will be used. Defaults to `None`.
#             value (Tensor, optional): The input value, has the same shape as
#                 `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
#                 `key` will be used. Defaults to `None`.
#             query_pos (Tensor, optional): The positional encoding for `query`,
#                 has the same shape as `query`. If not `None`, it will be added
#                 to `query` before forward function. Defaults to `None`.
#             key_pos (Tensor, optional): The positional encoding for `key`, has
#                 the same shape as `key`. If not `None`, it will be added to
#                 `key` before forward function. If None, and `query_pos` has the
#                 same shape as `key`, then `query_pos` will be used for
#                 `key_pos`. Defaults to None.
#             self_attn_mask (Tensor, optional): ByteTensor mask, has shape
#                 (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
#                 Defaults to None.
#             cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
#                 (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
#                 Defaults to None.
#             key_padding_mask (Tensor, optional): The `key_padding_mask` of
#                 `self_attn` input. ByteTensor, has shape (bs, num_value).
#                 Defaults to None.
#
#         Returns:
#             Tensor: forwarded results, has shape (bs, num_queries, dim).
#         """
#         T, B, C, N = query.shape
#         query_sa = self.self_attn(
#             query=query,
#             key=query,
#             value=query,
#             query_pos=query_pos,
#             key_pos=query_pos,
#             attn_mask=self_attn_mask,
#             **kwargs)
#         # query = query + query_sa * self.gamma1.unsqueeze(0).unsqueeze(0)
#
#         query = query + query_sa.reshape(T, B, C, N)
#
#         query_ca = self.cross_attn(
#             query=query,
#             key=key,
#             value=value,
#             query_pos=query_pos,
#             key_pos=key_pos,
#             attn_mask=cross_attn_mask,
#             key_padding_mask=key_padding_mask,
#             **kwargs)
#         # query = query + query_ca * self.gamma2.unsqueeze(0).unsqueeze(0)
#         query = query + query_ca.reshape(T, B, C, N)
#
#         query_ffn = self.ffn(query.flatten(0, 1))
#         # query = query + query_ffn * self.gamma3.unsqueeze(0).unsqueeze(0)
#         query = query + query_ffn.reshape(T, B, C, N)
#
#         return query
#
#
# # NOTE: Go This Branch
# class SpikeDetrTransformerEncoderLayer(BaseModule):
#     """Implements encoder layer in DETR transformer.
#
#     Args:
#         self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
#             attention.
#         ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
#         norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
#             normalization layers. All the layers will share the same
#             config. Defaults to `LN`.
#         init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
#             the initialization. Defaults to None.
#     """
#
#     def __init__(self,
#                  self_attn_cfg: OptConfigType = dict(
#                      embed_dims=256, num_heads=8, dropout=0.0),
#                  ffn_cfg: OptConfigType = dict(
#                      embed_dims=256,
#                      feedforward_channels=1024,
#                      num_fcs=2,
#                      ffn_drop=0.,
#                      act_cfg=dict(type='ReLU', inplace=True)),
#                  norm_cfg: OptConfigType = dict(type='LN'),
#                  init_cfg: OptConfigType = None) -> None:
#
#         super().__init__(init_cfg=init_cfg)
#
#         self.self_attn_cfg = self_attn_cfg
#         if 'batch_first' not in self.self_attn_cfg:
#             self.self_attn_cfg['batch_first'] = True
#         else:
#             assert self.self_attn_cfg['batch_first'] is True, 'First \
#             dimension of all DETRs in mmdet is `batch`, \
#             please set `batch_first` flag.'
#
#         self.ffn_cfg = ffn_cfg
#         self.norm_cfg = norm_cfg
#         self._init_layers()
#
#     def _init_layers(self) -> None:
#         """Initialize self-attention, FFN, and normalization."""
#         self.self_attn = MultiheadAttention(**self.self_attn_cfg)
#         self.embed_dims = self.self_attn.embed_dims
#         self.ffn = FFN(**self.ffn_cfg)
#
#     def forward(self, query: Tensor, query_pos: Tensor,
#                 key_padding_mask: Tensor, **kwargs) -> Tensor:
#         """Forward function of an encoder layer.
#
#         Args:
#             query (Tensor): The input query, has shape (bs, num_queries, dim).
#             query_pos (Tensor): The positional encoding for query, with
#                 the same shape as `query`.
#             key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
#                 input. ByteTensor. has shape (bs, num_queries).
#         Returns:
#             Tensor: forwarded results, has shape (bs, num_queries, dim).
#         """
#         # import pdb; pdb.set_trace()
#         query = self.self_attn(
#             query=query,
#             key=query,
#             value=query,
#             query_pos=query_pos,
#             key_pos=query_pos,
#             key_padding_mask=key_padding_mask,
#             **kwargs)
#         query = self.ffn(query)
#
#         return query
