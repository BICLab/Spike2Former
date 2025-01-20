# from visualizer import get_local
import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmengine.logging import print_log
from mmengine.runner import CheckpointLoader
from mmseg.models.utils.Qtrick import Multispike_norm
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode


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
            T=4,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.T = T
        self.spike1 = Multispike_norm()
        # self.spikex = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike2 = Multispike_norm()
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels,
                      bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike3 = Multispike_norm()
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        # import pdb; pdb.set_trace()
        x = self.spike1(x)

        x = self.pwconv1(x.flatten(0, 1)).reshape(T, B, -1, H, W)

        x = self.spike2(x)

        x = self.dwconv(x.flatten(0, 1)).reshape(T, B, -1, H, W)

        x = self.spike3(x)

        x = self.pwconv2(x.flatten(0, 1)).reshape(T, B, C, H, W)

        return x


class MS_ConvBlock_spike_SepConv(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.0,
            T=4,
    ):
        super().__init__()
        self.T = T
        self.Conv = SepConv_Spike(dim=dim)

        self.mlp_ratio = mlp_ratio

        self.spike1 = Multispike_norm()
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = Multispike_norm()
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
        self.fc1_spike = Multispike_norm()

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = Multispike_norm()

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


class MS_Attention_linear(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            sr_ratio=1,
            lamda_ratio=1,
    ):
        super().__init__()
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio

        self.head_spike = Multispike_norm()

        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = Multispike_norm()

        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.k_spike = Multispike_norm()

        self.v_conv = nn.Sequential(nn.Conv2d(dim, int(dim * lamda_ratio), 1, 1, bias=False),
                                    nn.BatchNorm2d(int(dim * lamda_ratio)))

        self.v_spike = Multispike_norm()

        self.attn_spike = Multispike_norm()

        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim * lamda_ratio, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        C_v = int(C * self.lamda_ratio)

        x = self.head_spike(x)

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C_v, H, W)

        q = self.q_spike(q)
        q = q.flatten(3)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k = self.k_spike(k)
        k = k.flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_spike(v)
        v = v.flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C_v // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        # import pdb;
        # pdb.set_trace()
        x = q @ k.transpose(-2, -1)
        x = (x @ v) * (self.scale * 2)

        x = x.transpose(3, 4).reshape(T, B, C_v, N).contiguous()
        x = self.attn_spike(x)
        x = x.reshape(T, B, C_v, H, W).flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)

        return x


class MS_Block_Spike_SepConv(nn.Module):
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
            init_values=1e-6
    ):
        super().__init__()

        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1)

        self.attn = MS_Attention_linear(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            lamda_ratio=4,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.attn(x)
        x = x + self.mlp(x)

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
            self.encode_spike = Multispike_norm()

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W)

        return x


@MODELS.register_module()
class Spiking_vit_MetaFormerv2(BaseModule):
    def __init__(
            self,
            img_size_h=128,
            img_size_w=128,
            patch_size=16,
            in_channels=2,
            num_classes=11,
            embed_dim=[64, 128, 256],
            num_heads=[1, 2, 4],
            mlp_ratios=[4, 4, 4],
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            depths=[6, 8, 6],
            sr_ratios=[8, 4, 2],
            T=1,
            decode_mode='QTrick',
            init_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            pretrained=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.depths = depths
        # embed_dim = [64, 128, 256, 512]
        self.T = T
        self.decode_mode = decode_mode
        self.freeze_bn_ = not norm_cfg['requires_grad']

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
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
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
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
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
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
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
                MS_Block_Spike_SepConv(
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
                for j in range(6)
            ]
        )

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,

        )

        self.block4 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    dim=embed_dim[3],
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
                for j in range(2)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            for name, param in m.named_parameters():
                param.requires_grad = True

    def init_weights(self):
        # logger = MMlogger.get_current_instance()
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                # 使用mmseg保存的checkpoint中包含backbone, neck, decode_head三个部分
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v
            # import pdb; pdb.set_trace()
            info = self.load_state_dict(state_dict, strict=False)
            print_log(info)
            print_log("--------------Successfully load checkpoint for BACKNONE------------")
            print_log("Time step: {:}".format(self.T))

    def forward_features(self, x):
        # import pdb; pdb.set_trace()
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        # BS, TC, H, W = x.shape
        # C = TC // self.T
        # x = x.reshape(self.T, BS, C, H, W)
        x = self.downsample1_1(x)  # [Conv + BN]
        # import pdb; pdb.set_trace()
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

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)
        x4 = x
        if self.decode_mode == 'QTrick':
            # import pdb; pdb.set_trace()
            return [x1, x2, x3, x4]
        return x  # T,B,C,N

    def forward(self, x):
        x = self.forward_features(x)  # B,C,H,W
        return x
