# from visualizer import get_local
import torch
import torchinfo
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial

class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value=0, max_value=8):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None

class SFA_neuron(nn.Module):
    def __init__(
        self,
        Norm = 8,
    ):
        super().__init__()
        self.spikeActFun = Quant()
        self.Norm = Norm
    
    def __repr__(self):
        return f"SFA_neuron(Norm={self.Norm})"
    
    def forward(self, input, init_v=None):
        self.batchSize = input.size()[0]

        if not hasattr(self, "h"):
            if init_v is None:
                self.h = torch.zeros_like(input ,device=input.device)
            else:
                self.h = init_v.clone()

        if input.device != self.h.device:
            input = input.to(self.h.device)

        u = self.h + input # 输入膜电位

        x = self.spikeActFun.apply(u) # 每个时间步上发放的脉冲总数
        
        self.h = u - x # soft reset后的状态

        return x / self.Norm

    def reset(self):
        if hasattr(self, "h"):
            del self.h


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
        self.spike1 = layer.MultiStepContainer(SFA_neuron())
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
            )
        self.spike2 = layer.MultiStepContainer(SFA_neuron())
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels, bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike3 = layer.MultiStepContainer(SFA_neuron())
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):

        T, B, C, H, W = x.shape
        
        x = self.spike1(x)
            
        x = self.pwconv1(x.flatten(0,1)).reshape(T, B, -1, H, W)
        
        x = self.spike2(x)
            
        x = self.dwconv(x.flatten(0,1)).reshape(T, B, -1, H, W)

        x = self.spike3(x)

        x = self.pwconv2(x.flatten(0,1)).reshape(T, B, C, H, W)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        
    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim)

        self.mlp_ratio = mlp_ratio

        self.spike1 = layer.MultiStepContainer(SFA_neuron())
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = layer.MultiStepContainer(SFA_neuron())
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)
        x = self.bn1(self.conv1(x.flatten(0,1))).reshape(T, B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)
        x = self.bn2(self.conv2(x.flatten(0,1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x

class MS_ConvBlock_1x1conv(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        
    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim)

        self.mlp_ratio = mlp_ratio

        self.spike1 = layer.MultiStepContainer(SFA_neuron())
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = layer.MultiStepContainer(SFA_neuron())
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)
        x = self.bn1(self.conv1(x.flatten(0,1))).reshape(T, B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)
        x = self.bn2(self.conv2(x.flatten(0,1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = layer.MultiStepContainer(SFA_neuron())

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = layer.MultiStepContainer(SFA_neuron())

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_spike(x)
        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
        x = self.fc2_spike(x)
        x = self.fc2_conv(x.flatten(0,1))
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
        self.scale = (dim//num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio

        self.head_spike = layer.MultiStepContainer(SFA_neuron())

        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = layer.MultiStepContainer(SFA_neuron())

        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.k_spike = layer.MultiStepContainer(SFA_neuron())

        self.v_conv = nn.Sequential(nn.Conv2d(dim, int(dim*lamda_ratio), 1, 1, bias=False), nn.BatchNorm2d(int(dim*lamda_ratio)))
        
        self.v_spike = layer.MultiStepContainer(SFA_neuron())

        self.attn_spike = layer.MultiStepContainer(SFA_neuron())

        # self.proj_conv = nn.Sequential(
        #     RepConv(dim*lamda_ratio, dim, bias=False), nn.BatchNorm2d(dim)
        # )

        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim*lamda_ratio, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )


    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        C_v = int(C*self.lamda_ratio)

        x = self.head_spike(x)

        q = self.q_conv(x.flatten(0,1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0,1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0,1)).reshape(T, B, C_v, H, W)

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

        x = q @ k.transpose(-2, -1)
        x = (x @ v) * (self.scale*2)

        x = x.transpose(3, 4).reshape(T, B, C_v, N).contiguous()
        x = self.attn_spike(x)
        x = x.reshape(T, B, C_v, H, W)
        x = self.proj_conv(x.flatten(0,1)).reshape(T, B, C, H, W)

        return x

class MS_Block(nn.Module):
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
        T=None
    ):
        super().__init__()

        self.attn = MS_Attention_linear(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            lamda_ratio=4
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.layer_scale1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.layer_scale2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.attn(x) * self.layer_scale1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = x + self.mlp(x) * self.layer_scale2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

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
            self.encode_spike = layer.MultiStepContainer(SFA_neuron())

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W)

        return x


class Efficient_SpikeFormer_scaling(nn.Module):
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
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        # embed_dim = [64, 128, 256, 512]
        self.T = 16

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

        self.block3 = nn.ModuleList(
            [
                MS_Block(
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
                for j in range(int(depths*0.75))
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
                MS_Block(
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
                for j in range(int(depths*0.25))
            ]
        )
        
        self.head = (
            nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        )
        self.spike = layer.MultiStepContainer(SFA_neuron(Norm=1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)

        return x  # T,B,C,N

    def forward(self, x):
        # x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = x.transpose(0,1)
        x = self.forward_features(x) # T,B,C,H,W
        x = x.flatten(3).mean(3)
        x = self.spike(x)
        x = self.head(x).mean(0)
        return x
    
def Efficient_Spiking_Transformer_scaling_4_10M_gesture(**kwargs):
    model = Efficient_SpikeFormer_scaling(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        num_classes=11,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=4,
        sr_ratios=1,
        **kwargs,
    )
    return model

def Efficient_Spiking_Transformer_scaling_4_10M_gait_day(**kwargs):
    model = Efficient_SpikeFormer_scaling(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        num_classes=20,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=4,
        sr_ratios=1,
        **kwargs,
    )
    return model

def Efficient_Spiking_Transformer_scaling_4_10M_gait_night(**kwargs):
    model = Efficient_SpikeFormer_scaling(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        num_classes=20,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=4,
        sr_ratios=1,
        **kwargs,
    )
    return model


def Efficient_Spiking_Transformer_scaling_8_30M(**kwargs):
    model = Efficient_SpikeFormer_scaling(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[96, 192, 384, 480],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model

def Efficient_Spiking_Transformer_scaling_8_50M(**kwargs):
    model = Efficient_SpikeFormer_scaling(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[120, 240, 480, 600],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model


import time
if __name__ == "__main__":
    model = Efficient_Spiking_Transformer_scaling_4_30M()
    print(model)
    x = torch.randn(1,2,128,128)
    y = model(x)
    torchinfo.summary(model, (1, 2, 128, 128), device='cpu')