import torch.nn as nn
from timm.models.layers import trunc_normal_
from mmdet.models.utils.Qtrick import MultiSpike_norm4, MultiSpike_4
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from Qtrick_architecture.clock_driven.neuron import Q_IFNode
from Qtrick_architecture.clock_driven.surrogate import Quant
import torch
import torch.nn.functional as F


class SepConv_Spike(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
            self,
            dim,
            expansion_ratio=2,
            T=4,
            act2_layer=nn.Identity,
            bias=False,
            kernel_size=7,
            padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.T = T
        self.expansion_ratio = expansion_ratio
        self.spike1 = Q_IFNode(surrogate_function=Quant())
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike2 = Q_IFNode(surrogate_function=Quant())
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels,
                      bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike3 = Q_IFNode(surrogate_function=Quant())
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, H, W, C = x.shape
        # import pdb; pdb.set_trace()
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = self.spike1(x)

        x = self.pwconv1(x.flatten(0, 1)).reshape(T, B, C * self.expansion_ratio, H, W)

        x = self.spike2(x)

        x = self.dwconv(x.flatten(0, 1)).reshape(T, B, C * self.expansion_ratio, H, W)

        x = self.spike3(x)

        x = self.pwconv2(x.flatten(0, 1)).reshape(T, B, C, H, W)

        return x.permute(0, 1, 3, 4, 2).contiguous()


class DW1x1(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
            self,
            dim,
            T=4,
    ):
        super().__init__()
        self.T = T
        self.spike1 = Q_IFNode(surrogate_function=Quant())
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        # N, H, W, C -> N, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.spike1(x)

        x = self.dwconv1(x)

        return x.permute(0, 2, 3, 1).contiguous()



class MLP(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            layer,
            quant_const=4,
            T=4,
    ):
        super().__init__()
        self.T = T
        self.fc1 = nn.Linear(in_dim, in_dim, bias=False)
        self.spike1 = Q_IFNode(surrogate_function=Quant())
        self.fc2 = nn.Linear(in_dim, in_dim, bias=False)
        self.spike2 = Q_IFNode(surrogate_function=Quant())
        self.fc_out = nn.Linear(in_dim, out_dim)
        self.quant_const = quant_const
        # import pdb; pdb.set_trace()
        nn.init.constant_(self.fc_out.bias, 0)
        trunc_normal_(self.fc_out.weight, std=0.02)

    def forward(self, x):
        # ln, t, bs, nq, embed_dim = x.shape * self.quant_const
        x = self.fc1(x)
        x = self.spike1(x) * self.quant_const
        x = self.fc2(x)
        x = self.spike2(x) * self.quant_const
        x = self.fc_out(x)
        return x

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
