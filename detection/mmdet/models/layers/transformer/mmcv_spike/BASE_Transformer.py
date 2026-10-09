# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/transformer.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

"""
Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from mmengine.model import BaseModule, ModuleList, caffe2_xavier_init

from mmdet.models.layers.transformer.mmcv_spike.transformer import MultiheadAttention, MSDA_FFN
from mmdet.models.utils.Qtrick import MultiSpike_norm4, MultiSpike_4


class Transformer(nn.Module):
    def __init__(
            self,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            Tokens=150,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = None
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.query_feat = nn.Embedding(Tokens, d_model)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                caffe2_xavier_init(p, bias=0)

    def forward(self, src, mask, query_embed, pos_embed, task_token=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(0, 2, 1)
        pos_embed = pos_embed.flatten(2)
        query_embed = query_embed.unsqueeze(0).repeat(bs, 1, 1)
        if mask is not None:
            mask = mask.flatten(1)

        # tgt = torch.zeros_like(query_embed)
        tgt = self.query_feat.weight.unsqueeze(0).repeat((bs, 1, 1))

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed
        )
        return hs, memory.permute(0, 2, 1).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        output = src

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
            self,
            tgt,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


# Spike-Drivern Manner
class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiheadAttention(embed_dims=d_model, num_heads=nhead, attn_type='norm')
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(embed_dims=d_model, num_heads=nhead, attn_type='norm')
        # Implementation of Feedforward model
        self.MLP = MSDA_FFN(embed_dims=d_model, feedforward_channels=dim_feedforward, num_fcs=2)
        self.layer_scale = True
        self.embed_dims = d_model
        self.layer_scale = 1e-6

        # Build layer scale to avoid the influence of query_initial
        if self.layer_scale:
            self.gamma1 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims),
                                       requires_grad=True)
            self.gamma3 = nn.Parameter(self.layer_scale * torch.ones(self.embed_dims),
                                       requires_grad=True)

    def forward_post(
            self,
            tgt,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
    ):
        tgt = tgt + self.gamma1 * self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=tgt_key_padding_mask
        )

        tgt = tgt + self.gamma2 * self.multihead_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            query_pos=query_pos,
            key_pos=pos.permute(0, 2, 1),
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.gamma3 * self.MLP(tgt)

        return tgt

    def forward(
            self,
            tgt,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
    ):
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return ModuleList([module for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
