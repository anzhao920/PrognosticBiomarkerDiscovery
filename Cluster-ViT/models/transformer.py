"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch.nn.functional as F
from torch import nn, Tensor

from .attention_layer import AdaMultiheadAttention

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu", 
                 normalize_before=False,group_Q=True, 
                 group_K=False, number_clusters = 128):
        super().__init__()

        encoder_layers = _get_clones(TransformerEncoderLayer(d_model, nhead, 
                                dim_feedforward, dropout, activation, normalize_before, 
                                group_Q, group_K, number_clusters), num_encoder_layers)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layers, encoder_norm)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:  
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                clusters: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # flatten NxCxHxW to HWxNxC

        output, A = self.encoder(src, mask, src_key_padding_mask,clusters,pos)

        return output, A


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layers, norm=None):
        super().__init__()
        self.layers = encoder_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                clusters: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, clusters = clusters)

        if self.norm is not None:
            output = (self.norm(output[0]),output[1])

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, 
                 group_Q=False, group_K=False,number_clusters=128):
        super().__init__()
        self.self_attn = AdaMultiheadAttention(d_model, nhead, 
                group_Q, group_K, number_clusters, dropout=dropout)
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

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                clusters: Optional[Tensor] = None):
        if type(src) is tuple:
            src = src[0]
        src2 = self.norm1(src)
        q = k = src2
        temp = self.self_attn(q, k, src2, clusters, key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        src2 = temp[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, temp[2]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        group_Q=args.group_Q,
        group_K=args.group_K
    )
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
