"""
Various positional encodings for the transformer.
"""
import torch
import torch
import torch.nn as nn
from einops import rearrange, repeat

class PositionEmbeddingLearned3D(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(
        self,
        dim_head
    ):
        super().__init__()
        self.resolution = 16
        scale = dim_head ** -0.5
        self.outputPos = nn.Parameter(torch.randn(1, dim_head) * scale)
        self.posEmbeddingList = nn.Parameter(torch.randn(self.resolution**3, dim_head) * scale)


    def forward(self, x, y, z):
        x= torch.round(x*(self.resolution-1)).long()
        y= torch.round(y*(self.resolution-1)).long()
        z= torch.round(z*(self.resolution-1)).long()
        b, n = x.shape
        ouputPos = repeat(self.outputPos, 'n d -> b n d', b = b)
        pos = x*self.resolution**2+y*self.resolution+z
        emb = self.posEmbeddingList[pos,:]
        emb = torch.cat((ouputPos, emb), dim=1)
        return emb

def build_position_encoding(args):
    dimhead = args.hidden_dim
    if args.position_embedding=='3Dlearned':
        position_embedding = PositionEmbeddingLearned3D(dimhead)
    else:
        raise ValueError(f"not supported {args.position_embedding}")
    return position_embedding