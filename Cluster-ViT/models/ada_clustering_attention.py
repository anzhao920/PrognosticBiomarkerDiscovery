from .extensions import broadcast, weighted_sum

import torch
from torch.nn import Module, Dropout

class WeightedSoftMax(Module):
    def __init__(self):
        super(WeightedSoftMax, self).__init__()
    
    def forward(self, x, dim=None, weight=None):
        ret = torch.softmax(x, dim=dim)
        if weight is not None:
            ret = ret * weight.unsqueeze(1)
            ret = ret / ret.sum(dim=-1, keepdim=True)
        return ret


class CalcCenter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clusters, counts):
        weights = 1/counts.float()
        center = weighted_sum(x, clusters, weights)
        ctx.save_for_backward(clusters, weights)

        return center

    @staticmethod
    def backward(ctx, grad_center):
        clusters, weights = ctx.saved_tensors
        grad = broadcast(grad_center, clusters, weights)

        return grad, None, None


class Broadcast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, center, clusters):
        B, C, D = center.shape
        weights = torch.ones((B, C), dtype=torch.float, device=center.device)
        x = broadcast(center, clusters, weights)
        ctx.save_for_backward(clusters,torch.tensor(C))
        
        return x

    @staticmethod
    def backward(ctx, grad):
        B, N, D = grad.shape
        clusters = ctx.saved_tensors[0]
        C = ctx.saved_tensors[1].int()
        weights = torch.ones((B, C), dtype=torch.float, device=grad.device)
        grad_center = weighted_sum(grad, clusters, weights)

        return grad_center, None, None

class AdaClusteringAttention(Module):
    """Use E2LSH to adaptively cluster queries or keys

    Arguments
    ---------
        group_Q: If true, use E2LSH to adaptively cluster queries
        group_K: If true, use E2LSH to adaptively cluster keys
        softmax_temp: The temperature to use for the softmax attention
        attention_dropout: The dropout rate to apply to the attention
    """
    def __init__(self, group_Q=False, group_K=False, softmax_temp=1, number_clusters=128, attention_dropout=0.0):
        super(AdaClusteringAttention, self).__init__()
        self.group_Q = group_Q
        self.group_K = group_K
        self.softmax_temp = softmax_temp
        self.number_clusters=number_clusters
        if attention_dropout > 0.0:
            self.dropout = Dropout(attention_dropout)
        else:
            self.dropout = None
        self.softmax = WeightedSoftMax()

    def _create_clusters(self, V, clusters):
        B, N, D = V.shape

        groups=clusters.int()
        C = self.number_clusters+1
        counts = torch.zeros((clusters.shape[0], C), dtype=torch.int, device=clusters.device)
        for i in range(0,groups.shape[0]):
            counts[i,:] = groups[i,:].bincount(minlength=C)
        groups = groups.repeat(int(B/clusters.shape[0]),1)
        counts = counts.repeat(int(B/clusters.shape[0]),1)
       
        return groups, counts.contiguous()

    def forward(self, queries, keys, values, clusters, key_padding_mask=None):
        
        if self.group_Q:
            q_groups, q_counts = self._create_clusters(queries, clusters)
            Q_center = CalcCenter.apply(queries, q_groups, q_counts)
            self.Q_clusters = q_counts.size(-1) # number of clusters
        else:
            Q_center = queries

        if self.group_K:
            k_groups, k_counts = self._create_clusters(keys, clusters)
            K_center = CalcCenter.apply(keys, k_groups, k_counts)
            V_center = CalcCenter.apply(values, k_groups, k_counts)
            self.K_clusters = k_counts.size(-1) # number of clusters
        else:
            K_center = keys
            V_center = values

        QK = torch.bmm(Q_center, K_center.permute(0, 2, 1))
        if key_padding_mask is not None:
            assert self.group_K is not True
            QK = QK.view(key_padding_mask.size(0), -1, Q_center.size(1), keys.size(1))
            QK = QK.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            QK = QK.view(-1, Q_center.size(1), keys.size(1))

        softmax_weight = k_counts if self.group_K else None
        A_full = self.softmax(self.softmax_temp * QK, dim=-1, weight=softmax_weight)
        if self.dropout:
            A = self.dropout(A_full)
        else:
            A = A_full
        
        V = torch.bmm(A, V_center)
        if self.group_Q:
            V = Broadcast.apply(V, q_groups)

        return V.contiguous(),A_full[:,:,0]
