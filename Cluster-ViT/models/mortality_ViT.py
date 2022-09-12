"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .transformer import build_transformer
from .position_encoding import build_position_encoding
from einops import rearrange, repeat
class mortality_ViT(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, transformer, position_embedding, max_num_cluster, withPosEmbedding, seq_pool,withLN=False,withEmbeddingPreNorm = False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.withLN = withLN
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.embedding_proj = nn.Linear(512,hidden_dim)
        self.position_embedding = position_embedding
        self.survival_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.withEmbeddingPreNorm = withEmbeddingPreNorm
        if not self.withLN:
            self.mlp_head = nn.Linear(hidden_dim, 1)
            self.mlp_head_MSE = nn.Linear(hidden_dim, 1)
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
            self.mlp_head_MSE = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
        self.max_num_cluster = max_num_cluster
        self.survival_score_cluster = nn.Parameter(torch.randn(self.max_num_cluster+1))
        self.maxSurvivalTime = 13
        self.timeInterval = 0.1
        self.baseline_hazard_survival = nn.Parameter(torch.rand(int(self.maxSurvivalTime/self.timeInterval+1)))
        self.norm_Embedding = nn.LayerNorm(hidden_dim)
        self.withPosEmbedding=withPosEmbedding
        self.seq_pool=seq_pool
        print(0)

    def predictSurvivalTime(self, survivalRisk):
        cumsum_baseline_hazard = torch.cumsum(self.baseline_hazard_survival, 0)
        baseline_survival = torch.exp(-cumsum_baseline_hazard*self.timeInterval)
        survival_t = torch.pow(repeat(baseline_survival,'t -> t b', b = len(survivalRisk)), repeat(torch.exp(survivalRisk),'b -> t b', t = len(baseline_survival)))
        predictedSurvivalTime = torch.sum(survival_t,dim=0)* self.timeInterval 
        return predictedSurvivalTime  

    def forward(self, patientEmbedding, pos, keyPaddingMask, cluster):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        patientEmbedding = self.embedding_proj(patientEmbedding)
        if not self.withEmbeddingPreNorm:
            x = patientEmbedding
        else:
            x=self.norm_Embedding(patientEmbedding)     
        b, n, _ = x.shape
        device = cluster.device

        survival_token = repeat(self.survival_token, '() n d -> b n d', b = b)
        x = torch.cat((survival_token, x), dim=1)
        cluster_survival_token = self.max_num_cluster*torch.ones([b,1],dtype=torch.int64).to(device)
        cluster = torch.cat((cluster_survival_token, cluster), dim=1)
        tokenPadding = torch.zeros(b,1).to(device)
        keyPaddingMask = torch.cat((tokenPadding, keyPaddingMask), dim=1)
        if self.withPosEmbedding:
            emb = self.position_embedding(pos[:,:,0],pos[:,:,1],pos[:,:,2])
            x = x+emb
        x = x.transpose(0, 1)
# temp type conversion
        cluster = cluster.long()
        keyPaddingMask = keyPaddingMask.bool()
        x, A = self.transformer(x, mask=None, src_key_padding_mask=keyPaddingMask,clusters = cluster, pos=pos)

        # survival token outputs survival time, others output survival risk
        if self.seq_pool:
            keyPaddingMask = rearrange(keyPaddingMask,'b n -> n b')
            keyPaddingMask = repeat(keyPaddingMask, 'n b -> n b d', d = 1)
            x = x[1:x.shape[0]]
            survival_score = self.mlp_head(x)
            survival_score = survival_score.masked_fill(keyPaddingMask[1:keyPaddingMask.shape[0]], -1e15)
            attentionWeight = F.softmax(survival_score, dim=0)
            attentionWeight = rearrange(attentionWeight, 'n b d -> b d n')
            survival_score_temp=rearrange(survival_score, 'n b d -> b n d')
            survival_score = survival_score.squeeze(2)
            survivalRisk = torch.matmul(attentionWeight, survival_score_temp).squeeze()            
        else:
            survival_score = self.mlp_head(x[1:x.shape[0]]).squeeze(2) 
            survivalRisk = (survival_score).mean(dim = 0)

        return survivalRisk,survival_score,A

class SetCriterion(nn.Module):
    def __init__(self):
        super(SetCriterion, self).__init__()
        self.loss_dict = {}
        self.weight_dict = {'neg_likelihood':1}
        self.survival_score_Loss = nn.MSELoss()

    def forward(self, outputs: Tensor, survivalTimes: Tensor, Dead: Tensor) -> Tensor:
        """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
        We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
        where h = exp(log_h) are the hazards and R is the risk set, and d is event.
        We just compute a cumulative sum, and not the true Risk sets. This is a
        limitation, but simple and fast.
        """
        survivalRisk,survival_score,A = outputs
        idx = survivalTimes.sort(descending=True)[1]
        events = Dead[idx]
        log_h = survivalRisk[idx]
        neg_likelihood=self.cox_ph_loss_sorted(log_h, events)
        self.loss_dict = {'neg_likelihood':neg_likelihood}

        return self.loss_dict

    def cox_ph_loss_sorted(self, log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
        """Requires the input to be sorted by descending duration time.
        See DatasetDurationSorted.
        We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
        where h = exp(log_h) are the hazards and R is the risk set, and d is event.
        We just compute a cumulative sum, and not the true Risk sets. This is a
        limitation, but simple and fast.
        """
        if events.dtype is torch.bool:
            events = events.float()
        events = events.view(-1)
        log_h = log_h.view(-1)
        gamma = log_h.max()
        log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
        return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum()+eps)

def build_mortality_ViT(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    device = torch.device(args.device)
    max_num_cluster = args.max_num_cluster

    transformer = build_transformer(args)

    position_embedding = build_position_encoding(args)
    model = mortality_ViT(transformer,position_embedding,max_num_cluster,args.withPosEmbedding, seq_pool=args.seq_pool, withLN=args.withLN, withEmbeddingPreNorm=args.withEmbeddingPreNorm)
    criterion = SetCriterion()
    criterion.to(device)
    return model, criterion  
