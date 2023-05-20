import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def assign_label_group(label_group_dict, hs, preds, labs):
    """
    h: (N, 1024)
    aro: [1-3) [3-5) [5-)
    dom: [1-3) [3-5) [5-)
    val: [1-3) [3-5) [5-)

    preds and labs are already normalized
    """
    def get_group_idx(score):
        if score < 3:
            return 0
        elif score < 5:
            return 1
        else:
            return 2

    batch_len = hs.size(0)
    for batch_idx in range(batch_len):
        cur_h = hs[batch_idx]
        cur_pred = preds[batch_idx]*6 + 1
        cur_lab = labs[batch_idx]*6 + 1
        pred_group_idx = get_group_idx(cur_pred[0])*9 + get_group_idx(cur_pred[1])*3 + get_group_idx(cur_pred[2])
        lab_group_idx = get_group_idx(cur_lab[0])*9 + get_group_idx(cur_lab[1])*3 + get_group_idx(cur_lab[2])

        label_group_dict[pred_group_idx][lab_group_idx].append(cur_h.detach()) #.cpu().numpy())

from collections import defaultdict
def generate_negative_idxs(lab, margin=0.5):
    # lab: [B, 3]
    # output: [B] => [indices]
    margin /= 6

    batch_num = lab.size(0)
    neg_idxs = defaultdict(list)
    for pos_idx in range(batch_num):
        pos_lab = lab[pos_idx]
        for neg_idx in range(pos_idx+1, batch_num):
            neg_lab = lab[neg_idx]

            is_neg = True
            for attr_idx in range(3):
                attr_diff = torch.abs(pos_lab[attr_idx]-neg_lab[attr_idx])
                if attr_diff < margin:
                    is_neg=False
            if is_neg:
                neg_idxs[pos_idx].append(neg_idx)
                neg_idxs[neg_idx].append(pos_idx)
    return neg_idxs


def get_negative_sample_per_query(label_group_dict, pred, lab):
    """
    cur_S_emb: (1024)
    emo_pred: (3)
    y: (3)

    negative samples = (num_negative, 1024)
    """
    def get_group_idx(score):
        if score < 3:
            return 0
        elif score < 5:
            return 1
        else:
            return 2

    
    pred = pred*6 + 1
    lab = lab*6 + 1
    pred_group_idx = get_group_idx(pred[0])*9 + get_group_idx(pred[1])*3 + get_group_idx(pred[2])
    lab_group_idx = get_group_idx(lab[0])*9 + get_group_idx(lab[1])*3 + get_group_idx(lab[2])

    negative_samples = []
    for lg_idx in range(27):
        if lg_idx == lab_group_idx:
            continue
        # cur_negatives = label_group_dict[pred_group_idx][lg_idx][:10]
        cur_negatives = label_group_dict[pred_group_idx][lg_idx]
        if len(cur_negatives) != 0:
            cur_negatives = torch.stack(cur_negatives, dim=0) # (N, 1024)
            negative_samples.append(cur_negatives)
    if len(negative_samples) != 0:
        negative_samples = torch.cat(negative_samples, dim=0)
        return negative_samples
    else:
        return None

        
def assign_label_group_cnt(label_group_dict, hs, preds, labs):
    """
    h: (N, 1024)
    aro: [1-3) [3-5) [5-)
    dom: [1-3) [3-5) [5-)
    val: [1-3) [3-5) [5-)

    preds and labs are already normalized
    """
    def get_group_idx(score):
        if score < 3:
            return 0
        elif score < 5:
            return 1
        else:
            return 2

    batch_len = hs.size(0)
    for batch_idx in range(batch_len):
        cur_h = hs[batch_idx]
        cur_pred = preds[batch_idx]*6 + 1
        cur_lab = labs[batch_idx]*6 + 1
        pred_group_idx = get_group_idx(cur_pred[0])*9 + get_group_idx(cur_pred[1])*3 + get_group_idx(cur_pred[2])
        lab_group_idx = get_group_idx(cur_lab[0])*9 + get_group_idx(cur_lab[1])*3 + get_group_idx(cur_lab[2])

        label_group_dict[pred_group_idx][lab_group_idx]+=1

def assign_prototype(prototype_dict, count_dict, hs, labs):
    """
    h: (N, 1024)
    aro: [1-3) [3-5) [5-)
    dom: [1-3) [3-5) [5-)
    val: [1-3) [3-5) [5-)

    preds and labs are already normalized
    """
    def get_group_idx(score, attr_indicator):
        bound_dict={
            "aro": [4.2, 4.8],
            "dom": [4.28, 4.85],
            "val": [3.8, 4.6]
        }
        if score < bound_dict[attr_indicator][0]:
            return 0
        elif score < bound_dict[attr_indicator][1]:
            return 1
        else:
            return 2

    batch_len = hs.size(0)
    for batch_idx in range(batch_len):
        cur_h = hs[batch_idx]
        cur_lab = labs[batch_idx]*6 + 1
        lab_group_idx = get_group_idx(cur_lab[0], "aro")*9 + get_group_idx(cur_lab[1], "dom")*3 + get_group_idx(cur_lab[2], "val")

        prototype_dict[lab_group_idx]+=(cur_h.detach())
        count_dict[lab_group_idx]+=1

def get_negative_prototype(prototype_dict, labs):
    """
    y: (N, 3)

    negative samples = (N, 26(num_negative), 1024)
    """
    def get_group_idx(score, attr_indicator):
        bound_dict={
            "aro": [4.2, 4.8],
            "dom": [4.28, 4.85],
            "val": [3.8, 4.6]
        }
        if score < bound_dict[attr_indicator][0]:
            return 0
        elif score < bound_dict[attr_indicator][1]:
            return 1
        else:
            return 2

    
    labs = labs*6 + 1
    batch_len = labs.size(0)
    negative_samples = []
    for batch_idx in range(batch_len):
        cur_lab = labs[batch_idx]
        lab_group_idx = get_group_idx(cur_lab[0], "aro")*9 + get_group_idx(cur_lab[1], "dom")*3 + get_group_idx(cur_lab[2], "val")
        
        cur_negatives = []
        for lg_idx in range(27):
            if lg_idx == lab_group_idx:
                continue
            cur_negatives.append(prototype_dict[lg_idx]+1e-10)
        cur_negatives = torch.stack(cur_negatives, dim=0)
        negative_samples.append(cur_negatives)
    if len(negative_samples) != 0:
        negative_samples = torch.stack(negative_samples, dim=0)
        return negative_samples
    else:
        return None

def assign_prototype_v2(prototype_dict, hs, labs):
    """
    h: (N, 1024)
    aro: [1-3) [3-5) [5-)
    dom: [1-3) [3-5) [5-)
    val: [1-3) [3-5) [5-)

    preds and labs are already normalized
    """
    def get_group_idx(score, attr_indicator):
        bound_dict={
            "aro": [4.2, 4.8],
            "dom": [4.28, 4.85],
            "val": [3.8, 4.6]
        }
        if score < bound_dict[attr_indicator][0]:
            return 0
        elif score < bound_dict[attr_indicator][1]:
            return 1
        else:
            return 2

    batch_len = hs.size(0)
    for batch_idx in range(batch_len):
        cur_h = hs[batch_idx]
        cur_lab = labs[batch_idx]*6 + 1
        lab_group_idx = get_group_idx(cur_lab[0], "aro")*9 + get_group_idx(cur_lab[1], "dom")*3 + get_group_idx(cur_lab[2], "val")

        prototype_dict[lab_group_idx].append(cur_h.detach()) #.cpu().numpy())

def get_negative_prototype_v2(prototype_dict, labs):
    """
    y: (N, 3)

    negative samples = (N, 26(num_negative), 1024)
    """
    def get_group_idx(score, attr_indicator):
        bound_dict={
            "aro": [4.2, 4.8],
            "dom": [4.28, 4.85],
            "val": [3.8, 4.6]
        }
        if score < bound_dict[attr_indicator][0]:
            return 0
        elif score < bound_dict[attr_indicator][1]:
            return 1
        else:
            return 2

    
    labs = labs*6 + 1
    batch_len = labs.size(0)
    negative_samples = []
    for batch_idx in range(batch_len):
        cur_lab = labs[batch_idx]
        lab_group_idx = get_group_idx(cur_lab[0], "aro")*9 + get_group_idx(cur_lab[1], "dom")*3 + get_group_idx(cur_lab[2], "val")
        
        cur_negatives = []
        for lg_idx in range(27):
            if lg_idx == lab_group_idx:
                continue
            
            neg_samps = torch.stack(prototype_dict[lab_group_idx], dim=0)
            repr_idxs = torch.randperm(neg_samps.size(0))
            repr_idxs = repr_idxs[:1000]
            neg_samps = neg_samps[repr_idxs]
            centroid = torch.mean(neg_samps, dim=0)
            cur_negatives.append(centroid)
        cur_negatives = torch.stack(cur_negatives, dim=0)
        negative_samples.append(cur_negatives)
    if len(negative_samples) != 0:
        negative_samples = torch.stack(negative_samples, dim=0)
        return negative_samples
    else:
        return None