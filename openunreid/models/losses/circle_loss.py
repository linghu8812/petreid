# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CosFace"]


class CosFace(nn.Module):
    def __init__(self, margin=0.25, gamma=32, cos_key="feat"):
        super(CosFace, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.cos_key = cos_key

    def forward(self, embedding, targets):
        embedding = F.normalize(embedding[self.cos_key], dim=1)

        dist_mat = torch.matmul(embedding, embedding.t())

        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        logit_p = -self.gamma * s_p + (-99999999.) * (1 - is_pos)
        logit_n = self.gamma * (s_n + self.margin) + (-99999999.) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss
