# -*- coding: utf-8 -*-
"""
Triplet Loss with Hard Mining

ReID 标准的 Triplet Loss 实现，使用 Batch Hard 策略。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss with Hard Mining
    
    Args:
        margin (float): Triplet margin (default: 0.3)
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        # L2 Normalize (Crucial for Triplet Loss with fixed margin)
        inputs = F.normalize(inputs, p=2, dim=1)
        
        n = inputs.size(0)
        
        # 1. 计算成对距离矩阵
        # inputs: (n, d)
        # dist: (n, n)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # 2. Hard Mining
        # For each anchor, find the hardest positive and hardest negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        dist_ap, dist_an = [], []
        for i in range(n):
            # Hardest Positive: 同类中距离最远的
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            
            # Hardest Negative: 异类中距离最近的
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # 3. Compute Ranking Loss
        # y = 1 means input1 (an) should be larger than input2 (ap) by margin
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # optional: return accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        
        return loss
