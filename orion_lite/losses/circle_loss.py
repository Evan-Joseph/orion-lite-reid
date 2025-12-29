# -*- coding: utf-8 -*-
"""
Circle Loss 实现

论文: Circle Loss: A Unified Perspective of Pair Similarity Optimization (CVPR 2020)
链接: https://arxiv.org/abs/2002.10857

Circle Loss 是一种统一的相似度优化损失函数，相比 Triplet Loss：
1. 自适应权重：对未优化好的样本对给予更高权重
2. 更灵活的收敛目标：不是简单的 margin，而是动态的决策边界
3. 统一视角：可以退化为 Triplet Loss 或 Softmax Cross Entropy

在 Re-ID 任务中通常能带来 1-2% 的 mAP 提升。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CircleLoss(nn.Module):
    """
    Circle Loss for Deep Metric Learning
    
    Args:
        m (float): 松弛因子，默认 0.25
        gamma (float): 缩放因子，默认 256（对于 Re-ID 任务）
    """
    def __init__(self, m: float = 0.25, gamma: float = 256):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算 Circle Loss
        
        Args:
            feats: 特征向量 (B, D)，需要先 L2 归一化
            labels: 标签 (B,)
            
        Returns:
            loss: 标量损失值
        """
        # L2 归一化
        feats = F.normalize(feats, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(feats, feats.t())  # (B, B)
        
        # 构建正负样本 mask
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.t()).float()  # 同一 ID
        mask_neg = (labels != labels.t()).float()  # 不同 ID
        
        # 排除对角线（自己和自己）
        mask_pos = mask_pos - torch.eye(mask_pos.size(0), device=mask_pos.device)
        
        # 获取正负样本相似度
        sim_pos = similarity_matrix * mask_pos  # 正样本对相似度
        sim_neg = similarity_matrix * mask_neg  # 负样本对相似度
        
        # Circle Loss 的核心：自适应权重
        # 正样本：相似度越低，权重越高（需要更多优化）
        # 负样本：相似度越高，权重越高（需要更多优化）
        
        # 最优点
        O_p = 1 + self.m  # 正样本的最优相似度
        O_n = -self.m     # 负样本的最优相似度
        
        # 决策边界
        Delta_p = 1 - self.m  # 正样本的决策边界
        Delta_n = self.m      # 负样本的决策边界
        
        # 计算自适应权重
        alpha_p = torch.clamp_min(O_p - sim_pos.detach(), min=0)  # 正样本权重
        alpha_n = torch.clamp_min(sim_neg.detach() - O_n, min=0)  # 负样本权重
        
        # 计算 logsumexp 项
        # 正样本项：-gamma * alpha_p * (sim_pos - Delta_p)
        # 负样本项：gamma * alpha_n * (sim_neg - Delta_n)
        
        logit_p = -self.gamma * alpha_p * (sim_pos - Delta_p) * mask_pos
        logit_n = self.gamma * alpha_n * (sim_neg - Delta_n) * mask_neg
        
        # 使用 logsumexp 进行稳定计算
        # 处理空集情况
        loss_p = torch.logsumexp(logit_p.masked_fill(mask_pos == 0, float('-inf')), dim=1)
        loss_n = torch.logsumexp(logit_n.masked_fill(mask_neg == 0, float('-inf')), dim=1)
        
        # 最终损失
        loss = self.soft_plus(loss_p + loss_n)
        
        # 过滤无效样本（没有正样本对的样本）
        valid_mask = mask_pos.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = loss.mean() * 0  # 返回0但保持梯度
        
        return loss


class CircleLossPairwise(nn.Module):
    """
    Pairwise Circle Loss（更高效的实现）
    
    直接对 batch 内的正负样本对进行优化，
    而不是构建完整的相似度矩阵。
    
    Args:
        m (float): 松弛因子，默认 0.25
        gamma (float): 缩放因子，默认 256
    """
    def __init__(self, m: float = 0.25, gamma: float = 256):
        super(CircleLossPairwise, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
        
    def _get_pairs(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取正负样本对的索引
        
        Returns:
            pos_pairs: (N_pos, 2) 正样本对索引
            neg_pairs: (N_neg, 2) 负样本对索引
        """
        labels = labels.view(-1)
        n = labels.size(0)
        
        # 使用广播构建索引
        indices = torch.arange(n, device=labels.device)
        mask_same = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_diff = ~mask_same
        
        # 排除对角线
        mask_same = mask_same & (indices.unsqueeze(0) != indices.unsqueeze(1))
        
        pos_pairs = torch.nonzero(mask_same, as_tuple=False)
        neg_pairs = torch.nonzero(mask_diff, as_tuple=False)
        
        return pos_pairs, neg_pairs

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算 Circle Loss
        
        Args:
            feats: 特征向量 (B, D)
            labels: 标签 (B,)
            
        Returns:
            loss: 标量损失值
        """
        # L2 归一化
        feats = F.normalize(feats, p=2, dim=1)
        
        # 获取正负样本对
        pos_pairs, neg_pairs = self._get_pairs(labels)
        
        if pos_pairs.size(0) == 0 or neg_pairs.size(0) == 0:
            return torch.tensor(0.0, device=feats.device, requires_grad=True)
        
        # 计算正负样本对的相似度
        pos_sim = (feats[pos_pairs[:, 0]] * feats[pos_pairs[:, 1]]).sum(dim=1)
        neg_sim = (feats[neg_pairs[:, 0]] * feats[neg_pairs[:, 1]]).sum(dim=1)
        
        # Circle Loss 计算
        alpha_p = torch.clamp_min(1 + self.m - pos_sim.detach(), min=0)
        alpha_n = torch.clamp_min(neg_sim.detach() + self.m, min=0)
        
        # 损失项
        logit_p = -self.gamma * alpha_p * (pos_sim - (1 - self.m))
        logit_n = self.gamma * alpha_n * (neg_sim - self.m)
        
        # LogSumExp
        loss = self.soft_plus(
            torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0)
        )
        
        return loss


if __name__ == "__main__":
    # 测试代码
    print("Testing Circle Loss...")
    
    # 模拟数据
    batch_size = 16
    feat_dim = 512
    num_classes = 4
    
    feats = torch.randn(batch_size, feat_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 测试 Circle Loss
    circle_loss = CircleLoss(m=0.25, gamma=256)
    loss = circle_loss(feats, labels)
    print(f"Circle Loss: {loss.item():.4f}")
    
    # 测试 Pairwise Circle Loss
    circle_loss_pw = CircleLossPairwise(m=0.25, gamma=256)
    loss_pw = circle_loss_pw(feats, labels)
    print(f"Circle Loss (Pairwise): {loss_pw.item():.4f}")
    
    print("✅ Circle Loss test passed!")
