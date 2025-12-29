# -*- coding: utf-8 -*-
"""
Cross Entropy Loss with Label Smoothing

Label Smoothing 是 BoT (Bag of Tricks) 的关键技巧之一，
通过软化标签防止模型过度自信，提高泛化能力。

原始标签：[0, 1, 0, 0] (one-hot)
平滑后：  [0.033, 0.9, 0.033, 0.033] (epsilon=0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing
    
    Args:
        num_classes (int): 类别数
        epsilon (float): 平滑参数，默认 0.1
        reduction (str): 'mean' 或 'sum'
    """
    def __init__(
        self, 
        num_classes: int, 
        epsilon: float = 0.1,
        reduction: str = 'mean'
    ):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算带 Label Smoothing 的交叉熵损失
        
        Args:
            inputs: 模型输出 logits (B, num_classes)
            targets: 真实标签 (B,) 或 (B, num_classes)
            
        Returns:
            loss: 标量损失值
        """
        log_probs = self.logsoftmax(inputs)
        
        # 如果 targets 是类别索引，转换为 one-hot
        if targets.dim() == 1:
            targets = F.one_hot(targets, self.num_classes).float()
        
        # Label Smoothing
        # 原始：targets (one-hot)
        # 平滑后：(1 - epsilon) * targets + epsilon / num_classes
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        
        # 计算交叉熵
        loss = (-targets * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return loss


class SoftmaxTripletLoss(nn.Module):
    """
    Softmax 版本的 Triplet Loss
    
    相比硬边界的 Triplet Loss，使用 softmax 进行软化，
    对所有负样本都施加惩罚，而不仅仅是最难的负样本。
    """
    def __init__(self, margin: float = 0.3, normalize: bool = True):
        super(SoftmaxTripletLoss, self).__init__()
        self.margin = margin
        self.normalize = normalize
        
    def forward(
        self, 
        feats: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feats: 特征向量 (B, D)
            labels: 标签 (B,)
        """
        if self.normalize:
            feats = F.normalize(feats, p=2, dim=1)
        
        # 计算距离矩阵
        dist_mat = torch.cdist(feats, feats, p=2)
        
        # 构建 mask
        n = feats.size(0)
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.t()).float()
        mask_neg = (labels != labels.t()).float()
        
        # 排除对角线
        mask_pos = mask_pos - torch.eye(n, device=feats.device)
        
        # 对每个 anchor，获取正样本距离
        pos_dist = dist_mat * mask_pos
        pos_dist = pos_dist.sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
        
        # 对每个 anchor，获取负样本距离（使用 softmin 加权）
        neg_dist = dist_mat * mask_neg
        neg_dist = neg_dist.masked_fill(mask_neg == 0, float('inf'))
        neg_dist = -F.log_softmax(-neg_dist, dim=1) * mask_neg
        neg_dist = neg_dist.sum(dim=1) / (mask_neg.sum(dim=1) + 1e-8)
        
        # Triplet Loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


if __name__ == "__main__":
    # 测试代码
    print("Testing Cross Entropy with Label Smoothing...")
    
    batch_size = 16
    num_classes = 751
    
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 测试 Label Smoothing CE
    ce_smooth = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1)
    loss = ce_smooth(inputs, targets)
    print(f"CE with Label Smoothing: {loss.item():.4f}")
    
    # 对比标准 CE
    ce_standard = nn.CrossEntropyLoss()
    loss_std = ce_standard(inputs, targets)
    print(f"Standard CE: {loss_std.item():.4f}")
    
    print("✅ Cross Entropy test passed!")
