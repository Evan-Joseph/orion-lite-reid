# -*- coding: utf-8 -*-
"""
Center Loss 实现

论文: A Discriminative Feature Learning Approach for Deep Face Recognition (ECCV 2016)
链接: https://ydwen.github.io/papers/WenECCV16.pdf

Center Loss 通过为每个类别维护一个可学习的中心，约束特征向中心聚集，
减小类内方差，与 Softmax/CrossEntropy/Circle Loss 结合使用效果显著。

关键思想：
1. 每个类别有一个可学习的中心向量
2. 训练时约束同类特征向中心靠拢
3. 间接增强类内紧凑性，提升检索性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """
    Center Loss for Deep Metric Learning
    
    Args:
        num_classes (int): 类别数
        feat_dim (int): 特征维度
        use_gpu (bool): 是否使用 GPU
    """
    def __init__(
        self, 
        num_classes: int, 
        feat_dim: int,
    ):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # 可学习的类别中心
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)
    
    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算 Center Loss
        
        Args:
            feat: 特征向量 (B, D)
            labels: 标签 (B,)
            
        Returns:
            loss: Center Loss 值
        """
        batch_size = feat.size(0)
        
        # 获取每个样本对应的中心
        centers_batch = self.centers[labels]  # (B, D)
        
        # 计算特征到中心的欧氏距离的平方
        # L = 1/2 * sum((feat - center)^2)
        dist = (feat - centers_batch).pow(2).sum(dim=1)  # (B,)
        
        # 平均
        loss = dist.mean() / 2.0
        
        return loss
    
    def get_centers(self) -> torch.Tensor:
        """返回所有中心"""
        return self.centers.data


class CenterLossWithUpdate(nn.Module):
    """
    Center Loss with Manual Center Update
    
    原论文中，中心的更新使用单独的学习率（通常是主学习率的 0.5）。
    这个版本通过梯度自动更新中心，但也提供手动更新接口。
    
    Args:
        num_classes (int): 类别数
        feat_dim (int): 特征维度
        alpha (float): 中心更新速率，默认 0.5
    """
    def __init__(
        self, 
        num_classes: int, 
        feat_dim: int,
        alpha: float = 0.5,
    ):
        super(CenterLossWithUpdate, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        
        # 类别中心（不作为参数，手动更新）
        self.register_buffer('centers', torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)
    
    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算 Center Loss
        
        Args:
            feat: 特征向量 (B, D)
            labels: 标签 (B,)
            
        Returns:
            loss: Center Loss 值
        """
        batch_size = feat.size(0)
        
        # 获取每个样本对应的中心
        centers_batch = self.centers[labels]  # (B, D)
        
        # 计算距离
        dist = (feat - centers_batch).pow(2).sum(dim=1)
        loss = dist.mean() / 2.0
        
        return loss
    
    @torch.no_grad()
    def update_centers(self, feat: torch.Tensor, labels: torch.Tensor):
        """
        手动更新中心
        
        使用指数移动平均更新中心位置
        """
        for label in labels.unique():
            mask = labels == label
            if mask.sum() > 0:
                # 计算该类别的平均特征
                class_feat = feat[mask].mean(dim=0)
                # 指数移动平均更新
                self.centers[label] = (
                    self.alpha * self.centers[label] + 
                    (1 - self.alpha) * class_feat
                )


if __name__ == "__main__":
    print("Testing Center Loss...")
    
    # 模拟数据
    batch_size = 16
    num_classes = 751
    feat_dim = 512
    
    feat = torch.randn(batch_size, feat_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 测试 CenterLoss
    center_loss = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
    loss = center_loss(feat, labels)
    print(f"Center Loss: {loss.item():.4f}")
    
    # 测试参数数量
    params = sum(p.numel() for p in center_loss.parameters())
    print(f"Center params: {params:,} ({params/1e6:.2f}M)")
    
    print("✅ Center Loss test passed!")
