# -*- coding: utf-8 -*-
"""
BNNeck 头部网络

论文: Bag of Tricks and a Strong Baseline for Deep Person Re-Identification (CVPR 2019)

BNNeck 的核心思想是在特征和分类层之间加入 BatchNorm 层，
解耦 ID Loss 和 Triplet Loss 对特征空间的约束冲突：
- Triplet Loss 作用于 BN 前的特征（保留幅度信息）
- ID Loss 作用于 BN 后的特征（归一化到超球面）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..layers.pooling import build_pooling_layer


class BNNeckHead(nn.Module):
    """
    BNNeck 头部网络
    
    结构: Backbone Feature -> Pool -> Reduce -> BN -> (FC for training)
    
    Args:
        in_channels (int): 输入特征通道数（来自 backbone）
        feat_dim (int): 输出特征维度，默认 512
        num_classes (int): 类别数（训练时需要设置）
        pool_type (str): 池化类型 'avg' / 'max' / 'gem'
        neck_feat (str): 推理时使用的特征位置
            - 'before': BN 前的特征
            - 'after': BN 后的特征（推荐用于 cosine 相似度）
    """
    def __init__(
        self,
        in_channels: int,
        feat_dim: int = 512,
        num_classes: int = 751,  # Market-1501 的类别数
        pool_type: str = 'avg',
        neck_feat: str = 'after',
    ):
        super(BNNeckHead, self).__init__()
        
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.neck_feat = neck_feat
        
        # 1. 全局池化
        self.pool = build_pooling_layer(pool_type)
        
        # 2. 降维层（如果需要）
        if in_channels != feat_dim:
            self.reduce = nn.Sequential(
                nn.Conv2d(in_channels, feat_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.reduce = nn.Identity()
        
        # 3. BNNeck - 核心模块
        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.bnneck.bias.requires_grad_(False)  # BN 的 bias 设为不可学习
        
        # 4. 分类器（仅训练时使用）
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        # BN 层初始化
        nn.init.constant_(self.bnneck.weight, 1)
        nn.init.constant_(self.bnneck.bias, 0)
        
        # 分类器初始化
        nn.init.normal_(self.classifier.weight, std=0.001)
    
    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            x: 输入特征图，形状 (B, C, H, W)
            labels: 标签，训练时需要提供
            
        Returns:
            训练模式: (cls_score, feat_before_bn, feat_after_bn)
            评估模式: feat（根据 neck_feat 选择 before 或 after）
        """
        # 1. 池化
        x = self.pool(x)  # (B, C, 1, 1)
        
        # 2. 降维
        x = self.reduce(x)  # (B, feat_dim, 1, 1)
        
        # 3. 展平
        feat = x.view(x.size(0), -1)  # (B, feat_dim)
        
        # 4. BNNeck
        feat_bn = self.bnneck(feat)  # (B, feat_dim)
        
        if self.training:
            # 训练模式：返回分类分数和两种特征
            cls_score = self.classifier(feat_bn)
            return cls_score, feat, feat_bn
        else:
            # 评估模式：只返回特征
            if self.neck_feat == 'before':
                return feat
            else:  # 'after'
                return feat_bn
    
    def set_num_classes(self, num_classes: int):
        """
        动态调整分类器的类别数（用于迁移学习）
        
        Args:
            num_classes: 新的类别数
        """
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)


if __name__ == "__main__":
    # 测试代码
    print("Testing BNNeck Head...")
    
    # 创建模拟输入（来自 backbone 的特征图）
    batch_size = 4
    in_channels = 960  # GhostNetV2 的输出通道数
    x = torch.randn(batch_size, in_channels, 16, 8)  # 模拟特征图
    
    # 创建 BNNeck
    head = BNNeckHead(
        in_channels=in_channels,
        feat_dim=512,
        num_classes=751,
        pool_type='avg',
        neck_feat='after',
    )
    
    # 训练模式测试
    head.train()
    labels = torch.randint(0, 751, (batch_size,))
    cls_score, feat_before, feat_after = head(x, labels)
    print(f"[Train Mode]")
    print(f"  Input:        {x.shape}")
    print(f"  Cls Score:    {cls_score.shape}")
    print(f"  Feat Before:  {feat_before.shape}")
    print(f"  Feat After:   {feat_after.shape}")
    
    # 评估模式测试
    head.eval()
    with torch.no_grad():
        feat = head(x)
    print(f"[Eval Mode]")
    print(f"  Output Feat:  {feat.shape}")
    
    # 参数量统计
    params = sum(p.numel() for p in head.parameters())
    print(f"Total params: {params:,}")
