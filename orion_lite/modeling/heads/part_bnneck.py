# -*- coding: utf-8 -*-
"""
Part-based BNNeck 头部网络

结合 MGN (Multiple Granularity Network) 和 BoT (Bag of Tricks) 的设计思想：
- 全局特征 + 局部切片特征 (Part-based)
- GeM Pooling (Generalized Mean Pooling)
- BNNeck 解耦设计

相比单一全局特征，Part-based 方法能：
1. 捕捉细粒度特征（如鞋子、配饰）
2. 提高对遮挡的鲁棒性
3. 增强类间可分离性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from ..layers.pooling import GeneralizedMeanPooling, GlobalAvgPool2d


class PartBNNeckHead(nn.Module):
    """
    Part-based BNNeck 头部网络
    
    结构：
        特征图 (H, W)
        ├── 全局池化 → feat_global
        ├── 上半部切片 → feat_part1
        └── 下半部切片 → feat_part2
        
        拼接 → 降维 → BNNeck → 分类器
    
    Args:
        in_channels (int): 输入特征通道数
        feat_dim (int): 每个分支输出特征维度，默认 256
        num_classes (int): 类别数
        num_parts (int): 切片数量，默认 2 (上/下)
        pool_type (str): 池化类型 'avg' / 'gem'
        neck_feat (str): 推理时使用的特征位置 'before' / 'after'
    """
    def __init__(
        self,
        in_channels: int,
        feat_dim: int = 256,
        num_classes: int = 751,
        num_parts: int = 2,
        pool_type: str = 'gem',
        neck_feat: str = 'after',
    ):
        super(PartBNNeckHead, self).__init__()
        
        self.in_channels = in_channels
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_parts = num_parts
        self.neck_feat = neck_feat
        
        # 总特征维度 = (1 + num_parts) * feat_dim
        self.total_feat_dim = (1 + num_parts) * feat_dim
        
        # 1. 池化层
        if pool_type == 'gem':
            self.global_pool = GeneralizedMeanPooling(p=3.0)
            self.part_pools = nn.ModuleList([
                GeneralizedMeanPooling(p=3.0) for _ in range(num_parts)
            ])
        else:
            self.global_pool = GlobalAvgPool2d()
            self.part_pools = nn.ModuleList([
                GlobalAvgPool2d() for _ in range(num_parts)
            ])
        
        # 2. 降维层（每个分支独立降维）
        self.global_reduce = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
        )
        
        self.part_reduces = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, feat_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(feat_dim),
            ) for _ in range(num_parts)
        ])
        
        # 3. BNNeck（对拼接后的总特征）
        self.bnneck = nn.BatchNorm1d(self.total_feat_dim)
        self.bnneck.bias.requires_grad_(False)
        
        # 4. 分类器
        self.classifier = nn.Linear(self.total_feat_dim, num_classes, bias=False)
        
        # 初始化
        self._init_weights()
        
        print(f"  Part-based Head: {num_parts} parts + global")
        print(f"    Feature dim: {feat_dim} x {1 + num_parts} = {self.total_feat_dim}")
        print(f"    Pool type: {pool_type}")
    
    def _init_weights(self):
        """权重初始化"""
        # BN 层初始化
        nn.init.constant_(self.bnneck.weight, 1)
        nn.init.constant_(self.bnneck.bias, 0)
        
        # 分类器初始化
        nn.init.normal_(self.classifier.weight, std=0.001)
        
        # 降维层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _split_parts(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        将特征图沿高度方向切分
        
        Args:
            x: 特征图 (B, C, H, W)
            
        Returns:
            切片列表 [(B, C, H/n, W), ...]
        """
        B, C, H, W = x.shape
        part_h = H // self.num_parts
        
        parts = []
        for i in range(self.num_parts):
            start_h = i * part_h
            end_h = (i + 1) * part_h if i < self.num_parts - 1 else H
            parts.append(x[:, :, start_h:end_h, :])
        
        return parts
    
    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            x: 输入特征图，形状 (B, C, H, W)
            labels: 标签（训练时需要）
            
        Returns:
            训练模式: (cls_score, feat_before_bn, feat_after_bn)
            评估模式: feat（根据 neck_feat 选择）
        """
        # 1. 全局特征
        feat_global = self.global_pool(x)  # (B, C, 1, 1)
        feat_global = self.global_reduce(feat_global)  # (B, feat_dim, 1, 1)
        feat_global = feat_global.view(feat_global.size(0), -1)  # (B, feat_dim)
        
        # 2. 局部特征
        parts = self._split_parts(x)
        feat_parts = []
        for i, (part, pool, reduce) in enumerate(zip(parts, self.part_pools, self.part_reduces)):
            feat_part = pool(part)  # (B, C, 1, 1)
            feat_part = reduce(feat_part)  # (B, feat_dim, 1, 1)
            feat_part = feat_part.view(feat_part.size(0), -1)  # (B, feat_dim)
            feat_parts.append(feat_part)
        
        # 3. 拼接所有特征
        feat = torch.cat([feat_global] + feat_parts, dim=1)  # (B, total_feat_dim)
        
        # 4. BNNeck
        feat_bn = self.bnneck(feat)
        
        if self.training:
            # 训练模式
            cls_score = self.classifier(feat_bn)
            return cls_score, feat, feat_bn
        else:
            # 评估模式
            if self.neck_feat == 'before':
                return feat
            else:
                return feat_bn
    
    def set_num_classes(self, num_classes: int):
        """动态调整分类器类别数"""
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.total_feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)


if __name__ == "__main__":
    # 测试代码
    print("Testing Part-based BNNeck Head...")
    
    batch_size = 4
    in_channels = 960
    x = torch.randn(batch_size, in_channels, 16, 8)
    
    head = PartBNNeckHead(
        in_channels=in_channels,
        feat_dim=256,
        num_classes=751,
        num_parts=2,
        pool_type='gem',
    )
    
    # 训练模式
    head.train()
    labels = torch.randint(0, 751, (batch_size,))
    cls_score, feat_before, feat_after = head(x, labels)
    print(f"[Train Mode]")
    print(f"  Input:        {x.shape}")
    print(f"  Cls Score:    {cls_score.shape}")
    print(f"  Feat Before:  {feat_before.shape}")
    print(f"  Feat After:   {feat_after.shape}")
    
    # 评估模式
    head.eval()
    with torch.no_grad():
        feat = head(x)
    print(f"[Eval Mode]")
    print(f"  Output Feat:  {feat.shape}")
    
    # 参数量
    params = sum(p.numel() for p in head.parameters())
    print(f"Total params: {params:,}")
