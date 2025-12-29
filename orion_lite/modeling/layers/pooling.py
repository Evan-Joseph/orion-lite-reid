# -*- coding: utf-8 -*-
"""
池化层模块

包含 Re-ID 常用的池化方法：
- GeneralizedMeanPooling (GeM)
- AdaptiveAvgMaxPool2d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedMeanPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM)
    
    GeM 池化是一种介于平均池化和最大池化之间的泛化形式：
    - p=1 时等价于平均池化
    - p→∞ 时趋近于最大池化
    
    在图像检索任务中，GeM 通常比简单的平均池化效果更好。
    
    Args:
        p (float): 池化参数，默认 3.0
        eps (float): 数值稳定性参数
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # Clamp 防止负数和零值
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.pow(1.0 / self.p)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.item():.4f})"


class AdaptiveAvgMaxPool2d(nn.Module):
    """
    自适应平均-最大混合池化
    
    将平均池化和最大池化的结果进行拼接或加权平均，
    结合两者的优势：平均池化捕捉整体信息，最大池化捕捉显著特征。
    
    Args:
        output_size: 输出尺寸，默认 (1, 1)
        mode (str): 'concat' 拼接 或 'avg' 加权平均
    """
    def __init__(self, output_size=(1, 1), mode='avg'):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size
        self.mode = mode
        
    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, self.output_size)
        max_pool = F.adaptive_max_pool2d(x, self.output_size)
        
        if self.mode == 'concat':
            return torch.cat([avg_pool, max_pool], dim=1)
        elif self.mode == 'avg':
            return (avg_pool + max_pool) / 2
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class GlobalAvgPool2d(nn.Module):
    """全局平均池化，输出 (B, C, 1, 1)"""
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))


class GlobalMaxPool2d(nn.Module):
    """全局最大池化，输出 (B, C, 1, 1)"""
    def __init__(self):
        super(GlobalMaxPool2d, self).__init__()
        
    def forward(self, x):
        return F.adaptive_max_pool2d(x, (1, 1))


def build_pooling_layer(pool_type='avg'):
    """
    构建池化层
    
    Args:
        pool_type (str): 池化类型
            - 'avg': 全局平均池化
            - 'max': 全局最大池化
            - 'gem': Generalized Mean Pooling
            - 'avgmax': 平均-最大混合池化
            
    Returns:
        nn.Module: 池化层
    """
    pool_type = pool_type.lower()
    
    if pool_type == 'avg':
        return GlobalAvgPool2d()
    elif pool_type == 'max':
        return GlobalMaxPool2d()
    elif pool_type == 'gem':
        return GeneralizedMeanPooling()
    elif pool_type == 'avgmax':
        return AdaptiveAvgMaxPool2d(mode='avg')
    else:
        raise ValueError(f"Unknown pool type: {pool_type}")


if __name__ == "__main__":
    # 测试代码
    x = torch.randn(2, 512, 16, 8)  # 模拟特征图
    
    # 测试各种池化
    for pool_type in ['avg', 'max', 'gem', 'avgmax']:
        pool = build_pooling_layer(pool_type)
        out = pool(x)
        print(f"{pool_type:8s} -> {out.shape}")
