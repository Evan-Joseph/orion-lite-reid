# -*- coding: utf-8 -*-
"""
Coordinate Attention 模块

论文: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)
论文链接: https://arxiv.org/abs/2103.02907

Coordinate Attention 通过将位置信息嵌入到通道注意力中，
使轻量级网络能够在更大区域上建模长距离依赖，同时保留精确的位置信息。
"""

import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    """Hard Sigmoid 激活函数"""
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard Swish 激活函数"""
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """
    Coordinate Attention 模块
    
    不同于 SE 模块的全局平均池化，CA 保留了位置信息：
    1. 分别沿 H 和 W 方向进行池化，生成方向感知特征
    2. 通过共享变换编码空间信息
    3. 分离并生成方向注意力权重
    
    Args:
        inp (int): 输入通道数
        oup (int): 输出通道数（通常等于输入）
        reduction (int): 通道降维比例，默认 32
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        
        # 自适应池化，分别压缩 H 和 W 方向
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, W)
        
        # 中间通道数，保证至少为 8
        mip = max(8, inp // reduction)
        
        # 共享的 1x1 卷积，用于通道降维和特征融合
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        # 分离卷积，恢复维度并生成注意力权重
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图，形状 (B, C, H, W)
            
        Returns:
            注意力加权后的特征图，形状 (B, C, H, W)
        """
        identity = x
        n, c, h, w = x.size()
        
        # 1. 坐标池化 - 生成方向感知特征
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1) -> (B, C, W, 1) for concat
        
        # 2. 拼接与共享变换
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # 3. 拆分 - 分离 H 和 W 方向的特征
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 恢复原始维度顺序
        
        # 4. 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()  # (B, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (B, C, 1, W)
        
        # 5. 施加注意力 - 通过广播机制乘积
        out = identity * a_w * a_h
        
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block（对比用）
    
    标准的通道注意力机制，通过全局平均池化压缩空间信息。
    缺点：丢失了位置信息，无法精确定位重要区域。
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        
    def forward(self, x):
        w = x.mean(dim=(2, 3), keepdim=True)
        w = torch.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


if __name__ == "__main__":
    # 测试代码
    import torch
    
    # 创建模拟输入
    x = torch.randn(2, 64, 32, 16)  # (B, C, H, W)
    
    # 测试 Coordinate Attention
    ca = CoordAtt(64, 64, reduction=32)
    out = ca(x)
    print(f"CoordAtt Input:  {x.shape}")
    print(f"CoordAtt Output: {out.shape}")
    
    # 参数量统计
    params = sum(p.numel() for p in ca.parameters())
    print(f"CoordAtt Params: {params:,}")
