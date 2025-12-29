# -*- coding: utf-8 -*-
"""
GhostNetV2 骨干网络

使用 timm 库加载预训练的 GhostNetV2，并进行 Re-ID 任务适配：
1. 移除分类头，只保留特征提取部分
2. 可选：注入 Coordinate Attention 模块

论文: GhostNetV2: Enhance Cheap Operation with Long-Range Attention (NeurIPS 2022)
"""

import torch
import torch.nn as nn
import timm
from typing import Optional

from ..layers.coord_attention import CoordAtt


class GhostNetV2Backbone(nn.Module):
    """
    GhostNetV2 骨干网络
    
    Args:
        model_name (str): timm 模型名称，如 'ghostnetv2_100', 'ghostnetv2_130', 'ghostnetv2_160'
        pretrained (bool): 是否加载 ImageNet 预训练权重
        last_stride (int): 最后一个 stage 的步长，1 或 2（目前未使用，timm不支持直接修改）
        with_coord_att (bool): 是否在最后添加 Coordinate Attention
        coord_att_reduction (int): Coordinate Attention 的降维比例
    """
    def __init__(
        self,
        model_name: str = 'ghostnetv2_100',
        pretrained: bool = True,
        last_stride: int = 1,
        with_coord_att: bool = True,
        coord_att_reduction: int = 32,
    ):
        super(GhostNetV2Backbone, self).__init__()
        
        self.model_name = model_name
        
        # 使用 timm 加载完整模型，然后移除分类头
        full_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头，返回特征
        )
        
        # 保存特征提取部分
        self.backbone = full_model
        
        # 获取输出通道数（通过前向传播获取）
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 128)
            dummy_output = self.backbone(dummy_input)
            if len(dummy_output.shape) == 2:
                # 输出是 (B, C) 形式，说明已经做了全局池化
                self._output_is_pooled = True
                self.out_channels = dummy_output.shape[1]
            else:
                # 输出是 (B, C, H, W) 形式
                self._output_is_pooled = False
                self.out_channels = dummy_output.shape[1]
        
        # 可选的 Coordinate Attention
        self.with_coord_att = with_coord_att and not self._output_is_pooled
        if self.with_coord_att:
            self.coord_att = CoordAtt(
                self.out_channels, 
                self.out_channels, 
                reduction=coord_att_reduction
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像，形状 (B, 3, H, W)
            
        Returns:
            特征，如果 backbone 已池化则 (B, C)，否则 (B, C, H', W')
        """
        feat = self.backbone(x)
        
        # 如果输出已经是池化后的向量，需要reshape以适配后续的head
        if self._output_is_pooled:
            # 转换为 (B, C, 1, 1) 以保持和head的兼容性
            feat = feat.view(feat.size(0), feat.size(1), 1, 1)
        
        # 可选的注意力增强
        if self.with_coord_att:
            feat = self.coord_att(feat)
        
        return feat
    
    def get_out_channels(self) -> int:
        """返回输出通道数"""
        return self.out_channels


def build_ghostnetv2_backbone(
    model_name: str = 'ghostnetv2_100',
    pretrained: bool = True,
    last_stride: int = 1,
    with_coord_att: bool = True,
    coord_att_reduction: int = 32,
) -> GhostNetV2Backbone:
    """
    构建 GhostNetV2 骨干网络
    
    可选的模型变体：
    - ghostnetv2_100: 基础版本，宽度乘子 1.0 (~6M 参数)
    - ghostnetv2_130: 加宽版本，宽度乘子 1.3
    - ghostnetv2_160: 更宽版本，宽度乘子 1.6
    """
    return GhostNetV2Backbone(
        model_name=model_name,
        pretrained=pretrained,
        last_stride=last_stride,
        with_coord_att=with_coord_att,
        coord_att_reduction=coord_att_reduction,
    )


if __name__ == "__main__":
    # 测试代码
    print("Testing GhostNetV2 Backbone...")
    
    # 检查可用的 GhostNetV2 模型
    available_models = [m for m in timm.list_models() if 'ghostnetv2' in m.lower()]
    print(f"Available GhostNetV2 models: {available_models}")
    
    # 创建模型
    backbone = build_ghostnetv2_backbone(
        model_name='ghostnetv2_100',
        pretrained=True,
        last_stride=1,
        with_coord_att=False,  # 先测试不带 CA
    )
    
    # 测试前向传播
    x = torch.randn(2, 3, 256, 128)  # Re-ID 标准输入尺寸
    feat = backbone(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {feat.shape}")
    print(f"Output channels: {backbone.get_out_channels()}")
    
    # 统计参数量
    params = sum(p.numel() for p in backbone.parameters())
    print(f"Total params: {params:,} ({params/1e6:.2f}M)")
