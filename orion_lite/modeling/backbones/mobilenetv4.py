# -*- coding: utf-8 -*-
"""
MobileNetV4 骨干网络

MobileNetV4 是 Google 于 2024 年 9 月发布的最新轻量级骨干网络，
专门为移动端和边缘设备优化，在 Apple Neural Engine 上表现尤为出色。

核心创新：
1. Universal Inverted Bottleneck (UIB) - 统一的反向瓶颈搜索块
2. Mobile Multi-Query Attention (MQA) - 移动端优化的多查询注意力

论文: MobileNetV4 - Universal Models for the Mobile Ecosystem (ECCV 2024)
链接: https://arxiv.org/abs/2404.10518
"""

import torch
import torch.nn as nn
import timm
from typing import Optional

from ..layers.coord_attention import CoordAtt


class MobileNetV4Backbone(nn.Module):
    """
    MobileNetV4 骨干网络
    
    Args:
        model_name (str): timm 模型名称
            - 'mobilenetv4_conv_small': 最轻量，2.49M 参数
            - 'mobilenetv4_conv_medium': 中等规模
            - 'mobilenetv4_conv_large': 较大规模
            - 'mobilenetv4_hybrid_*': 混合架构（含 MQA 注意力）
        pretrained (bool): 是否加载 ImageNet 预训练权重
        with_coord_att (bool): 是否在最后添加 Coordinate Attention
        coord_att_reduction (int): Coordinate Attention 的降维比例
    """
    def __init__(
        self,
        model_name: str = 'mobilenetv4_conv_small',
        pretrained: bool = True,
        with_coord_att: bool = True,
        coord_att_reduction: int = 32,
        last_stride: int = 1,  # BoT trick: stride=1 保留更多空间细节
    ):
        super(MobileNetV4Backbone, self).__init__()
        
        self.model_name = model_name
        
        # 使用 timm 加载完整模型
        # 注意：我们需要使用 forward_features 来获取空间特征图
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            features_only=False,
        )
        
        # BoT Trick: Last Stride = 1
        # 将 Block 3 的下采样 stride 从 2 改为 1
        # 这使特征图从 8x4 变为 16x8，保留 4 倍的空间信息
        if last_stride == 1:
            try:
                # Block 3 的第一个 UniversalInvertedResidual 包含下采样
                block3_first = self.backbone.blocks[3][0]
                if hasattr(block3_first, 'dw_mid') and hasattr(block3_first.dw_mid, 'conv'):
                    original_stride = block3_first.dw_mid.conv.stride
                    block3_first.dw_mid.conv.stride = (1, 1)
                    print(f"  Last Stride: {original_stride} → (1, 1) [BoT Trick Enabled]")
            except Exception as e:
                print(f"  Warning: Could not modify last stride: {e}")
        
        # 获取输出通道数
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 256, 128)
            self.backbone.eval()
            dummy_output = self.backbone.forward_features(dummy_input)
            self.out_channels = dummy_output.shape[1]
            self._feature_map_size = (dummy_output.shape[2], dummy_output.shape[3])
            print(f"  MobileNetV4 Feature Map: {self.out_channels} channels, {self._feature_map_size} spatial")
        
        # 可选的 Coordinate Attention
        self.with_coord_att = with_coord_att and self._feature_map_size[0] > 1
        if self.with_coord_att:
            self.coord_att = CoordAtt(
                inp=self.out_channels,
                oup=self.out_channels,
                reduction=coord_att_reduction,
            )
            print(f"  Coordinate Attention: Enabled (reduction={coord_att_reduction})")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像，形状 (B, 3, H, W)
            
        Returns:
            特征图，形状 (B, C, H', W')
        """
        # 使用 forward_features 获取空间特征图
        feat = self.backbone.forward_features(x)  # (B, C, H', W')
        
        # 应用 Coordinate Attention (如果启用)
        if self.with_coord_att:
            feat = self.coord_att(feat)
        
        return feat
    
    def get_out_channels(self) -> int:
        """返回输出通道数"""
        return self.out_channels


def build_mobilenetv4_backbone(
    model_name: str = 'mobilenetv4_conv_small',
    pretrained: bool = True,
    with_coord_att: bool = True,
    coord_att_reduction: int = 32,
) -> MobileNetV4Backbone:
    """
    构建 MobileNetV4 骨干网络
    
    推荐的模型变体（按参数量排序）：
    - mobilenetv4_conv_small_035: ~1M 参数（超轻量）
    - mobilenetv4_conv_small_050: ~1.5M 参数
    - mobilenetv4_conv_small: ~2.5M 参数（推荐，精度/效率平衡）
    - mobilenetv4_conv_medium: ~4M 参数
    - mobilenetv4_conv_large: ~8M 参数
    - mobilenetv4_hybrid_medium: ~6M 参数（含 MQA 注意力）
    """
    return MobileNetV4Backbone(
        model_name=model_name,
        pretrained=pretrained,
        with_coord_att=with_coord_att,
        coord_att_reduction=coord_att_reduction,
    )


if __name__ == "__main__":
    # 测试代码
    print("Testing MobileNetV4 Backbone...")
    
    # 检查可用的 MobileNetV4 模型
    available_models = [m for m in timm.list_models() if 'mobilenetv4' in m.lower()]
    print(f"Available MobileNetV4 models: {available_models}")
    
    # 创建模型
    backbone = build_mobilenetv4_backbone(
        model_name='mobilenetv4_conv_small',
        pretrained=True,
    )
    
    # 测试前向传播
    x = torch.randn(2, 3, 256, 128)  # Re-ID 标准输入尺寸
    backbone.eval()
    with torch.no_grad():
        feat = backbone(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {feat.shape}")
    print(f"Output channels: {backbone.get_out_channels()}")
    
    # 统计参数量
    params = sum(p.numel() for p in backbone.parameters())
    print(f"Total params: {params:,} ({params/1e6:.2f}M)")
