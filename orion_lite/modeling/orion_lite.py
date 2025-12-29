# -*- coding: utf-8 -*-
"""
Orion-Lite 模型主体

整合 GhostNetV2 骨干网络 + Coordinate Attention + BNNeck 头部
的完整 Re-ID 模型。
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union

from .backbones import build_ghostnetv2_backbone
from .heads import BNNeckHead


class OrionLite(nn.Module):
    """
    Orion-Lite: 轻量化行人重识别模型
    
    架构: GhostNetV2 (Backbone) -> CoordAtt -> BNNeck (Head)
    
    Args:
        cfg: 配置对象
        num_classes (int): 类别数（训练身份数）
    """
    def __init__(
        self,
        backbone_name: str = 'ghostnetv2_100',
        pretrained: bool = True,
        last_stride: int = 1,
        with_coord_att: bool = True,
        coord_att_reduction: int = 32,
        feat_dim: int = 512,
        num_classes: int = 751,
        pool_type: str = 'avg',
        neck_feat: str = 'after',
    ):
        super(OrionLite, self).__init__()
        
        self.num_classes = num_classes
        
        # 1. 骨干网络
        self.backbone = build_ghostnetv2_backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            last_stride=last_stride,
            with_coord_att=with_coord_att,
            coord_att_reduction=coord_att_reduction,
        )
        
        # 2. 头部网络
        self.head = BNNeckHead(
            in_channels=self.backbone.get_out_channels(),
            feat_dim=feat_dim,
            num_classes=num_classes,
            pool_type=pool_type,
            neck_feat=neck_feat,
        )
        
        # 保存配置
        self.feat_dim = feat_dim
    
    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, 3, 256, 128)
            labels: 标签（训练时需要）
            
        Returns:
            训练模式: dict with 'cls_score', 'feat_before_bn', 'feat_after_bn'
            评估模式: feat tensor (B, feat_dim)
        """
        # 提取特征
        feat_map = self.backbone(x)  # (B, C, H', W')
        
        # 通过头部
        if self.training:
            cls_score, feat, feat_bn = self.head(feat_map, labels)
            return {
                'cls_score': cls_score,
                'feat': feat,           # Triplet Loss 用
                'feat_bn': feat_bn,     # 归一化后特征
            }
        else:
            return self.head(feat_map)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        特征提取接口（用于推理/检索）
        
        Args:
            x: 输入图像
            
        Returns:
            特征向量 (B, feat_dim)
        """
        self.eval()
        with torch.no_grad():
            feat = self(x)
        return feat
    
    def get_params_groups(self, base_lr: float = 1e-4, weight_decay: float = 5e-4):
        """
        获取参数分组（用于差异化学习率）
        
        骨干网络使用较小的学习率（预训练权重）
        头部使用较大的学习率（随机初始化）
        
        Returns:
            list of param groups for optimizer
        """
        # Backbone 参数
        backbone_params = []
        backbone_params_no_decay = []
        
        # Head 参数
        head_params = []
        head_params_no_decay = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'backbone' in name:
                if 'bn' in name or 'bias' in name:
                    backbone_params_no_decay.append(param)
                else:
                    backbone_params.append(param)
            else:  # head
                if 'bn' in name or 'bias' in name:
                    head_params_no_decay.append(param)
                else:
                    head_params.append(param)
        
        return [
            {'params': backbone_params, 'lr': base_lr * 0.1, 'weight_decay': weight_decay},
            {'params': backbone_params_no_decay, 'lr': base_lr * 0.1, 'weight_decay': 0.0},
            {'params': head_params, 'lr': base_lr, 'weight_decay': weight_decay},
            {'params': head_params_no_decay, 'lr': base_lr, 'weight_decay': 0.0},
        ]


def build_model(cfg) -> OrionLite:
    """
    从配置构建模型
    
    Args:
        cfg: 配置对象（yacs CfgNode）
        
    Returns:
        OrionLite 模型实例
    """
    model = OrionLite(
        backbone_name=cfg.MODEL.BACKBONE.NAME,
        pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
        last_stride=cfg.MODEL.BACKBONE.LAST_STRIDE,
        with_coord_att=cfg.MODEL.ATTENTION.ENABLED,
        coord_att_reduction=cfg.MODEL.ATTENTION.REDUCTION,
        feat_dim=cfg.MODEL.HEAD.FEAT_DIM,
        num_classes=751,  # 将在训练开始时动态设置
        pool_type=cfg.MODEL.HEAD.POOL_TYPE,
        neck_feat=cfg.MODEL.HEAD.NECK_FEAT,
    )
    
    return model


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Testing Orion-Lite Model")
    print("=" * 60)
    
    # 创建模型
    model = OrionLite(
        backbone_name='ghostnetv2_100',
        pretrained=True,
        last_stride=1,
        with_coord_att=True,
        feat_dim=512,
        num_classes=751,
    )
    
    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 测试输入
    x = torch.randn(4, 3, 256, 128)  # Re-ID 标准尺寸
    labels = torch.randint(0, 751, (4,))
    
    # 训练模式
    model.train()
    outputs = model(x, labels)
    print(f"\n[Train Mode]")
    print(f"  Input:      {x.shape}")
    print(f"  Cls Score:  {outputs['cls_score'].shape}")
    print(f"  Feat:       {outputs['feat'].shape}")
    print(f"  Feat BN:    {outputs['feat_bn'].shape}")
    
    # 评估模式
    model.eval()
    with torch.no_grad():
        feat = model(x)
    print(f"\n[Eval Mode]")
    print(f"  Output:     {feat.shape}")
    
    # MPS 设备测试（如果可用）
    if torch.backends.mps.is_available():
        print(f"\n[MPS Test]")
        device = torch.device('mps')
        model = model.to(device)
        x = x.to(device)
        
        with torch.no_grad():
            feat = model(x)
        print(f"  MPS Output: {feat.shape} on {feat.device}")
    
    print("=" * 60)
    print("✅ Orion-Lite Model Test Passed!")
    print("=" * 60)
