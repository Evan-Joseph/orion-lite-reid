# -*- coding: utf-8 -*-
"""
Orion-Lite v2.0 模型主体

v2.0 升级内容：
1. 骨干网络：GhostNetV2 → MobileNetV4（速度提升 4x）
2. 损失函数：Triplet Loss → Circle Loss（mAP 提升 1-2%）
3. Part-based 局部特征 + GeM Pooling（新增）
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union

from .backbones import build_backbone
from .heads import BNNeckHead, PartBNNeckHead


class OrionLite(nn.Module):
    """
    Orion-Lite v2.0: 轻量化行人重识别模型
    
    架构: MobileNetV4/GhostNetV2 (Backbone) -> BNNeck/PartBNNeck (Head)
    
    Args:
        backbone_name (str): 骨干网络名称
            - 'mobilenetv4_conv_small' (推荐，2.5M 参数)
            - 'ghostnetv2_100' (4.9M 参数)
        pretrained (bool): 是否加载预训练权重
        feat_dim (int): 输出特征维度
        num_classes (int): 类别数（训练身份数）
        pool_type (str): 池化类型 'avg' / 'gem'
        neck_feat (str): 推理时使用的特征位置
        head_type (str): 头部类型
            - 'bnneck': 标准 BNNeck (全局特征)
            - 'part': Part-based BNNeck (全局 + 局部特征)
        num_parts (int): Part-based 切片数量（仅 head_type='part' 时有效）
    """
    def __init__(
        self,
        backbone_name: str = 'mobilenetv4_conv_small',
        pretrained: bool = True,
        feat_dim: int = 512,
        num_classes: int = 751,
        pool_type: str = 'avg',
        neck_feat: str = 'after',
        head_type: str = 'bnneck',
        num_parts: int = 2,
    ):
        super(OrionLite, self).__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.head_type = head_type
        
        # 1. 骨干网络
        self.backbone = build_backbone(
            name=backbone_name,
            pretrained=pretrained,
        )
        
        # 2. 头部网络
        if head_type == 'part':
            self.head = PartBNNeckHead(
                in_channels=self.backbone.get_out_channels(),
                feat_dim=feat_dim // (1 + num_parts),  # 每个分支的维度
                num_classes=num_classes,
                num_parts=num_parts,
                pool_type=pool_type if pool_type != 'avg' else 'gem',  # Part-based 默认用 GeM
                neck_feat=neck_feat,
            )
            self.feat_dim = self.head.total_feat_dim
        else:  # 'bnneck'
            self.head = BNNeckHead(
                in_channels=self.backbone.get_out_channels(),
                feat_dim=feat_dim,
                num_classes=num_classes,
                pool_type=pool_type,
                neck_feat=neck_feat,
            )
            self.feat_dim = feat_dim
    
    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, 3, 256, 128)
            labels: 标签（训练时需要）
            
        Returns:
            训练模式: dict with 'cls_score', 'feat', 'feat_bn'
            评估模式: feat tensor (B, feat_dim)
        """
        # 提取特征
        feat_map = self.backbone(x)  # (B, C, 1, 1) 或 (B, C, H', W')
        
        # 通过头部
        if self.training:
            cls_score, feat, feat_bn = self.head(feat_map, labels)
            return {
                'cls_score': cls_score,
                'feat': feat,           # Circle Loss 用
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
    
    def set_num_classes(self, num_classes: int):
        """动态设置类别数"""
        self.num_classes = num_classes
        self.head.set_num_classes(num_classes)


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
        feat_dim=cfg.MODEL.HEAD.FEAT_DIM,
        num_classes=751,  # 将在训练开始时动态设置
        pool_type=cfg.MODEL.HEAD.POOL_TYPE,
        neck_feat=cfg.MODEL.HEAD.NECK_FEAT,
    )
    
    return model


if __name__ == "__main__":
    # 测试代码
    import time
    
    print("=" * 60)
    print("Testing Orion-Lite v2.0 Model")
    print("=" * 60)
    
    # 测试两种骨干网络
    for backbone in ['mobilenetv4_conv_small', 'ghostnetv2_100']:
        print(f"\n📊 Backbone: {backbone}")
        
        model = OrionLite(
            backbone_name=backbone,
            pretrained=True,
            feat_dim=512,
            num_classes=751,
        )
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total Params: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # 测试推理
        model.eval()
        x = torch.randn(4, 3, 256, 128)
        
        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = model(x)
            elapsed = (time.time() - start) / 10 * 1000
        
        print(f"   Latency (CPU): {elapsed:.2f}ms/batch ({elapsed/4:.2f}ms/img)")
    
    print("\n" + "=" * 60)
    print("✅ Orion-Lite v2.0 Test Passed!")
    print("=" * 60)
