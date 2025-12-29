# -*- coding: utf-8 -*-
"""损失函数模块"""

from .circle_loss import CircleLoss, CircleLossPairwise
from .cross_entropy import CrossEntropyLabelSmooth, SoftmaxTripletLoss


def build_losses(cfg):
    """
    根据配置构建损失函数
    
    Args:
        cfg: 配置对象
        
    Returns:
        dict: 损失函数字典
    """
    losses = {}
    
    # ID Loss (Classification)
    if cfg.LOSS.ID.ENABLED:
        losses['id_loss'] = CrossEntropyLabelSmooth(
            num_classes=751,  # 将在训练时动态设置
            epsilon=cfg.LOSS.ID.LABEL_SMOOTH,
        )
    
    # Metric Loss (Circle Loss 替代 Triplet Loss)
    if cfg.LOSS.TRIPLET.ENABLED:
        # 使用 Circle Loss 替代传统 Triplet Loss
        losses['metric_loss'] = CircleLoss(
            m=0.25,
            gamma=256,
        )
    
    return losses
