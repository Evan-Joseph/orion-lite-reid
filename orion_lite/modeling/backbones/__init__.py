# -*- coding: utf-8 -*-
"""骨干网络模块"""

from .ghostnetv2 import build_ghostnetv2_backbone, GhostNetV2Backbone
from .mobilenetv4 import build_mobilenetv4_backbone, MobileNetV4Backbone


def build_backbone(name: str, **kwargs):
    """
    统一的骨干网络构建接口
    
    Args:
        name: 骨干网络名称
            - 'ghostnetv2_100', 'ghostnetv2_130', 'ghostnetv2_160'
            - 'mobilenetv4_conv_small', 'mobilenetv4_conv_medium', etc.
        **kwargs: 其他参数
        
    Returns:
        骨干网络实例
    """
    name = name.lower()
    
    if 'ghostnetv2' in name:
        return build_ghostnetv2_backbone(model_name=name, **kwargs)
    elif 'mobilenetv4' in name:
        return build_mobilenetv4_backbone(model_name=name, **kwargs)
    else:
        raise ValueError(f"Unknown backbone: {name}")
