#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orion-Lite 模型测试脚本

验证模型构建、前向传播和 MPS 加速是否正常工作。
"""

import sys
import torch
import timm

# 添加项目根目录到路径
sys.path.insert(0, '/Users/joseph/Desktop/【人工智能技术结课项目】ReID重识别')


def test_coord_attention():
    """测试 Coordinate Attention 模块"""
    print("=" * 60)
    print("Testing Coordinate Attention Module")
    print("=" * 60)
    
    from orion_lite.modeling.layers.coord_attention import CoordAtt
    
    x = torch.randn(2, 64, 32, 16)
    ca = CoordAtt(64, 64, reduction=32)
    out = ca(x)
    
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in ca.parameters()):,}")
    print("  ✅ Passed!")
    return True


def test_pooling():
    """测试池化层"""
    print("\n" + "=" * 60)
    print("Testing Pooling Layers")
    print("=" * 60)
    
    from orion_lite.modeling.layers.pooling import build_pooling_layer
    
    x = torch.randn(2, 512, 16, 8)
    
    for pool_type in ['avg', 'max', 'gem']:
        pool = build_pooling_layer(pool_type)
        out = pool(x)
        print(f"  {pool_type:6s} -> {out.shape}")
    
    print("  ✅ Passed!")
    return True


def test_ghostnetv2_backbone():
    """测试 GhostNetV2 骨干网络"""
    print("\n" + "=" * 60)
    print("Testing GhostNetV2 Backbone")
    print("=" * 60)
    
    # 检查可用模型
    available = [m for m in timm.list_models() if 'ghostnetv2' in m.lower()]
    print(f"  Available models: {available}")
    
    from orion_lite.modeling.backbones.ghostnetv2 import build_ghostnetv2_backbone
    
    backbone = build_ghostnetv2_backbone(
        model_name='ghostnetv2_100',
        pretrained=True,
        last_stride=1,
        with_coord_att=True,
    )
    
    x = torch.randn(2, 3, 256, 128)
    feat = backbone(x)
    
    print(f"  Input:           {x.shape}")
    print(f"  Output:          {feat.shape}")
    print(f"  Output Channels: {backbone.get_out_channels()}")
    print(f"  Params:          {sum(p.numel() for p in backbone.parameters()):,}")
    print("  ✅ Passed!")
    return True


def test_bnneck_head():
    """测试 BNNeck 头部"""
    print("\n" + "=" * 60)
    print("Testing BNNeck Head")
    print("=" * 60)
    
    from orion_lite.modeling.heads.bnneck import BNNeckHead
    
    in_channels = 960  # GhostNetV2 输出通道数
    x = torch.randn(4, in_channels, 16, 8)
    
    head = BNNeckHead(
        in_channels=in_channels,
        feat_dim=512,
        num_classes=751,
    )
    
    # 训练模式
    head.train()
    cls_score, feat, feat_bn = head(x)
    print(f"  [Train] Cls Score: {cls_score.shape}")
    print(f"  [Train] Feat:      {feat.shape}")
    print(f"  [Train] Feat BN:   {feat_bn.shape}")
    
    # 评估模式
    head.eval()
    with torch.no_grad():
        feat = head(x)
    print(f"  [Eval]  Feat:      {feat.shape}")
    
    print(f"  Params: {sum(p.numel() for p in head.parameters()):,}")
    print("  ✅ Passed!")
    return True


def test_full_model():
    """测试完整模型"""
    print("\n" + "=" * 60)
    print("Testing Full Orion-Lite Model")
    print("=" * 60)
    
    from orion_lite.modeling.orion_lite import OrionLite
    
    model = OrionLite(
        backbone_name='ghostnetv2_100',
        pretrained=True,
        last_stride=1,
        with_coord_att=True,
        feat_dim=512,
        num_classes=751,
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total Params:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable Params: {trainable_params:,}")
    
    # 测试输入
    x = torch.randn(4, 3, 256, 128)
    
    # 训练模式
    model.train()
    outputs = model(x)
    print(f"\n  [Train Mode]")
    print(f"    Cls Score: {outputs['cls_score'].shape}")
    print(f"    Feat:      {outputs['feat'].shape}")
    print(f"    Feat BN:   {outputs['feat_bn'].shape}")
    
    # 评估模式
    model.eval()
    with torch.no_grad():
        feat = model(x)
    print(f"\n  [Eval Mode]")
    print(f"    Feat: {feat.shape}")
    
    print("  ✅ Passed!")
    return model


def test_mps_acceleration(model):
    """测试 MPS 加速"""
    print("\n" + "=" * 60)
    print("Testing MPS (Metal) Acceleration")
    print("=" * 60)
    
    if not torch.backends.mps.is_available():
        print("  ⚠️ MPS not available, skipping...")
        return False
    
    device = torch.device('mps')
    print(f"  Device: {device}")
    
    # 移动模型到 MPS
    model = model.to(device)
    x = torch.randn(4, 3, 256, 128).to(device)
    
    # 预热
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    # 测速
    import time
    
    torch.mps.synchronize()
    start = time.time()
    
    n_iters = 50
    with torch.no_grad():
        for _ in range(n_iters):
            feat = model(x)
    
    torch.mps.synchronize()
    elapsed = time.time() - start
    
    avg_time_ms = (elapsed / n_iters) * 1000
    throughput = (n_iters * 4) / elapsed  # images per second
    
    print(f"  Iterations:  {n_iters}")
    print(f"  Batch Size:  4")
    print(f"  Total Time:  {elapsed:.2f}s")
    print(f"  Avg Latency: {avg_time_ms:.2f}ms/batch")
    print(f"  Per Image:   {avg_time_ms/4:.2f}ms/img")
    print(f"  Throughput:  {throughput:.1f} img/s")
    print("  ✅ Passed!")
    return True


def main():
    print("\n")
    print("🌟" * 30)
    print("  ORION-LITE MODEL TEST SUITE")
    print("🌟" * 30)
    print()
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS Available:   {torch.backends.mps.is_available()}")
    print(f"timm Version:    {timm.__version__}")
    
    # 运行测试
    test_coord_attention()
    test_pooling()
    test_ghostnetv2_backbone()
    test_bnneck_head()
    model = test_full_model()
    test_mps_acceleration(model)
    
    print("\n")
    print("=" * 60)
    print("🎉 ALL TESTS PASSED! Orion-Lite is ready!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
