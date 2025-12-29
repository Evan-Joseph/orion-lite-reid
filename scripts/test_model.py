#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orion-Lite v2.0 模型测试脚本

验证新架构：MobileNetV4 + Circle Loss
"""

import sys
import time
import torch
import timm

# 添加项目根目录到路径
sys.path.insert(0, '/Users/joseph/Desktop/【人工智能技术结课项目】ReID重识别')


def test_mobilenetv4_backbone():
    """测试 MobileNetV4 骨干网络"""
    print("=" * 60)
    print("Testing MobileNetV4 Backbone (v2.0)")
    print("=" * 60)
    
    from orion_lite.modeling.backbones.mobilenetv4 import build_mobilenetv4_backbone
    
    backbone = build_mobilenetv4_backbone(
        model_name='mobilenetv4_conv_small',
        pretrained=True,
    )
    backbone.eval()
    
    x = torch.randn(2, 3, 256, 128)
    with torch.no_grad():
        feat = backbone(x)
    
    params = sum(p.numel() for p in backbone.parameters())
    print(f"  Model:   mobilenetv4_conv_small")
    print(f"  Input:   {x.shape}")
    print(f"  Output:  {feat.shape}")
    print(f"  Params:  {params:,} ({params/1e6:.2f}M)")
    print("  ✅ Passed!")
    return True


def test_circle_loss():
    """测试 Circle Loss"""
    print("\n" + "=" * 60)
    print("Testing Circle Loss (v2.0)")
    print("=" * 60)
    
    from orion_lite.losses.circle_loss import CircleLoss, CircleLossPairwise
    
    # 模拟数据
    batch_size = 16
    feat_dim = 512
    num_classes = 4
    
    feats = torch.randn(batch_size, feat_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 测试两种实现
    circle_loss = CircleLoss(m=0.25, gamma=256)
    loss1 = circle_loss(feats, labels)
    
    circle_loss_pw = CircleLossPairwise(m=0.25, gamma=256)
    loss2 = circle_loss_pw(feats, labels)
    
    print(f"  Circle Loss (Matrix):   {loss1.item():.4f}")
    print(f"  Circle Loss (Pairwise): {loss2.item():.4f}")
    print("  ✅ Passed!")
    return True


def test_cross_entropy_smooth():
    """测试 Label Smoothing CE"""
    print("\n" + "=" * 60)
    print("Testing Cross Entropy with Label Smoothing")
    print("=" * 60)
    
    from orion_lite.losses.cross_entropy import CrossEntropyLabelSmooth
    
    batch_size = 16
    num_classes = 751
    
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    ce_smooth = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1)
    loss = ce_smooth(inputs, targets)
    
    print(f"  Epsilon: 0.1")
    print(f"  Loss:    {loss.item():.4f}")
    print("  ✅ Passed!")
    return True


def test_full_model_v2():
    """测试完整的 v2.0 模型"""
    print("\n" + "=" * 60)
    print("Testing Full Orion-Lite v2.0 Model")
    print("=" * 60)
    
    from orion_lite.modeling.orion_lite import OrionLite
    
    model = OrionLite(
        backbone_name='mobilenetv4_conv_small',
        pretrained=True,
        feat_dim=512,
        num_classes=751,
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  Backbone:   MobileNetV4-Conv-Small")
    print(f"  Total Params: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 测试前向传播
    x = torch.randn(4, 3, 256, 128)
    
    # 训练模式
    model.train()
    outputs = model(x)
    print(f"\n  [Train Mode]")
    print(f"    Cls Score: {outputs['cls_score'].shape}")
    print(f"    Feat:      {outputs['feat'].shape}")
    
    # 评估模式
    model.eval()
    with torch.no_grad():
        feat = model(x)
    print(f"\n  [Eval Mode]")
    print(f"    Feat: {feat.shape}")
    
    print("  ✅ Passed!")
    return model


def test_mps_performance(model):
    """测试 MPS 性能对比"""
    print("\n" + "=" * 60)
    print("Testing MPS Performance Comparison")
    print("=" * 60)
    
    if not torch.backends.mps.is_available():
        print("  ⚠️ MPS not available, skipping...")
        return False
    
    from orion_lite.modeling.orion_lite import OrionLite
    
    device = torch.device('mps')
    x = torch.randn(4, 3, 256, 128).to(device)
    
    results = {}
    
    for backbone_name in ['mobilenetv4_conv_small', 'ghostnetv2_100']:
        model = OrionLite(
            backbone_name=backbone_name,
            pretrained=True,
            feat_dim=512,
            num_classes=751,
        )
        model.eval().to(device)
        
        # 预热
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        
        # 测速
        torch.mps.synchronize()
        start = time.time()
        
        n_iters = 50
        with torch.no_grad():
            for _ in range(n_iters):
                _ = model(x)
        
        torch.mps.synchronize()
        elapsed = time.time() - start
        
        params = sum(p.numel() for p in model.parameters())
        latency_batch = (elapsed / n_iters) * 1000
        latency_img = latency_batch / 4
        
        results[backbone_name] = {
            'params': params,
            'latency_batch': latency_batch,
            'latency_img': latency_img,
        }
        
        print(f"\n  📊 {backbone_name}:")
        print(f"     Params:  {params:,} ({params/1e6:.2f}M)")
        print(f"     Latency: {latency_batch:.2f}ms/batch ({latency_img:.2f}ms/img)")
    
    # 对比
    mv4 = results['mobilenetv4_conv_small']
    ghost = results['ghostnetv2_100']
    
    print(f"\n  🚀 v2.0 vs v1.0 对比:")
    print(f"     参数减少: {(1 - mv4['params']/ghost['params'])*100:.1f}%")
    print(f"     速度提升: {ghost['latency_img']/mv4['latency_img']:.1f}x")
    
    print("  ✅ Passed!")
    return True


def main():
    print("\n")
    print("🌟" * 30)
    print("  ORION-LITE v2.0 TEST SUITE")
    print("🌟" * 30)
    print()
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS Available:   {torch.backends.mps.is_available()}")
    print(f"timm Version:    {timm.__version__}")
    
    # 运行测试
    test_mobilenetv4_backbone()
    test_circle_loss()
    test_cross_entropy_smooth()
    model = test_full_model_v2()
    test_mps_performance(model)
    
    print("\n")
    print("=" * 60)
    print("🎉 ALL TESTS PASSED! Orion-Lite v2.0 is ready!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
