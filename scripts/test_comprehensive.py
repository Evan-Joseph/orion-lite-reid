#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orion-Lite 全面测试脚本

测试所有核心组件，包括用多摄像头数据验证评估逻辑。
"""

import sys
import os
import tempfile
from PIL import Image
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, '/Users/joseph/Desktop/【人工智能技术结课项目】ReID重识别')

import torch


def test_evaluator_logic():
    """
    专门测试评估器的 CMC/mAP 计算逻辑
    使用已知答案的模拟数据验证正确性
    """
    print("=" * 60)
    print("Testing Evaluator Logic with Known Answer")
    print("=" * 60)
    
    from orion_lite.engine.evaluator import Evaluator
    
    # 创建一个简单的评估器（不需要真实模型）
    evaluator = Evaluator(model=None, device=torch.device('cpu'))
    
    # 构造已知答案的测试数据
    # 场景：3 个 query，10 个 gallery
    # Query 0 (pid=1, cam=1) 应该匹配 Gallery 中 pid=1 的样本（但排除 cam=1）
    # Query 1 (pid=2, cam=2) 应该匹配 Gallery 中 pid=2 的样本（但排除 cam=2）
    # Query 2 (pid=3, cam=1) 应该匹配 Gallery 中 pid=3 的样本（但排除 cam=1）
    
    num_query = 3
    num_gallery = 10
    
    # Query 信息
    q_pids = np.array([1, 2, 3])
    q_camids = np.array([1, 2, 1])
    
    # Gallery 信息
    # Gallery 包含:
    # - pid=1: cam=1 (排除), cam=2 (匹配)
    # - pid=2: cam=1 (匹配), cam=2 (排除)
    # - pid=3: cam=1 (排除), cam=2 (匹配)
    # - pid=4,5: 干扰项
    g_pids = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    g_camids = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    
    # 构造距离矩阵
    # 我们让每个 query 与其正确的 gallery 样本距离最小
    distmat = np.ones((num_query, num_gallery)) * 10.0  # 默认都是大距离
    
    # Query 0 (pid=1): 让 gallery index 1 (pid=1, cam=2) 距离最小
    distmat[0, 1] = 0.1  # 正确匹配
    distmat[0, 0] = 0.05  # 同 cam，应被排除
    
    # Query 1 (pid=2): 让 gallery index 2 (pid=2, cam=1) 距离最小
    distmat[1, 2] = 0.1  # 正确匹配
    distmat[1, 3] = 0.05  # 同 cam，应被排除
    
    # Query 2 (pid=3): 让 gallery index 5 (pid=3, cam=2) 距离最小
    distmat[2, 5] = 0.1  # 正确匹配
    distmat[2, 4] = 0.05  # 同 cam，应被排除
    
    print("  构造的测试场景:")
    print(f"    Query PIDs:   {q_pids}")
    print(f"    Query CamIDs: {q_camids}")
    print(f"    Gallery PIDs:   {g_pids}")
    print(f"    Gallery CamIDs: {g_camids}")
    
    # 计算 CMC 和 mAP
    cmc, mAP = evaluator.compute_metrics(
        distmat=distmat,
        q_pids=q_pids,
        g_pids=g_pids,
        q_camids=q_camids,
        g_camids=g_camids,
        max_rank=10,
    )
    
    print(f"\n  计算结果:")
    print(f"    Rank-1: {cmc[0]*100:.2f}%")
    print(f"    mAP:    {mAP*100:.2f}%")
    
    # 验证结果
    # 预期：所有 3 个 query 都应该在 rank-1 找到正确匹配
    expected_rank1 = 1.0  # 100%
    expected_mAP = 1.0    # 100% (每个 query 只有一个正确匹配，且都在 rank-1)
    
    if abs(cmc[0] - expected_rank1) < 0.01 and abs(mAP - expected_mAP) < 0.01:
        print(f"\n  ✅ 评估逻辑正确! (预期 Rank-1=100%, mAP=100%)")
        return True
    else:
        print(f"\n  ❌ 评估结果不符预期!")
        print(f"    预期: Rank-1={expected_rank1*100}%, mAP={expected_mAP*100}%")
        return False


def test_single_camera_scenario():
    """
    测试单摄像头场景（应该触发警告并返回 0）
    """
    print("\n" + "=" * 60)
    print("Testing Single Camera Scenario (Expected: 0%)")
    print("=" * 60)
    
    from orion_lite.engine.evaluator import Evaluator
    
    evaluator = Evaluator(model=None, device=torch.device('cpu'))
    
    # 所有样本来自同一摄像头
    num_query = 3
    num_gallery = 10
    
    q_pids = np.array([1, 2, 3])
    q_camids = np.array([1, 1, 1])  # 全是 cam=1
    
    g_pids = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    g_camids = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 全是 cam=1
    
    distmat = np.random.rand(num_query, num_gallery)
    
    print("  所有样本 camid=1，根据 Market-1501 协议会被排除")
    
    cmc, mAP = evaluator.compute_metrics(
        distmat=distmat,
        q_pids=q_pids,
        g_pids=g_pids,
        q_camids=q_camids,
        g_camids=g_camids,
        max_rank=10,
    )
    
    print(f"\n  结果: Rank-1={cmc[0]*100:.2f}%, mAP={mAP*100:.2f}%")
    
    if cmc[0] == 0 and mAP == 0:
        print("  ✅ 正确! 单摄像头场景下无有效匹配")
        return True
    else:
        print("  ❌ 错误! 单摄像头应返回 0%")
        return False


def test_multi_camera_mock_dataset():
    """
    使用多摄像头模拟数据集测试完整流程
    """
    print("\n" + "=" * 60)
    print("Testing Multi-Camera Mock Dataset Pipeline")
    print("=" * 60)
    
    from orion_lite.modeling import OrionLite
    from orion_lite.data import Market1501, build_transforms
    from orion_lite.data.dataloader import collate_fn
    from orion_lite.engine import Evaluator
    from torch.utils.data import DataLoader
    
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"  Device: {device}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建多摄像头模拟数据集
        train_dir = os.path.join(tmpdir, 'bounding_box_train')
        query_dir = os.path.join(tmpdir, 'query')
        gallery_dir = os.path.join(tmpdir, 'bounding_box_test')
        
        for d in [train_dir, query_dir, gallery_dir]:
            os.makedirs(d, exist_ok=True)
        
        # 创建数据：5 个 PID，每个 PID 在 2 个摄像头各有图片
        num_pids = 5
        for pid in range(1, num_pids + 1):
            for cam in [1, 2]:  # 两个摄像头
                for i in range(3):  # 每个摄像头 3 张
                    filename = f"{pid:04d}_c{cam}s1_{i:06d}_00.jpg"
                    img = Image.new('RGB', (64, 128), color=(pid*50, cam*100, i*50))
                    
                    # 训练集和 gallery 都有
                    img.save(os.path.join(train_dir, filename))
                    img.save(os.path.join(gallery_dir, filename))
                    
                    # Query: 每个 PID 的 cam=1 第一张作为 query
                    if cam == 1 and i == 0:
                        img.save(os.path.join(query_dir, filename))
        
        print(f"  Created dataset: {num_pids} PIDs × 2 cameras")
        
        # 构建数据集
        transform = build_transforms(is_train=False)
        
        query_dataset = Market1501(root=tmpdir, mode='query', transform=transform, verbose=False)
        gallery_dataset = Market1501(root=tmpdir, mode='gallery', transform=transform, verbose=False)
        
        print(f"  Query: {len(query_dataset)} samples")
        print(f"  Gallery: {len(gallery_dataset)} samples")
        
        # 验证摄像头分布
        query_cams = set([s[2] for s in query_dataset.samples])
        gallery_cams = set([s[2] for s in gallery_dataset.samples])
        print(f"  Query cameras: {query_cams}")
        print(f"  Gallery cameras: {gallery_cams}")
        
        # 构建 DataLoader
        query_loader = DataLoader(
            query_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        gallery_loader = DataLoader(
            gallery_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        
        # 构建模型
        model = OrionLite(
            backbone_name='mobilenetv4_conv_small',
            pretrained=True,
            feat_dim=512,
            num_classes=num_pids,
        )
        model = model.to(device)
        model.eval()
        
        # 评估
        evaluator = Evaluator(model=model, device=device, feat_norm=True)
        
        print("\n  Running evaluation...")
        metrics = evaluator.evaluate(query_loader, gallery_loader)
        
        # 验证结果不为 0（因为有多摄像头）
        if metrics['Rank-1'] > 0:
            print(f"\n  ✅ 多摄像头场景评估正常工作!")
            return True
        else:
            print(f"\n  ⚠️ Rank-1 为 0，检查评估逻辑")
            return False


def test_all_components():
    """运行所有组件测试"""
    print("\n" + "=" * 60)
    print("Testing All Core Components")
    print("=" * 60)
    
    results = {}
    
    # 1. 模型测试
    print("\n📦 Testing Model...")
    try:
        from orion_lite.modeling import OrionLite
        model = OrionLite(backbone_name='mobilenetv4_conv_small', pretrained=True, num_classes=10)
        x = torch.randn(2, 3, 256, 128)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 512), f"Output shape mismatch: {out.shape}"
        results['Model'] = True
        print("  ✅ Model OK")
    except Exception as e:
        results['Model'] = False
        print(f"  ❌ Model Error: {e}")
    
    # 2. 数据加载测试
    print("\n📦 Testing Data Loading...")
    try:
        from orion_lite.data import build_transforms, RandomErasing
        transform = build_transforms(is_train=True)
        img = Image.new('RGB', (100, 200), 'red')
        out = transform(img)
        assert out.shape == (3, 256, 128), f"Transform output mismatch: {out.shape}"
        results['Data'] = True
        print("  ✅ Data Loading OK")
    except Exception as e:
        results['Data'] = False
        print(f"  ❌ Data Error: {e}")
    
    # 3. 损失函数测试
    print("\n📦 Testing Loss Functions...")
    try:
        from orion_lite.losses import CircleLoss, CrossEntropyLabelSmooth
        
        feats = torch.randn(16, 512)
        labels = torch.randint(0, 4, (16,))
        
        circle_loss = CircleLoss()
        loss1 = circle_loss(feats, labels)
        
        logits = torch.randn(16, 10)
        ce_loss = CrossEntropyLabelSmooth(num_classes=10)
        loss2 = ce_loss(logits, labels)
        
        assert loss1.item() >= 0, "Circle Loss should be non-negative"
        assert loss2.item() >= 0, "CE Loss should be non-negative"
        results['Losses'] = True
        print("  ✅ Loss Functions OK")
    except Exception as e:
        results['Losses'] = False
        print(f"  ❌ Loss Error: {e}")
    
    # 4. 学习率调度器测试
    print("\n📦 Testing Scheduler...")
    try:
        from orion_lite.optim import build_scheduler
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = build_scheduler(optimizer, 'warmup_cosine', total_epochs=60, warmup_epochs=10)
        
        lrs = [scheduler.get_last_lr()[0]]
        for _ in range(60):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
        
        # 检查 warmup: lr 应该先增加
        assert lrs[10] > lrs[0], "LR should increase during warmup"
        # 检查 cosine: lr 应该在 warmup 后下降
        assert lrs[-1] < lrs[10], "LR should decrease after warmup"
        results['Scheduler'] = True
        print("  ✅ Scheduler OK")
    except Exception as e:
        results['Scheduler'] = False
        print(f"  ❌ Scheduler Error: {e}")
    
    return results


def main():
    print("\n")
    print("🌟" * 30)
    print("  ORION-LITE COMPREHENSIVE TEST SUITE")
    print("🌟" * 30)
    print()
    
    all_passed = True
    
    # 1. 组件测试
    component_results = test_all_components()
    for name, passed in component_results.items():
        all_passed &= passed
    
    # 2. 评估逻辑测试（已知答案）
    all_passed &= test_evaluator_logic()
    
    # 3. 单摄像头场景测试
    all_passed &= test_single_camera_scenario()
    
    # 4. 多摄像头完整流程测试
    all_passed &= test_multi_camera_mock_dataset()
    
    # 总结
    print("\n")
    print("=" * 60)
    if all_passed:
        print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("   评估逻辑正确:")
        print("   - 已知答案测试: 100% Rank-1/mAP ✅")
        print("   - 单摄像头场景: 正确返回 0% ✅")
        print("   - 多摄像头场景: 正常工作 ✅")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
