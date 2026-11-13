#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orion-Lite 数据模块测试脚本

测试 Market-1501 数据集加载、数据变换和采样器。
"""

import sys
import os

# 添加项目根目录到路径
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


def test_transforms():
    """测试数据变换"""
    print("=" * 60)
    print("Testing Data Transforms")
    print("=" * 60)
    
    from PIL import Image
    from orion_lite.data.transforms import build_transforms, RandomErasing
    
    # 创建测试图像
    img = Image.new('RGB', (100, 200), color='blue')
    
    # 测试训练变换
    train_transform = build_transforms(
        height=256,
        width=128,
        is_train=True,
        random_erasing_prob=0.5,
    )
    train_out = train_transform(img)
    print(f"  Train Transform: {img.size} -> {tuple(train_out.shape)}")
    assert train_out.shape == (3, 256, 128), "输出尺寸错误"
    
    # 测试测试变换
    test_transform = build_transforms(
        height=256,
        width=128,
        is_train=False,
    )
    test_out = test_transform(img)
    print(f"  Test Transform:  {img.size} -> {tuple(test_out.shape)}")
    
    # 测试 Random Erasing
    import torch
    re = RandomErasing(probability=1.0)
    original = torch.randn(3, 256, 128)
    erased = re(original.clone())
    is_different = not torch.equal(original, erased)
    print(f"  Random Erasing:  Applied = {is_different}")
    
    print("  ✅ Passed!")
    return True


def test_pk_sampler():
    """测试 PK Sampler"""
    print("\n" + "=" * 60)
    print("Testing PK Sampler")
    print("=" * 60)
    
    from collections import defaultdict
    from orion_lite.data.samplers import RandomIdentitySampler
    
    # 模拟数据集
    class MockDataset:
        def __init__(self):
            self.samples = []
            # 创建 20 个 PID，每个有 3-10 张图片
            import random
            idx = 0
            for pid in range(20):
                count = random.randint(3, 10)
                for _ in range(count):
                    self.samples.append((f"img_{idx}.jpg", pid, 1))
                    idx += 1
        
        def __len__(self):
            return len(self.samples)
        
        def get_pid_to_indices(self):
            pid_to_indices = defaultdict(list)
            for idx, (_, pid, _) in enumerate(self.samples):
                pid_to_indices[pid].append(idx)
            return dict(pid_to_indices)
    
    dataset = MockDataset()
    print(f"  Mock Dataset: {len(dataset)} samples, 20 PIDs")
    
    # 创建 PK Sampler: P=8, K=4 -> batch=32
    sampler = RandomIdentitySampler(
        data_source=dataset,
        batch_size=32,
        num_instances=4,
    )
    
    # 采样
    indices = list(sampler)
    print(f"  Sampled: {len(indices)} indices")
    
    # 验证 batch 中的 PID 分布
    batch_size = 32
    for i in range(0, min(64, len(indices)), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_pids = [dataset.samples[idx][1] for idx in batch_indices]
        unique_pids = len(set(batch_pids))
        print(f"  Batch {i//batch_size}: {unique_pids} unique PIDs")
    
    print("  ✅ Passed!")
    return True


def test_market1501_mock():
    """测试 Market-1501 数据集（模拟模式）"""
    print("\n" + "=" * 60)
    print("Testing Market-1501 Dataset (Mock)")
    print("=" * 60)
    
    import tempfile
    import os
    from PIL import Image
    from orion_lite.data.datasets import Market1501
    
    # 创建临时目录模拟 Market-1501 结构
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建子目录
        train_dir = os.path.join(tmpdir, 'bounding_box_train')
        query_dir = os.path.join(tmpdir, 'query')
        gallery_dir = os.path.join(tmpdir, 'bounding_box_test')
        
        os.makedirs(train_dir)
        os.makedirs(query_dir)
        os.makedirs(gallery_dir)
        
        # 创建模拟图像文件
        # 格式: 0001_c1s1_001051_00.jpg
        for pid in range(1, 6):  # 5 个 PID
            for cam in range(1, 3):  # 2 个摄像头
                for frame in range(1, 4):  # 每个 3 张图
                    filename = f"{pid:04d}_c{cam}s1_{frame:06d}_00.jpg"
                    img = Image.new('RGB', (64, 128), color=(pid*50, cam*50, frame*50))
                    img.save(os.path.join(train_dir, filename))
                    img.save(os.path.join(gallery_dir, filename))
                    if frame == 1:  # query 只取第一张
                        img.save(os.path.join(query_dir, filename))
        
        print(f"  Created mock data in: {tmpdir}")
        
        # 测试训练集
        train_dataset = Market1501(root=tmpdir, mode='train', verbose=False)
        print(f"  Train: {len(train_dataset)} samples, {train_dataset.num_classes} classes")
        
        # 测试查询集
        query_dataset = Market1501(root=tmpdir, mode='query', verbose=False)
        print(f"  Query: {len(query_dataset)} samples")
        
        # 测试样本获取
        sample = train_dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        assert 'img' in sample, "缺少 'img' 键"
        assert 'pid' in sample, "缺少 'pid' 键"
        assert 'camid' in sample, "缺少 'camid' 键"
        
        # 验证 pid2label 映射
        assert train_dataset.num_classes == 5, f"类别数错误: {train_dataset.num_classes}"
        
        print("  ✅ Passed!")
        return True


def test_dataloader_mock():
    """测试 DataLoader（模拟模式）"""
    print("\n" + "=" * 60)
    print("Testing DataLoader (Mock)")
    print("=" * 60)
    
    import tempfile
    import os
    from PIL import Image
    from orion_lite.data import build_train_dataloader
    
    # 创建临时目录模拟 Market-1501
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = os.path.join(tmpdir, 'bounding_box_train')
        os.makedirs(train_dir)
        
        # 创建模拟图像
        for pid in range(1, 11):  # 10 个 PID
            for i in range(5):  # 每个 5 张
                filename = f"{pid:04d}_c1s1_{i:06d}_00.jpg"
                img = Image.new('RGB', (64, 128), color=(pid*25, 100, 100))
                img.save(os.path.join(train_dir, filename))
        
        print(f"  Created 50 mock images")
        
        # 构建 DataLoader
        train_loader, num_classes = build_train_dataloader(
            root=tmpdir,
            batch_size=16,
            num_instances=4,
            num_workers=0,  # macOS 必须设为 0
            random_erasing_prob=0.5,
        )
        
        print(f"  Num classes: {num_classes}")
        print(f"  Num batches: {len(train_loader)}")
        
        # 获取一个 batch
        batch = next(iter(train_loader))
        print(f"  Batch 'img' shape: {batch['img'].shape}")
        print(f"  Batch 'pid': {batch['pid'].tolist()}")
        
        assert batch['img'].shape == (16, 3, 256, 128), "Batch shape 错误"
        assert len(batch['pid']) == 16, "PID 数量错误"
        
        print("  ✅ Passed!")
        return True


def main():
    print("\n")
    print("🌟" * 30)
    print("  ORION-LITE DATA MODULE TEST")
    print("🌟" * 30)
    print()
    
    all_passed = True
    
    # 运行测试
    all_passed &= test_transforms()
    all_passed &= test_pk_sampler()
    all_passed &= test_market1501_mock()
    all_passed &= test_dataloader_mock()
    
    print("\n")
    print("=" * 60)
    if all_passed:
        print("🎉 ALL DATA MODULE TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)
    print()
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
