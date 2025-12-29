#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orion-Lite 训练流程快速测试

使用模拟数据验证整个训练管道是否正常工作。
"""

import sys
import os
import tempfile
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, '/Users/joseph/Desktop/【人工智能技术结课项目】ReID重识别')

import torch


def create_mock_dataset(tmpdir: str, num_pids: int = 10, samples_per_pid: int = 5):
    """创建模拟的 Market-1501 数据集"""
    train_dir = os.path.join(tmpdir, 'bounding_box_train')
    query_dir = os.path.join(tmpdir, 'query')
    gallery_dir = os.path.join(tmpdir, 'bounding_box_test')
    
    for d in [train_dir, query_dir, gallery_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 创建模拟图像
    for pid in range(1, num_pids + 1):
        for i in range(samples_per_pid):
            filename = f"{pid:04d}_c1s1_{i:06d}_00.jpg"
            img = Image.new('RGB', (64, 128), color=(pid*25 % 256, 100, 100))
            img.save(os.path.join(train_dir, filename))
            img.save(os.path.join(gallery_dir, filename))
            if i == 0:
                img.save(os.path.join(query_dir, filename))
    
    print(f"  Created mock dataset with {num_pids} PIDs, {samples_per_pid} samples each")
    return tmpdir


def test_training_pipeline():
    """测试完整的训练流程"""
    print("=" * 60)
    print("Testing Complete Training Pipeline")
    print("=" * 60)
    
    from orion_lite.modeling import OrionLite
    from orion_lite.data import build_train_dataloader, build_test_dataloader
    from orion_lite.losses import CircleLoss, CrossEntropyLabelSmooth
    from orion_lite.engine import Trainer, Evaluator
    from orion_lite.optim import build_scheduler
    
    # 设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"  Device: MPS")
    else:
        device = torch.device('cpu')
        print(f"  Device: CPU")
    
    # 创建临时数据集
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n📦 Creating mock dataset...")
        data_root = create_mock_dataset(tmpdir, num_pids=10, samples_per_pid=6)
        
        # 构建数据加载器
        print("\n📦 Building DataLoaders...")
        train_loader, num_classes = build_train_dataloader(
            root=data_root,
            batch_size=16,
            num_instances=4,
            num_workers=0,
        )
        print(f"  Train: {len(train_loader)} batches, {num_classes} classes")
        
        query_loader = build_test_dataloader(
            root=data_root,
            mode='query',
            batch_size=16,
            num_workers=0,
        )
        
        gallery_loader = build_test_dataloader(
            root=data_root,
            mode='gallery',
            batch_size=16,
            num_workers=0,
        )
        
        # 构建模型
        print("\n🏗️ Building Model...")
        model = OrionLite(
            backbone_name='mobilenetv4_conv_small',
            pretrained=True,
            feat_dim=512,
            num_classes=num_classes,
        )
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {total_params:,}")
        
        # 构建损失函数
        print("\n📉 Building Loss Functions...")
        id_loss_fn = CrossEntropyLabelSmooth(
            num_classes=num_classes,
            epsilon=0.1,
        ).to(device)
        
        metric_loss_fn = CircleLoss(m=0.25, gamma=256).to(device)
        print("  ✅ Loss functions ready")
        
        # 构建优化器和调度器
        print("\n⚡ Building Optimizer...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00035)
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_name='warmup_cosine',
            total_epochs=3,
            warmup_epochs=1,
        )
        print("  ✅ Optimizer ready")
        
        # 构建训练器
        checkpoint_dir = os.path.join(tmpdir, 'checkpoints')
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            id_loss_fn=id_loss_fn,
            metric_loss_fn=metric_loss_fn,
            device=device,
            log_period=10,
            checkpoint_dir=checkpoint_dir,
        )
        
        # 构建评估器
        evaluator = Evaluator(
            model=model,
            device=device,
            feat_norm=True,
        )
        
        # 训练 2 个 epoch
        print("\n🚀 Training for 2 epochs...")
        for epoch in range(1, 3):
            metrics = trainer.train_one_epoch(epoch)
            print(f"  Epoch {epoch}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.2f}%")
        
        # 评估
        print("\n📊 Evaluating...")
        eval_metrics = evaluator.evaluate(query_loader, gallery_loader)
        
        # 保存检查点
        print("\n💾 Saving checkpoint...")
        trainer.save_checkpoint(epoch=2, is_best=True, metrics=eval_metrics)
        
        # 验证检查点存在
        assert os.path.exists(os.path.join(checkpoint_dir, 'latest.pth'))
        assert os.path.exists(os.path.join(checkpoint_dir, 'best.pth'))
        print("  ✅ Checkpoints saved")
    
    print("\n" + "=" * 60)
    print("🎉 Training Pipeline Test PASSED!")
    print("=" * 60)
    return True


def main():
    print("\n")
    print("🌟" * 30)
    print("  ORION-LITE TRAINING PIPELINE TEST")
    print("🌟" * 30)
    print()
    
    success = test_training_pipeline()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
