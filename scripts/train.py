#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orion-Lite 训练脚本

使用方法：
    python scripts/train.py --data_root ./data/Market-1501

可选参数：
    --data_root: Market-1501 数据集路径
    --epochs: 训练轮数 (默认 60)
    --batch_size: 批大小 (默认 64)
    --lr: 学习率 (默认 0.00035)
    --backbone: 骨干网络 (默认 'mobilenetv4_conv_small')
    --resume: 恢复训练的检查点路径
    --eval_only: 仅评估模式
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orion_lite.modeling import OrionLite
from orion_lite.data import build_train_dataloader, build_test_dataloader
from orion_lite.losses import CircleLoss, CrossEntropyLabelSmooth
from orion_lite.engine import Trainer, Evaluator
from orion_lite.optim import build_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Orion-Lite Training')
    
    # 数据相关
    parser.add_argument('--data_root', type=str, 
                        default='./data/Market-1501',
                        help='Market-1501 数据集路径')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=60,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小')
    parser.add_argument('--num_instances', type=int, default=4,
                        help='PK 采样中每个 ID 的图片数')
    parser.add_argument('--lr', type=float, default=0.00035,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减')
    
    # 模型相关
    parser.add_argument('--backbone', type=str, 
                        default='mobilenetv4_conv_small',
                        choices=['mobilenetv4_conv_small', 'ghostnetv2_100'],
                        help='骨干网络')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='特征维度')
    
    # 损失函数相关
    parser.add_argument('--label_smooth', type=float, default=0.1,
                        help='Label Smoothing 参数')
    parser.add_argument('--circle_m', type=float, default=0.25,
                        help='Circle Loss 的 m 参数')
    parser.add_argument('--circle_gamma', type=float, default=256,
                        help='Circle Loss 的 gamma 参数')
    
    # 训练控制
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模式')
    parser.add_argument('--eval_period', type=int, default=5,
                        help='评估间隔 (epochs)')
    
    # 输出相关
    parser.add_argument('--output_dir', type=str, 
                        default='./logs/train',
                        help='输出目录')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./checkpoints',
                        help='检查点目录')
    parser.add_argument('--log_period', type=int, default=50,
                        help='日志打印间隔 (iterations)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 打印配置
    print("\n")
    print("=" * 60)
    print("🌟 Orion-Lite v2.0 Training")
    print("=" * 60)
    
    # 设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"📱 Device: Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🎮 Device: CUDA")
    else:
        device = torch.device('cpu')
        print(f"💻 Device: CPU")
    
    print(f"📂 Data: {args.data_root}")
    print(f"🔧 Backbone: {args.backbone}")
    print(f"📊 Batch Size: {args.batch_size} (P={args.batch_size//args.num_instances}, K={args.num_instances})")
    print(f"📈 Epochs: {args.epochs}")
    print(f"📐 Learning Rate: {args.lr}")
    print("=" * 60)
    
    # 检查数据集
    if not os.path.exists(args.data_root):
        print(f"\n❌ 数据集不存在: {args.data_root}")
        print("请先下载 Market-1501 数据集并解压到上述路径")
        print("\n下载链接:")
        print("  - 官方: https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html")
        print("  - Kaggle: https://www.kaggle.com/datasets/pengcw1/market-1501")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 构建训练数据加载器
    print("\n📦 Building DataLoaders...")
    train_loader, num_classes = build_train_dataloader(
        root=args.data_root,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
        num_workers=0,  # macOS MPS 建议设为 0
        random_erasing_prob=0.5,
    )
    print(f"  Training: {len(train_loader)} batches, {num_classes} classes")
    
    # 构建测试数据加载器
    query_loader = build_test_dataloader(
        root=args.data_root,
        mode='query',
        batch_size=args.batch_size * 2,
        num_workers=0,
    )
    gallery_loader = build_test_dataloader(
        root=args.data_root,
        mode='gallery',
        batch_size=args.batch_size * 2,
        num_workers=0,
    )
    print(f"  Query: {len(query_loader)} batches")
    print(f"  Gallery: {len(gallery_loader)} batches")
    
    # 构建模型
    print("\n🏗️ Building Model...")
    model = OrionLite(
        backbone_name=args.backbone,
        pretrained=True,
        feat_dim=args.feat_dim,
        num_classes=num_classes,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable: {trainable_params:,}")
    
    # 构建损失函数
    print("\n📉 Building Loss Functions...")
    id_loss_fn = CrossEntropyLabelSmooth(
        num_classes=num_classes,
        epsilon=args.label_smooth,
    ).to(device)
    
    metric_loss_fn = CircleLoss(
        m=args.circle_m,
        gamma=args.circle_gamma,
    ).to(device)
    
    print(f"  ID Loss: CrossEntropy + LabelSmooth (eps={args.label_smooth})")
    print(f"  Metric Loss: Circle Loss (m={args.circle_m}, gamma={args.circle_gamma})")
    
    # 构建优化器
    print("\n⚡ Building Optimizer...")
    param_groups = model.get_params_groups(
        base_lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print(f"  Optimizer: Adam (lr={args.lr}, wd={args.weight_decay})")
    
    # 构建学习率调度器
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name='warmup_cosine',
        total_epochs=args.epochs,
        warmup_epochs=10,
        warmup_factor=0.01,
    )
    print(f"  Scheduler: Warmup Cosine (warmup=10 epochs)")
    
    # 构建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        id_loss_fn=id_loss_fn,
        metric_loss_fn=metric_loss_fn,
        device=device,
        log_period=args.log_period,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # 构建评估器
    evaluator = Evaluator(
        model=model,
        device=device,
        feat_norm=True,
    )
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 仅评估模式
    if args.eval_only:
        print("\n🔍 Evaluation Only Mode")
        metrics = evaluator.evaluate(query_loader, gallery_loader)
        return
    
    # 训练循环
    print("\n🚀 Starting Training...")
    print("-" * 60)
    
    start_time = time.time()
    start_epoch = trainer.current_epoch + 1
    
    for epoch in range(start_epoch, args.epochs + 1):
        # 训练一个 epoch
        train_metrics = trainer.train_one_epoch(epoch)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.2f}%")
        print(f"  LR: {train_metrics['lr']:.6f}")
        
        # 定期评估
        if epoch % args.eval_period == 0 or epoch == args.epochs:
            print(f"\n📊 Evaluating at epoch {epoch}...")
            eval_metrics = evaluator.evaluate(query_loader, gallery_loader)
            
            # 检查是否为最佳
            is_best = trainer.update_best(eval_metrics['Rank-1'])
            
            # 保存检查点
            trainer.save_checkpoint(epoch, is_best=is_best, metrics=eval_metrics)
    
    # 训练完成
    total_time = time.time() - start_time
    print("\n")
    print("=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print(f"  Total Time: {total_time/3600:.2f} hours")
    print(f"  Best Rank-1: {trainer.best_rank1:.2f}%")
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
