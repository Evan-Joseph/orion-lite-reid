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
from orion_lite.losses import CircleLoss, CrossEntropyLabelSmooth, TripletLoss, CenterLoss
from orion_lite.engine import Trainer, Evaluator
from orion_lite.optim import build_scheduler
from orion_lite.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Orion-Lite Training')
    
    # ... (args 省略，保持不变) ...
    parser.add_argument('--data_root', type=str, 
                        default='./data/Market-1501',
                        help='Market-1501 数据集路径')
    # 训练相关
    parser.add_argument('--epochs', type=int, default=120,
                        help='训练轮数 (建议 120)')
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
    parser.add_argument('--loss_type', type=str, default='triplet',
                        choices=['triplet', 'circle'],
                        help='度量损失类型')
    parser.add_argument('--label_smooth', type=float, default=0.1,
                        help='Label Smoothing 参数')
    parser.add_argument('--margin', type=float, default=0.3,
                        help='Triplet Loss Margin')
    parser.add_argument('--circle_m', type=float, default=0.25,
                        help='Circle Loss 的 m 参数')
    parser.add_argument('--circle_gamma', type=float, default=64,
                        help='Circle Loss 的 gamma 参数')
    parser.add_argument('--id_loss_weight', type=float, default=1.0,
                        help='ID Loss 权重')
    parser.add_argument('--metric_loss_weight', type=float, default=1.0,
                        help='Metric Loss 权重')
    
    # 头部网络相关
    parser.add_argument('--head_type', type=str, default='bnneck',
                        choices=['bnneck', 'part'],
                        help='头部类型: bnneck (全局) 或 part (全局+局部)')
    parser.add_argument('--num_parts', type=int, default=2,
                        help='Part-based 切片数量')
    parser.add_argument('--pool_type', type=str, default='avg',
                        choices=['avg', 'gem'],
                        help='池化类型')
    
    # Center Loss 相关
    parser.add_argument('--use_center_loss', action='store_true',
                        help='启用 Center Loss')
    parser.add_argument('--center_loss_weight', type=float, default=0.0005,
                        help='Center Loss 权重')
    parser.add_argument('--center_lr', type=float, default=0.5,
                        help='Center Loss 学习率')
    
    parser.add_argument('--resume', type=str, default='', help='恢复训练')
    parser.add_argument('--eval_only', action='store_true', help='仅评估')
    parser.add_argument('--eval_period', type=int, default=5, help='评估间隔')
    
    parser.add_argument('--output_dir', type=str, default='./logs/train', help='输出目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='检查点目录')
    parser.add_argument('--log_period', type=int, default=50, help='日志频率')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 初始化 Logger
    logger = setup_logger(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("🌟 Orion-Lite v2.0 Training")
    logger.info("=" * 60)
    
    # 设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info(f"📱 Device: Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"🎮 Device: CUDA")
    else:
        device = torch.device('cpu')
        logger.info(f"💻 Device: CPU")
    
    logger.info(f"📂 Data: {args.data_root}")
    logger.info(f"🔧 Backbone: {args.backbone}")
    logger.info(f"📊 Batch Size: {args.batch_size} (P={args.batch_size//args.num_instances}, K={args.num_instances})")
    logger.info(f"📈 Epochs: {args.epochs}")
    logger.info(f"📐 Learning Rate: {args.lr}")
    logger.info(f"⚖️  Loss Type: {args.loss_type}")
    logger.info("=" * 60)
    
    # 检查数据集
    if not os.path.exists(args.data_root):
        logger.error(f"❌ 数据集不存在: {args.data_root}")
        return
    
    # 构建训练数据加载器
    logger.info("📦 Building DataLoaders...")
    train_loader, num_classes = build_train_dataloader(
        root=args.data_root,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
        num_workers=0,  # macOS MPS 建议设为 0
        random_erasing_prob=0.5,
    )
    logger.info(f"  Training: {len(train_loader)} batches, {num_classes} classes")
    
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
    logger.info(f"  Query: {len(query_loader)} batches")
    logger.info(f"  Gallery: {len(gallery_loader)} batches")
    
    # 构建模型
    logger.info("🏗️ Building Model...")
    model = OrionLite(
        backbone_name=args.backbone,
        pretrained=True,
        feat_dim=args.feat_dim,
        num_classes=num_classes,
        pool_type=args.pool_type,
        head_type=args.head_type,
        num_parts=args.num_parts,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total Params: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"  Trainable: {trainable_params:,}")
    
    # 构建损失函数
    logger.info("📉 Building Loss Functions...")
    id_loss_fn = CrossEntropyLabelSmooth(
        num_classes=num_classes,
        epsilon=args.label_smooth,
    ).to(device)
    
    if args.loss_type == 'circle':
        metric_loss_fn = CircleLoss(
            m=args.circle_m,
            gamma=args.circle_gamma,
        ).to(device)
        logger.info(f"  Metric Loss: Circle Loss (m={args.circle_m}, gamma={args.circle_gamma})")
    else:
        metric_loss_fn = TripletLoss(margin=args.margin).to(device)
        logger.info(f"  Metric Loss: Triplet Loss (margin={args.margin})")
        
    logger.info(f"  ID Loss: CrossEntropy + LabelSmooth (eps={args.label_smooth})")
    
    # 构建 Center Loss (可选)
    if args.use_center_loss:
        center_loss_fn = CenterLoss(
            num_classes=num_classes,
            feat_dim=model.feat_dim,
        ).to(device)
        logger.info(f"  Center Loss: Enabled (weight={args.center_loss_weight}, lr={args.center_lr})")
    else:
        center_loss_fn = None
    
    # 构建优化器
    logger.info("⚡ Building Optimizer...")
    param_groups = model.get_params_groups(
        base_lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    logger.info(f"  Optimizer: Adam (lr={args.lr}, wd={args.weight_decay})")
    
    # 构建学习率调度器 - Warmup Cosine (优化版)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name='warmup_cosine',
        total_epochs=args.epochs,
        warmup_epochs=10,
        warmup_factor=0.1,  # 更激进的预热 (之前是 0.01)
    )
    logger.info(f"  Scheduler: Warmup Cosine (warmup=10, warmup_factor=0.1)")
    
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
        id_loss_weight=args.id_loss_weight,
        metric_loss_weight=args.metric_loss_weight,
        center_loss_fn=center_loss_fn,
        center_loss_weight=args.center_loss_weight,
        center_lr=args.center_lr,
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
        logger.info("🔍 Evaluation Only Mode")
        metrics = evaluator.evaluate(query_loader, gallery_loader)
        logger.info(f"Results: {metrics}")
        return
    
    # 训练循环
    logger.info("🚀 Starting Training...")
    logger.info("-" * 60)
    
    start_time = time.time()
    start_epoch = trainer.current_epoch + 1
    
    for epoch in range(start_epoch, args.epochs + 1):
        # 训练一个 epoch
        train_metrics = trainer.train_one_epoch(epoch)
        
        logger.info(f"Epoch {epoch} Summary: "
                    f"Loss={train_metrics['loss']:.4f}, "
                    f"Acc={train_metrics['accuracy']:.2f}%, "
                    f"LR={train_metrics['lr']:.6f}")
        
        # 定期评估
        if epoch % args.eval_period == 0 or epoch == args.epochs:
            logger.info(f"📊 Evaluating at epoch {epoch}...")
            eval_metrics = evaluator.evaluate(query_loader, gallery_loader)
            
            # 记录详细指标
            logger.info(f"  Rank-1: {eval_metrics['Rank-1']:.2f}%")
            logger.info(f"  mAP:    {eval_metrics['mAP']:.2f}%")
            
            # 检查是否为最佳
            is_best = trainer.update_best(eval_metrics['Rank-1'])
            if is_best:
                logger.info(f"  🔥 New Best Rank-1: {trainer.best_rank1:.2f}%")
            
            # 保存检查点
            trainer.save_checkpoint(epoch, is_best=is_best, metrics=eval_metrics)
    
    # 训练完成
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("🎉 Training Complete!")
    logger.info("=" * 60)
    logger.info(f"  Total Time: {total_time/3600:.2f} hours")
    logger.info(f"  Best Rank-1: {trainer.best_rank1:.2f}%")
    logger.info(f"  Checkpoints: {args.checkpoint_dir}")
    logger.info(f"  Logs: {os.path.join(args.output_dir, 'train_log.txt')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
