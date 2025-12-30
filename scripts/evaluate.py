#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本 - 支持 Re-ranking

使用方法:
    # 普通评估
    python scripts/evaluate.py --checkpoint checkpoints/best.pth
    
    # 带 Re-ranking 评估
    python scripts/evaluate.py --checkpoint checkpoints/best.pth --rerank
"""

import argparse
import os
import sys
import time

import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orion_lite.modeling import OrionLite
from orion_lite.data import build_test_dataloader
from orion_lite.engine import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Orion-Lite Evaluation')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data_root', type=str, 
                        default='./data/Market-1501',
                        help='数据集路径')
    
    # 模型相关
    parser.add_argument('--backbone', type=str, 
                        default='mobilenetv4_conv_small',
                        help='骨干网络')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='特征维度')
    
    # Re-ranking
    parser.add_argument('--rerank', action='store_true',
                        help='启用 K-Reciprocal Re-ranking')
    parser.add_argument('--rerank_k1', type=int, default=20,
                        help='Re-ranking 参数 k1')
    parser.add_argument('--rerank_k2', type=int, default=6,
                        help='Re-ranking 参数 k2')
    parser.add_argument('--rerank_lambda', type=float, default=0.3,
                        help='Re-ranking 原始距离权重')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批大小')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🔍 Orion-Lite Evaluation")
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
    print(f"📦 Checkpoint: {args.checkpoint}")
    print(f"🔄 Re-ranking: {'Enabled' if args.rerank else 'Disabled'}")
    print("=" * 60)
    
    # 检查检查点
    if not os.path.exists(args.checkpoint):
        print(f"❌ 检查点不存在: {args.checkpoint}")
        return
    
    # 加载检查点获取类别数
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # 从权重推断类别数
    classifier_weight = checkpoint['model_state_dict'].get(
        'head.classifier.weight', 
        checkpoint['model_state_dict'].get('head.classifier.0.weight')
    )
    if classifier_weight is not None:
        num_classes = classifier_weight.shape[0]
    else:
        num_classes = 751  # Market-1501 默认
    
    print(f"📊 Classes: {num_classes}")
    
    # 构建模型
    print("🏗️ Building Model...")
    model = OrionLite(
        backbone_name=args.backbone,
        pretrained=False,  # 不需要预训练，我们会加载检查点
        feat_dim=args.feat_dim,
        num_classes=num_classes,
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint and checkpoint['metrics']:
        print(f"   Previous metrics: {checkpoint['metrics']}")
    
    # 构建数据加载器
    print("📦 Building DataLoaders...")
    query_loader = build_test_dataloader(
        root=args.data_root,
        mode='query',
        batch_size=args.batch_size,
        num_workers=0,
    )
    gallery_loader = build_test_dataloader(
        root=args.data_root,
        mode='gallery',
        batch_size=args.batch_size,
        num_workers=0,
    )
    
    # 构建评估器
    evaluator = Evaluator(
        model=model,
        device=device,
        feat_norm=True,
    )
    
    # 评估
    print("🚀 Starting Evaluation...")
    start_time = time.time()
    
    metrics = evaluator.evaluate(
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        rerank=args.rerank,
        rerank_k1=args.rerank_k1,
        rerank_k2=args.rerank_k2,
        rerank_lambda=args.rerank_lambda,
    )
    
    elapsed = time.time() - start_time
    
    print("=" * 60)
    print("📊 Final Results")
    print("=" * 60)
    print(f"  Rank-1:  {metrics['Rank-1']:.2f}%")
    print(f"  Rank-5:  {metrics['Rank-5']:.2f}%")
    print(f"  Rank-10: {metrics['Rank-10']:.2f}%")
    print(f"  mAP:     {metrics['mAP']:.2f}%")
    print(f"  Time:    {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
