# -*- coding: utf-8 -*-
"""
训练器 (Trainer)

实现 Orion-Lite 的训练逻辑，包括：
- 前向传播与损失计算
- 反向传播与参数更新
- 学习率调度
- 日志记录与检查点保存
"""

import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class Trainer:
    """
    Orion-Lite 训练器
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        id_loss_fn: ID 损失函数 (分类)
        metric_loss_fn: 度量损失函数 (Circle Loss)
        device: 训练设备
        log_period: 日志打印间隔
        checkpoint_dir: 检查点保存目录
        id_loss_weight: ID 损失权重
        metric_loss_weight: 度量损失权重
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        id_loss_fn: nn.Module,
        metric_loss_fn: nn.Module,
        device: torch.device,
        log_period: int = 50,
        checkpoint_dir: str = './checkpoints',
        id_loss_weight: float = 1.0,
        metric_loss_weight: float = 1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.id_loss_fn = id_loss_fn
        self.metric_loss_fn = metric_loss_fn
        self.device = device
        self.log_period = log_period
        self.checkpoint_dir = checkpoint_dir
        self.id_loss_weight = id_loss_weight
        self.metric_loss_weight = metric_loss_weight
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.best_rank1 = 0.0
        
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            epoch: 当前 epoch 编号
            
        Returns:
            metrics: 训练指标字典
        """
        self.model.train()
        self.current_epoch = epoch
        
        # 统计
        total_loss = 0.0
        total_id_loss = 0.0
        total_metric_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 获取数据
            imgs = batch['img'].to(self.device)
            pids = batch['pid'].to(self.device)
            
            # 前向传播
            outputs = self.model(imgs, pids)
            cls_score = outputs['cls_score']
            feat = outputs['feat']
            
            # 计算损失
            id_loss = self.id_loss_fn(cls_score, pids)
            metric_loss = self.metric_loss_fn(feat, pids)
            
            loss = self.id_loss_weight * id_loss + self.metric_loss_weight * metric_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_id_loss += id_loss.item()
            total_metric_loss += metric_loss.item()
            
            # 计算准确率
            _, predicted = cls_score.max(1)
            total_correct += predicted.eq(pids).sum().item()
            total_samples += pids.size(0)
            
            # 日志
            if (batch_idx + 1) % self.log_period == 0:
                elapsed = time.time() - start_time
                speed = (batch_idx + 1) * imgs.size(0) / elapsed
                
                print(f"Epoch [{epoch}][{batch_idx + 1}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"(ID: {id_loss.item():.4f}, Metric: {metric_loss.item():.4f}) "
                      f"Acc: {100. * total_correct / total_samples:.2f}% "
                      f"Speed: {speed:.1f} img/s "
                      f"LR: {self.get_lr():.6f}")
        
        # 更新学习率
        self.scheduler.step()
        
        # 计算平均指标
        num_batches = len(self.train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'id_loss': total_id_loss / num_batches,
            'metric_loss': total_metric_loss / num_batches,
            'accuracy': 100. * total_correct / total_samples,
            'lr': self.get_lr(),
        }
        
        return metrics
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def save_checkpoint(
        self, 
        epoch: int, 
        is_best: bool = False,
        metrics: Optional[Dict] = None
    ):
        """
        保存检查点
        
        Args:
            epoch: 当前 epoch
            is_best: 是否为最佳模型
            metrics: 训练指标
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_rank1': self.best_rank1,
            'metrics': metrics,
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存定期检查点
        if epoch % 10 == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model with Rank-1: {self.best_rank1:.2f}%")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_rank1 = checkpoint.get('best_rank1', 0.0)
        
        print(f"✅ Loaded checkpoint from epoch {self.current_epoch}")
    
    def update_best(self, rank1: float) -> bool:
        """
        更新最佳指标
        
        Args:
            rank1: 当前 Rank-1 准确率
            
        Returns:
            is_best: 是否为新的最佳
        """
        is_best = rank1 > self.best_rank1
        if is_best:
            self.best_rank1 = rank1
        return is_best
